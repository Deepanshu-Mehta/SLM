"""Phase 12 jobs on Modal: RLAIF for SLM-125M-RAFT ("slm125") and Gemma 3 270M RAFT ("gemma3") (docs/13-rlaif.md).

    modal run rlaif_app.py::upload_prompts                                     # .smoke/rlaif/prompts.jsonl -> Volume
    modal run --detach rlaif_app.py::headroom_check --confirm --max-usd 0.6   # 12.1, 1 x L40S per track
    modal run --detach rlaif_app.py::sample_check --confirm --max-usd 1.0     # 12.3, 1 x L40S per track
    modal run --detach rlaif_app.py::dpo_check --track slm125 --confirm --max-usd 1.5   # 12.5, 1 x H100

Policies are the RAFT SFT checkpoints (SLM-125M: 7.8's n20000.pt on the Phase 5 weights; Gemma 3: 11.4's
n20000-lr2e-05.pt). Sampling: raft_decode.sample_batch (decision drawn from the model, copy2, FormatConstraint,
temperature on content only). DPO: dpo.train_dpo (reference = the SFT checkpoint). Spend is projected from the
wall clock (+60 s) and checked before every pass; results are written to the Volume as they finish.
"""

from __future__ import annotations

import json
import shutil
import time
import zlib
from pathlib import Path

import modal

import config

RL = config.RLAIF
app = modal.App(f"{config.PROJECT}-rlaif")
SRC = ("config", "model", "data", "train", "cleaning", "raft_data", "raft_decode", "eval_raft", "eval_forgetting", "sft",
       "gemma_sft", "dpo", "teacher", "raft_prompts", "rlaif_data")
slm_image = (modal.Image.debian_slim(python_version="3.12").apt_install("gcc")
             .pip_install("torch==2.5.1", "transformers==4.46.3", "numpy==2.1.3", "safetensors==0.4.5")
             .add_local_python_source(*SRC))
g3_image = (modal.Image.debian_slim(python_version="3.12").apt_install("gcc")
            .pip_install("torch==2.5.1", f"transformers=={config.GEMMA3.transformers}", "numpy==2.1.3", "safetensors==0.4.5")
            .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
            .add_local_python_source(*SRC))
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
ROOT = Path(config.RLAIF_DIR)
SCALE_DIR = Path(config.RAFT_DIR) / "datasets" / "scale"
G3_DIR = Path(config.GEMMA3_DIR)
LOCAL = Path(".smoke/rlaif")
TRACKS = ("slm125", "gemma3")


def _rate(gpu: str) -> float:
    return config.GPU_RATE_PER_SEC[gpu] * config.COST_OVERHEAD_FACTOR


def _clock(gpu: str, log_path: Path):
    t0 = time.time()
    spent = lambda: (time.time() - t0 + 60.0) * _rate(gpu)

    def log(m: str) -> None:
        line = f"[{time.time() - t0:7.0f}s ${spent():.2f}] {m}"
        print(line, flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")
    return spent, log


def load_policy(track: str, device):
    """(fp32 model at the SFT checkpoint, tokenizer, tau, stored-prompt converter)."""
    import torch
    from transformers import AutoTokenizer

    import raft_decode as RD
    if track == "slm125":
        import sft
        net = sft.load_init(config.RAFT_INIT_CKPT).to(device)
        net.load_state_dict(torch.load(Path(config.RAFT_DIR) / "runs" / "full" / "n20000.pt", map_location=device, weights_only=False))
        tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
        tau = json.loads((Path(config.RAFT_DIR) / "runs" / "full" / "final.json").read_text())["tau"]
        return net, tok, tau, (lambda p: p)
    import gemma_sft as GS
    base = G3_DIR / "models" / "gemma-3-270m"
    net = GS.load_base(str(base), device, config.GEMMA3.attn, dtype=torch.float32)
    net.load_state_dict(torch.load(G3_DIR / "runs" / "full" / "n20000-lr2e-05.pt", map_location="cpu", weights_only=False))
    tok = AutoTokenizer.from_pretrained(base)
    tau = json.loads((G3_DIR / "runs" / "full" / "final.json").read_text())["tau"]
    return net, tok, tau, RD.GEMMA.convert


def _sample(track: str, rows: list[dict], k_of, out_path: Path, max_usd: float, gpu: str, log, spent) -> dict:
    """K samples per row (k_of(row)), saved as {id: [samples]}; resumes from a partial file."""
    import torch

    import raft_decode as RD
    from train import autocast
    device = torch.device("cuda")
    net, tok, tau, conv = load_policy(track, device)
    done = {}
    if out_path.exists():
        done = {json.loads(ln)["id"]: json.loads(ln) for ln in out_path.read_text().splitlines() if ln.strip()}
    todo = [r for r in rows if r["id"] not in done]
    log(f"{track}: {len(done)} rows already sampled, {len(todo)} to go")
    t_chunk = 0.0
    for c in range(0, len(todo), 64):
        if spent() + 1.5 * t_chunk * _rate(gpu) > max_usd:
            log(f"budget: stopping with {len(done)} rows sampled")
            break
        t1 = time.time()
        part = todo[c:c + 64]
        for k in sorted({k_of(r) for r in part}):
            sub = [r for r in part if k_of(r) == k]
            res = RD.sample_batch(net, tok, [conv(r["prompt"]) for r in sub], [len(r["passages"] or []) for r in sub],
                                  [r["passages"] or [] for r in sub], device, k=k, temperature=RL.temperature,
                                  seed=zlib.crc32(sub[0]["id"].encode()) % (2**31), batch=128, autocast=lambda: autocast(device))
            with out_path.open("a") as fh:
                for r, ss in zip(sub, res):
                    rec = {"id": r["id"], "samples": ss}
                    done[r["id"]] = rec
                    fh.write(json.dumps(rec) + "\n")
        volume.commit()
        t_chunk = max(t_chunk, time.time() - t1)
    n = sum(len(v["samples"]) for v in done.values())
    valid = sum(RD.parse_output(s["text"])["valid"] for v in done.values() for s in v["samples"])
    log(f"{track}: {len(done)} rows, {n} samples, format validity {valid / max(1, n):.4f}")
    return {"rows": len(done), "samples": n, "format_validity": valid / max(1, n), "spent_usd": spent()}


def _headroom(track: str, max_usd: float) -> dict:
    spent, log = _clock(RL.sample_gpu, ROOT / track / "log.txt")
    rows = [json.loads(ln) for ln in (SCALE_DIR / "dev.jsonl").read_text().splitlines() if ln.strip()]
    rows = [r for r in rows if r["bucket"] in ("A", "B1", "B2", "B3")]
    return _sample(track, rows, lambda r: RL.headroom_k, ROOT / track / "headroom.jsonl", max_usd, RL.sample_gpu, log, spent)


def _sample_prompts(track: str, max_usd: float) -> dict:
    spent, log = _clock(RL.sample_gpu, ROOT / track / "log.txt")
    rows = [json.loads(ln) for ln in (ROOT / "prompts.jsonl").read_text().splitlines() if ln.strip()]
    return _sample(track, rows, lambda r: r["k_samples"], ROOT / track / "samples_r1.jsonl", max_usd, RL.sample_gpu, log, spent)


@app.function(image=slm_image, gpu=RL.sample_gpu, volumes=VOLUMES, cpu=8.0, memory=32_768, timeout=60 * 40)
def headroom_slm(max_usd: float) -> dict:
    return _headroom("slm125", max_usd)


@app.function(image=g3_image, gpu=RL.sample_gpu, volumes=VOLUMES, cpu=8.0, memory=32_768, timeout=60 * 40)
def headroom_g3(max_usd: float) -> dict:
    return _headroom("gemma3", max_usd)


@app.function(image=slm_image, gpu=RL.sample_gpu, volumes=VOLUMES, cpu=8.0, memory=32_768, timeout=60 * 60)
def sample_slm(max_usd: float) -> dict:
    return _sample_prompts("slm125", max_usd)


@app.function(image=g3_image, gpu=RL.sample_gpu, volumes=VOLUMES, cpu=8.0, memory=32_768, timeout=60 * 60)
def sample_g3(max_usd: float) -> dict:
    return _sample_prompts("gemma3", max_usd)


# ------------------------------------------------------------------------------------- DPO
def _dpo(track: str, max_usd: float) -> dict:
    """12.5 pilot: one DPO run per beta from the SFT checkpoint; per run keep='select' (dev copy2 proxy among
    gate-passing evals), tau re-chosen on dev, dev generations saved."""
    import numpy as np
    import torch

    import dpo as DPO
    import eval_forgetting as EF
    import eval_raft as E
    import gemma_sft as GS
    import sft
    from train import autocast
    gpu = RL.train_gpu
    run_dir = ROOT / track / "dpo"
    run_dir.mkdir(parents=True, exist_ok=True)
    spent, log = _clock(gpu, run_dir / "log.txt")
    device = torch.device("cuda")
    ac = lambda: autocast(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    net, tok, tau_sft, conv = load_policy(track, device)
    sft_state = {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}
    rows = {json.loads(ln)["id"]: json.loads(ln) for ln in (ROOT / "prompts.jsonl").read_text().splitlines() if ln.strip()}
    prs = [json.loads(ln) for ln in (ROOT / track / "pairs_r1.jsonl").read_text().splitlines() if ln.strip()]
    enc = DPO.encode_pairs(prs, rows, tok)
    pad = tok.pad_token_id
    DPO.reference_logps(net, enc, pad, device, ac)
    dev_rows = [json.loads(ln) for ln in (SCALE_DIR / "dev.jsonl").read_text().splitlines() if ln.strip()]
    prompts, cps = [conv(r["prompt"]) for r in dev_rows], [r["passages"] or [] for r in dev_rows]
    ks = [len(c) for c in cps]
    val_dir, subset = Path(config.VAL_TOKENS_DIR), EF.heldout_subset(Path(config.VAL_TOKENS_DIR))
    replay = None
    if track == "slm125":
        from data import WindowStore
        shutil.copytree(config.TRAIN_TOKENS_DIR, "/tmp/tokens/train", dirs_exist_ok=True)
        store = WindowStore.open("/tmp/tokens/train", config.SEQ_LEN)
        rng = np.random.default_rng(1337)
        replay = lambda: torch.from_numpy(store.gather(rng.integers(0, len(store), 2))[:, :config.SEQ_LEN].astype(np.int64))
        base_bpb = json.loads((Path(config.RAFT_DIR) / "runs" / "pilot" / "guards_base.json").read_text())["heldout"]["pooled_bpb"]
        bpb_fn = lambda: EF.heldout_bpb(net, tok, val_dir, subset, device, ac, batch=16)["pooled_bpb"]
    else:
        src_tok = __import__("transformers").AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
        bpb_fn = lambda: GS.heldout_bpb(net, tok, src_tok, val_dir, subset, device, ac, batch=8)["pooled_bpb"]
        base_bpb = None
    sft_bpb = bpb_fn()
    gate_ref = min(x for x in (base_bpb, sft_bpb) if x is not None) if base_bpb is not None else sft_bpb
    log(f"{track}: {len(enc)} pairs; SFT bpb {sft_bpb:.4f}; gate reference {gate_ref:.4f} (+{config.RAFT.max_bpb_increase})")

    def extra() -> dict:
        bpb = bpb_fn()
        g = E.generate_batch(net, tok, prompts, ks, device, tau=tau_sft, batch=64, autocast=ac, copy_passages=cps)
        m = E.proxy_metrics(dev_rows, g)
        return {"heldout_pooled_bpb": bpb, "bpb_delta_vs_gate_ref": bpb - gate_ref, "bpb_delta_vs_sft": bpb - sft_bpb,
                "gate_ok": bpb - gate_ref <= config.RAFT.max_bpb_increase, "select_score": m["selection_score"],
                **{f"proxy_{k}": v for k, v in m.items()}}

    res_path = run_dir / "results.jsonl"
    done = {json.loads(ln)["key"] for ln in res_path.read_text().splitlines() if ln.strip()} if res_path.exists() else set()
    for beta in RL.betas:
        key = f"beta{beta:g}"
        if key in done:
            continue
        if spent() > max_usd * 0.55 and done:
            log(f"budget: stopping before {key}")
            break
        net.load_state_dict(sft_state)
        t1 = time.time()
        res = DPO.train_dpo(net, enc, device, beta=beta, lr=RL.lr[track], epochs=RL.epochs, pairs_per_step=RL.pairs_per_step,
                            micro=RL.micro_pairs, alpha=RL.alpha_sft, replay=replay,
                            replay_weight=RL.replay_weight_slm if track == "slm125" else 0.0, n_evals=RL.evals,
                            extra_eval=extra, budget_ok=lambda: spent() < max_usd * 0.92, log=log, autocast=ac)
        if res["best_state"] is None:
            log(f"{key}: no eval finished ({res['status']})")
            break
        net.load_state_dict(res["best_state"])
        torch.save(res["best_state"], run_dir / f"{key}.pt")
        if track == "slm125":
            dv = sft.tokenize(dev_rows, tok, config.SEQ_LEN)
            dec = E.decision_ids(tok)
            s = E.teacher_forced(net, [r[:m].tolist() for r, m in zip(dv.ids, dv.length)], dv.decision_pos.tolist(),
                                 list(dv.mask), device, dec, 16, ac)["s"]
            refuse, buckets = dv.refusal.tolist(), dv.buckets
        else:
            dv = GS.encode_rows(dev_rows, tok)
            s = GS.teacher_forced(net, dv, E.decision_ids(tok), device, autocast=ac)["s"]
            refuse, buckets = dv.refusal.tolist(), dv.buckets
        tau = E.choose_tau(E.tau_table(s, refuse, buckets))
        gens, prox = {}, {}
        for mode in ("plain", "copy2"):
            g = E.generate_batch(net, tok, prompts, ks, device, tau=tau["tau"], batch=64, autocast=ac,
                                 copy_passages=cps if mode == "copy2" else None)
            gens[mode] = [{"id": r["id"], "bucket": r["bucket"], "s": o["s"], "choice": o["choice"], "text": o["text"]}
                          for r, o in zip(dev_rows, g)]
            prox[mode] = E.proxy_metrics(dev_rows, g)
        best = res["best"]
        row = {"key": key, "beta": beta, "lr": RL.lr[track], "status": res["status"], "steps": res["steps"],
               "selected_step": best["step"], "gate_ok": best.get("gate_ok"), "bpb_delta_vs_gate_ref": best.get("bpb_delta_vs_gate_ref"),
               "bpb_delta_vs_sft": best.get("bpb_delta_vs_sft"), "tau": tau, "proxies": prox,
               "history": [{k: h.get(k) for k in ("step", "train_dpo", "train_margin", "bpb_delta_vs_gate_ref", "select_score", "gate_ok")}
                           for h in res["history"]], "gens": gens, "seconds": round(time.time() - t1, 1)}
        with res_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        volume.commit()
        log(f"{track} {key}: step {best['step']}/{res['steps']} gate {best.get('gate_ok')} | tau {tau['tau']} | copy2 "
            f"{prox['copy2']['selection_score']:+.3f} grounded {prox['copy2']['proxy_grounded_acc']:.3f} halluc {prox['copy2']['proxy_hallucination']:.3f}")
    rows_out = [json.loads(ln) for ln in res_path.read_text().splitlines() if ln.strip()] if res_path.exists() else []
    return {"status": "completed", "runs": rows_out, "sft_bpb": sft_bpb, "gate_ref": gate_ref, "spent_usd": spent()}


@app.function(image=slm_image, gpu=RL.train_gpu, volumes=VOLUMES, cpu=8.0, memory=65_536, timeout=60 * 60)
def dpo_slm(max_usd: float) -> dict:
    return _dpo("slm125", max_usd)


@app.function(image=g3_image, gpu=RL.train_gpu, volumes=VOLUMES, cpu=8.0, memory=65_536, timeout=60 * 60)
def dpo_g3(max_usd: float) -> dict:
    return _dpo("gemma3", max_usd)


# ----------------------------------------------------------------------------- entrypoints
def _gate(confirm: bool, max_usd: float, stage: str) -> None:
    if not confirm:
        raise SystemExit(f"Step {stage} costs money. Re-run with --confirm once approved.")
    cap = config.RLAIF_STAGE_CAPS_USD[stage]
    if max_usd > cap:
        raise SystemExit(f"--max-usd {max_usd} exceeds the approved cap ${cap} for {stage}")


@app.local_entrypoint()
def upload_prompts():
    with volume.batch_upload(force=True) as up:
        up.put_file(str(LOCAL / "prompts.jsonl"), "rlaif/prompts.jsonl")
    print("uploaded rlaif/prompts.jsonl")


def _both(fn_slm, fn_g3, max_usd: float, name: str) -> None:
    """Both tracks in parallel, each with half the cap; results written locally."""
    calls = {"slm125": fn_slm.spawn(max_usd / 2), "gemma3": fn_g3.spawn(max_usd / 2)}
    for t, c in calls.items():
        rep = c.get()
        print(t, json.dumps(rep))
        (LOCAL / t).mkdir(parents=True, exist_ok=True)
        (LOCAL / t / f"{name}_report.json").write_text(json.dumps(rep, indent=2))


@app.local_entrypoint()
def headroom_check(confirm: bool = False, max_usd: float = 0.6):
    """12.1 (paid: 1 x L40S per track)."""
    _gate(confirm, max_usd, "12.1-headroom-gpu")
    _both(headroom_slm, headroom_g3, max_usd, "headroom")


@app.local_entrypoint()
def sample_check(confirm: bool = False, max_usd: float = 1.0):
    """12.3 (paid: 1 x L40S per track)."""
    _gate(confirm, max_usd, "12.3-sample-gpu")
    _both(sample_slm, sample_g3, max_usd, "sample")


@app.local_entrypoint()
def dpo_check(track: str = "", confirm: bool = False, max_usd: float = 1.5):
    """12.5 (paid: 1 x H100). --track slm125 | gemma3."""
    _gate(confirm, max_usd, "12.5-dpo-gpu")
    fn = {"slm125": dpo_slm, "gemma3": dpo_g3}[track]
    rep = fn.remote(max_usd)
    d = LOCAL / track / "dpo"
    d.mkdir(parents=True, exist_ok=True)
    for r in rep.get("runs", []):
        for mode, g in r.pop("gens").items():
            (d / f"dev_{r['key']}_{mode}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in g))
    (d / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    for r in rep.get("runs", []):
        print(f"{r['key']}: step {r['selected_step']}/{r['steps']} gate {r['gate_ok']} | tau {r['tau']['tau']} | "
              f"copy2 grounded {r['proxies']['copy2']['proxy_grounded_acc']:.3f} halluc {r['proxies']['copy2']['proxy_hallucination']:.3f}")
    print(f"spent ${rep['spent_usd']:.2f}")
