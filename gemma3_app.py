"""Phase 11 jobs on Modal: RAFT on google/gemma-3-270m with full fine-tuning (docs/12-gemma3-270m.md).

    modal run gemma3_app.py::fetch_check                                        # pinned weights -> Volume (CPU)
    modal run --detach gemma3_app.py::probe_check --confirm --max-usd 0.4       # 11.2, 1 x H100
    modal run --detach gemma3_app.py::pilot_check --confirm --max-usd 2.0       # 11.3, 1 x H100
    modal run --detach gemma3_app.py::full_check --confirm --lr <pick> --max-usd 2.5   # 11.4, 1 x H100

Recipe: Phase 9's data/format/evaluation for Gemma (plain text, <bos> prefix, no replay, target-only logits,
tokenizer-free bits/byte gate, CaseHOLD on the seeded 1,200 subset), with FULL fine-tuning (fp32 master
weights + bf16 autocast, AdamW wd 0.1 on matrices) as SLM-125M and the 500M had. Guards and budget as
gemma_app.py: spend projected from the wall clock (+60 s), checked before each run and pass and every 10
training steps; results appended per run; each test decoding mode saved as it finishes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

import config

G3 = config.GEMMA3
app = modal.App(f"{config.PROJECT}-gemma3")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("gcc")
    .pip_install("torch==2.5.1", f"transformers=={G3.transformers}", "numpy==2.1.3", "safetensors==0.4.5", f"peft=={G3.peft}")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("config", "model", "data", "train", "cleaning", "raft_data", "raft_decode",
                             "eval_raft", "eval_forgetting", "sft", "gemma_sft")
)
cpu_image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub").add_local_python_source("config")
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
SECRETS = [modal.Secret.from_name(config.HF_SECRET_NAME)]
ROOT = Path(config.GEMMA3_DIR)
BASE_DIR = ROOT / "models" / "gemma-3-270m"
SCALE_DIR = Path(config.RAFT_DIR) / "datasets" / "scale"
CASEHOLD = Path(config.RAFT_DIR) / "eval" / "casehold_test.jsonl"
LOCAL = Path(".smoke/gemma3")


def _rate() -> float:
    return config.GPU_RATE_PER_SEC[G3.gpu] * config.COST_OVERHEAD_FACTOR


def _clock(log_path: Path):
    t0 = time.time()
    spent = lambda: (time.time() - t0 + 60.0) * _rate()

    def log(m: str) -> None:
        line = f"[{time.time() - t0:7.0f}s ${spent():.2f}] {m}"
        print(line, flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")
    return spent, log


def _casehold_items() -> list[dict]:
    """Phase 9's seeded 1,200-item subset (gemma_app._casehold_items), so the two Gemmas are comparable."""
    import numpy as np
    items = [json.loads(ln) for ln in CASEHOLD.read_text().splitlines() if ln.strip()]
    pick = np.sort(np.random.default_rng([config.GEMMA.seed, 11]).choice(len(items), config.GEMMA.casehold_items, replace=False))
    return [items[i] for i in pick]


def _gens(rows, out) -> list[dict]:
    return [{"id": r["id"], "bucket": r["bucket"], "s": o["s"], "choice": o["choice"], "text": o["text"]}
            for r, o in zip(rows, out)]


def closed_book_prompt(shots: list[dict], question: str) -> str:
    """Few-shot closed-book Q/A for the BASE model (as slm500_app.closed_book_prompt, with <bos>)."""
    ex = "".join(f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots)
    return f"<bos>{ex}Question: {question}\nAnswer:"


# ------------------------------------------------------------------------------- weights
@app.function(image=cpu_image, volumes=VOLUMES, secrets=SECRETS, cpu=2.0, memory=8_192, timeout=60 * 20)
def fetch() -> dict:
    import os
    from huggingface_hub import snapshot_download
    if not (BASE_DIR / "model.safetensors").exists():
        snapshot_download(G3.base, revision=G3.base_revision, local_dir=str(BASE_DIR), token=os.environ["HF_TOKEN"],
                          allow_patterns=["*.json", "*.safetensors", "tokenizer.model"])
    volume.commit()
    return {p.name: p.stat().st_size for p in sorted(BASE_DIR.iterdir()) if p.is_file()}


@app.local_entrypoint()
def fetch_check():
    print(json.dumps(fetch.remote(), indent=2))


# ------------------------------------------------------------------------------- shared
def _stage(device, dtype=None):
    import torch
    from transformers import AutoTokenizer

    import eval_forgetting as EF
    import gemma_sft as GS
    import raft_decode as RD
    import sft
    tok = AutoTokenizer.from_pretrained(BASE_DIR)
    dec = RD.decision_ids(tok)
    if dec["format"] != "gemma":
        raise SystemExit(f"unexpected format {dec['format']}")
    net = GS.load_base(str(BASE_DIR), device, G3.attn, dtype=dtype or torch.float32)   # LoRA: frozen bf16 base
    return {"tok": tok, "dec": dec, "net": net, "src_tok": AutoTokenizer.from_pretrained(config.TOKENIZER_DIR),
            "pool": sft.load_rows(SCALE_DIR / "train.jsonl"), "dev": GS.encode_rows(sft.load_rows(SCALE_DIR / "dev.jsonl"), tok),
            "subset": EF.heldout_subset(Path(config.VAL_TOKENS_DIR)), "items": _casehold_items()}


def _guards(net, S, device, ac, casehold: bool = True) -> dict:
    import eval_forgetting as EF
    import gemma_sft as GS
    g = {"heldout": GS.heldout_bpb(net, S["tok"], S["src_tok"], Path(config.VAL_TOKENS_DIR), S["subset"], device, ac, batch=8)}
    if casehold:
        g["casehold"] = EF.score_casehold(GS.causal_lm(net), S["tok"], S["items"], device, ac, items_per_batch=16,
                                          prefix_id=S["tok"].bos_token_id, softcap=None)
    return g


# ------------------------------------------------------------------------------- 11.2
@app.function(image=image, gpu=G3.gpu, volumes=VOLUMES, cpu=8.0, memory=32_768, timeout=60 * 20)
def probe(max_usd: float) -> dict:
    """11.2: forward+backward time and memory per micro-batch token budget, an optimizer step, eval timings,
    the full base guards (saved for 11.3/11.4), and a few-shot closed-book check of the base model."""
    import numpy as np
    import torch

    import eval_raft as E
    import gemma_sft as GS
    import sft
    from train import autocast
    spent, log = _clock(ROOT / "probe" / "log.txt")
    device = torch.device("cuda")
    ac = lambda: autocast(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    S = _stage(device)
    net, dec, tok, dev = S["net"], S["dec"], S["tok"], S["dev"]
    rep: dict = {"params": sum(p.numel() for p in net.parameters())}
    init = GS.trainable_state(net)
    train = GS.encode_rows(S["pool"][:3_000], tok)
    order = np.argsort([-len(s) for s in train.seqs])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-6, fused=True)
    net.train()
    for budget in (16_384, 32_768):
        rows_n = max(1, budget // len(train.seqs[order[0]]))
        idx = order[:rows_n]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times = []
        try:
            for _ in range(4):
                x, m = GS.pad_batch([train.seqs[i] for i in idx], train.plen[idx].tolist(), dec["pad"])
                x, m = x.to(device), m.to(device)
                torch.cuda.synchronize()
                t = time.time()
                with ac():
                    r = GS.loss_sums(net, x, m, dec)
                r["sft"].backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                times.append(time.time() - t)
            rep[f"tokens{budget}"] = {"rows": int(rows_n), "padded_tokens": int(x.numel()), "s_fwd_bwd_opt": float(np.median(times[1:])),
                                      "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            rep[f"tokens{budget}"] = {"oom": True}
        log(f"budget {budget}: {rep[f'tokens{budget}']}")
    GS.load_trainable(net, init)
    sub = GS.GemmaData(dev.seqs[:128], dev.plen[:128], dev.refusal[:128], dev.buckets[:128], dev.rows[:128])
    t = time.time()
    GS.teacher_forced(net, sub, dec, device, autocast=ac)
    rep["teacher_forced128_s"] = time.time() - t
    cps = [r["passages"] or [] for r in dev.rows[:64]]
    t = time.time()
    g = E.generate_batch(net, tok, [r["prompt"] for r in dev.rows[:64]], [len(c) for c in cps], device, tau=0.5, batch=64,
                         autocast=ac, copy_passages=cps)
    rep["gen64_copy2_s"], rep["gen64_copy2_valid"] = time.time() - t, sum(E.parse_output(o["text"])["valid"] for o in g)
    t = time.time()
    base = _guards(net, S, device, ac)
    rep["guards_s"] = time.time() - t
    rep["base_bpb"], rep["base_casehold_pmi"] = base["heldout"]["pooled_bpb"], base["casehold"]["acc_pmi"]
    (ROOT / "guards_base.json").write_text(json.dumps({**base, "gpu": G3.gpu}, default=str))
    volume.commit()
    log(f"timings {rep}")
    test_a = [r for r in sft.load_rows(SCALE_DIR / "test.jsonl") if r["bucket"] == "A"]
    shots = [r for r in S["pool"] if r["bucket"] == "A" and len(r["answer"]) < 40][:3]   # as slm500_app's probe
    nl = tok.encode("\n", add_special_tokens=False)
    out = []
    tok.padding_side = "left"
    for i in range(0, len(test_a), 64):
        if spent() > max_usd * 0.9:
            log("budget: stopping the closed-book check early")
            break
        part = test_a[i:i + 64]
        enc = tok([closed_book_prompt(shots, r["question"]) for r in part], add_special_tokens=False, return_tensors="pt", padding=True)
        with torch.no_grad(), ac():
            y = net.generate(**{k: v.to(device) for k, v in enc.items()}, do_sample=False, max_new_tokens=32,
                             eos_token_id=[tok.eos_token_id] + nl[-1:], pad_token_id=tok.pad_token_id)
        for r, row in zip(part, y[:, enc["input_ids"].shape[1]:].tolist()):
            out.append({"id": r["id"], "answer": tok.decode(row, skip_special_tokens=True).split("\n")[0].strip()})
    rep["closed_book_shots"], rep["spent_usd"] = [s["id"] for s in shots], spent()
    (ROOT / "probe" / "probe.json").write_text(json.dumps(rep, indent=2))
    volume.commit()
    return {**rep, "closed_book": out}


@app.local_entrypoint()
def probe_check(confirm: bool = False, max_usd: float = 0.4):
    _gate(confirm, max_usd, "11.2-probe")
    rep = probe.remote(max_usd)
    d = LOCAL / "probe"
    d.mkdir(parents=True, exist_ok=True)
    (d / "closed_book_base.jsonl").write_text("".join(json.dumps(o) + "\n" for o in rep.pop("closed_book")))
    (d / "probe.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


# ------------------------------------------------------------------------------- 11.3/11.4
def _runs(max_usd: float, which: str, runs: list[dict], test: bool, lora: bool = False, gated: bool = True) -> dict:
    """`gated=False` (D-35, Gemma 3 full fine-tuning at 20K): the bits/byte gate is reported, not enforced:
    keep='select' takes the best dev proxy over all evals; `gate_passed` records the real gate result."""
    import torch

    import eval_raft as E
    import gemma_sft as GS
    import sft
    from train import autocast
    if not (G3.probe_step_s and G3.probe_eval_s):
        raise SystemExit("run the 11.2 probe first and record probe_step_s / probe_eval_s in config.GEMMA3")
    run_dir = ROOT / "runs" / which
    run_dir.mkdir(parents=True, exist_ok=True)
    spent, log = _clock(run_dir / "log.txt")
    device = torch.device("cuda")
    ac = lambda: autocast(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    S = _stage(device, torch.bfloat16 if lora else torch.float32)
    net, dec, tok, dev = S["net"], S["dec"], S["tok"], S["dev"]
    gpath = ROOT / ("guards_base_bf16.json" if lora else "guards_base.json")   # LoRA's base runs in bf16: its own baseline
    if gpath.exists():
        base = json.loads(gpath.read_text())
    else:
        base = _guards(net, S, device, ac)
        gpath.write_text(json.dumps(base, default=str))
        volume.commit()
    base_bpb = base["heldout"]["pooled_bpb"]
    log(f"staged: pool {len(S['pool'])}, dev {len(dev)}; base bpb {base_bpb:.4f}, CaseHOLD PMI {base['casehold']['acc_pmi']:.4f}")
    if lora:
        net, init = GS.add_lora(net)                    # config.GEMMA's LoRA (r 32, all linear), as Gemma 2B
    else:
        init = GS.trainable_state(net)
    prompts, cps = [r["prompt"] for r in dev.rows], [r["passages"] or [] for r in dev.rows]
    ks = [len(c) for c in cps]

    def extra() -> dict:
        bpb = _guards(net, S, device, ac, casehold=False)["heldout"]["pooled_bpb"]
        g = E.generate_batch(net, tok, prompts, ks, device, tau=config.SFT.tau_pilot, batch=64, autocast=ac, copy_passages=cps)
        m = E.proxy_metrics(dev.rows, g)
        passed = bpb - base_bpb <= config.RAFT.max_bpb_increase
        return {"heldout_pooled_bpb": bpb, "bpb_delta": bpb - base_bpb, "gate_ok": passed or not gated, "gate_passed": passed,
                "select_score": m["selection_score"], **{f"proxy_{k}": v for k, v in m.items()}}

    res_path = run_dir / "results.jsonl"
    done = {json.loads(ln)["key"] for ln in res_path.read_text().splitlines() if ln.strip()} if res_path.exists() else set()
    for run in runs:
        rk = f"n{run['n']}-lr{run['lr']:g}"
        if rk in done:
            continue
        plan = sft.make_plan(run["n"], config.GEMMA.epochs, config.GEMMA.max_len, config.GEMMA.sft_rows_per_step,
                             config.GEMMA.micro_batch, sft_per_replay=0)
        est = (plan.total_steps * (G3.lora_step_s if lora else G3.probe_step_s) + (run["evals"] + 2) * G3.probe_eval_s + 60) * _rate()
        if spent() + est > max_usd:
            log(f"budget: run {rk} (est ${est:.2f}) would cross --max-usd {max_usd}; stopping")
            break
        t1 = time.time()
        train = GS.encode_rows(sft.select_mix(S["pool"], run["n"], config.SFT.refusal_share, config.SFT.seed), tok)
        reserve = (2 * G3.probe_eval_s + 60) * _rate()
        res = GS.train_run(net, init, train, dev, dec, device, lr=run["lr"], n_evals=run["evals"], extra_eval=extra,
                           budget_ok=lambda: spent() + reserve < max_usd, log=log, micro_tokens=G3.micro_tokens)
        if res["best_state"] is None:
            log(f"{rk}: no eval finished ({res['status']}); stopping")
            break
        GS.load_trainable(net, res["best_state"])
        torch.save(res["best_state"], run_dir / f"{rk}.pt")
        tf = GS.teacher_forced(net, dev, dec, device, autocast=ac)
        tau = E.choose_tau(E.tau_table(tf["s"], dev.refusal.tolist(), dev.buckets))
        gens, prox = {}, {}
        for mode in ("plain", "copy2"):
            g = E.generate_batch(net, tok, prompts, ks, device, tau=tau["tau"], batch=64, autocast=ac,
                                 copy_passages=cps if mode == "copy2" else None)
            gens[mode], prox[mode] = _gens(dev.rows, g), E.proxy_metrics(dev.rows, g)
        best = res["best"]
        row = {"key": rk, **run, "status": res["status"], "steps": res["steps"], "selected_step": best["step"],
               "gate_ok": best.get("gate_ok"), "bpb_delta": best.get("bpb_delta"), "select_score": best.get("select_score"),
               "dev_macro_f1": best.get("dev_macro_f1"), "dev_auroc": best.get("dev_auroc"), "tau": tau, "proxies": prox,
               "gated": gated, "history": [{k: h.get(k) for k in ("step", "dev_macro_f1", "dev_target_loss", "bpb_delta", "select_score", "gate_ok", "gate_passed")}
                           for h in res["history"]], "gens": gens, "seconds": round(time.time() - t1, 1)}
        with res_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        volume.commit()
        log(f"{rk}: step {best['step']}/{res['steps']} gate {best.get('gate_ok')} bpb {best.get('bpb_delta', 0):+.4f} | tau {tau['tau']} | "
            f"copy2 score {prox['copy2']['selection_score']:+.3f} grounded {prox['copy2']['proxy_grounded_acc']:.3f} ({row['seconds']}s)")

    rows = [json.loads(ln) for ln in res_path.read_text().splitlines() if ln.strip()] if res_path.exists() else []
    out = {"status": "completed", "runs": rows, "base_guards": {"bpb": base_bpb, "casehold_pmi": base["casehold"]["acc_pmi"]}}
    if not test or not rows:
        return {**out, "spent_usd": spent()}
    top = max(rows, key=lambda r: r["proxies"]["copy2"]["selection_score"])
    final_path = run_dir / "final.json"
    if final_path.exists():
        return {**out, "final": json.loads(final_path.read_text()), "spent_usd": spent()}
    GS.load_trainable(net, torch.load(run_dir / f"{top['key']}.pt", map_location="cpu", weights_only=False))
    tdata = GS.encode_rows(sft.load_rows(SCALE_DIR / "test.jsonl"), tok)
    tcps = [r["passages"] or [] for r in tdata.rows]
    tgens = {}
    for mode in ("plain", "copy2"):
        part = run_dir / f"test_{mode}.jsonl"
        if part.exists():
            tgens[mode] = [json.loads(ln) for ln in part.read_text().splitlines() if ln.strip()]
            continue
        if spent() + 2 * G3.probe_eval_s * _rate() > max_usd:
            log(f"budget: stopping before test {mode}")
            return {**out, "status": "budget_stop", "spent_usd": spent()}
        g = E.generate_batch(net, tok, [r["prompt"] for r in tdata.rows], [len(c) for c in tcps], device, tau=top["tau"]["tau"],
                             batch=64, autocast=ac, copy_passages=tcps if mode == "copy2" else None)
        tgens[mode] = _gens(tdata.rows, g)
        part.write_text("".join(json.dumps(x) + "\n" for x in tgens[mode]))
        volume.commit()
        log(f"test {mode}: done")
    tft = GS.teacher_forced(net, tdata, dec, device, autocast=ac)
    guards = _guards(net, S, device, ac)
    hf = run_dir / "hf"
    if lora:
        net.save_pretrained(run_dir / "adapter")
        net = net.merge_and_unload()                     # bf16 base + adapter -> one bf16 model
    net.to(torch.bfloat16).save_pretrained(hf, safe_serialization=True)          # last use of the training copy
    tok.save_pretrained(hf)
    (hf / "raft_format.json").write_text(json.dumps({
        "prompt": "<bos>Passage 1: ...\\nPassage k: ...\\nQuestion: ...\\nSource:", "refusal": config.RAFT.refusal,
        "decision": "s = P(' none') / (P(' none') + P(' Passage'))", "tau": top["tau"]["tau"]}, indent=2))
    final = {"key": top["key"], "tau": top["tau"]["tau"], "test_gens": tgens,
             "test_decision": E.decision_metrics(tft["s"], tdata.refusal.tolist(), tdata.buckets, top["tau"]["tau"]),
             "guards": guards, "bpb_delta_pooled": guards["heldout"]["pooled_bpb"] - base_bpb,
             "bpb_delta_by_source": {s: guards["heldout"][s]["bpb"] - base["heldout"][s]["bpb"] for s in ("case-law", "sec", "fineweb-edu")},
             "casehold_pmi_delta_points": 100 * (guards["casehold"]["acc_pmi"] - base["casehold"]["acc_pmi"])}
    final_path.write_text(json.dumps(final, default=str))
    volume.commit()
    log(f"final {top['key']}: tau {final['tau']} | test AUROC {final['test_decision']['auroc']:.3f} | "
        f"bpb {final['bpb_delta_pooled']:+.4f} | casehold {final['casehold_pmi_delta_points']:+.1f}")
    return {**out, "final": final, "spent_usd": spent()}


@app.function(image=image, gpu=G3.gpu, volumes=VOLUMES, cpu=8.0, memory=32_768, timeout=60 * 90)
def pilot(max_usd: float) -> dict:
    return _runs(max_usd, "pilot", [{"n": G3.pilot_size, "lr": lr, "evals": G3.pilot_evals} for lr in G3.pilot_lrs], False)


@app.function(image=image, gpu=G3.gpu, volumes=VOLUMES, cpu=8.0, memory=32_768, timeout=60 * 120)
def full(max_usd: float, lr: float) -> dict:
    """D-35: full fine-tuning at 20K with the gate reported, not enforced (no tested setting passed it, 11.3/11.3b)."""
    return _runs(max_usd, "full", [{"n": G3.full_size, "lr": lr, "evals": G3.full_evals}], True, gated=False)


@app.function(image=image, gpu=G3.gpu, volumes=VOLUMES, cpu=8.0, memory=32_768, timeout=60 * 90)
def lora_pilot(max_usd: float) -> dict:
    return _runs(max_usd, "lora_pilot", [{"n": G3.pilot_size, "lr": lr, "evals": G3.pilot_evals} for lr in G3.lora_lrs], False, lora=True)


@app.function(image=image, gpu=G3.gpu, volumes=VOLUMES, cpu=8.0, memory=32_768, timeout=60 * 120)
def lora_full(max_usd: float, lr: float) -> dict:
    return _runs(max_usd, "lora_full", [{"n": G3.full_size, "lr": lr, "evals": G3.full_evals}], True, lora=True)


@app.local_entrypoint()
def lora_pilot_check(confirm: bool = False, max_usd: float = 1.2):
    _gate(confirm, max_usd, "11.3b-lora-pilot")
    _report("lora_pilot", lora_pilot.remote(max_usd))


@app.local_entrypoint()
def lora_full_check(confirm: bool = False, lr: float = 0.0, max_usd: float = 2.5):
    _gate(confirm, max_usd, "11.4-full")
    if lr not in G3.lora_lrs:
        raise SystemExit(f"--lr must be one of {G3.lora_lrs} (the 11.3b pre-registered pick)")
    _report("lora_full", lora_full.remote(max_usd, lr))


@app.local_entrypoint()
def pilot_check(confirm: bool = False, max_usd: float = 2.0):
    _gate(confirm, max_usd, "11.3-pilot")
    _report("pilot", pilot.remote(max_usd))


@app.local_entrypoint()
def full_check(confirm: bool = False, lr: float = 0.0, max_usd: float = 2.5):
    _gate(confirm, max_usd, "11.4-full")
    if lr not in G3.pilot_lrs:
        raise SystemExit(f"--lr must be one of {G3.pilot_lrs} (the 11.3 pre-registered pick)")
    _report("full", full.remote(max_usd, lr))


def _gate(confirm: bool, max_usd: float, stage: str) -> None:
    if not confirm:
        raise SystemExit(f"Step {stage} uses a GPU and costs money. Re-run with --confirm once approved.")
    cap = config.GEMMA3_STAGE_CAPS_USD[stage]
    if max_usd > cap:
        raise SystemExit(f"--max-usd {max_usd} exceeds the approved cap ${cap} for {stage}")


def _report(which: str, rep: dict) -> None:
    d = LOCAL / which
    d.mkdir(parents=True, exist_ok=True)
    for r in rep.get("runs", []):
        for mode, g in r.pop("gens").items():
            (d / f"dev_{r['key']}_{mode}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in g))
    if rep.get("final"):
        for mode, g in rep["final"].pop("test_gens").items():
            (d / f"test_{mode}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in g))
    (d / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    for r in rep.get("runs", []):
        print(f"{r['key']}: step {r['selected_step']}/{r['steps']} gate {r['gate_ok']} | tau {r['tau']['tau']} | "
              f"copy2 {r['proxies']['copy2']['selection_score']:+.3f} grounded {r['proxies']['copy2']['proxy_grounded_acc']:.3f}")
    print(f"status {rep['status']}; spent ${rep['spent_usd']:.2f} -> {d}")
