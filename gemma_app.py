"""Phase 9 jobs on Modal: Gemma 2 2B on the RAFT data (docs/10-gemma.md §4). GPU steps need --confirm.

    modal run gemma_app.py::fetch_check                                         # weights -> Volume (CPU, cents)
    modal run --detach gemma_app.py::probe_check --confirm --max-usd 0.5        # 9.2, 1 x H100
    modal run --detach gemma_app.py::baselines_check --confirm --max-usd 1.0    # 9.3, 1 x L40S
    modal run --detach gemma_app.py::pilot_check --confirm --max-usd 3.0        # 9.4, 1 x H100
    modal run --detach gemma_app.py::full_check --confirm --lr <9.4 pick> --max-usd 6.0   # 9.5, 1 x H100

Every GPU function projects spend from its own wall clock (+60 s start allowance) and stops before a
run, eval or pass that would cross --max-usd; inside a training run the guard is re-checked every
10 steps. Results are appended to the Volume as each run finishes, so a restarted container resumes
at the next unfinished run. The 125M image and apps are untouched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

import config

G = config.GEMMA
app = modal.App(f"{config.PROJECT}-gemma")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("gcc")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "numpy==2.1.3", "safetensors==0.4.5",
                 "peft==0.13.2", "accelerate==1.1.1")
    .add_local_python_source("config", "model", "data", "train", "cleaning", "raft_data", "raft_decode",
                             "eval_raft", "eval_forgetting", "sft", "gemma_sft")
)
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
SECRETS = [modal.Secret.from_name(config.HF_SECRET_NAME)]
ROOT = Path(config.GEMMA_DIR)
BASE_DIR, INSTRUCT_DIR = ROOT / "models" / "gemma-2-2b-bf16", ROOT / "models" / "gemma-2-2b-it"
SCALE_DIR = Path(config.RAFT_DIR) / "datasets" / "scale"
CASEHOLD = Path(config.RAFT_DIR) / "eval" / "casehold_test.jsonl"
LOCAL = Path(".smoke/gemma")


def _rate(gpu: str) -> float:
    return config.GPU_RATE_PER_SEC[gpu] * config.COST_OVERHEAD_FACTOR


def _clock(gpu: str, log_path: Path | None = None):
    """(spent(), log()) for one container."""
    t0 = time.time()
    spent = lambda: (time.time() - t0 + 60.0) * _rate(gpu)

    def log(m: str) -> None:
        line = f"[{time.time() - t0:7.0f}s ${spent():.2f}] {m}"
        print(line, flush=True)
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as fh:
                fh.write(line + "\n")
    return spent, log


def _gens(rows: list[dict], out: list[dict]) -> list[dict]:
    return [{"id": r["id"], "bucket": r["bucket"], "s": o["s"], "choice": o["choice"], "text": o["text"]}
            for r, o in zip(rows, out)]


def _stage(device):
    """Tokenizers, decision ids and the frozen datasets, shared by every GPU step."""
    from transformers import AutoTokenizer

    import gemma_sft as GS
    import raft_decode as RD
    import sft
    tok = AutoTokenizer.from_pretrained(BASE_DIR)
    dec = RD.decision_ids(tok)
    if dec["format"] != "gemma":
        raise SystemExit(f"unexpected format {dec['format']}")
    src_tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    pool = sft.load_rows(SCALE_DIR / "train.jsonl")
    dev = GS.encode_rows(sft.load_rows(SCALE_DIR / "dev.jsonl"), tok)
    return tok, dec, src_tok, pool, dev


# ------------------------------------------------------------------------------ weights
@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, cpu=4.0, memory=32_768, timeout=60 * 30)
def fetch() -> dict:
    """Pinned Gemma weights onto the Volume: the base (stored in fp32 on the Hub) re-saved in bf16,
    and the instruct model as published (bf16). Idempotent."""
    import os

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    token = os.environ["HF_TOKEN"]
    out = {}
    if not (BASE_DIR / "config.json").exists():
        src = snapshot_download(G.base, revision=G.base_revision, token=token, cache_dir="/tmp/hf")
        net = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16)
        net.save_pretrained(BASE_DIR, safe_serialization=True)
        AutoTokenizer.from_pretrained(src).save_pretrained(BASE_DIR)
        out["base_params"] = sum(p.numel() for p in net.parameters())
    if not (INSTRUCT_DIR / "config.json").exists():
        snapshot_download(G.instruct, revision=G.instruct_revision, token=token, local_dir=str(INSTRUCT_DIR),
                          allow_patterns=["*.json", "*.safetensors", "tokenizer.model"])
    (ROOT / "models" / "manifest.json").write_text(json.dumps(
        {"base": G.base, "base_revision": G.base_revision, "base_dtype": "bfloat16 (re-saved)",
         "instruct": G.instruct, "instruct_revision": G.instruct_revision}, indent=2))
    volume.commit()
    for d in (BASE_DIR, INSTRUCT_DIR):
        out[d.name] = {p.name: p.stat().st_size for p in sorted(d.iterdir())}
    return out


@app.local_entrypoint()
def fetch_check():
    """CPU only (a few cents): download the pinned Gemma weights to the Volume."""
    print(json.dumps(fetch.remote(), indent=2))


# -------------------------------------------------------------------------------- 9.2
@app.function(image=image, gpu=G.gpu, volumes=VOLUMES, cpu=8.0, memory=65_536, timeout=60 * 20)
def probe(max_usd: float) -> dict:
    """9.2: eager vs SDPA, LoRA step time and peak memory, and timings of every eval pass, from which
    the 9.4/9.5 estimates are re-derived (docs/10-gemma.md §4)."""
    import numpy as np
    import torch

    import eval_forgetting as EF
    import eval_raft as E
    import gemma_sft as GS
    from train import autocast
    spent, log = _clock(G.gpu, ROOT / "probe" / "log.txt")
    device = torch.device("cuda")
    ac = lambda: autocast(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    tok, dec, src_tok, pool, dev = _stage(device)
    rep: dict = {"gpu": G.gpu}

    def guard(what: str) -> None:
        if spent() > max_usd * 0.9:
            raise SystemExit(f"budget: stopping before {what} (spent ${spent():.2f} of --max-usd {max_usd})")

    # Timings on subsets, extrapolated by the caller (keeps the probe near its cap).
    # eager vs SDPA on 16 dev prompts: the decision score at the end of the prompt
    prompts = [r["prompt"] for r in dev.rows[:16]]
    eager = GS.load_base(str(BASE_DIR), device, "eager")
    s_eager = GS.prompt_scores(eager, tok, prompts, device, autocast=ac)
    sdpa = GS.load_base(str(BASE_DIR), device, "sdpa")
    s_sdpa = GS.prompt_scores(sdpa, tok, prompts, device, autocast=ac)
    del sdpa
    torch.cuda.empty_cache()
    rep["eager_vs_sdpa_max_abs_ds"] = float(np.max(np.abs(np.array(s_eager) - np.array(s_sdpa))))
    log(f"eager vs sdpa: max |ds| {rep['eager_vs_sdpa_max_abs_ds']:.4f}")

    pm, init = GS.add_lora(eager)
    rep["trainable"] = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    train = GS.encode_rows(pool[:2_000], tok)
    longest = np.argsort([-len(s) for s in train.seqs])
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-5)
    pm.train()
    for micro in (13, 26):
        for kind, idx in (("longest", longest[:micro]), ("typical", np.arange(micro))):
            if micro == 26 and kind == "typical":
                continue
            guard(f"micro {micro}")
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
                        r = GS.loss_sums(pm, x, m, dec)
                    r["sft"].backward()
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    times.append(time.time() - t)
                rep[f"micro{micro}_{kind}"] = {"s_per_micro": float(np.median(times[1:])), "tokens": int(x.numel()),
                                               "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                rep[f"micro{micro}_{kind}"] = {"oom": True}
            log(f"micro {micro} {kind}: {rep[f'micro{micro}_{kind}']}")
    GS.load_adapter(pm, init)

    guard("teacher-forced dev")
    sub = GS.GemmaData(dev.seqs[:128], dev.plen[:128], dev.refusal[:128], dev.buckets[:128], dev.rows[:128])
    t = time.time()
    GS.teacher_forced(pm, sub, dec, device, autocast=ac)
    rep["teacher_forced128_s"] = time.time() - t
    cps = [r["passages"] or [] for r in dev.rows[:G.gen_batch]]
    for mode in ("plain", "copy2"):
        guard(f"generation {mode}")
        t = time.time()
        g = E.generate_batch(pm, tok, [r["prompt"] for r in dev.rows[:G.gen_batch]], [len(c) for c in cps], device,
                             tau=0.5, batch=G.gen_batch, autocast=ac, copy_passages=cps if mode == "copy2" else None)
        rep[f"gen{G.gen_batch}_{mode}_s"] = time.time() - t
        rep[f"gen{G.gen_batch}_{mode}_valid"] = sum(E.parse_output(o["text"])["valid"] for o in g)
    guard("held-out bits/byte")
    subset = EF.heldout_subset(Path(config.VAL_TOKENS_DIR))
    small = {k: {**v, "indices": v["indices"][:10]} for k, v in subset.items()}
    t = time.time()
    bpb = GS.heldout_bpb(pm, tok, src_tok, Path(config.VAL_TOKENS_DIR), small, device, ac)
    rep["bpb30_s"], rep["bpb30_pooled"] = time.time() - t, bpb["pooled_bpb"]
    guard("CaseHOLD")
    items = [json.loads(ln) for ln in CASEHOLD.read_text().splitlines() if ln.strip()][:48]
    t = time.time()
    ch = EF.score_casehold(GS.causal_lm(pm), tok, items, device, ac, items_per_batch=8,
                           prefix_id=tok.bos_token_id, softcap=eager.config.final_logit_softcapping)
    rep["casehold48_s"], rep["casehold48_pmi"] = time.time() - t, ch["acc_pmi"]
    rep["spent_usd"] = spent()
    (ROOT / "probe").mkdir(parents=True, exist_ok=True)
    (ROOT / "probe" / "probe.json").write_text(json.dumps(rep, indent=2))
    volume.commit()
    log(json.dumps(rep))
    return rep


@app.local_entrypoint()
def probe_check(confirm: bool = False, max_usd: float = 0.5):
    """9.2 (paid: 1 x H100). Requires --confirm; cap config.GEMMA_STAGE_CAPS_USD['9.2-probe']."""
    _gate(confirm, max_usd, "9.2-probe")
    rep = probe.remote(max_usd)
    _save("probe", {"probe.json": rep})
    print(json.dumps(rep, indent=2))


# -------------------------------------------------------------------------------- 9.3
@app.function(image=image, gpu=G.eval_gpu, volumes=VOLUMES, cpu=8.0, memory=32_768, timeout=60 * 40)
def baselines(max_usd: float) -> dict:
    """9.3: gemma-2-2b-it few-shot (tau on dev by the pre-registered rule, then test once, plain and
    copy2) and the closed-book probe on the 500 test A questions."""
    import torch
    from transformers import AutoTokenizer

    import eval_raft as E
    import gemma_sft as GS
    import sft
    from train import autocast
    spent, log = _clock(G.eval_gpu, ROOT / "baselines" / "log.txt")
    device = torch.device("cuda")
    ac = lambda: autocast(device)
    tok = AutoTokenizer.from_pretrained(INSTRUCT_DIR)
    net = GS.load_base(str(INSTRUCT_DIR), device)
    shots = GS.pick_shots(sft.load_rows(SCALE_DIR / "train.jsonl"))
    out_dir = ROOT / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep: dict = {"shots": [s["id"] for s in shots]}
    for group in ("dev", "test"):
        rows = sft.load_rows(SCALE_DIR / f"{group}.jsonl")
        prompts = [GS.instruct_prompt(r, shots) for r in rows]
        cps = [r["passages"] or [] for r in rows]
        s = GS.prompt_scores(net, tok, prompts, device, batch=8, autocast=ac)
        refuse = [r["bucket"] in sft.REFUSAL_BUCKETS for r in rows]
        buckets = [r["bucket"] for r in rows]
        if group == "dev":
            rep["tau"] = E.choose_tau(E.tau_table(s, refuse, buckets))
        tau = rep["tau"]["tau"]
        rep[f"{group}_decision"] = E.decision_metrics(s, refuse, buckets, tau)
        for mode in ("plain", "copy2"):
            if spent() > max_usd * 0.85:
                raise SystemExit(f"budget: stopping before {group} {mode} (spent ${spent():.2f})")
            g = E.generate_batch(net, tok, prompts, [len(c) for c in cps], device, tau=tau, batch=8, autocast=ac,
                                 copy_passages=cps if mode == "copy2" else None)
            rep[f"{group}_{mode}_proxy"] = E.proxy_metrics(rows, g)
            (out_dir / f"{group}_{mode}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in _gens(rows, g)))
            log(f"{group} {mode}: tau {tau} {rep[f'{group}_{mode}_proxy']}")
        volume.commit()
    test_a = [r for r in sft.load_rows(SCALE_DIR / "test.jsonl") if r["bucket"] == "A"]
    cb = []
    tok.padding_side = "left"
    for i in range(0, len(test_a), 16):
        part = test_a[i:i + 16]
        enc = tok([GS.closed_book_prompt(r["question"]) for r in part], add_special_tokens=False,
                  return_tensors="pt", padding=True)
        with torch.no_grad(), ac():
            y = net.generate(**{k: v.to(device) for k, v in enc.items()}, do_sample=False, max_new_tokens=48,
                             eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")],
                             pad_token_id=tok.pad_token_id)
        for r, row in zip(part, y[:, enc["input_ids"].shape[1]:].tolist()):
            cb.append({"id": r["id"], "question": r["question"], "answer": tok.decode(row, skip_special_tokens=True).strip()})
    (out_dir / "closed_book_test_A.jsonl").write_text("".join(json.dumps(x) + "\n" for x in cb))
    rep["closed_book_n"], rep["spent_usd"] = len(cb), spent()
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2))
    volume.commit()
    return rep


@app.local_entrypoint()
def baselines_check(confirm: bool = False, max_usd: float = 1.0):
    """9.3 (paid: 1 x L40S). Requires --confirm."""
    _gate(confirm, max_usd, "9.3-baselines")
    rep = baselines.remote(max_usd)
    _save("baselines", {"report.json": rep})
    print(json.dumps({k: v for k, v in rep.items() if not k.endswith("_proxy")}, indent=2, default=str))


# ------------------------------------------------------------------------ 9.4 and 9.5
def _lora_runs(max_usd: float, which: str, runs: list[dict]) -> dict:
    """Shared body of 9.4 (pilot) and 9.5 (full): base guards (once), then each LoRA run with
    keep='select', tau on dev, dev generations; for 9.5 also test, guards, adapter and merged export."""
    import numpy as np
    import torch

    import eval_forgetting as EF
    import eval_raft as E
    import gemma_sft as GS
    import sft
    from train import autocast
    if not (G.probe_step_s and G.probe_eval_s):
        raise SystemExit("run the 9.2 probe first and record probe_step_s / probe_eval_s in config.GEMMA")
    run_dir = ROOT / "runs" / which
    run_dir.mkdir(parents=True, exist_ok=True)
    spent, log = _clock(G.gpu, run_dir / "log.txt")
    rate = _rate(G.gpu)
    device = torch.device("cuda")
    ac = lambda: autocast(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    tok, dec, src_tok, pool, dev = _stage(device)
    net = GS.load_base(str(BASE_DIR), device)
    val_dir = Path(config.VAL_TOKENS_DIR)
    subset = EF.heldout_subset(val_dir)
    items = [json.loads(ln) for ln in CASEHOLD.read_text().splitlines() if ln.strip()]
    cap = net.config.final_logit_softcapping
    guards_path = ROOT / "guards_base.json"
    if guards_path.exists():
        base = json.loads(guards_path.read_text())
    else:
        base = {"heldout": GS.heldout_bpb(net, tok, src_tok, val_dir, subset, device, ac),
                "casehold": EF.score_casehold(net, tok, items, device, ac, items_per_batch=8,
                                              prefix_id=tok.bos_token_id, softcap=cap), "gpu": G.gpu}
        guards_path.write_text(json.dumps(base, default=str))
        volume.commit()
    base_bpb = base["heldout"]["pooled_bpb"]
    log(f"staged: pool {len(pool)}, dev {len(dev)}; Gemma base bpb {base_bpb:.4f}, "
        f"CaseHOLD PMI {base['casehold']['acc_pmi']:.4f}")
    pm, init = GS.add_lora(net)
    prompts, cps = [r["prompt"] for r in dev.rows], [r["passages"] or [] for r in dev.rows]
    ks = [len(c) for c in cps]

    def extra() -> dict:                      # every eval: the bits/byte gate + the copy2 proxy on dev
        bpb = GS.heldout_bpb(pm, tok, src_tok, val_dir, subset, device, ac)["pooled_bpb"]
        g = E.generate_batch(pm, tok, prompts, ks, device, tau=config.SFT.tau_pilot, batch=G.gen_batch,
                             autocast=ac, copy_passages=cps)
        m = E.proxy_metrics(dev.rows, g)
        return {"heldout_pooled_bpb": bpb, "bpb_delta": bpb - base_bpb,
                "gate_ok": bpb - base_bpb <= config.RAFT.max_bpb_increase,
                "select_score": m["selection_score"], **{f"proxy_{k}": v for k, v in m.items()}}

    res_path = run_dir / "results.jsonl"
    done = {json.loads(ln)["key"] for ln in res_path.read_text().splitlines() if ln.strip()} if res_path.exists() else set()
    for run in runs:
        key = f"n{run['n']}-lr{run['lr']:g}"
        if key in done:
            continue
        plan = sft.make_plan(run["n"], G.epochs, G.max_len, G.sft_rows_per_step, G.micro_batch, sft_per_replay=0)
        est = (plan.total_steps * G.probe_step_s + (run["evals"] + 2) * G.probe_eval_s + 60) * rate
        if spent() + est > max_usd:
            log(f"budget: run {key} (est ${est:.2f}) would cross --max-usd {max_usd}; stopping")
            break
        t1 = time.time()
        train = GS.encode_rows(sft.select_mix(pool, run["n"], config.SFT.refusal_share, config.SFT.seed), tok)
        reserve = (2 * G.probe_eval_s + 60) * rate          # tau + dev generations after the run
        res = GS.train_run(pm, init, train, dev, dec, device, lr=run["lr"], n_evals=run["evals"], extra_eval=extra,
                           budget_ok=lambda: spent() + reserve < max_usd, log=log)
        if res["best_state"] is None:
            log(f"{key}: no eval finished ({res['status']}); stopping")
            break
        GS.load_adapter(pm, res["best_state"])
        torch.save(res["best_state"], run_dir / f"{key}.pt")
        tf = GS.teacher_forced(pm, dev, dec, device, autocast=ac)
        tau = E.choose_tau(E.tau_table(tf["s"], dev.refusal.tolist(), dev.buckets))
        gens, prox = {}, {}
        for mode in ("plain", "copy2"):
            g = E.generate_batch(pm, tok, prompts, ks, device, tau=tau["tau"], batch=G.gen_batch, autocast=ac,
                                 copy_passages=cps if mode == "copy2" else None)
            gens[mode], prox[mode] = _gens(dev.rows, g), E.proxy_metrics(dev.rows, g)
        best = res["best"]
        row = {"key": key, **run, "status": res["status"], "steps": res["steps"], "selected_step": best["step"],
               "gate_ok": best.get("gate_ok"), "bpb_delta": best.get("bpb_delta"), "select_score": best.get("select_score"),
               "dev_macro_f1": best.get("dev_macro_f1"), "dev_auroc": best.get("dev_auroc"), "tau": tau, "proxies": prox,
               "history": [{k: h.get(k) for k in ("step", "dev_macro_f1", "dev_target_loss", "bpb_delta", "select_score",
                                                  "gate_ok")} for h in res["history"]],
               "gens": gens, "seconds": round(time.time() - t1, 1)}
        with res_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        volume.commit()
        log(f"{key}: step {best['step']}/{res['steps']} gate {best.get('gate_ok')} bpb {best.get('bpb_delta', 0):+.4f} | "
            f"tau {tau['tau']} | copy2 score {prox['copy2']['selection_score']:+.3f} grounded "
            f"{prox['copy2']['proxy_grounded_acc']:.3f} | plain {prox['plain']['selection_score']:+.3f} ({row['seconds']}s)")

    rows = [json.loads(ln) for ln in res_path.read_text().splitlines() if ln.strip()] if res_path.exists() else []
    out = {"status": "completed", "runs": rows, "base_guards": {k: v for k, v in base.items() if k != "casehold"} |
           {"casehold_pmi": base["casehold"]["acc_pmi"]}}
    if which != "full" or not rows:
        return {**out, "spent_usd": spent()}

    # 9.5: test once, guards, adapter + merged export
    top = rows[0]
    final_path = run_dir / "final.json"
    if final_path.exists():
        return {**out, "final": json.loads(final_path.read_text()), "spent_usd": spent()}
    if spent() + (3 * G.probe_eval_s + 600) * rate > max_usd:
        log("budget: no room for the test pass; stopping before it")
        return {**out, "status": "budget_stop", "spent_usd": spent()}
    GS.load_adapter(pm, torch.load(run_dir / f"{top['key']}.pt", map_location="cpu", weights_only=False))
    test = GS.encode_rows(sft.load_rows(SCALE_DIR / "test.jsonl"), tok)
    tcps = [r["passages"] or [] for r in test.rows]
    tgens = {}
    for mode in ("plain", "copy2"):
        g = E.generate_batch(pm, tok, [r["prompt"] for r in test.rows], [len(c) for c in tcps], device,
                             tau=top["tau"]["tau"], batch=G.gen_batch, autocast=ac,
                             copy_passages=tcps if mode == "copy2" else None)
        tgens[mode] = _gens(test.rows, g)
    tft = GS.teacher_forced(pm, test, dec, device, autocast=ac)
    lm = GS.causal_lm(pm)
    guards = {"heldout": GS.heldout_bpb(pm, tok, src_tok, val_dir, subset, device, ac),
              "casehold": EF.score_casehold(lm, tok, items, device, ac, items_per_batch=8,
                                            prefix_id=tok.bos_token_id, softcap=cap)}
    pm.save_pretrained(run_dir / "adapter")
    s_before = GS.prompt_scores(pm, tok, prompts[:32], device, autocast=ac)
    merged = pm.merge_and_unload()
    s_after = GS.prompt_scores(merged, tok, prompts[:32], device, autocast=ac)
    merged.save_pretrained(run_dir / "merged", safe_serialization=True)
    tok.save_pretrained(run_dir / "merged")
    (run_dir / "merged" / "raft_format.json").write_text(json.dumps({
        "prompt": "<bos>Passage 1: ...\\nPassage k: ...\\nQuestion: ...\\nSource:", "refusal": config.RAFT.refusal,
        "decision": "s = P(' none') / (P(' none') + P(' Passage'))", "tau": top["tau"]["tau"]}, indent=2))
    final = {"key": top["key"], "tau": top["tau"]["tau"], "test_gens": tgens,
             "test_decision": E.decision_metrics(tft["s"], test.refusal.tolist(), test.buckets, top["tau"]["tau"]),
             "guards": guards,
             "bpb_delta_by_source": {s: guards["heldout"][s]["bpb"] - base["heldout"][s]["bpb"]
                                     for s in ("case-law", "sec", "fineweb-edu")},
             "bpb_delta_pooled": guards["heldout"]["pooled_bpb"] - base_bpb,
             "casehold_pmi_delta_points": 100 * (guards["casehold"]["acc_pmi"] - base["casehold"]["acc_pmi"]),
             "merge_max_abs_ds": float(np.max(np.abs(np.array(s_before) - np.array(s_after))))}
    final_path.write_text(json.dumps(final, default=str))
    volume.commit()
    log(f"final {top['key']}: tau {final['tau']} | test AUROC {final['test_decision']['auroc']:.3f} | "
        f"bpb {final['bpb_delta_pooled']:+.4f} | casehold {final['casehold_pmi_delta_points']:+.1f} | "
        f"merge |ds| {final['merge_max_abs_ds']:.4f}")
    return {**out, "final": final, "spent_usd": spent()}


@app.function(image=image, gpu=G.gpu, volumes=VOLUMES, cpu=8.0, memory=65_536, timeout=60 * 90)
def pilot(max_usd: float) -> dict:
    return _lora_runs(max_usd, "pilot", [{"n": G.pilot_size, "lr": lr, "evals": G.pilot_evals} for lr in G.pilot_lrs])


@app.function(image=image, gpu=G.gpu, volumes=VOLUMES, cpu=8.0, memory=65_536, timeout=60 * 150)
def full(max_usd: float, lr: float) -> dict:
    return _lora_runs(max_usd, "full", [{"n": G.full_size, "lr": lr, "evals": G.full_evals}])


@app.local_entrypoint()
def pilot_check(confirm: bool = False, max_usd: float = 3.0):
    """9.4 (paid: 1 x H100). Requires --confirm."""
    _gate(confirm, max_usd, "9.4-pilot")
    _report("pilot", pilot.remote(max_usd))


@app.local_entrypoint()
def full_check(confirm: bool = False, lr: float = 0.0, max_usd: float = 6.0):
    """9.5 (paid: 1 x H100). Requires --confirm and the LR picked by the 9.4 rule."""
    _gate(confirm, max_usd, "9.5-full")
    if lr not in G.pilot_lrs:
        raise SystemExit(f"--lr must be one of the pilot LRs {G.pilot_lrs} (the 9.4 pre-registered pick)")
    _report("full", full.remote(max_usd, lr))


# ------------------------------------------------------------------------------ local
def _gate(confirm: bool, max_usd: float, stage: str) -> None:
    if not confirm:
        raise SystemExit(f"Step {stage} uses a GPU and costs money. Re-run with --confirm once approved.")
    cap = config.GEMMA_STAGE_CAPS_USD[stage]
    if max_usd > cap:
        raise SystemExit(f"--max-usd {max_usd} exceeds the approved cap ${cap} for {stage}")


def _save(which: str, files: dict) -> Path:
    d = LOCAL / which
    d.mkdir(parents=True, exist_ok=True)
    for name, obj in files.items():
        (d / name).write_text(obj if isinstance(obj, str) else json.dumps(obj, indent=2, default=str))
    return d


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
