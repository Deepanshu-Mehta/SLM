"""Phase 7 GPU steps on Modal (docs/08-raft.md §5, §7). Nothing here runs without --confirm.

    modal run --detach raft_gpu_app.py::sweep --confirm [--max-usd 4]      # step 7.6, 1 x H100
    modal run --detach raft_gpu_app.py::lambda_check --confirm --max-usd 1.6   # step 7.6b, 1 x H100
    modal run --detach raft_gpu_app.py::confirm_check --confirm --max-usd 1.4  # step 7.6c, 1 x H100
    modal run --detach raft_gpu_app.py::casehold_check --confirm --max-usd 0.3 # step 7.6d, 1 x L40S
    modal run --detach raft_gpu_app.py::pilot_eval_check --confirm --max-usd 0.75  # step 7.7a, 1 x H100
    modal run --detach raft_gpu_app.py::pilot_regen_check --confirm --max-usd 0.3  # 7.7a addendum, 1 x L40S
    modal run --detach raft_gpu_app.py::full_check --confirm --max-usd 2.0       # step 7.8, 1 x H100

One container does the whole pilot sweep:
  1. stage the train token shards on local disk (replay windows are random 2 KB reads);
  2. official before-SFT baselines on the fp32 init weights, bf16 autocast: held-out bits/byte on
     the seeded 600-window subset (each source its own bytes/token) and CaseHOLD (acc / acc_norm /
     PMI, 3,600 items);
  3. compile once; 3 LRs x 3 refusal shares at 2,000 examples, then 500- and 1,000-example curve
     points at the best setting. Every run reloads the base weights in place (sft.run_sweep);
  4. on the best run's weights: the forgetting guards again, and a tau table on dev.
A container-level guard stops before any run that would cross --max-usd. Results are appended
to the Volume as they finish, so a preempted container resumes at the next unfinished run.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import modal

import config

app = modal.App(f"{config.PROJECT}-raft-gpu")
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("gcc")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "numpy==2.1.3", "safetensors==0.4.5")
    .add_local_python_source("config", "model", "data", "train", "cleaning", "raft_data", "eval_raft",
                             "eval_forgetting", "sft")
)
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
RUN_DIR = Path(config.RAFT_DIR) / "runs" / "pilot"
DATA_DIR = Path(config.RAFT_DIR) / "datasets" / "pilot"
SCALE_DIR = Path(config.RAFT_DIR) / "datasets" / "scale"      # 7.8: 20K train + the frozen pilot dev/test
FULL_DIR = Path(config.RAFT_DIR) / "runs" / "full"
CASEHOLD = Path(config.RAFT_DIR) / "eval" / "casehold_test.jsonl"
LOCAL_TRAIN = "/tmp/tokens/train"
SPENT_76_USD = 3.58        # measured: 7.6 $1.37 + 7.6b $1.18 + 7.6c $0.52 + 7.6d $0.125 + 7.7a $0.32 + regen $0.063


def _rate_per_s() -> float:
    return config.GPU_RATE_PER_SEC[config.SFT.gpu] * config.SFT.gpu_count * config.COST_OVERHEAD_FACTOR


@app.function(image=gpu_image, gpu=f"{config.SFT.gpu}:{config.SFT.gpu_count}", volumes=VOLUMES,
              timeout=60 * 75, cpu=8.0, memory=32_768)
def pilot_sweep(max_usd: float, compile: bool = True) -> dict:
    import torch
    from transformers import AutoTokenizer

    import eval_forgetting as EF
    import eval_raft as E
    import sft
    from data import WindowStore
    from train import autocast

    t0 = time.time()
    spent = lambda: (time.time() - t0 + 60.0) * _rate_per_s()        # + container start allowance
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_DIR / "log.txt"

    def log(msg: str) -> None:
        line = f"[{time.time() - t0:7.0f}s ${spent():.2f}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    ac = lambda: autocast(device)
    shutil.copytree(config.TRAIN_TOKENS_DIR, LOCAL_TRAIN, dirs_exist_ok=True)
    store = WindowStore.open(LOCAL_TRAIN, config.SEQ_LEN)
    log(f"staged {len(store):,} train windows")
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    dec = E.decision_ids(tok)
    pool = sft.load_rows(DATA_DIR / "train.jsonl")
    dev = sft.tokenize(sft.load_rows(DATA_DIR / "dev.jsonl"), tok, config.SEQ_LEN)
    raw = sft.load_init(config.RAFT_INIT_CKPT).to(device)
    base_state = {k: v.detach().clone() for k, v in raw.state_dict().items()}
    val_dir = Path(config.VAL_TOKENS_DIR)
    subset = EF.heldout_subset(val_dir)
    items = [json.loads(ln) for ln in CASEHOLD.read_text().splitlines() if ln.strip()]

    def guards(tag: str) -> dict:
        r = {"heldout": EF.heldout_bpb(raw, tok, val_dir, subset, device, ac, batch=16),
             "casehold": EF.score_casehold(raw, tok, items, device, ac, items_per_batch=16)}
        (RUN_DIR / f"guards_{tag}.json").write_text(json.dumps(r, indent=2))
        volume.commit()
        log(f"guards[{tag}] pooled bpb {r['heldout']['pooled_bpb']:.4f} | casehold acc {r['casehold']['acc']:.3f} "
            f"norm {r['casehold']['acc_norm']:.3f} pmi {r['casehold']['acc_pmi']:.3f}")
        return r

    base_guards = (json.loads((RUN_DIR / "guards_base.json").read_text())
                   if (RUN_DIR / "guards_base.json").exists() else guards("base"))
    raw.load_state_dict(base_state)
    net = torch.compile(raw) if compile else raw
    base_cfg = sft.RunConfig(epochs=config.SFT.pilot_max_epochs, gpu=config.SFT.gpu, max_usd=max_usd,
                             log_every=10_000, replay_weight=0.25, keep="best")   # the 7.6 recipe, pinned
    run_cost = 120 * _rate_per_s()                                     # one 2,000-example run (~95 s estimated) + margin
    budget_ok = lambda: spent() + run_cost < max_usd - 0.6            # keep room for the final guards
    S = config.SFT
    grid = [{"n": S.pilot_size, "lr": lr, "refusal_share": r} for lr in S.sweep_lrs for r in S.sweep_refusal_shares]
    rows = sft.run_sweep(raw, net, base_state, pool, dev, store, tok, dec, device, grid, RUN_DIR / "results.jsonl",
                         base_cfg, budget_ok, log)
    volume.commit()
    grid_rows = [r for r in rows if r["n"] == S.pilot_size]
    if len(grid_rows) < len(grid):
        return {"status": "budget_stop", "results": rows, "spent_usd": spent()}
    best = sft.best_run(grid_rows)
    curve = [{"n": n, "lr": best["lr"], "refusal_share": best["refusal_share"]} for n in S.curve_sizes]
    rows = sft.run_sweep(raw, net, base_state, pool, dev, store, tok, dec, device, curve, RUN_DIR / "results.jsonl",
                         base_cfg, budget_ok, log)
    volume.commit()

    # forgetting guards and tau table on the best run's weights (retrained deterministically)
    raw.load_state_dict(base_state)
    sft.init_chat_tokens(raw, base_cfg.seed)
    train = sft.tokenize(sft.select_mix(pool, best["n"], best["refusal_share"], base_cfg.seed), tok, config.SEQ_LEN)
    res = sft.train_one(sft.RunConfig(**{**sft.asdict(base_cfg), "lr": best["lr"]}), raw, net, train, dev, store,
                        dec, device, None, log=lambda *_: None)
    raw.load_state_dict(res["best_state"])
    torch.save(res["best_state"], RUN_DIR / "best_pilot.pt")
    after = guards("best")
    tf = E.teacher_forced(raw, [r[:n].tolist() for r, n in zip(dev.ids, dev.length)], dev.decision_pos.tolist(),
                          list(dev.mask), device, dec, 16, ac)
    taus = {f"{t:.1f}": E.decision_metrics(tf["s"], dev.refusal.tolist(), dev.buckets, t)
            for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)}
    pooled_delta = after["heldout"]["pooled_bpb"] - base_guards["heldout"]["pooled_bpb"]
    report = {"status": "completed", "best": best, "results": rows, "taus": taus,
              "forgetting": {"bpb_delta_pooled": pooled_delta,
                             "bpb_delta_by_source": {s: after["heldout"][s]["bpb"] - base_guards["heldout"][s]["bpb"]
                                                     for s in ("case-law", "sec", "fineweb-edu")},
                             "casehold_delta_points": {k: 100 * (after["casehold"][k] - base_guards["casehold"][k])
                                                       for k in ("acc", "acc_norm", "acc_pmi")},
                             "bpb_guard_ok": pooled_delta <= config.RAFT.max_bpb_increase},
              "guards_base": base_guards, "guards_best": after, "spent_usd": spent(),
              "wall_seconds": time.time() - t0}
    (RUN_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
    volume.commit()
    log(f"done: best {best['key']} score {best['selection_score']:.3f}; bpb delta {pooled_delta:+.4f}")
    return report


@app.function(image=gpu_image, gpu=f"{config.SFT.gpu}:{config.SFT.gpu_count}", volumes=VOLUMES,
              timeout=60 * 45, cpu=8.0, memory=32_768)
def lambda_sweep(max_usd: float, arms: list[dict], refusal_share: float, n: int, compile: bool = True,
                 per_epoch_casehold: bool = False, judge: str = "best", out_name: str = "pilot_lambda") -> dict:
    """Step 7.6b: the forgetting fix. The pilot's selected setting at several (LR, replay weight)
    arms, with held-out bits/byte after EVERY epoch (the forgetting slope, to project to 7.8's
    longer run), CaseHOLD PMI at the best epoch, and dev proxies with plain and copy-constrained
    decoding. Results are appended per arm, so a restarted container skips finished arms.
    Step 7.6c reuses it with CaseHOLD PMI every epoch too, judged on the FINAL epoch's weights
    (`judge="final"`): 7.6b showed forgetting is transient and best-epoch picks are noisy."""
    assert judge in ("best", "final")
    import torch
    from transformers import AutoTokenizer

    import eval_forgetting as EF
    import eval_raft as E
    import sft
    from data import WindowStore
    from train import autocast

    t0 = time.time()
    spent = lambda: (time.time() - t0 + 60.0) * _rate_per_s()
    out_dir = Path(config.RAFT_DIR) / "runs" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[{time.time() - t0:7.0f}s ${spent():.2f}] {m}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    ac = lambda: autocast(device)
    shutil.copytree(config.TRAIN_TOKENS_DIR, LOCAL_TRAIN, dirs_exist_ok=True)
    store = WindowStore.open(LOCAL_TRAIN, config.SEQ_LEN)
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    dec = E.decision_ids(tok)
    pool = sft.load_rows(DATA_DIR / "train.jsonl")
    dev = sft.tokenize(sft.load_rows(DATA_DIR / "dev.jsonl"), tok, config.SEQ_LEN)
    raw = sft.load_init(config.RAFT_INIT_CKPT).to(device)
    base_state = {k: v.detach().clone() for k, v in raw.state_dict().items()}
    val_dir = Path(config.VAL_TOKENS_DIR)
    subset = EF.heldout_subset(val_dir)
    items = [json.loads(ln) for ln in CASEHOLD.read_text().splitlines() if ln.strip()]
    base = json.loads((RUN_DIR / "guards_base.json").read_text())
    log(f"staged; base pooled bpb {base['heldout']['pooled_bpb']:.4f}, casehold PMI {base['casehold']['acc_pmi']:.3f}")
    net = torch.compile(raw) if compile else raw
    res_path = out_dir / "results.jsonl"
    done = {json.loads(ln)["key"] for ln in res_path.read_text().splitlines() if ln.strip()} if res_path.exists() else set()
    run_cost = (480 if per_epoch_casehold else 240) * _rate_per_s()   # 8 CaseHOLD passes add ~280 s
    rows = []
    for arm in arms:
        lr, lam = arm["lr"], arm["lambda"]
        key = f"lr{lr:g}-lam{lam:g}"
        if key in done:
            continue
        if spent() + run_cost > max_usd:
            log(f"budget reached before {key}")
            break
        t1 = time.time()
        raw.load_state_dict(base_state)
        sft.init_chat_tokens(raw, config.SFT.seed)
        train = sft.tokenize(sft.select_mix(pool, n, refusal_share, config.SFT.seed), tok, config.SEQ_LEN)
        cfg = sft.RunConfig(lr=lr, epochs=config.SFT.pilot_max_epochs, replay_weight=lam, gpu=config.SFT.gpu,
                            max_usd=max_usd, log_every=10_000, keep="best")   # judge= picks best or final
        def extra() -> dict:
            r = {"heldout_pooled_bpb": EF.heldout_bpb(raw, tok, val_dir, subset, device, ac, batch=16)["pooled_bpb"]}
            if per_epoch_casehold:
                r["casehold_pmi"] = EF.score_casehold(raw, tok, items, device, ac, items_per_batch=16)["acc_pmi"]
            return r

        res = sft.train_one(cfg, raw, net, train, dev, store, dec, device, None, log=lambda *_: None, extra_eval=extra)
        if judge == "best":
            raw.load_state_dict(res["best_state"])
        ref = res["best"] if judge == "best" else res["history"][-1]      # raw holds the final weights otherwise
        ch = ({"acc_pmi": ref["casehold_pmi"]} if per_epoch_casehold
              else EF.score_casehold(raw, tok, items, device, ac, items_per_batch=16))
        prompts, ks = [r["prompt"] for r in dev.rows], [len(r["passages"] or []) for r in dev.rows]
        plain = E.proxy_metrics(dev.rows, E.generate_batch(raw, tok, prompts, ks, device, tau=0.5, batch=64, autocast=ac))
        copy = E.proxy_metrics(dev.rows, E.generate_batch(raw, tok, prompts, ks, device, tau=0.5, batch=64, autocast=ac,
                                                          copy_passages=[r["passages"] or [] for r in dev.rows]))
        best = res["best"]
        row = {"key": key, "lambda": lam, "lr": lr, "refusal_share": refusal_share, "n": n,
               "best_step": best["step"], "steps_per_epoch": res["plan"]["steps_per_epoch"],
               "best_dev_macro_f1": best["dev_macro_f1"], "best_dev_auroc": best["dev_auroc"],
               "bpb_delta_at_best": best["heldout_pooled_bpb"] - base["heldout"]["pooled_bpb"],
               "judge": judge, "judged_step": ref["step"], "judged_dev_macro_f1": ref["dev_macro_f1"],
               "bpb_delta_judged": ref["heldout_pooled_bpb"] - base["heldout"]["pooled_bpb"],
               "casehold_pmi": ch["acc_pmi"], "casehold_pmi_delta_points": 100 * (ch["acc_pmi"] - base["casehold"]["acc_pmi"]),
               "history": [{"step": h["step"], "macro_f1": h["dev_macro_f1"],
                            "bpb_delta": h["heldout_pooled_bpb"] - base["heldout"]["pooled_bpb"],
                            **({"casehold_pmi_delta_points": 100 * (h["casehold_pmi"] - base["casehold"]["acc_pmi"])}
                               if "casehold_pmi" in h else {})} for h in res["history"]],
               "plain": plain, "copy": copy, "seconds": round(time.time() - t1, 1)}
        with res_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        volume.commit()
        rows.append(row)
        log(f"{key}: {judge} step {ref['step']} macroF1 {ref['dev_macro_f1']:.3f} | bpb delta {row['bpb_delta_judged']:+.4f} "
            f"| casehold PMI {ch['acc_pmi']:.3f} ({row['casehold_pmi_delta_points']:+.1f}) | score plain {plain['selection_score']:+.3f} "
            f"copy {copy['selection_score']:+.3f} ({row['seconds']}s)")
    all_rows = [json.loads(ln) for ln in res_path.read_text().splitlines() if ln.strip()]
    return {"results": all_rows, "spent_usd": spent(), "wall_seconds": time.time() - t0}


def casehold_decompose(base: list[dict], tuned: list[dict]) -> dict:
    """Split a CaseHOLD PMI change into its two terms. PMI picks argmax(cond - null) over the 5 holdings;
    swapping in one model's term at a time shows whether the change comes from the citing-context
    term (cond) or the context-free prior term (null). Margins: correct holding minus the mean of the
    other four; a falling cond margin or a rising null margin both hurt PMI."""
    import numpy as np

    def acc(cond_from, null_from):
        return float(np.mean([int(np.argmax(np.array(c["cond"]) - np.array(u["null"]))) == c["label"]
                              for c, u in zip(cond_from, null_from)]))

    def margin(rows, term):
        m = []
        for r in rows:
            v = np.array(r[term])
            m.append(v[r["label"]] - np.delete(v, r["label"]).mean())
        return np.array(m)

    pb = [int(np.argmax(np.array(r["cond"]) - np.array(r["null"]))) == r["label"] for r in base]
    pt = [int(np.argmax(np.array(r["cond"]) - np.array(r["null"]))) == r["label"] for r in tuned]
    d_cond, d_null = margin(tuned, "cond") - margin(base, "cond"), margin(tuned, "null") - margin(base, "null")
    se = lambda d: float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"n": len(base),
            "pmi_acc": {"base": acc(base, base), "tuned": acc(tuned, tuned),
                        "tuned_cond_base_null": acc(tuned, base), "base_cond_tuned_null": acc(base, tuned)},
            "plain_acc": {"base": float(np.mean([int(np.argmax(r["cond"])) == r["label"] for r in base])),
                          "tuned": float(np.mean([int(np.argmax(r["cond"])) == r["label"] for r in tuned]))},
            "flips": {"right_to_wrong": int(sum(a and not b for a, b in zip(pb, pt))),
                      "wrong_to_right": int(sum(b and not a for a, b in zip(pb, pt)))},
            "margin_change_nats": {"cond": float(d_cond.mean()), "cond_se": se(d_cond),
                                   "null": float(d_null.mean()), "null_se": se(d_null),
                                   "pmi": float((d_cond - d_null).mean()), "pmi_se": se(d_cond - d_null)}}


DIAG_GPU = "L40S"


@app.function(image=gpu_image, gpu=DIAG_GPU, volumes=VOLUMES, timeout=60 * 20, cpu=4.0, memory=16_384)
def casehold_diag(max_usd: float) -> dict:
    """Step 7.6d: CaseHOLD PMI decomposition, base weights vs the 7.6 best pilot weights (best_pilot.pt)."""
    import torch
    from transformers import AutoTokenizer

    import eval_forgetting as EF
    import sft
    from train import autocast

    t0 = time.time()
    rate = config.GPU_RATE_PER_SEC[DIAG_GPU] * config.COST_OVERHEAD_FACTOR
    spent = lambda: (time.time() - t0 + 60.0) * rate
    device = torch.device("cuda")
    ac = lambda: autocast(device)
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    items = [json.loads(ln) for ln in CASEHOLD.read_text().splitlines() if ln.strip()]
    raw = sft.load_init(config.RAFT_INIT_CKPT).to(device)
    base = EF.score_casehold(raw, tok, items, device, ac, items_per_batch=16, return_scores=True)
    print(f"[{time.time() - t0:.0f}s ${spent():.3f}] base PMI {base['acc_pmi']:.3f}", flush=True)
    if spent() * 2 > max_usd:
        raise SystemExit(f"budget: the second pass would cross --max-usd {max_usd}")
    raw.load_state_dict(torch.load(RUN_DIR / "best_pilot.pt", map_location=device))
    tuned = EF.score_casehold(raw, tok, items, device, ac, items_per_batch=16, return_scores=True)
    print(f"[{time.time() - t0:.0f}s ${spent():.3f}] tuned PMI {tuned['acc_pmi']:.3f}", flush=True)
    rep = {"decomposition": casehold_decompose(base["scores"], tuned["scores"]),
           "base": {k: v for k, v in base.items() if k != "scores"},
           "tuned": {k: v for k, v in tuned.items() if k != "scores"},
           "gpu": DIAG_GPU, "spent_usd": spent(), "wall_seconds": time.time() - t0}
    out = Path(config.RAFT_DIR) / "runs" / "pilot" / "casehold_diag.json"
    out.write_text(json.dumps(rep, indent=2))
    volume.commit()
    return rep


@app.local_entrypoint()
def casehold_check(confirm: bool = False, max_usd: float = 0.3):
    """Step 7.6d (paid: 1 x L40S). Requires --confirm; draws on the 7.6 cap."""
    if not confirm:
        raise SystemExit("Step 7.6d uses a GPU and costs money. Re-run with --confirm once approved.")
    left = config.RAFT_STAGE_CAPS_USD["7.6-pilot-gpu"] - SPENT_76_USD
    if max_usd > left:
        raise SystemExit(f"--max-usd {max_usd} exceeds what is left of the 7.6 cap (${left:.2f})")
    rep = casehold_diag.remote(max_usd)
    out = Path(".smoke/raft/pilot/casehold_diag.json")
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep["decomposition"], indent=1))
    print(f"spent ${rep['spent_usd']:.3f} -> {out}")


@app.function(image=gpu_image, gpu=f"{config.SFT.gpu}:{config.SFT.gpu_count}", volumes=VOLUMES,
              timeout=60 * 30, cpu=8.0, memory=32_768)
def pilot_eval(max_usd: float, compile: bool = True) -> dict:
    """Step 7.7a: retrain the chosen pilot setting (config.SFT: LR, lambda, refusal share, keep='final'),
    save the weights, measure the guards, pick tau by the pre-registered rule on a teacher-forced
    table, and generate on dev at that tau with plain and copy-constrained decoding."""
    import torch
    from transformers import AutoTokenizer

    import eval_forgetting as EF
    import eval_raft as E
    import sft
    from data import WindowStore
    from train import autocast

    t0 = time.time()
    spent = lambda: (time.time() - t0 + 60.0) * _rate_per_s()
    log = lambda m: print(f"[{time.time() - t0:7.0f}s ${spent():.2f}] {m}", flush=True)
    S = config.SFT
    out_dir = Path(config.RAFT_DIR) / "runs" / "pilot_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    ac = lambda: autocast(device)
    shutil.copytree(config.TRAIN_TOKENS_DIR, LOCAL_TRAIN, dirs_exist_ok=True)
    store = WindowStore.open(LOCAL_TRAIN, config.SEQ_LEN)
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    dec = E.decision_ids(tok)
    pool = sft.load_rows(DATA_DIR / "train.jsonl")
    dev = sft.tokenize(sft.load_rows(DATA_DIR / "dev.jsonl"), tok, config.SEQ_LEN)
    raw = sft.load_init(config.RAFT_INIT_CKPT).to(device)
    val_dir = Path(config.VAL_TOKENS_DIR)
    subset = EF.heldout_subset(val_dir)
    items = [json.loads(ln) for ln in CASEHOLD.read_text().splitlines() if ln.strip()]
    base = json.loads((RUN_DIR / "guards_base.json").read_text())
    log(f"staged; base pooled bpb {base['heldout']['pooled_bpb']:.4f}")
    if spent() + 300 * _rate_per_s() > max_usd:
        raise SystemExit(f"budget: training would cross --max-usd {max_usd}")

    sft.init_chat_tokens(raw, S.seed)
    train = sft.tokenize(sft.select_mix(pool, S.pilot_size, S.refusal_share, S.seed), tok, config.SEQ_LEN)
    cfg = sft.RunConfig(epochs=S.pilot_max_epochs, gpu=S.gpu, max_usd=max_usd, log_every=10_000)
    log(f"train: n {len(train)} lr {cfg.lr:g} lambda {cfg.replay_weight:g} refusal {S.refusal_share} keep {cfg.keep}")
    net = torch.compile(raw) if compile else raw
    res = sft.train_one(cfg, raw, net, train, dev, store, dec, device, None, log=lambda *_: None)
    torch.save(res["best_state"], out_dir / "final.pt")
    volume.commit()
    final = res["best"]
    log(f"trained: step {final['step']} dev macro-F1 {final['dev_macro_f1']:.3f} AUROC {final['dev_auroc']:.3f}")

    guards = {"heldout": EF.heldout_bpb(raw, tok, val_dir, subset, device, ac, batch=16),
              "casehold": EF.score_casehold(raw, tok, items, device, ac, items_per_batch=16)}
    bpb_delta = guards["heldout"]["pooled_bpb"] - base["heldout"]["pooled_bpb"]
    ch_delta = 100 * (guards["casehold"]["acc_pmi"] - base["casehold"]["acc_pmi"])
    log(f"guards: bpb delta {bpb_delta:+.4f} | casehold PMI {ch_delta:+.1f} points")

    tf = E.teacher_forced(raw, [r[:n].tolist() for r, n in zip(dev.ids, dev.length)], dev.decision_pos.tolist(),
                          list(dev.mask), device, dec, 16, ac)
    table = E.tau_table(tf["s"], dev.refusal.tolist(), dev.buckets)
    tau = E.choose_tau(table)
    log(f"tau {tau['tau']}: macro-F1 {tau['macro_f1']:.3f} over-refusal A {tau['over_refusal_A']:.3f} "
        f"recall B {tau['recall_B']:.3f} meets targets {tau['meets_targets']}")

    prompts, ks = [r["prompt"] for r in dev.rows], [len(r["passages"] or []) for r in dev.rows]
    gens, proxies = {}, {}
    for mode in ("plain", "copy"):
        cp = [r["passages"] or [] for r in dev.rows] if mode == "copy" else None
        g = E.generate_batch(raw, tok, prompts, ks, device, tau=tau["tau"], batch=64, autocast=ac, copy_passages=cp)
        gens[mode] = [{"id": r["id"], "bucket": r["bucket"], "s": o["s"], "choice": o["choice"], "text": o["text"]}
                      for r, o in zip(dev.rows, g)]
        proxies[mode] = E.proxy_metrics(dev.rows, g)
        log(f"{mode}: score {proxies[mode]['selection_score']:+.3f} grounded {proxies[mode]['proxy_grounded_acc']:.3f} "
            f"halluc {proxies[mode]['proxy_hallucination']:.3f} format {proxies[mode]['format_validity']:.3f}")
        with (out_dir / f"dev_gen_{mode}.jsonl").open("w") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in gens[mode])
    report = {"config": {"n": S.pilot_size, "refusal_share": S.refusal_share, "lr": cfg.lr,
                         "lambda": cfg.replay_weight, "epochs": cfg.epochs, "keep": cfg.keep},
              "final": final, "history": res["history"], "tau": tau, "tau_table": table,
              "guards": guards, "bpb_delta": bpb_delta, "casehold_pmi_delta_points": ch_delta,
              "proxies": proxies, "spent_usd": spent(), "wall_seconds": time.time() - t0}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    volume.commit()
    return {"report": report, "gens": gens}


@app.function(image=gpu_image, gpu=DIAG_GPU, volumes=VOLUMES, timeout=60 * 20, cpu=8.0, memory=16_384)
def pilot_regen(max_usd: float) -> dict:
    """7.7a addendum: regenerate dev with the fixed CopyConstraint (forced scaffold, forced close,
    240-char cap) from the saved 7.7a weights, at the 7.7a tau. Generation only."""
    import torch
    from transformers import AutoTokenizer

    import eval_raft as E
    import sft
    from train import autocast

    t0 = time.time()
    spent = lambda: (time.time() - t0 + 60.0) * config.GPU_RATE_PER_SEC[DIAG_GPU] * config.COST_OVERHEAD_FACTOR
    d = Path(config.RAFT_DIR) / "runs" / "pilot_final"
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    dev = sft.tokenize(sft.load_rows(DATA_DIR / "dev.jsonl"), tok, config.SEQ_LEN)
    raw = sft.load_init(config.RAFT_INIT_CKPT).to(device)
    raw.load_state_dict(torch.load(d / "final.pt", map_location=device, weights_only=False))
    tau = json.loads((d / "report.json").read_text())["tau"]["tau"]
    if spent() * 3 > max_usd:
        raise SystemExit(f"budget: generation would cross --max-usd {max_usd}")
    g = E.generate_batch(raw, tok, [r["prompt"] for r in dev.rows], [len(r["passages"] or []) for r in dev.rows],
                         device, tau=tau, batch=64, autocast=lambda: autocast(device),
                         copy_passages=[r["passages"] or [] for r in dev.rows])
    gens = [{"id": r["id"], "bucket": r["bucket"], "s": o["s"], "choice": o["choice"], "text": o["text"]}
            for r, o in zip(dev.rows, g)]
    with (d / "dev_gen_copy2.jsonl").open("w") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in gens)
    volume.commit()
    return {"gens": gens, "proxies": E.proxy_metrics(dev.rows, g), "tau": tau, "spent_usd": spent()}


@app.local_entrypoint()
def pilot_regen_check(confirm: bool = False, max_usd: float = 0.3):
    """7.7a addendum (paid: 1 x L40S). Requires --confirm; draws on what is left of the 7.6 GPU cap."""
    if not confirm:
        raise SystemExit("This uses a GPU and costs money. Re-run with --confirm once approved.")
    left = config.RAFT_STAGE_CAPS_USD["7.6-pilot-gpu"] - SPENT_76_USD
    if max_usd > left:
        raise SystemExit(f"--max-usd {max_usd} exceeds what is left of the 7.6 cap (${left:.2f})")
    out = pilot_regen.remote(max_usd)
    d = Path(".smoke/raft/pilot/eval")
    (d / "dev_gen_copy2.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out["gens"]))
    print(f"tau {out['tau']}: " + " ".join(f"{k} {v:.3f}" for k, v in out["proxies"].items()))
    print(f"spent ${out['spent_usd']:.3f} -> {d / 'dev_gen_copy2.jsonl'}")


@app.function(image=gpu_image, gpu=f"{config.SFT.gpu}:{config.SFT.gpu_count}", volumes=VOLUMES,
              timeout=60 * 50, cpu=8.0, memory=32_768)
def full_run(max_usd: float, compile: bool = True) -> dict:
    """Step 7.8: the learning curve (config.SFT.full_sizes) on the 20K pool, each run keeping the
    checkpoint with the best dev proxy score under copy2 among evals that pass the bits/byte gate
    (D-27). Per run: tau by the pre-registered rule, dev generations (plain, copy2). For the best
    run: test generations, CaseHOLD, per-source bits/byte, HF export. Results are appended per run;
    a restarted container skips finished runs."""
    import torch
    from transformers import AutoTokenizer

    import eval_forgetting as EF
    import eval_raft as E
    import sft
    from data import WindowStore
    from train import autocast

    t0 = time.time()
    spent = lambda: (time.time() - t0 + 60.0) * _rate_per_s()
    S = config.SFT
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    log_path = FULL_DIR / "log.txt"

    def log(m: str) -> None:
        line = f"[{time.time() - t0:7.0f}s ${spent():.2f}] {m}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    ac = lambda: autocast(device)
    shutil.copytree(config.TRAIN_TOKENS_DIR, LOCAL_TRAIN, dirs_exist_ok=True)
    store = WindowStore.open(LOCAL_TRAIN, config.SEQ_LEN)
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    dec = E.decision_ids(tok)
    pool = sft.load_rows(SCALE_DIR / "train.jsonl")
    dev = sft.tokenize(sft.load_rows(SCALE_DIR / "dev.jsonl"), tok, config.SEQ_LEN)
    raw = sft.load_init(config.RAFT_INIT_CKPT).to(device)
    base_state = {k: v.detach().clone() for k, v in raw.state_dict().items()}
    val_dir = Path(config.VAL_TOKENS_DIR)
    subset = EF.heldout_subset(val_dir)
    base = json.loads((RUN_DIR / "guards_base.json").read_text())
    base_bpb = base["heldout"]["pooled_bpb"]
    prompts, ks = [r["prompt"] for r in dev.rows], [len(r["passages"] or []) for r in dev.rows]
    cps = [r["passages"] or [] for r in dev.rows]
    log(f"staged: pool {len(pool)} rows, dev {len(dev)}; base bpb {base_bpb:.4f}")
    net = torch.compile(raw) if compile else raw
    res_path = FULL_DIR / "results.jsonl"
    done = {json.loads(ln)["n"] for ln in res_path.read_text().splitlines() if ln.strip()} if res_path.exists() else set()

    def extra() -> dict:                       # every eval: bits/byte gate + copy2 proxy score on dev
        bpb = EF.heldout_bpb(raw, tok, val_dir, subset, device, ac, batch=16)["pooled_bpb"]
        g = E.generate_batch(raw, tok, prompts, ks, device, tau=S.tau_pilot, batch=64, autocast=ac, copy_passages=cps)
        m = E.proxy_metrics(dev.rows, g)
        return {"heldout_pooled_bpb": bpb, "bpb_delta": bpb - base_bpb,
                "gate_ok": bpb - base_bpb <= config.RAFT.max_bpb_increase,
                "select_score": m["selection_score"], **{f"proxy_{k}": v for k, v in m.items()}}

    for n, n_evals in zip(S.full_sizes, S.full_evals):
        if n in done:
            continue
        plan = sft.make_plan(n, S.epochs, config.SEQ_LEN)
        est = (plan.total_steps * 0.3 + n_evals * 35 + 90) * _rate_per_s()
        if spent() + est > max_usd:
            log(f"budget: run n={n} (est ${est:.2f}) would cross --max-usd {max_usd}; stopping")
            break
        t1 = time.time()
        raw.load_state_dict(base_state)
        sft.init_chat_tokens(raw, S.seed)
        train = sft.tokenize(sft.select_mix(pool, n, S.refusal_share, S.seed), tok, config.SEQ_LEN)
        cfg = sft.RunConfig(epochs=S.epochs, gpu=S.gpu, max_usd=max_usd, log_every=10_000, keep=S.keep,
                            eval_every=max(1, round(plan.total_steps / n_evals)))
        log(f"run n={n}: {plan.total_steps} steps, eval every {cfg.eval_every}, lr {cfg.lr:g} lambda {cfg.replay_weight:g}")
        res = sft.train_one(cfg, raw, net, train, dev, store, dec, device, None, log=lambda *_: None, extra_eval=extra)
        raw.load_state_dict(res["best_state"])
        torch.save(res["best_state"], FULL_DIR / f"n{n}.pt")
        tf = E.teacher_forced(raw, [r[:m].tolist() for r, m in zip(dev.ids, dev.length)], dev.decision_pos.tolist(),
                              list(dev.mask), device, dec, 16, ac)
        tau = E.choose_tau(E.tau_table(tf["s"], dev.refusal.tolist(), dev.buckets))
        gens, prox = {}, {}
        for mode in ("plain", "copy2"):
            g = E.generate_batch(raw, tok, prompts, ks, device, tau=tau["tau"], batch=64, autocast=ac,
                                 copy_passages=cps if mode == "copy2" else None)
            gens[mode] = [{"id": r["id"], "bucket": r["bucket"], "s": o["s"], "choice": o["choice"], "text": o["text"]}
                          for r, o in zip(dev.rows, g)]
            prox[mode] = E.proxy_metrics(dev.rows, g)
        best = res["best"]
        row = {"n": n, "steps": res["steps"], "selected_step": best["step"], "gate_ok": best.get("gate_ok"),
               "bpb_delta": best.get("bpb_delta"), "select_score": best.get("select_score"),
               "dev_macro_f1": best["dev_macro_f1"], "dev_auroc": best["dev_auroc"], "tau": tau,
               "proxies": prox, "history": [{k: h.get(k) for k in ("step", "dev_macro_f1", "dev_target_loss",
                                                                     "bpb_delta", "select_score", "gate_ok")}
                                            for h in res["history"]],
               "gens": gens, "seconds": round(time.time() - t1, 1)}
        with res_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        volume.commit()
        log(f"n={n}: step {best['step']}/{res['steps']} gate {best.get('gate_ok')} bpb {best.get('bpb_delta', 0):+.4f} | "
            f"tau {tau['tau']} | copy2 score {prox['copy2']['selection_score']:+.3f} grounded "
            f"{prox['copy2']['proxy_grounded_acc']:.3f} | plain {prox['plain']['selection_score']:+.3f} ({row['seconds']}s)")

    rows = [json.loads(ln) for ln in res_path.read_text().splitlines() if ln.strip()]
    if not rows:
        return {"status": "no_runs", "spent_usd": spent()}
    top = max(rows, key=lambda r: r["proxies"]["copy2"]["selection_score"])
    final_path = FULL_DIR / "final.json"
    if final_path.exists():
        final = json.loads(final_path.read_text())
    elif spent() + 150 * _rate_per_s() > max_usd:
        log("budget: no room for the test pass; stopping before it")
        return {"status": "budget_stop", "runs": rows, "spent_usd": spent()}
    else:
        raw.load_state_dict(torch.load(FULL_DIR / f"n{top['n']}.pt", map_location=device, weights_only=False))
        test = sft.tokenize(sft.load_rows(SCALE_DIR / "test.jsonl"), tok, config.SEQ_LEN)
        tgens = {}
        for mode in ("plain", "copy2"):
            g = E.generate_batch(raw, tok, [r["prompt"] for r in test.rows], [len(r["passages"] or []) for r in test.rows],
                                 device, tau=top["tau"]["tau"], batch=64, autocast=ac,
                                 copy_passages=[r["passages"] or [] for r in test.rows] if mode == "copy2" else None)
            tgens[mode] = [{"id": r["id"], "bucket": r["bucket"], "s": o["s"], "choice": o["choice"], "text": o["text"]}
                           for r, o in zip(test.rows, g)]
        tft = E.teacher_forced(raw, [r[:m].tolist() for r, m in zip(test.ids, test.length)],
                               test.decision_pos.tolist(), list(test.mask), device, dec, 16, ac)
        items = [json.loads(ln) for ln in CASEHOLD.read_text().splitlines() if ln.strip()]
        guards = {"heldout": EF.heldout_bpb(raw, tok, val_dir, subset, device, ac, batch=16),
                  "casehold": EF.score_casehold(raw, tok, items, device, ac, items_per_batch=16)}
        sft.export(raw, None, FULL_DIR / "hf", config.TOKENIZER_DIR, tau=top["tau"]["tau"])
        final = {"n": top["n"], "tau": top["tau"]["tau"], "test_gens": tgens,
                 "test_decision": E.decision_metrics(tft["s"], test.refusal.tolist(), test.buckets, top["tau"]["tau"]),
                 "guards": guards,
                 "bpb_delta_by_source": {s: guards["heldout"][s]["bpb"] - base["heldout"][s]["bpb"]
                                         for s in ("case-law", "sec", "fineweb-edu")},
                 "bpb_delta_pooled": guards["heldout"]["pooled_bpb"] - base_bpb,
                 "casehold_pmi_delta_points": 100 * (guards["casehold"]["acc_pmi"] - base["casehold"]["acc_pmi"])}
        final_path.write_text(json.dumps(final, default=str))
        volume.commit()
        log(f"final n={top['n']}: tau {final['tau']} | test AUROC {final['test_decision']['auroc']:.3f} | "
            f"bpb {final['bpb_delta_pooled']:+.4f} | casehold {final['casehold_pmi_delta_points']:+.1f}")
    return {"status": "completed", "runs": rows, "final": final, "spent_usd": spent(), "wall_seconds": time.time() - t0}


@app.local_entrypoint()
def full_check(confirm: bool = False, max_usd: float = 2.0, compile: bool = True):
    """Step 7.8 GPU (paid: 1 x H100). Requires --confirm; cap config.RAFT_STAGE_CAPS_USD['7.8-full-gpu']."""
    if not confirm:
        raise SystemExit("Step 7.8 uses a GPU and costs money. Re-run with --confirm once approved.")
    cap = config.RAFT_STAGE_CAPS_USD["7.8-full-gpu"]
    if max_usd > cap:
        raise SystemExit(f"--max-usd {max_usd} exceeds the approved cap ${cap}")
    rep = full_run.remote(max_usd, compile)
    d = Path(".smoke/raft/scale/eval")
    d.mkdir(parents=True, exist_ok=True)
    for r in rep.get("runs", []):
        for mode, g in r.pop("gens").items():
            (d / f"dev_n{r['n']}_{mode}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in g))
    if rep.get("final"):
        for mode, g in rep["final"].pop("test_gens").items():
            (d / f"test_{mode}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in g))
    (d / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    for r in rep.get("runs", []):
        print(f"n={r['n']}: step {r['selected_step']}/{r['steps']} gate {r['gate_ok']} | tau {r['tau']['tau']} | "
              f"copy2 {r['proxies']['copy2']['selection_score']:+.3f} grounded {r['proxies']['copy2']['proxy_grounded_acc']:.3f}")
    print(f"status {rep['status']}; spent ${rep['spent_usd']:.2f} -> {d}")


SPENT_78_GPU_USD = 2.07    # measured: 7.8 full_run $1.82 + 7.8b regen $0.245 (over the $2 cap, C-15)


@app.function(image=gpu_image, gpu=DIAG_GPU, volumes=VOLUMES, timeout=60 * 20, cpu=8.0, memory=16_384)
def answer_copy_regen(max_usd: float, group: str = "dev") -> dict:
    """7.8b: generations with copy3 (copy2 + answer copy) from the 20K weights at the 7.8 tau."""
    import torch
    from transformers import AutoTokenizer

    import eval_raft as E
    import sft
    from train import autocast

    t0 = time.time()
    spent = lambda: (time.time() - t0 + 60.0) * config.GPU_RATE_PER_SEC[DIAG_GPU] * config.COST_OVERHEAD_FACTOR
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    data = sft.tokenize(sft.load_rows(SCALE_DIR / f"{group}.jsonl"), tok, config.SEQ_LEN)
    raw = sft.load_init(config.RAFT_INIT_CKPT).to(device)
    raw.load_state_dict(torch.load(FULL_DIR / "n20000.pt", map_location=device, weights_only=False))
    tau = json.loads((FULL_DIR / "final.json").read_text())["tau"]
    g, t_chunk = [], 0.0
    for i in range(0, len(data.rows), 64):         # C-15: re-check the budget before every chunk
        if spent() + t_chunk * config.GPU_RATE_PER_SEC[DIAG_GPU] * config.COST_OVERHEAD_FACTOR > max_usd:
            raise SystemExit(f"budget: the next chunk would cross --max-usd {max_usd} ({len(g)} rows done)")
        t1, part = time.time(), data.rows[i:i + 64]
        g += E.generate_batch(raw, tok, [r["prompt"] for r in part], [len(r["passages"] or []) for r in part],
                              device, tau=tau, batch=64, autocast=lambda: autocast(device),
                              copy_passages=[r["passages"] or [] for r in part], answer_copy=True)
        t_chunk = max(t_chunk, time.time() - t1)
    gens = [{"id": r["id"], "bucket": r["bucket"], "s": o["s"], "choice": o["choice"], "text": o["text"]}
            for r, o in zip(data.rows, g)]
    return {"gens": gens, "proxies": E.proxy_metrics(data.rows, g), "tau": tau, "spent_usd": spent()}


@app.local_entrypoint()
def answer_copy_check(confirm: bool = False, max_usd: float = 0.15, group: str = "dev"):
    """7.8b (paid: 1 x L40S). Requires --confirm; draws on what is left of the 7.8 GPU cap."""
    if not confirm:
        raise SystemExit("This uses a GPU and costs money. Re-run with --confirm once approved.")
    left = config.RAFT_STAGE_CAPS_USD["7.8-full-gpu"] - SPENT_78_GPU_USD
    if max_usd > left:
        raise SystemExit(f"--max-usd {max_usd} exceeds what is left of the 7.8 GPU cap (${left:.2f})")
    out = answer_copy_regen.remote(max_usd, group)
    name = "dev_n20000_copy3" if group == "dev" else f"{group}_copy3"
    path = Path(".smoke/raft/scale/eval") / f"{name}.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in out["gens"]))
    print(f"tau {out['tau']}: " + " ".join(f"{k} {v:.3f}" for k, v in out["proxies"].items()))
    print(f"spent ${out['spent_usd']:.3f} -> {path}")


@app.local_entrypoint()
def lambda_check(confirm: bool = False, max_usd: float = 1.5):
    """Step 7.6b (paid: 1 x H100). Requires --confirm; draws on the 7.6 cap."""
    if not confirm:
        raise SystemExit("Step 7.6b uses a GPU and costs money. Re-run with --confirm once approved.")
    left = config.RAFT_STAGE_CAPS_USD["7.6-pilot-gpu"] - SPENT_76_USD
    if max_usd > left:
        raise SystemExit(f"--max-usd {max_usd} exceeds what is left of the 7.6 cap (${left:.2f})")
    best = json.loads(Path(".smoke/raft/pilot/sweep_report.json").read_text())["best"]   # the 7.6 selection
    arms = [{"lr": lr, "lambda": lam} for lr, lam in config.SFT.forgetting_arms]
    print(f"7.6b at n {best['n']}, refusal share {best['refusal_share']}: {arms}")
    rep = lambda_sweep.remote(max_usd, arms, best["refusal_share"], best["n"])
    out = Path(".smoke/raft/pilot/lambda_report.json")
    out.write_text(json.dumps(rep, indent=2, default=str))
    for r in rep["results"]:
        print(f"{r['key']}: bpb delta {r['bpb_delta_at_best']:+.4f} casehold {r['casehold_pmi_delta_points']:+.1f} "
              f"score plain {r['plain']['selection_score']:+.3f} copy {r['copy']['selection_score']:+.3f}")
    print(f"spent ${rep['spent_usd']:.2f} -> {out}")


@app.local_entrypoint()
def confirm_check(confirm: bool = False, max_usd: float = 1.4):
    """Step 7.6c (paid: 1 x H100): the recommended remedy, LR 5e-5 / lambda 1.0, with held-out bits/byte
    and CaseHOLD PMI every epoch, judged on the final epoch. Requires --confirm; draws on the 7.6 cap."""
    if not confirm:
        raise SystemExit("Step 7.6c uses a GPU and costs money. Re-run with --confirm once approved.")
    left = config.RAFT_STAGE_CAPS_USD["7.6-pilot-gpu"] - SPENT_76_USD
    if max_usd > left:
        raise SystemExit(f"--max-usd {max_usd} exceeds what is left of the 7.6 cap (${left:.2f})")
    best = json.loads(Path(".smoke/raft/pilot/sweep_report.json").read_text())["best"]
    lr, lam = config.SFT.confirm_arm
    print(f"7.6c at n {best['n']}, refusal share {best['refusal_share']}: lr {lr:g}, lambda {lam:g}, judged on the final epoch")
    rep = lambda_sweep.remote(max_usd, [{"lr": lr, "lambda": lam}], best["refusal_share"], best["n"],
                              per_epoch_casehold=True, judge="final", out_name="pilot_confirm")
    out = Path(".smoke/raft/pilot/confirm_report.json")
    out.write_text(json.dumps(rep, indent=2, default=str))
    for r in rep["results"]:
        ok = r["bpb_delta_judged"] <= config.RAFT.max_bpb_increase and r["casehold_pmi_delta_points"] >= -2.0
        print(f"{r['key']} final step {r['judged_step']}: bpb delta {r['bpb_delta_judged']:+.4f} "
              f"casehold {r['casehold_pmi_delta_points']:+.1f} copy score {r['copy']['selection_score']:+.3f} "
              f"-> {'PASS' if ok else 'FAIL'}")
    print(f"spent ${rep['spent_usd']:.2f} -> {out}")


@app.local_entrypoint()
def pilot_eval_check(confirm: bool = False, max_usd: float = 0.75):
    """Step 7.7a (paid: 1 x H100). Requires --confirm; draws on what is left of the 7.6 GPU cap."""
    if not confirm:
        raise SystemExit("Step 7.7a uses a GPU and costs money. Re-run with --confirm once approved.")
    left = config.RAFT_STAGE_CAPS_USD["7.6-pilot-gpu"] - SPENT_76_USD
    if max_usd > left:
        raise SystemExit(f"--max-usd {max_usd} exceeds what is left of the 7.6 cap (${left:.2f})")
    out = pilot_eval.remote(max_usd)
    d = Path(".smoke/raft/pilot/eval")
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(out["report"], indent=2, default=str))
    for mode, rows in out["gens"].items():
        (d / f"dev_gen_{mode}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    r = out["report"]
    print(f"final step {r['final']['step']}: macro-F1 {r['final']['dev_macro_f1']:.3f}; bpb {r['bpb_delta']:+.4f}; "
          f"casehold {r['casehold_pmi_delta_points']:+.1f}; tau {r['tau']['tau']} (meets {r['tau']['meets_targets']})")
    for mode, m in r["proxies"].items():
        print(f"  {mode}: " + " ".join(f"{k} {v:.3f}" for k, v in m.items()))
    print(f"spent ${r['spent_usd']:.2f} -> {d}")


@app.local_entrypoint()
def sweep(confirm: bool = False, max_usd: float = 4.0, compile: bool = True):
    """Step 7.6 (paid: 1 x H100). Requires --confirm."""
    if not confirm:
        raise SystemExit("Step 7.6 uses a GPU and costs money. Re-run with --confirm once approved.")
    cap = config.RAFT_STAGE_CAPS_USD["7.6-pilot-gpu"]
    if max_usd > cap:
        raise SystemExit(f"--max-usd {max_usd} exceeds the approved cap ${cap}")
    local = Path(".smoke/casehold_test.jsonl")
    with volume.batch_upload(force=True) as up:          # the eval data the container needs
        up.put_file(str(local), str(CASEHOLD.relative_to(config.DATA_ROOT)))
    rep = pilot_sweep.remote(max_usd, compile)
    out = Path(".smoke/raft/pilot/sweep_report.json")
    out.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep.get(k) for k in ("status", "best", "forgetting", "spent_usd")}, indent=1, default=str))
    print(f"-> {out}")
