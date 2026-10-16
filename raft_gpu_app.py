"""Phase 7 GPU steps on Modal (docs/08-raft.md §5, §7). Nothing here runs without --confirm.

    modal run --detach raft_gpu_app.py::sweep --confirm [--max-usd 4]      # step 7.6, 1 x H100

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
CASEHOLD = Path(config.RAFT_DIR) / "eval" / "casehold_test.jsonl"
LOCAL_TRAIN = "/tmp/tokens/train"


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
                             log_every=10_000)
    run_cost = 90 * _rate_per_s()                                      # one 2,000-example run, measured upper bound
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
