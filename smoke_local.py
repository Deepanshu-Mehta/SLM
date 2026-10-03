"""Phase 5 Step 1: local smoke test (CPU, fp32, $0). Exits non-zero on any failure.

  A. unit checks: LR schedule, data layout invariants, real-run step arithmetic
  B. reference run: 20 steps, uninterrupted
  C. crash test: same run SIGKILLed after step 13, resumed -> must match B exactly
  D. DDP: 2 processes (gloo) -> per-step losses must match B
  E. HF export: from_pretrained round-trip, token ids, tied weights, bf16 logits

Runs on CPU, not MPS, because CPU is deterministic, which is what lets C demand an
exact match. CUDA-only paths (bf16 autocast, fused AdamW, NCCL, compile-on-CUDA)
are first exercised by the Step 2 calibration.
"""

from __future__ import annotations

import json
import math
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import config
from data import StepLayout, StreamOrder, WindowStore, epoch_permutation
from train import lr_at

ROOT = Path(".smoke")
PY = sys.executable
TORCHRUN = str(Path(PY).with_name("torchrun"))
COMMON = ["--tokens-dir", str(ROOT / "tokens"), "--tokenizer-dir", str(ROOT / "tokenizer"),
          "--device", "cpu", "--gpu", "none", "--micro-batch", "1",
          "--global-batch-tokens", "2048", "--log-every", "1", "--ckpt-every", "5",
          "--eval-every", "10", "--eval-max-batches", "4", "--startup-allowance-s", "0",
          "--warmup-steps", "5"]
STEPS = 20
failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {what}", flush=True)
    if not ok:
        failures.append(what)


def train_rows(run: Path) -> dict[int, dict]:
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    return {r["step"]: r for r in rows if "loss" in r}


def run_train(run: Path, extra: list[str] = (), launcher: list[str] | None = None) -> int:
    cmd = (launcher or [PY, "-m", "train"]) + ["--ckpt-dir", str(run), *COMMON, *extra]
    return subprocess.run(cmd).returncode


# ----------------------------------------------------------------------------- A
def unit_checks() -> None:
    print("\n=== A. unit checks ===")
    total, warm, hi, lo = 15_564, 381, config.TRAIN.lr, config.TRAIN.min_lr
    lrs = [lr_at(s, total, warm, hi, lo) for s in range(total)]
    check(abs(lrs[0] - hi / warm) < 1e-12 and abs(lrs[warm - 1] - hi) < 1e-12,
          f"warmup ramps linearly to peak {hi} at step {warm - 1}")
    check(all(a >= b for a, b in zip(lrs[warm - 1:], lrs[warm:])),
          "LR never increases after warmup (one cosine, no restarts)")
    check(abs(lr_at(total, total, warm, hi, lo) - lo) < 1e-12 and lrs[-1] - lo < 1e-9,
          f"LR reaches min_lr {lo} at the end of the schedule")

    n = 1_000
    perm = epoch_permutation(n, 0, 1337)
    check(np.array_equal(np.sort(perm), np.arange(n)), "each epoch visits every window exactly once")
    check(not np.array_equal(perm, epoch_permutation(n, 1, 1337)), "each epoch has a different order")
    order = StreamOrder(n, 1337)
    for layouts in [(StepLayout(2, 1, 2), StepLayout(1, 2, 2), StepLayout(1, 4, 1))]:
        sets = []
        for lay in layouts:
            for step in (0, 7, 249, 250):  # 250 * 4 = 1000 -> crosses an epoch boundary
                pos = np.concatenate([lay.positions(step, m, r)
                                      for m in range(lay.accum) for r in range(lay.world)])
                assert len(np.unique(pos)) == len(pos)
                sets.append((step, tuple(sorted(order.windows(pos)))))
        by_step = {}
        for step, s in sets:
            by_step.setdefault(step, set()).add(s)
        check(all(len(v) == 1 for v in by_step.values()),
              "a step reads the same windows for any (world, micro-batch) with equal batch")
    lay = StepLayout(1, 2, 2)
    pos = [tuple(lay.positions(3, m, r)) for m in range(2) for r in range(2)]
    check(len(set(pos)) == 4, "ranks and micro-steps read disjoint positions")

    store = WindowStore.open(str(ROOT / "tokens" / "val"), config.SEQ_LEN)
    raw = np.fromfile(store.paths[0], dtype=np.uint16).reshape(-1, config.SEQ_LEN)
    idx = np.array([0, 5, len(store) - 1])
    check(np.array_equal(store.gather(idx), raw[idx]), "WindowStore.gather matches the raw file")

    real_train = 2_040_088_576 // config.SEQ_LEN
    steps = real_train * config.PRETRAIN_EPOCHS // StepLayout.for_batch(
        config.TRAIN.global_batch_tokens, config.SEQ_LEN, 32, 8).windows_per_step
    check(steps == config.pretrain_steps(2_040_088_576) == 15_564,
          f"real run: {steps:,} steps from the layout = config.pretrain_steps()")
    check(StepLayout.for_batch(524_288, 1024, 32, 8).accum == 2
          and StepLayout.for_batch(524_288, 1024, 32, 1).accum == 16,
          "grad accumulation: 2 on 8 GPUs, 16 on 1 GPU (same 524,288-token batch)")


# ------------------------------------------------------------------------- B, C
def reference_and_crash() -> None:
    print("\n=== B. reference run (20 steps, uninterrupted) ===")
    ref = ROOT / "ref"
    t0 = time.time()
    rc = run_train(ref, ["--max-steps", str(STEPS)])
    print(f"  ({time.time() - t0:.0f}s)")
    check(rc == 0, "reference run exits 0")
    rows = train_rows(ref)
    losses = [rows[s]["loss"] for s in range(1, STEPS + 1)]
    check(sorted(rows) == list(range(1, STEPS + 1)), "one metrics row per step 1..20")
    check(all(math.isfinite(x) for x in losses), "all losses finite")
    check(losses[0] > 9.0, f"step-1 loss {losses[0]:.3f} ~ ln(16384) = {math.log(16384):.3f} "
                           "(random init over the vocab)")
    check(np.mean(losses[-5:]) < losses[0] - 1.0,
          f"loss falls: {losses[0]:.3f} -> mean of last 5 {np.mean(losses[-5:]):.3f}")
    status = json.loads((ref / "run_status.json").read_text())
    check(status["status"] == "max_steps" and status["end_step"] == STEPS, "status = max_steps @ 20")
    check((ref / "samples.json").exists(), "samples generated (prompted with <|eos|>)")

    print("\n=== C. crash test: SIGKILL after step 13, then resume ===")
    crash = ROOT / "crash"
    cmd = [PY, "-m", "train", "--ckpt-dir", str(crash), *COMMON, "--max-steps", str(STEPS)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    killed_at = None
    while proc.poll() is None:
        time.sleep(0.3)
        mpath = crash / "metrics.jsonl"
        if mpath.exists():
            done = [s for s in train_rows(crash)] if mpath.stat().st_size else []
            if done and max(done) >= 13:
                killed_at = max(done)
                proc.send_signal(signal.SIGKILL)
                proc.wait()
    check(killed_at is not None and 13 <= killed_at < 15,
          f"killed mid-run at step {killed_at} (last checkpoint: step 10)")
    out = subprocess.run(cmd, capture_output=True, text=True)
    check(out.returncode == 0, "resumed run exits 0")
    check("[resume] from step 10" in out.stdout, "resume picked up the step-10 checkpoint")
    rows_c = train_rows(crash)
    check(sorted(rows_c) == list(range(1, STEPS + 1)),
          "metrics truncated on resume: exactly one row per step, no repeats")
    exact = all(rows_c[s]["loss"] == rows[s]["loss"] for s in range(1, STEPS + 1))
    worst = max(abs(rows_c[s]["loss"] - rows[s]["loss"]) for s in range(1, STEPS + 1))
    check(exact, f"every per-step loss identical to the uninterrupted run (max diff {worst:.2e})")
    lr_same = all(rows_c[s]["lr"] == rows[s]["lr"] for s in range(1, STEPS + 1))
    check(lr_same, "LR identical at every step: the schedule continued, no re-warm")

    import torch
    a = torch.load(ref / "ckpt.pt", weights_only=False)["model"]
    b = torch.load(crash / "ckpt.pt", weights_only=False)["model"]
    same = all(torch.equal(a[k], b[k]) for k in a)
    check(same, "final weights bit-identical to the uninterrupted run")


# ----------------------------------------------------------------------------- D
def ddp_check() -> None:
    print("\n=== D. DDP: 2 processes (gloo), same 2048-token batch ===")
    run = ROOT / "ddp"
    t0 = time.time()
    rc = run_train(run, ["--max-steps", "6", "--no-samples"],
                   launcher=[TORCHRUN, "--standalone", "--nproc_per_node=2", "-m", "train"])
    print(f"  ({time.time() - t0:.0f}s)")
    check(rc == 0, "torchrun --nproc_per_node=2 exits 0")
    ref, ddp = train_rows(ROOT / "ref"), train_rows(run)
    diffs = [abs(ddp[s]["loss"] - ref[s]["loss"]) for s in range(1, 7)]
    check(max(diffs) < 2e-3, f"per-step loss matches single-process run (max diff {max(diffs):.2e})")
    st = json.loads((run / "run_status.json").read_text())
    check(st["world"] == 2 and st["accum"] == 1, "world=2, accum=1 (vs accum=2 single-process)")


# ----------------------------------------------------------------------------- E
def export_check() -> None:
    print("\n=== E. HF export round-trip ===")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    import model as M

    base = ROOT / "ref" / "base"
    files = sorted(p.name for p in base.iterdir())
    need = {"config.json", "model.safetensors", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json"}
    check(need <= set(files), f"export has {sorted(need)}")
    cfg = json.loads((base / "config.json").read_text())
    check((cfg["bos_token_id"], cfg["eos_token_id"], cfg["pad_token_id"]) == (0, 1, 2),
          "config.json: bos 0 / eos 1 / pad 2 (not LlamaConfig's 1/2 defaults)")
    gen = GenerationConfig.from_pretrained(base)
    check((gen.bos_token_id, gen.eos_token_id, gen.pad_token_id) == (0, 1, 2),
          "generation_config.json: bos 0 / eos 1 / pad 2")
    hf = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    n = sum(p.numel() for p in hf.parameters())
    check(n == config.MODEL.approx_params(), f"loaded model has {n:,} params")
    check(hf.lm_head.weight.data_ptr() == hf.model.embed_tokens.weight.data_ptr(),
          "lm_head re-tied to embed_tokens after load")
    tok = AutoTokenizer.from_pretrained(base)
    check(tok.convert_tokens_to_ids("<|eos|>") == M.EOS_TOKEN_ID, "exported tokenizer: <|eos|> = 1")

    trained = M.build_model()
    trained.load_state_dict(torch.load(ROOT / "ref" / "ckpt.pt", weights_only=False)["model"])
    x = torch.from_numpy(WindowStore.open(str(ROOT / "tokens" / "val"), config.SEQ_LEN)
                         .gather(np.array([0])).astype(np.int64))
    with torch.no_grad():
        a = trained(x).logits.log_softmax(-1)
        b = hf(x).logits.log_softmax(-1)
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    check(agree > 0.95, f"bf16 export agrees with fp32 weights on {agree:.1%} of next-token argmaxes")
    samples = json.loads((ROOT / "ref" / "samples.json").read_text())
    for s in samples:
        print(f"    sample (20 steps — gibberish expected): {s['prompt']} >>> {s['completion'][:70]!r}")


if __name__ == "__main__":
    for d in ("ref", "crash", "ddp"):
        shutil.rmtree(ROOT / d, ignore_errors=True)
    t_start = time.time()
    unit_checks()
    reference_and_crash()
    ddp_check()
    export_check()
    print(f"\n=== {'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED'} "
          f"({time.time() - t_start:.0f}s) ===")
    for f in failures:
        print(f"  FAIL: {f}")
    sys.exit(1 if failures else 0)
