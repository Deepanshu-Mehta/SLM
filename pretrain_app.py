"""Modal GPU launcher for Phase 5. Nothing here runs without an explicit --confirm.

    modal run --detach pretrain_app.py::calibrate --confirm            # Step 2, 1xH100, < $1
    modal run --detach pretrain_app.py::pretrain --confirm --max-usd 49 [--compile]   # Step 3

Each function stages the token shards from the Volume onto local disk (training reads
random 2 KB windows, which a network volume serves slowly), launches train.py under
torchrun, and commits the Volume from a background thread whenever the run directory
changes — train.py runs in torchrun subprocesses, which cannot call volume.commit()
themselves. A preempted container restarts with the same input and train.py resumes
from the last committed ckpt.pt.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time

import modal

import config

app = modal.App(f"{config.PROJECT}-pretrain")

# All apt/pip steps before add_local_python_source (Modal rule). gcc: Triton needs a
# C compiler for its runtime helpers when torch.compile is used.
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("gcc")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "numpy==2.1.3", "safetensors==0.4.5")
    .add_local_python_source("config", "model", "data", "train")
)
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}

LOCAL_TOKENS = "/tmp/tokens"
CALIB_DIR = f"{config.CKPT_DIR}/calibration"


def _stage_tokens() -> str:
    """Copy train/val shards to container-local disk; fall back to the Volume path."""
    t0 = time.time()
    try:
        for split in ("train", "val"):
            shutil.copytree(f"{config.TOKENS_DIR}/{split}", f"{LOCAL_TOKENS}/{split}",
                            dirs_exist_ok=True)
        print(f"[stage] tokens copied to {LOCAL_TOKENS} in {time.time() - t0:.0f}s", flush=True)
        return LOCAL_TOKENS
    except OSError as exc:
        print(f"[stage] local copy failed ({exc}); reading from the Volume", flush=True)
        return config.TOKENS_DIR


def _signature(run_dirs: list[str]) -> tuple:
    sig = []
    for d in run_dirs:
        for name in ("ckpt.pt", "spend.json", "metrics.jsonl", "run_status.json"):
            path = os.path.join(d, name)
            if os.path.exists(path):
                st = os.stat(path)
                sig.append((path, st.st_mtime_ns, st.st_size))
    return tuple(sig)


def _committer(stop: threading.Event, run_dirs: list[str], period_s: float = 30.0) -> None:
    last: tuple = ()
    while not stop.wait(period_s):
        try:
            sig = _signature(run_dirs)
            if sig != last:
                volume.commit()
                last = sig
        except Exception as exc:  # a failed commit is retried on the next tick
            print(f"[commit] retrying later: {exc}", flush=True)


def _launch(runs: list[list[str]], n_gpus: int) -> list[dict]:
    """Run train.py once per argv list, sequentially, in this one container."""
    tokens_dir = _stage_tokens()
    run_dirs = [argv[argv.index("--ckpt-dir") + 1] for argv in runs]
    stop = threading.Event()
    thread = threading.Thread(target=_committer, args=(stop, run_dirs), daemon=True)
    thread.start()
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join([os.path.dirname(config.__file__),
                                          os.environ.get("PYTHONPATH", "")]),
           # Each rank compiles independently; inductor's default of up to 32 worker
           # processes per rank would mean ~256 torch-importing processes on 8 GPUs.
           "TORCHINDUCTOR_COMPILE_THREADS": "4"}
    results = []
    try:
        for argv, run_dir in zip(runs, run_dirs):
            cmd = ["torchrun", "--standalone", f"--nproc_per_node={n_gpus}", "-m", "train",
                   "--tokens-dir", tokens_dir, "--gpu", config.PRETRAIN_GPU, *argv]
            print(f"[launch] {' '.join(cmd)}", flush=True)
            rc = subprocess.run(cmd, env=env).returncode
            status_path = os.path.join(run_dir, "run_status.json")
            status = json.load(open(status_path)) if os.path.exists(status_path) else {}
            results.append({"argv": argv, "returncode": rc, **status})
            volume.commit()
    finally:
        stop.set()
        thread.join(timeout=5)
        volume.commit()
    return results


@app.function(image=gpu_image, gpu="H100", volumes=VOLUMES, timeout=45 * 60)
def calibrate_1xh100() -> list[dict]:
    """Step 2: eager 30 steps -> resume to 50 (GPU resume test) -> compiled 50 steps."""
    shared = ["--max-usd", "2", "--eval-every", "100000", "--eval-max-batches", "20",
              "--ckpt-every", "100000", "--log-every", "5"]
    eager, compiled = f"{CALIB_DIR}/eager", f"{CALIB_DIR}/compiled"
    for d in (eager, compiled):  # calibration always starts from scratch
        shutil.rmtree(d, ignore_errors=True)
    return _launch([
        ["--ckpt-dir", eager, "--max-steps", "30", "--no-samples", *shared],
        ["--ckpt-dir", eager, "--max-steps", "50", *shared],
        ["--ckpt-dir", compiled, "--max-steps", "50", "--compile", *shared],
    ], n_gpus=1)


# Explicit host resources: 8 ranks + their compile workers must not be squeezed into
# Modal's tiny default request. ~$0.0004/s, inside COST_OVERHEAD_FACTOR's 15%.
@app.function(image=gpu_image, gpu=f"{config.PRETRAIN_GPU}:{config.PRETRAIN_GPU_COUNT}",
              volumes=VOLUMES, timeout=6 * 60 * 60, cpu=16.0, memory=96 * 1024)
def pretrain_8xh100(max_usd: float, compile: bool) -> dict:
    """Step 3: the full run. Resumes automatically if /data/checkpoints/ckpt.pt exists."""
    argv = ["--ckpt-dir", config.CKPT_DIR, "--max-usd", str(max_usd)]
    if compile:
        argv.append("--compile")
    return _launch([argv], n_gpus=config.PRETRAIN_GPU_COUNT)[0]


def _require(confirm: bool, what: str) -> None:
    if not confirm:
        raise SystemExit(f"Refusing to start {what} without --confirm (this spends GPU money).")


@app.local_entrypoint()
def calibrate(confirm: bool = False):
    _require(confirm, "calibration on 1xH100")
    results = calibrate_1xh100.remote()
    print("\n=== CALIBRATION (1 x H100) ===")
    best = None
    for r in results:
        last = r.get("last", {})
        mfu, full = last.get("mfu"), last.get("full_run_usd")
        print(f"  {' '.join(r['argv'][:4]):<60} rc={r['returncode']} status={r.get('status')} "
              f"steps {r.get('start_step')}->{r.get('end_step')} "
              f"tok/s={last.get('tok_per_s', 0):,.0f} mfu={(mfu or 0):.1%} "
              f"4-epoch est=${(full or 0):.2f} spent=${r.get('spend_usd', 0):.2f} "
              f"val_bpb={last.get('val_bpb', float('nan')):.4f}")
        if full and (best is None or full < best[0]):
            best = (full, mfu, "--compile" in r["argv"])
    if best:
        cost, mfu, compiled = best
        verdict = ("PROCEED to Step 3 (needs approval)" if cost <= 42
                   else "STOP: too close to call" if cost <= config.BUDGET_CAP_USD
                   else "DO NOT LAUNCH: over budget")
        print(f"\n  best: {'compiled' if compiled else 'eager'} mfu={mfu:.1%} -> "
              f"4-epoch ${cost:.2f} -> {verdict}")


@app.local_entrypoint()
def pretrain(confirm: bool = False, max_usd: float = config.BUDGET_CAP_USD,
             compile: bool = False):
    _require(confirm, f"pretraining on {config.PRETRAIN_GPU_COUNT}x{config.PRETRAIN_GPU}")
    if max_usd > config.BUDGET_CAP_USD:
        raise SystemExit(f"--max-usd {max_usd} exceeds config.BUDGET_CAP_USD "
                         f"{config.BUDGET_CAP_USD}")
    print(json.dumps(pretrain_8xh100.remote(max_usd, compile), indent=2))
