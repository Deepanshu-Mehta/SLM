"""Pretraining loop for the 125M SLM — single process, or DDP under torchrun.

Precision: master weights and AdamW state stay fp32; forward/backward run under
bf16 autocast on CUDA. The model is never cast with model.to(bfloat16): bf16's
8-bit mantissa would round small updates to RMSNorm gains (and most weights) to
zero. The LR, the epoch and the data position are pure functions of the global
step, so resuming from ckpt.pt continues the single cosine schedule exactly.

Exit codes: 0 finished (schedule end or --max-steps), 3 budget stop, 4 non-finite loss.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

import config
import model as M
from data import StepLayout, StreamOrder, WindowStore

EXIT_OK, EXIT_BUDGET, EXIT_NONFINITE = 0, 3, 4
BPB_FACTOR = 1.0 / (math.log(2) * config.VAL_BYTES_PER_TOKEN)
SAMPLE_PROMPTS = (
    "The court held that the defendant",
    "The Company's net revenues for the fiscal year",
)


# --------------------------------------------------------------------------- args
@dataclass(frozen=True)
class Args:
    tokens_dir: str
    tokenizer_dir: str
    ckpt_dir: str
    device: str
    gpu: str
    micro_batch: int
    global_batch_tokens: int
    epochs: int
    max_steps: int
    log_every: int
    eval_every: int
    eval_max_batches: int
    ckpt_every: int
    max_usd: float
    guard_after: int
    warmup_steps: int
    startup_allowance_s: float
    compile: bool
    attn: str
    samples: bool
    seed: int


def parse_args(argv: list[str] | None = None) -> Args:
    t = config.TRAIN
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tokens-dir", default=config.TOKENS_DIR)
    p.add_argument("--tokenizer-dir", default=config.TOKENIZER_DIR)
    p.add_argument("--ckpt-dir", default=config.CKPT_DIR)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--gpu", default=config.PRETRAIN_GPU,
                   help="key into config.GPU_RATE_PER_SEC for spend; 'none' = free")
    p.add_argument("--micro-batch", type=int, default=t.micro_batch_size)
    p.add_argument("--global-batch-tokens", type=int, default=t.global_batch_tokens)
    p.add_argument("--epochs", type=int, default=config.PRETRAIN_EPOCHS)
    p.add_argument("--max-steps", type=int, default=0,
                   help="stop after this many completed steps (0 = end of schedule); "
                        "the LR schedule is always the full run's")
    p.add_argument("--log-every", type=int, default=t.log_every_steps)
    p.add_argument("--eval-every", type=int, default=t.eval_every_steps)
    p.add_argument("--eval-max-batches", type=int, default=0, help="0 = whole val split")
    p.add_argument("--ckpt-every", type=int, default=t.ckpt_every_steps)
    p.add_argument("--max-usd", type=float, default=config.BUDGET_CAP_USD)
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="0 = config.TRAIN.warmup_tokens / global batch (381 for the real run)")
    p.add_argument("--guard-after", type=int, default=20,
                   help="steps into a launch before the spend projection is trusted")
    p.add_argument("--startup-allowance-s", type=float, default=120.0,
                   help="billed-but-untimed container start per launch, charged to the ledger")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--no-samples", dest="samples", action="store_false")
    p.add_argument("--seed", type=int, default=t.seed)
    return Args(**vars(p.parse_args(argv)))


# ------------------------------------------------------------------------ runtime
@dataclass(frozen=True)
class Runtime:
    rank: int
    local_rank: int
    world: int
    device: torch.device
    distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_runtime(device_arg: str) -> Runtime:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        if torch.cuda.is_available() and device_arg in ("auto", "cuda"):
            torch.cuda.set_device(local)
            device, backend = torch.device("cuda", local), "nccl"
        else:
            device, backend = torch.device("cpu"), "gloo"
        dist.init_process_group(backend=backend)
        return Runtime(rank, local, world, device, True)
    if device_arg == "auto":
        device_arg = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
    return Runtime(0, 0, 1, torch.device(device_arg), False)


def autocast(device: torch.device) -> contextlib.AbstractContextManager:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def log(rt: Runtime, msg: str) -> None:
    if rt.is_main:
        print(msg, flush=True)


# ----------------------------------------------------------------- schedule / opt
def lr_at(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup then one cosine decay to min_lr. A pure function of the step."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_optimizer(net: torch.nn.Module, device: torch.device) -> torch.optim.AdamW:
    """AdamW; weight decay on matrices/embeddings only, never on RMSNorm gains."""
    decay = [p for p in net.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in net.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": config.TRAIN.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    extra = {"fused": True} if device.type == "cuda" else {}
    return torch.optim.AdamW(groups, lr=config.TRAIN.lr,
                             betas=(config.TRAIN.beta1, config.TRAIN.beta2), eps=1e-8, **extra)


# ------------------------------------------------------------------- spend ledger
@dataclass(frozen=True)
class Ledger:
    """Wall-clock billed so far for one run directory, across all launches."""

    prior_seconds: float
    launches: int
    launch_start: float

    @classmethod
    def open(cls, path: Path, allowance_s: float) -> "Ledger":
        prior, launches = 0.0, 0
        if path.exists():
            saved = json.loads(path.read_text())
            prior, launches = float(saved["wall_seconds"]), int(saved["launches"])
        return cls(prior + allowance_s, launches + 1, time.time())

    def seconds(self) -> float:
        return self.prior_seconds + (time.time() - self.launch_start)

    def write(self, path: Path) -> None:
        atomic_write_text(path, json.dumps(
            {"wall_seconds": self.seconds(), "launches": self.launches, "updated": time.time()}))


def spend_usd(seconds: float, gpu: str, world: int) -> float:
    if gpu not in config.GPU_RATE_PER_SEC:
        return 0.0
    return config.projected_cost_usd(seconds, gpu=gpu, gpu_count=world)


def mfu(tokens_per_s: float, world: int, gpu: str) -> float | None:
    peak = config.GPU_PEAK_BF16_FLOPS.get(gpu)
    if not peak:
        return None
    return 6 * config.MODEL.approx_params() * tokens_per_s / (world * peak)


# -------------------------------------------------------------------------- files
def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_torch_save(obj: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def truncate_metrics(path: Path, keep_through_step: int) -> None:
    """On resume, drop rows logged after the checkpoint so the curve has no repeats."""
    if not path.exists():
        return
    rows = [ln for ln in path.read_text().splitlines()
            if ln.strip() and json.loads(ln).get("step", 0) <= keep_through_step]
    atomic_write_text(path, "".join(r + "\n" for r in rows))


def fingerprint(args: Args, total_steps: int, warmup_steps: int, n_train: int) -> dict[str, Any]:
    """Everything that must match for a resume to continue the same run."""
    t = config.TRAIN
    return {"total_steps": total_steps, "warmup_steps": warmup_steps, "n_train_windows": n_train,
            "global_batch_tokens": args.global_batch_tokens, "seq_len": config.SEQ_LEN,
            "epochs": args.epochs, "seed": args.seed, "lr": t.lr, "min_lr": t.min_lr,
            "params": config.MODEL.approx_params()}


# ---------------------------------------------------------------- eval / export
@torch.no_grad()
def evaluate(raw: torch.nn.Module, store: WindowStore, rt: Runtime, micro_batch: int,
             max_batches: int) -> dict[str, float]:
    raw.eval()
    indices = np.arange(rt.rank, len(store), rt.world, dtype=np.int64)
    if max_batches:
        indices = indices[: max_batches * micro_batch]
    loss_sum = n_tok = 0.0
    for start in range(0, len(indices), micro_batch):
        x = to_device(store.gather(indices[start:start + micro_batch]), rt.device)
        with autocast(rt.device):
            loss = raw(input_ids=x, labels=x, use_cache=False).loss
        n = x.shape[0] * (x.shape[1] - 1)  # labels are shifted inside the model
        loss_sum += float(loss) * n
        n_tok += n
    if rt.distributed:
        t = torch.tensor([loss_sum, n_tok], dtype=torch.float64, device=rt.device)
        dist.all_reduce(t)
        loss_sum, n_tok = float(t[0]), float(t[1])
    raw.train()
    val = loss_sum / n_tok
    return {"val_loss": val, "val_ppl": math.exp(val), "val_bpb": val * BPB_FACTOR,
            "val_tokens": int(n_tok)}


def export_hf(raw: torch.nn.Module, out_dir: Path, tokenizer_dir: str) -> None:
    """HF directory: config.json, bf16 model.safetensors, generation_config.json, tokenizer.

    The bf16 copy is made tensor-by-tensor on CPU; the training model is untouched.
    """
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, GenerationConfig

    out_dir.mkdir(parents=True, exist_ok=True)
    weights = {k: v.detach().to("cpu", torch.bfloat16).contiguous()
               for k, v in raw.state_dict().items()}
    if raw.config.tie_word_embeddings:
        weights.pop("lm_head.weight", None)  # re-tied from embed_tokens on load
    save_file(weights, str(out_dir / "model.safetensors"), metadata={"format": "pt"})

    cfg = copy.deepcopy(raw.config)
    cfg.torch_dtype = "bfloat16"
    cfg.use_cache = True
    cfg.bos_token_id, cfg.eos_token_id, cfg.pad_token_id = (
        M.BOS_TOKEN_ID, M.EOS_TOKEN_ID, M.PAD_TOKEN_ID)
    cfg.save_pretrained(out_dir)
    GenerationConfig(
        bos_token_id=M.BOS_TOKEN_ID, eos_token_id=M.EOS_TOKEN_ID, pad_token_id=M.PAD_TOKEN_ID,
        do_sample=True, temperature=0.8, top_k=50, top_p=0.95,
        max_new_tokens=128, min_new_tokens=16,
    ).save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(tokenizer_dir).save_pretrained(out_dir)


@torch.no_grad()
def generate_samples(raw: torch.nn.Module, tokenizer_dir: str, device: torch.device,
                     seed: int) -> list[dict[str, str]]:
    """Completions prompted with <|eos|> + text, matching how documents start in training."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    raw.eval()
    out = []
    for prompt in SAMPLE_PROMPTS:
        torch.manual_seed(seed)
        ids = [M.PROMPT_PREFIX_TOKEN_ID] + tok.encode(prompt, add_special_tokens=False)
        x = torch.tensor([ids], device=device)
        with autocast(device):
            y = raw.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=60,
                             min_new_tokens=20, do_sample=True, temperature=0.8, top_k=50,
                             top_p=0.95, pad_token_id=M.PAD_TOKEN_ID, eos_token_id=M.EOS_TOKEN_ID)
        out.append({"prompt": prompt,
                    "completion": tok.decode(y[0, len(ids):].tolist(), skip_special_tokens=True)})
    raw.train()
    return out


def to_device(batch: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(batch.astype(np.int64))
    if device.type == "cuda":
        return x.pin_memory().to(device, non_blocking=True)
    return x.to(device)


# -------------------------------------------------------------------------- train
def train(args: Args) -> int:
    rt = init_runtime(args.device)
    try:
        return _train(args, rt)
    finally:
        if rt.distributed and dist.is_initialized():
            dist.destroy_process_group()


def _train(args: Args, rt: Runtime) -> int:
    ckpt_dir = Path(args.ckpt_dir)
    paths = {"ckpt": ckpt_dir / "ckpt.pt", "metrics": ckpt_dir / "metrics.jsonl",
             "status": ckpt_dir / "run_status.json", "ledger": ckpt_dir / "spend.json",
             "samples": ckpt_dir / "samples.json", "base": ckpt_dir / "base"}
    if rt.is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if rt.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_store = WindowStore.open(os.path.join(args.tokens_dir, "train"), config.SEQ_LEN)
    val_store = WindowStore.open(os.path.join(args.tokens_dir, "val"), config.SEQ_LEN)
    layout = StepLayout.for_batch(args.global_batch_tokens, config.SEQ_LEN, args.micro_batch,
                                  rt.world)
    order = StreamOrder(len(train_store), args.seed)
    total_steps = (len(train_store) * args.epochs) // layout.windows_per_step
    warmup_steps = args.warmup_steps or max(1, config.TRAIN.warmup_tokens // args.global_batch_tokens)
    end_step = min(total_steps, args.max_steps) if args.max_steps else total_steps
    fp = fingerprint(args, total_steps, warmup_steps, len(train_store))

    # model -> (resume) -> optimizer -> compile -> DDP
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    raw = M.build_model(attn_implementation=args.attn).to(rt.device)
    opt = build_optimizer(raw, rt.device)
    start_step = 0
    if paths["ckpt"].exists():
        state = torch.load(paths["ckpt"], map_location=rt.device, weights_only=False)
        mismatched = {k: (state["fingerprint"].get(k), v) for k, v in fp.items()
                      if state["fingerprint"].get(k) != v}
        if mismatched:
            raise SystemExit(f"checkpoint belongs to a different run: {mismatched}")
        raw.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        torch.set_rng_state(state["rng"]["torch"].cpu())
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
        del state
        log(rt, f"[resume] from step {start_step}")
        if rt.is_main:
            truncate_metrics(paths["metrics"], start_step)
    net: torch.nn.Module = torch.compile(raw) if args.compile else raw
    if rt.distributed:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[rt.local_rank] if rt.device.type == "cuda" else None)

    ledger = Ledger.open(paths["ledger"], args.startup_allowance_s) if rt.is_main else None
    log(rt, f"[run] world={rt.world} device={rt.device} micro_batch={layout.micro_batch} "
            f"accum={layout.accum} windows/step={layout.windows_per_step} "
            f"train_windows={len(train_store):,} val_windows={len(val_store):,} "
            f"steps {start_step}->{end_step} of {total_steps:,} warmup={warmup_steps} "
            f"compile={args.compile} attn={args.attn} params={config.MODEL.approx_params():,}")
    if start_step >= end_step:
        log(rt, "[run] nothing to do: checkpoint is already at the target step")
        return EXIT_OK

    tokens_per_step = layout.windows_per_step * config.SEQ_LEN
    status, exit_code, last_log = "max_steps" if end_step < total_steps else "completed", EXIT_OK, {}
    t_last, steps_since_log = time.time(), 0
    t_guard, guard_from = None, start_step + args.guard_after
    raw.train()

    for step in range(start_step, end_step):
        lr = lr_at(step, total_steps, warmup_steps, config.TRAIN.lr, config.TRAIN.min_lr)
        for group in opt.param_groups:
            group["lr"] = lr
        loss_acc = torch.zeros((), device=rt.device)
        for micro in range(layout.accum):
            x = to_device(train_store.gather(order.windows(layout.positions(step, micro, rt.rank))),
                          rt.device)
            sync = (net.no_sync() if rt.distributed and micro < layout.accum - 1
                    else contextlib.nullcontext())
            with sync:
                with autocast(rt.device):
                    loss = net(input_ids=x, labels=x, use_cache=False).loss / layout.accum
                loss.backward()
            loss_acc += loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), config.TRAIN.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        done = step + 1
        steps_since_log += 1
        if done == guard_from:
            t_guard = time.time()

        is_last = done == end_step
        stop = False
        if done % args.log_every == 0 or is_last:
            if rt.distributed:  # SUM / world: ReduceOp.AVG exists only on NCCL, not gloo
                dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM)
                loss_acc /= rt.world
            train_loss, gnorm = float(loss_acc), float(grad_norm)
            now = time.time()
            tok_s = steps_since_log * tokens_per_step / (now - t_last)
            t_last, steps_since_log = now, 0
            flag = torch.zeros((), device=rt.device)
            if rt.is_main:
                assert ledger is not None
                ledger.write(paths["ledger"])
                spent = spend_usd(ledger.seconds(), args.gpu, rt.world)
                sec_per_step = ((now - t_guard) / (done - guard_from)
                                if t_guard is not None and done > guard_from else None)
                projected = (spent + spend_usd(sec_per_step * (end_step - done), args.gpu, rt.world)
                             if sec_per_step else None)
                full_run = (spend_usd(sec_per_step * total_steps, args.gpu, rt.world)
                            if sec_per_step else None)
                epoch, offset = order.locate(done * layout.windows_per_step)
                last_log = {"step": done, "epoch": epoch, "epoch_offset": offset,
                            "tokens_seen": done * tokens_per_step, "loss": train_loss,
                            "lr": lr, "grad_norm": gnorm, "tok_per_s": tok_s,
                            "mfu": mfu(tok_s, rt.world, args.gpu),
                            "sec_per_step": sec_per_step, "elapsed_s": ledger.seconds(),
                            "spend_usd": spent, "projected_usd": projected,
                            "full_run_usd": full_run}
                append_metrics(paths["metrics"], last_log)
                m = last_log["mfu"]
                print(f"step {done:>6}/{total_steps} ep {epoch} loss {train_loss:.4f} lr {lr:.2e} "
                      f"gnorm {gnorm:.3f} {tok_s:,.0f} tok/s"
                      + (f" mfu {m:.1%}" if m is not None else "")
                      + f" ${spent:.2f}"
                      + (f" proj ${projected:.2f} full-run ${full_run:.2f}" if projected else ""),
                      flush=True)
                if not (math.isfinite(train_loss) and math.isfinite(gnorm)):
                    flag.fill_(EXIT_NONFINITE)
                elif spent > args.max_usd or (projected is not None and projected > args.max_usd):
                    flag.fill_(EXIT_BUDGET)
            if rt.distributed:
                dist.broadcast(flag, src=0)
            code = int(flag)
            if code == EXIT_NONFINITE:
                log(rt, f"[stop] non-finite loss/grad at step {done}; last checkpoint is clean")
                status, exit_code = "nonfinite", EXIT_NONFINITE
                break  # never checkpoint a poisoned model
            if code == EXIT_BUDGET:
                log(rt, f"[stop] budget: projected spend exceeds --max-usd {args.max_usd}")
                status, exit_code, stop = "budget_stop", EXIT_BUDGET, True

        if done % args.eval_every == 0 or is_last:
            ev = evaluate(raw, val_store, rt, args.micro_batch, args.eval_max_batches)
            if rt.is_main:
                append_metrics(paths["metrics"], {"step": done, **ev})
                print(f"[eval] step {done} val_loss {ev['val_loss']:.4f} "
                      f"ppl {ev['val_ppl']:.2f} bpb {ev['val_bpb']:.4f} "
                      f"({ev['val_tokens']:,} tok)", flush=True)
                last_log = {**last_log, **ev}

        if done % args.ckpt_every == 0 or is_last or stop:
            if rt.is_main:
                t0 = time.time()
                epoch, offset = order.locate(done * layout.windows_per_step)
                atomic_torch_save({
                    "model": raw.state_dict(), "optimizer": opt.state_dict(), "step": done,
                    "fingerprint": fp, "data_position": {"epoch": epoch, "offset": offset},
                    "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                            "python": random.getstate()},
                }, paths["ckpt"])
                print(f"[ckpt] step {done} saved in {time.time() - t0:.1f}s", flush=True)
            if rt.distributed:
                dist.barrier()
        if stop:
            break

    if rt.is_main:
        if status in ("completed", "max_steps"):
            export_hf(raw, paths["base"], args.tokenizer_dir)
            print(f"[export] HF model -> {paths['base']}", flush=True)
            if args.samples:
                samples = generate_samples(raw, args.tokenizer_dir, rt.device, args.seed)
                atomic_write_text(paths["samples"], json.dumps(samples, indent=2))
                for s in samples:
                    print(f"[sample] {s['prompt']} >>> {s['completion']}", flush=True)
        assert ledger is not None
        ledger.write(paths["ledger"])
        atomic_write_text(paths["status"], json.dumps({
            "status": status, "exit_code": exit_code, "world": rt.world, "gpu": args.gpu,
            "compile": args.compile, "micro_batch": layout.micro_batch, "accum": layout.accum,
            "total_steps": total_steps, "start_step": start_step, "end_step": end_step,
            "spend_usd": spend_usd(ledger.seconds(), args.gpu, rt.world), "last": last_log,
        }, indent=2))
    if rt.distributed:
        dist.barrier()
    return exit_code


def main(argv: list[str] | None = None) -> None:
    sys.exit(train(parse_args(argv)))


if __name__ == "__main__":
    main()
