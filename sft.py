"""Phase 7 SFT trainer (docs/08-raft.md §7). One process: one GPU, or CPU for the local smoke test.

Loss (§7.1): L = L_sft + lambda * L_replay. Each term is a per-token mean of fp32 cross-entropy
over its OWN target tokens in the WHOLE global batch: the token counts are computed before the
first backward, and every micro-batch's sums are divided by them, so gradient accumulation cannot
reweight tokens. Within SFT the mean stays per-token: every example's single decision token weighs
the same, so the refusal share is the decision prior (a per-example mean would roughly double it).

Every micro-batch has a fixed shape: `sft_micro` SFT rows + `replay_micro` raw pretraining windows,
all of length `seq_len` (1,024 on the GPU, so torch.compile compiles once). SFT rows are right
padded; with causal attention no attention mask is needed.

The SFT row order (StreamOrder), the replay windows (seeded by step) and the LR are pure functions
of the step, so resuming from ckpt.pt continues bit-identically.

Exit codes: 0 finished, 3 budget stop, 4 non-finite loss.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import config
import eval_raft as E
import model as M
import raft_data as R
from data import StreamOrder, WindowStore
from train import (EXIT_BUDGET, EXIT_NONFINITE, EXIT_OK, Ledger, append_metrics, atomic_torch_save,
                   atomic_write_text, autocast, build_optimizer, export_hf, init_runtime, lr_at,
                   spend_usd, truncate_metrics)

S = config.SFT
CHAT_ROWS = (4, 5, 6)
UNTRAINED_ROWS = (0, 2, 3, 4, 5, 6)


# ------------------------------------------------------------------------------- data
@dataclass
class SFTData:
    ids: np.ndarray            # (N, seq_len) int64, right-padded with <|pad|>
    mask: np.ndarray           # (N, seq_len) bool: target tokens (decision token .. final <|eos|>)
    length: np.ndarray         # (N,)
    decision_pos: np.ndarray   # (N,) index of the first target token
    refusal: np.ndarray        # (N,) bool
    buckets: list[str]
    rows: list[dict]

    def __len__(self) -> int:
        return len(self.rows)


def load_rows(path: str | Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


REFUSAL_BUCKETS = ("B1", "B2", "B3")


def select_mix(rows: list[dict], n: int, refusal_share: float, seed: int = S.seed) -> list[dict]:
    """n rows with the given refusal share; within each side the buckets keep the config's ratios
    (B1:B2:B3 = 15:7.5:2.5, A:C = 55:20). Fails if a bucket is too small."""
    mix = dict(config.RAFT.mix)
    rng = random.Random(f"{seed}:{n}:{refusal_share}")
    out = []
    for side, buckets, share in ((True, REFUSAL_BUCKETS, refusal_share),
                                 (False, ("A", "C"), 1 - refusal_share)):
        total = sum(mix[b] for b in buckets)
        want = round(n * share)
        counts = {b: round(want * mix[b] / total) for b in buckets}
        counts[buckets[0]] += want - sum(counts.values())            # rounding remainder
        for b, c in counts.items():
            pool = [r for r in rows if r["bucket"] == b]
            if c > len(pool):
                raise SystemExit(f"select_mix: need {c} {b} rows, have {len(pool)}")
            out += rng.sample(pool, c)
    rng.shuffle(out)
    return out


def tokenize(rows: list[dict], tok, seq_len: int) -> SFTData:
    n = len(rows)
    ids = np.full((n, seq_len), M.PAD_TOKEN_ID, dtype=np.int64)
    mask = np.zeros((n, seq_len), dtype=bool)
    length = np.zeros(n, dtype=np.int64)
    dpos = np.zeros(n, dtype=np.int64)
    for i, r in enumerate(rows):
        x, labels = R.encode_example(tok, r["prompt"], r["target"])
        if len(x) > seq_len:
            raise SystemExit(f"{r['id']}: {len(x)} tokens > seq_len {seq_len}")
        p = labels.count(-100)
        ids[i, :len(x)] = x
        mask[i, p:len(x)] = True
        length[i], dpos[i] = len(x), p
    return SFTData(ids, mask, length, dpos, np.array([r["bucket"] in REFUSAL_BUCKETS for r in rows]),
                   [r["bucket"] for r in rows], rows)


# ------------------------------------------------------------------------------ model
def load_init(path: str, attn: str = "sdpa") -> torch.nn.Module:
    """fp32 model from the Phase 5 ckpt.pt (state['model'] only) or an HF directory."""
    if path.endswith(".pt"):
        raw = M.build_model(attn_implementation=attn)
        state = torch.load(path, map_location="cpu", weights_only=False)
        raw.load_state_dict(state["model"])
        del state
    else:
        from transformers import AutoModelForCausalLM
        raw = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32,
                                                   attn_implementation=attn)
    if raw.config.tie_word_embeddings:
        raw.tie_weights()
    return raw


@torch.no_grad()
def init_chat_tokens(raw: torch.nn.Module, seed: int = S.seed) -> dict:
    """Rows 4-6 <- samples from N(mu, Sigma) of the trained embedding rows (§7.1; E-19: the
    untrained rows had collapsed onto one direction). Tied, so the output side follows."""
    E_ = raw.get_input_embeddings().weight
    keep = torch.ones(E_.shape[0], dtype=torch.bool)
    keep[list(UNTRAINED_ROWS)] = False
    T = E_.detach().cpu()[keep].double()             # statistics and sampling on CPU in float64
                                                     # (the generator is CPU; MPS has no float64)
    mu = T.mean(0)
    cov = torch.cov(T.T) + 1e-6 * torch.eye(T.shape[1], dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    z = torch.randn((len(CHAT_ROWS), T.shape[1]), generator=g, dtype=torch.float64)
    sample = mu + z @ torch.linalg.cholesky(cov).T
    E_[list(CHAT_ROWS)] = sample.to(E_.dtype).to(E_.device)
    rows = E_[list(CHAT_ROWS)].float()
    cos = F.cosine_similarity(rows[:, None], rows[None], dim=-1)
    return {"pairwise_cos_max": float(cos[~torch.eye(3, dtype=torch.bool)].max()),
            "norms": [round(float(v), 3) for v in rows.norm(dim=1)]}


# --------------------------------------------------------------------------- schedule
@dataclass(frozen=True)
class Plan:
    seq_len: int
    micro: int
    sft_micro: int
    replay_micro: int
    accum: int
    steps_per_epoch: int
    total_steps: int
    warmup_steps: int

    @property
    def sft_per_step(self) -> int:
        return self.sft_micro * self.accum


def make_plan(n_sft: int, epochs: int, seq_len: int, batch_sequences: int = S.batch_sequences,
              micro: int = S.micro_batch, sft_per_replay: int = S.sft_per_replay,
              warmup_fraction: float = S.warmup_fraction) -> Plan:
    if batch_sequences % micro:
        raise SystemExit(f"batch {batch_sequences} not divisible by micro {micro}")
    replay_micro = round(micro / (sft_per_replay + 1)) if sft_per_replay else 0
    sft_micro = micro - replay_micro
    accum = batch_sequences // micro
    spe = math.ceil(n_sft / (sft_micro * accum))
    total = spe * epochs
    return Plan(seq_len, micro, sft_micro, replay_micro, accum, spe, total,
                max(1, round(warmup_fraction * total)))


def micro_batch(step: int, micro: int, data: SFTData, order: StreamOrder, store: WindowStore | None,
                plan: Plan, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(sft_ids, sft_mask, replay_ids) for one micro-batch: pure functions of (step, micro)."""
    base = step * plan.sft_per_step + micro * plan.sft_micro
    idx = order.windows(np.arange(base, base + plan.sft_micro, dtype=np.int64))
    rep = np.zeros((0, plan.seq_len), dtype=np.int64)
    if plan.replay_micro and store is not None:
        rng = np.random.default_rng([seed, 7, step, micro])
        w = rng.integers(0, len(store), plan.replay_micro)
        rep = store.gather(w)[:, :plan.seq_len].astype(np.int64)
    return data.ids[idx], data.mask[idx], rep


def loss_sums(net, x: torch.Tensor, n_sft: int, mask: torch.Tensor, dec: dict) -> dict:
    """Summed fp32 CE over SFT target tokens and over replay tokens, plus decision-token stats."""
    logits = net(input_ids=x, use_cache=False).logits
    lg, tg = logits[:, :-1].float(), x[:, 1:]
    m = mask[:n_sft, 1:]                              # SFT rows only; target at t+1 predicted from t
    sft = F.cross_entropy(lg[:n_sft][m], tg[:n_sft][m], reduction="sum")
    rep = (F.cross_entropy(lg[n_sft:].reshape(-1, lg.shape[-1]), tg[n_sft:].reshape(-1), reduction="sum")
           if x.shape[0] > n_sft else lg.new_zeros(()))
    first = m.float().argmax(dim=1)                   # decision token = first target position
    sel = lg[torch.arange(n_sft, device=x.device), first]
    gold_none = tg[torch.arange(n_sft, device=x.device), first] == dec["none"]
    two = torch.log_softmax(sel[:, [dec["none"], dec["pass"]]], dim=-1)
    dce = -torch.where(gold_none, two[:, 0], two[:, 1])
    return {"sft": sft, "rep": rep, "dec_ce_sum": dce.sum().detach(),
            "dec_correct": ((two[:, 0] > two[:, 1]) == gold_none).sum().detach()}


# ------------------------------------------------------------------------------- eval
def evaluate(raw, dev: SFTData, dec: dict, device: torch.device, batch: int = 16) -> dict:
    tf = E.teacher_forced(raw, [r[:n].tolist() for r, n in zip(dev.ids, dev.length)],
                          dev.decision_pos.tolist(), list(dev.mask), device, dec, batch,
                          autocast=lambda: autocast(device))
    m = E.decision_metrics(tf["s"], dev.refusal.tolist(), dev.buckets)
    raw.train()
    return {"dev_target_loss": tf["target_loss"], **{f"dev_{k}": v for k, v in m.items()}}


# ------------------------------------------------------------------------------ train
@dataclass(frozen=True)
class RunConfig:
    lr: float = S.lr
    epochs: int = S.epochs
    replay_weight: float = S.replay_weight
    keep: str = "final"                # 'final', 'best' (dev macro-F1) or 'select' (D-27: needs extra_eval with
                                       # select_score + gate_ok; 7.8 passes config.SFT.keep)
    seq_len: int = config.SEQ_LEN
    batch_sequences: int = S.batch_sequences
    micro: int = S.micro_batch
    sft_per_replay: int = S.sft_per_replay
    eval_every: int = 0                # steps; 0 = at the end of every epoch
    ckpt_every: int = 100
    log_every: int = 10
    max_steps: int = 0
    max_usd: float = 2.0
    gpu: str = S.gpu
    seed: int = S.seed
    die_after_step: int = 0            # smoke-test hook: hard exit (no cleanup) after this step


def fingerprint(cfg: RunConfig, plan: Plan, n_train: int) -> dict:
    """Everything that changes the math; cadence, stop step and budget may differ on resume."""
    skip = ("max_usd", "die_after_step", "log_every", "max_steps", "ckpt_every", "eval_every", "gpu")
    return {**{k: v for k, v in asdict(cfg).items() if k not in skip}, **asdict(plan), "n_train": n_train}


def train_one(cfg: RunConfig, raw: torch.nn.Module, net: torch.nn.Module, train: SFTData, dev: SFTData | None,
              store: WindowStore | None, dec: dict, device: torch.device, out: Path | None,
              log=print, extra_eval=None) -> dict:
    """Train one run. With `out`, writes metrics/ckpt/ledger and resumes from out/ckpt.pt. Returns
    {'best': {...}, 'best_state': CPU state dict or None, 'status', 'steps', 'plan', 'history': every dev
    eval}. 'best' is the best dev macro-F1 epoch, or the final epoch with keep='final'. `extra_eval()` (optional) returns more metrics to merge into each dev eval."""
    if cfg.keep == "select" and extra_eval is None:
        raise ValueError("keep='select' needs extra_eval returning select_score and gate_ok")
    plan = make_plan(len(train), cfg.epochs, cfg.seq_len, cfg.batch_sequences, cfg.micro, cfg.sft_per_replay)
    order = StreamOrder(len(train), cfg.seed)
    end = min(plan.total_steps, cfg.max_steps) if cfg.max_steps else plan.total_steps
    eval_every = cfg.eval_every or plan.steps_per_epoch
    fp = fingerprint(cfg, plan, len(train))
    opt = build_optimizer(raw, device)
    start, best = 0, {"dev_macro_f1": -1.0, "step": 0}
    best_state, history = None, []
    paths = {k: out / v for k, v in (("ckpt", "ckpt.pt"), ("metrics", "metrics.jsonl"),
                                     ("ledger", "spend.json"), ("best", "best.pt"))} if out else {}
    if out:
        out.mkdir(parents=True, exist_ok=True)
        if paths["ckpt"].exists():
            st = torch.load(paths["ckpt"], map_location=device, weights_only=False)
            bad = {k: (st["fingerprint"].get(k), v) for k, v in fp.items() if st["fingerprint"].get(k) != v}
            if bad:
                raise SystemExit(f"checkpoint belongs to a different run: {bad}")
            raw.load_state_dict(st["model"])
            opt.load_state_dict(st["optimizer"])
            start, best = int(st["step"]), st["best"]
            truncate_metrics(paths["metrics"], start)
            log(f"[resume] step {start}")
    ledger = Ledger.open(paths["ledger"], 60.0) if out else None
    log(f"[plan] {asdict(plan)} | steps {start}->{end} | lr {cfg.lr} | lambda {cfg.replay_weight}")
    raw.train()
    status, t0, toks = "completed", time.time(), 0
    for step in range(start, end):
        lr = lr_at(step, plan.total_steps, plan.warmup_steps, cfg.lr, cfg.lr * S.min_lr_fraction)
        for g in opt.param_groups:
            g["lr"] = lr
        parts = [micro_batch(step, m, train, order, store, plan, cfg.seed) for m in range(plan.accum)]
        n_sft_tok = sum(int(mk[:, 1:].sum()) for _, mk, _ in parts)
        n_rep_tok = sum(rp.shape[0] * (plan.seq_len - 1) for _, _, rp in parts)
        acc = {"sft": 0.0, "rep": 0.0, "dec_ce": 0.0, "dec_ok": 0, "n_dec": 0}
        for sid, mk, rp in parts:
            x = torch.from_numpy(np.concatenate([sid, rp])).to(device)
            mask = torch.from_numpy(np.concatenate([mk, np.zeros_like(rp, dtype=bool)])).to(device)
            with autocast(device):
                r = loss_sums(net, x, sid.shape[0], mask, dec)
            loss = r["sft"] / max(1, n_sft_tok)
            if n_rep_tok:
                loss = loss + cfg.replay_weight * r["rep"] / n_rep_tok
            loss.backward()
            acc["sft"] += float(r["sft"]); acc["rep"] += float(r["rep"])
            acc["dec_ce"] += float(r["dec_ce_sum"]); acc["dec_ok"] += int(r["dec_correct"]); acc["n_dec"] += sid.shape[0]
            toks += x.numel()
        gnorm = float(torch.nn.utils.clip_grad_norm_(raw.parameters(), S.grad_clip))
        opt.step()
        opt.zero_grad(set_to_none=True)
        done = step + 1
        l_sft = acc["sft"] / max(1, n_sft_tok)
        l_rep = acc["rep"] / n_rep_tok if n_rep_tok else 0.0
        if cfg.die_after_step and done == cfg.die_after_step:
            os._exit(17)                                  # simulated hard kill: no checkpoint, no cleanup
        if not (math.isfinite(l_sft) and math.isfinite(gnorm)):
            status = "nonfinite"
            log(f"[stop] non-finite at step {done}")
            break
        row = {"step": done, "epoch": step // plan.steps_per_epoch, "lr": lr, "l_sft": l_sft, "l_replay": l_rep,
               "loss": l_sft + cfg.replay_weight * l_rep, "dec_ce": acc["dec_ce"] / max(1, acc["n_dec"]),
               "dec_acc": acc["dec_ok"] / max(1, acc["n_dec"]), "grad_norm": gnorm}
        stop = False
        if done % cfg.log_every == 0 or done == end:
            row["tok_per_s"] = toks / max(1e-9, time.time() - t0)
            t0, toks = time.time(), 0
            if ledger:
                ledger.write(paths["ledger"])
                row["spend_usd"] = spend_usd(ledger.seconds(), cfg.gpu, 1)
                if row["spend_usd"] > cfg.max_usd:
                    status, stop = "budget_stop", True
            log(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
            if out:
                append_metrics(paths["metrics"], row)
        if dev is not None and (done % eval_every == 0 or done == end):
            ev = evaluate(raw, dev, dec, device)
            if extra_eval is not None:                 # e.g. held-out bits/byte for forgetting curves
                ev.update(extra_eval())
                raw.train()
            ev["step"] = done
            history.append(ev)
            log(f"[eval] step {done} " + " ".join(f"{k} {v:.4f}" for k, v in ev.items()
                                                  if isinstance(v, float)))
            if out:
                append_metrics(paths["metrics"], ev)
            if cfg.keep == "select":                   # D-27: proxy score among gate-passing evals
                better = ev["gate_ok"] and (not best.get("gate_ok") or ev["select_score"] > best["select_score"])
                if not best.get("gate_ok") and not ev["gate_ok"]:   # nothing passed yet: lowest bpb so far
                    better = ev["heldout_pooled_bpb"] < best.get("heldout_pooled_bpb", math.inf)
                if better:
                    best = {**ev}
                    best_state = {k: v.detach().to("cpu", copy=True) for k, v in raw.state_dict().items()}
                    if out:
                        atomic_torch_save(best_state, paths["best"])
            elif ev["dev_macro_f1"] > best["dev_macro_f1"]:
                best = {**ev}
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in raw.state_dict().items()}
                if out:
                    atomic_torch_save(best_state, paths["best"])
        if out and (done % cfg.ckpt_every == 0 or done == end or stop):
            atomic_torch_save({"model": raw.state_dict(), "optimizer": opt.state_dict(), "step": done,
                               "fingerprint": fp, "best": best}, paths["ckpt"])
        if stop:
            log(f"[stop] budget: spend over --max-usd {cfg.max_usd}")
            break
    if cfg.keep == "final" and status == "completed":      # D-23: the last epoch, not the noisy best
        best = {**history[-1]} if history else {"step": end}
        best_state = {k: v.detach().to("cpu", copy=True) for k, v in raw.state_dict().items()}
        if out:
            atomic_torch_save(best_state, paths["best"])
    if ledger:
        ledger.write(paths["ledger"])
    return {"status": status, "steps": end, "best": best, "best_state": best_state, "plan": asdict(plan),
            "history": history}


def export(raw: torch.nn.Module, state: dict | None, out_dir: Path, tokenizer_dir: str, tau: float = 0.5) -> None:
    """HF export of the best weights with a greedy generation config and the format contract."""
    from transformers import GenerationConfig
    if state is not None:
        raw.load_state_dict(state)
    export_hf(raw, out_dir, tokenizer_dir)
    GenerationConfig(bos_token_id=M.BOS_TOKEN_ID, eos_token_id=M.EOS_TOKEN_ID, pad_token_id=M.PAD_TOKEN_ID,
                     do_sample=False, num_beams=1, max_new_tokens=S.max_new_tokens, min_new_tokens=0,
                     suppress_tokens=list(E.SUPPRESS)).save_pretrained(out_dir)
    atomic_write_text(out_dir / "raft_format.json", json.dumps({
        "prompt": "<|eos|><|user|>Passage 1: ...\\nPassage k: ...\\nQuestion: ...<|assistant|>Source:",
        "refusal": config.RAFT.refusal, "decision": "s = P(' none') / (P(' none') + P(' Pass'))",
        "tau": tau, "suppress_ids": list(E.SUPPRESS)}, indent=2))


# ------------------------------------------------------------------------------ sweep
def run_sweep(raw: torch.nn.Module, net: torch.nn.Module, base_state: dict, pool: list[dict], dev: SFTData,
              store: WindowStore | None, tok, dec: dict, device: torch.device, grid: list[dict],
              results_path: Path, base_cfg: RunConfig, budget_ok=lambda: True, log=print,
              gen_batch: int = 64) -> list[dict]:
    """The pilot sweep (§7.6): every run reuses the SAME (compiled) module. Base weights are copied
    in place (load_state_dict keeps the Parameter objects, so Dynamo's guards still hold), chat
    rows re-initialised, optimizer fresh. Each run keeps its best epoch by dev macro-F1, then
    generates on dev for the deterministic proxies (§7.2). Results are appended per run, so a
    restarted container skips finished runs. `budget_ok()` is checked before each run."""
    done = {}
    if results_path.exists():
        for ln in results_path.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                done[r["key"]] = r
    rows_out = list(done.values())
    for g in grid:
        key = f"n{g['n']}-lr{g['lr']:g}-r{g['refusal_share']:g}"
        if key in done:
            continue
        if not budget_ok():
            log(f"[sweep] budget reached before {key}; stopping")
            break
        t0 = time.time()
        raw.load_state_dict(base_state)
        init_chat_tokens(raw, base_cfg.seed)
        train = tokenize(select_mix(pool, g["n"], g["refusal_share"], base_cfg.seed), tok, base_cfg.seq_len)
        cfg = RunConfig(**{**asdict(base_cfg), "lr": g["lr"]})
        res = train_one(cfg, raw, net, train, dev, store, dec, device, None, log=lambda *_: None)
        raw.load_state_dict(res["best_state"])
        gens = E.generate_batch(raw, tok, [r["prompt"] for r in dev.rows], [len(r["passages"] or []) for r in dev.rows],
                                device, tau=0.5, batch=gen_batch, autocast=lambda: autocast(device))
        prox = E.proxy_metrics(dev.rows, gens)
        row = {"key": key, **g, "best_step": res["best"]["step"],
               "best_epoch": res["best"]["step"] / max(1, res["plan"]["steps_per_epoch"]),
               **{k: v for k, v in res["best"].items() if k.startswith("dev_")}, **prox,
               "seconds": round(time.time() - t0, 1), "steps": res["steps"]}
        with results_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        rows_out.append(row)
        log(f"[sweep] {key}: score {prox['selection_score']:.3f} grounded {prox['proxy_grounded_acc']:.3f} "
            f"halluc {prox['proxy_hallucination']:.3f} over-refusal {prox['over_refusal_A']:.3f} "
            f"macroF1 {res['best']['dev_macro_f1']:.3f} best epoch {row['best_epoch']:.0f} ({row['seconds']}s)")
    return rows_out


def best_run(rows: list[dict]) -> dict:
    """Highest selection score; ties -> lower proxy hallucination (§7.2)."""
    return max(rows, key=lambda r: (round(r["selection_score"], 6), -r["proxy_hallucination"]))


# -------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train", required=True)
    p.add_argument("--dev", default="")
    p.add_argument("--tokenizer-dir", default=config.TOKENIZER_DIR)
    p.add_argument("--init", default=config.RAFT_INIT_CKPT)
    p.add_argument("--tokens-dir", default=config.TRAIN_TOKENS_DIR, help="replay windows ('' = none)")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--n-train", type=int, default=0, help="0 = all rows")
    p.add_argument("--refusal-share", type=float, default=-1.0, help="with --n-train; <0 = as given")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--export", action="store_true")
    for f in RunConfig.__dataclass_fields__.values():
        p.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    a = p.parse_args(argv)
    cfg = RunConfig(**{k: getattr(a, k) for k in RunConfig.__dataclass_fields__})
    rt = init_runtime(a.device)
    if rt.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    dec = E.decision_ids(tok)
    rows = load_rows(a.train)
    if a.n_train:
        rows = select_mix(rows, a.n_train, a.refusal_share, cfg.seed) if a.refusal_share >= 0 \
            else random.Random(cfg.seed).sample(rows, a.n_train)
    train = tokenize(rows, tok, cfg.seq_len)
    dev = tokenize(load_rows(a.dev), tok, cfg.seq_len) if a.dev else None
    store = WindowStore.open(a.tokens_dir, config.SEQ_LEN) if a.tokens_dir else None
    out = Path(a.out)
    raw = load_init(a.init).to(rt.device)
    if not (out / "ckpt.pt").exists():
        print(f"[init] chat tokens {init_chat_tokens(raw, cfg.seed)}")
    net = torch.compile(raw) if a.compile else raw
    res = train_one(cfg, raw, net, train, dev, store, dec, rt.device, out)
    atomic_write_text(out / "run_status.json", json.dumps({k: v for k, v in res.items() if k != "best_state"},
                                                          indent=2, default=str))
    if a.export and res["status"] == "completed":
        state = res["best_state"]
        if state is None and (out / "best.pt").exists():     # after a resume the best is on disk
            state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        export(raw, state, out / "hf", a.tokenizer_dir)
    return {"completed": EXIT_OK, "budget_stop": EXIT_BUDGET, "nonfinite": EXIT_NONFINITE}[res["status"]]


if __name__ == "__main__":
    sys.exit(main())
