"""Phase 12 DPO trainer (docs/13-rlaif.md §3), for SLM-125M-RAFT and Gemma 3 270M RAFT alike.

loss = DPO + alpha * SFT(chosen) [+ lambda * replay LM loss, SLM-125M only]
  DPO   = -log sigmoid(beta * [(log pi(c) - log ref(c)) - (log pi(r) - log ref(r))]), mean over pairs
  SFT   = per-token mean NLL of the chosen targets (keeps the chosen likelihood from falling)
  log pi(.) = sum of target-token log-probs (the text after `Source:` plus the stop token), computed only at
  target positions (gemma_sft.head: no 262K x T logits). The reference is the SFT checkpoint: its log-probs
  are computed once before training, so only one model is ever in memory.
Each optimizer step averages the DPO term over its pairs and the SFT term over its chosen tokens across
micro-batches, so accumulation does not reweight. Checkpoint choice as Phase 7 (keep='select'): the best dev
copy2 proxy among evals that pass the bits/byte gate (`gemma_sft.better`).
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import config
import gemma_sft as GS
import raft_data as R
import raft_decode as RD
from train import lr_at


def encode_pairs(pairs: list[dict], rows: dict[str, dict], tok, max_len: int = config.SEQ_LEN) -> list[dict]:
    """Stored (SLM-spelled) prompt in the model's spelling + chosen/rejected targets (`text` + stop token)."""
    fmt = RD.format_for(tok)
    out = []
    for p in pairs:
        prompt = fmt.convert(rows[p["id"]]["prompt"])
        enc = {}
        for side in ("chosen", "rejected"):
            x, lab = R.encode_example(tok, prompt, p[side] + fmt.eos)
            enc[side] = (np.asarray(x, dtype=np.int64), lab.count(-100))
        if max(len(enc["chosen"][0]), len(enc["rejected"][0])) <= max_len:
            out.append({**p, "c": enc["chosen"], "r": enc["rejected"]})
    return out


def target_logps(net, seqs: list[np.ndarray], plens: list[int], pad_id: int, device, chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """(sum of target log-probs per row, target tokens per row), gradient-carrying."""
    lm = GS.causal_lm(net)
    x, m = GS.pad_batch(seqs, plens, pad_id)
    x, m = x.to(device), m.to(device)
    h = lm.model(input_ids=x, use_cache=False).last_hidden_state
    mm = m[:, 1:]
    hs, tg = h[:, :-1][mm], x[:, 1:][mm]
    rows = torch.arange(x.shape[0], device=device).unsqueeze(1).expand(-1, x.shape[1] - 1)[mm]
    lp = torch.cat([-F.cross_entropy(GS.head(lm, hs[i:i + chunk]), tg[i:i + chunk], reduction="none")
                    for i in range(0, hs.shape[0], chunk)])
    out = torch.zeros(x.shape[0], device=device, dtype=lp.dtype).index_add(0, rows, lp)
    return out, mm.sum(1)


@torch.no_grad()
def reference_logps(net, enc: list[dict], pad_id: int, device, autocast, batch: int = 16) -> None:
    """Fill p['ref_c'], p['ref_r'] with the frozen reference's log-probs (call before any update)."""
    net.eval()
    for i in range(0, len(enc), batch):
        part = enc[i:i + batch]
        with autocast():
            lp, _ = target_logps(net, [p["c"][0] for p in part] + [p["r"][0] for p in part],
                                 [p["c"][1] for p in part] + [p["r"][1] for p in part], pad_id, device)
        for j, p in enumerate(part):
            p["ref_c"], p["ref_r"] = float(lp[j]), float(lp[len(part) + j])


def dpo_terms(lp_c, lp_r, ref_c, ref_r, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pair DPO loss and implicit reward margin (beta * [chosen - rejected log-ratio])."""
    margin = beta * ((lp_c - ref_c) - (lp_r - ref_r))
    return -F.logsigmoid(margin), margin.detach()


def train_dpo(net, enc: list[dict], device, *, beta: float, lr: float, epochs: int = 1, pairs_per_step: int = 32,
              micro: int = 8, alpha: float = 0.2, replay=None, replay_weight: float = 0.0, n_evals: int = 2,
              extra_eval=None, budget_ok=lambda: True, log=print, seed: int = 0, autocast=None) -> dict:
    """DPO from the current weights (which must equal the reference used for p['ref_*']). `replay()` returns a
    (B, T) int64 tensor of pretraining windows or None. Returns {'status', 'steps', 'best', 'best_state', 'history'}."""
    autocast = autocast or (lambda: torch.autocast("cpu", enabled=False))
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(len(enc)) for _ in range(epochs)])
    steps = max(1, math.ceil(len(order) / pairs_per_step))
    warm = max(1, round(0.1 * steps))
    eval_every = max(1, round(steps / n_evals))
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(config.TRAIN.beta1, config.TRAIN.beta2), eps=1e-8, weight_decay=0.0,
                            **({"fused": True} if device.type == "cuda" else {}))
    pad_id = getattr(GS.causal_lm(net).config, "pad_token_id", None)
    pad_id = 0 if pad_id is None else pad_id
    best, best_state, history, status = {"step": 0}, None, [], "completed"
    log(f"[dpo] pairs {len(enc)}, epochs {epochs}, steps {steps}, beta {beta}, lr {lr:g}, alpha {alpha}, replay {replay_weight}")
    net.train()
    t0 = time.time()
    for step in range(steps):
        cur = lr_at(step, steps, warm, lr, lr * 0.1)
        for g in opt.param_groups:
            g["lr"] = cur
        idx = order[step * pairs_per_step:(step + 1) * pairs_per_step]
        batch = [enc[i] for i in idx]
        n_c = sum(int(len(p["c"][0]) - p["c"][1]) for p in batch)
        acc = {"dpo": 0.0, "sft": 0.0, "rep": 0.0, "margin": 0.0, "acc": 0}
        for i in range(0, len(batch), micro):
            part = batch[i:i + micro]
            with autocast():
                lp, _ = target_logps(net, [p["c"][0] for p in part] + [p["r"][0] for p in part],
                                     [p["c"][1] for p in part] + [p["r"][1] for p in part], pad_id, device)
                lc, lrj = lp[:len(part)].float(), lp[len(part):].float()
                rc = torch.tensor([p["ref_c"] for p in part], device=device)
                rr = torch.tensor([p["ref_r"] for p in part], device=device)
                d, margin = dpo_terms(lc, lrj, rc, rr, beta)
                loss = d.sum() / len(batch) + alpha * (-lc.sum()) / max(1, n_c)
                if replay is not None and replay_weight > 0:
                    w = replay()
                    if w is not None:
                        out = net(input_ids=w.to(device), labels=w.to(device), use_cache=False)
                        loss = loss + replay_weight * out.loss * (len(part) / len(batch))
                        acc["rep"] += float(out.loss) * len(part) / len(batch)
            loss.backward()
            acc["dpo"] += float(d.sum()) / len(batch)
            acc["sft"] += float(-lc.sum()) / max(1, n_c)
            acc["margin"] += float(margin.sum()) / len(batch)
            acc["acc"] += int((margin > 0).sum())
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        done = step + 1
        if not (math.isfinite(acc["dpo"]) and math.isfinite(gnorm)):
            status = "nonfinite"
            log(f"[stop] non-finite at step {done}")
            break
        if done % 10 == 0 or done == steps:
            log(f"step {done}/{steps} lr {cur:.3g} dpo {acc['dpo']:.4f} sft {acc['sft']:.4f} rep {acc['rep']:.4f} "
                f"margin {acc['margin']:.3f} pair_acc {acc['acc'] / len(batch):.3f} gnorm {gnorm:.3g} ({time.time() - t0:.0f}s)")
            if not budget_ok():
                status = "budget_stop"
                log("[stop] budget")
                break
        if extra_eval is not None and (done % eval_every == 0 or done == steps):
            ev = {"step": done, "train_dpo": acc["dpo"], "train_margin": acc["margin"], **extra_eval()}
            net.train()
            history.append(ev)
            log(f"[eval] step {done} " + " ".join(f"{k} {v:.4f}" for k, v in ev.items() if isinstance(v, float)))
            if GS.better(ev, best):
                best, best_state = {**ev}, GS.trainable_state(net)
    return {"status": status, "steps": steps, "best": best, "best_state": best_state, "history": history}
