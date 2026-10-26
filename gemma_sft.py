"""Phase 9: LoRA SFT of Gemma 2 2B on the RAFT rows (docs/10-gemma.md §2-§4).

The same contract as sft.py where it matters for the comparison:
- loss: a per-token mean of fp32 cross-entropy over the target tokens of the WHOLE optimizer step
  (counts taken before the first backward, so gradient accumulation cannot reweight tokens);
- the row order (StreamOrder) and the LR (lr_at: 3% warmup, cosine to 10%) are pure functions of the
  step; 52 SFT rows per step, Phase 7's SFT share of its 64-sequence batch;
- checkpoint choice keep="select" (D-27): the best dev copy2 proxy score among evals that pass the
  bits/byte gate; if none passes, the lowest bits/byte, flagged.

Differences (docs/10-gemma.md §3): no replay; LoRA (frozen bf16 base, fp32 adapters, bf16 autocast);
dynamic right padding; logits only at target positions, with Gemma 2's final soft-cap applied; held-out
bits/byte scored per UTF-8 byte of the decoded windows, so it does not depend on the tokenizer.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
import eval_raft as E
import raft_data as R
import raft_decode as RD
import sft
from data import StreamOrder
from train import lr_at

G = config.GEMMA
S = config.SFT


# ------------------------------------------------------------------------------- data
@dataclass
class GemmaData:
    seqs: list[np.ndarray]     # prompt + target ids (Gemma spelling), unpadded
    plen: np.ndarray           # (N,) prompt length = index of the first target (decision) token
    refusal: np.ndarray        # (N,) bool
    buckets: list[str]
    rows: list[dict]           # rows with `prompt` / `target` in the Gemma spelling

    def __len__(self) -> int:
        return len(self.rows)


def convert_rows(rows: list[dict], fmt: RD.Format = RD.GEMMA) -> list[dict]:
    return [{**r, "prompt": fmt.convert(r["prompt"]), "target": fmt.convert(r["target"])} for r in rows]


def encode_rows(rows: list[dict], tok, max_len: int = G.max_len) -> GemmaData:
    """Rows in the stored (SLM) spelling -> Gemma ids. Fails if a row is too long or its prompt does
    not start with exactly one <bos>."""
    conv = convert_rows(rows)
    seqs, plen = [], []
    for r in conv:
        x, labels = R.encode_example(tok, r["prompt"], r["target"])
        if len(x) > max_len:
            raise SystemExit(f"{r['id']}: {len(x)} tokens > max_len {max_len}")
        if x[0] != tok.bos_token_id or x.count(tok.bos_token_id) != 1:
            raise SystemExit(f"{r['id']}: prompt must start with exactly one <bos>")
        seqs.append(np.asarray(x, dtype=np.int64))
        plen.append(labels.count(-100))
    return GemmaData(seqs, np.asarray(plen), np.array([r["bucket"] in sft.REFUSAL_BUCKETS for r in conv]),
                     [r["bucket"] for r in conv], conv)


def pad_batch(seqs: list[np.ndarray], plens: list[int], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded ids (B, W) and the target mask (positions plen..len-1). Causal attention: no mask."""
    w = max(len(s) for s in seqs)
    x = torch.full((len(seqs), w), pad_id, dtype=torch.long)
    m = torch.zeros((len(seqs), w), dtype=torch.bool)
    for i, (s, p) in enumerate(zip(seqs, plens)):
        x[i, :len(s)] = torch.from_numpy(s)
        m[i, p:len(s)] = True
    return x, m


# ------------------------------------------------------------------------------ model
def load_base(path: str, device: torch.device, attn: str = G.attn) -> torch.nn.Module:
    """Gemma 2 in bf16: the frozen base for LoRA (docs/10-gemma.md §2; not the 125M fp32 recipe)."""
    from transformers import AutoModelForCausalLM
    net = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation=attn)
    return net.to(device)


def add_lora(net: torch.nn.Module, seed: int = G.seed):
    """PEFT LoRA on every attention and MLP projection; adapters in fp32 (autocast_adapter_dtype).
    Returns (peft model, initial adapter state) so every run starts from the same adapter init."""
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    torch.manual_seed(seed)
    cfg = LoraConfig(r=G.lora_rank, lora_alpha=G.lora_alpha, lora_dropout=G.lora_dropout,
                     target_modules=list(G.lora_targets), bias="none", task_type="CAUSAL_LM")
    pm = get_peft_model(net, cfg, autocast_adapter_dtype=True)
    init = {k: v.detach().to("cpu", copy=True) for k, v in get_peft_model_state_dict(pm).items()}
    return pm, init


def adapter_state(pm) -> dict:
    from peft import get_peft_model_state_dict
    return {k: v.detach().to("cpu", copy=True) for k, v in get_peft_model_state_dict(pm).items()}


def load_adapter(pm, state: dict) -> None:
    from peft import set_peft_model_state_dict
    res = set_peft_model_state_dict(pm, state)
    if getattr(res, "unexpected_keys", None):
        raise RuntimeError(f"adapter state has unexpected keys: {res.unexpected_keys[:4]}")


def causal_lm(net):
    """The underlying *ForCausalLM (LoRA layers stay injected) for a PEFT model or a plain model."""
    return net.get_base_model() if hasattr(net, "get_base_model") else net


def head(lm, h: torch.Tensor) -> torch.Tensor:
    """fp32 logits from hidden states, with Gemma 2's final logit soft-cap (as in its forward)."""
    logits = lm.lm_head(h).float()
    cap = getattr(lm.config, "final_logit_softcapping", None)
    return torch.tanh(logits / cap) * cap if cap else logits


def chunked_ce(lm, h: torch.Tensor, targets: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Summed CE over rows of h (N, H) without materialising (N, 256K) at once."""
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.shape[0], chunk):
        total = total + F.cross_entropy(head(lm, h[i:i + chunk]), targets[i:i + chunk], reduction="sum")
    return total


def loss_sums(net, x: torch.Tensor, mask: torch.Tensor, dec: dict) -> dict:
    """Summed CE over target tokens plus decision-token stats (sft.loss_sums without replay)."""
    lm = causal_lm(net)
    h = lm.model(input_ids=x, use_cache=False).last_hidden_state
    m = mask[:, 1:]                                   # target at t+1 predicted from t
    ce = chunked_ce(lm, h[:, :-1][m], x[:, 1:][m])
    first = m.float().argmax(dim=1)                   # decision token = first target position
    rows = torch.arange(x.shape[0], device=x.device)
    sel = head(lm, h[rows, first])
    gold_none = x[rows, first + 1] == dec["none"]
    two = torch.log_softmax(sel[:, [dec["none"], dec["pass"]]], dim=-1)
    dce = -torch.where(gold_none, two[:, 0], two[:, 1])
    return {"sft": ce, "dec_ce_sum": dce.sum().detach(),
            "dec_correct": ((two[:, 0] > two[:, 1]) == gold_none).sum().detach()}


# ------------------------------------------------------------------------------- eval
@torch.no_grad()
def teacher_forced(net, data: GemmaData, dec: dict, device: torch.device, batch: int = 16, autocast=None) -> dict:
    """Decision scores s and the mean target-token loss (E.teacher_forced, target positions only)."""
    net.eval()
    lm = causal_lm(net)
    scores, tot, n = [], 0.0, 0
    for i in range(0, len(data), batch):
        x, m = pad_batch(data.seqs[i:i + batch], data.plen[i:i + batch].tolist(), dec["pad"])
        x, m = x.to(device), m.to(device)
        with (autocast() if autocast else torch.autocast("cpu", enabled=False)):
            h = lm.model(input_ids=x, use_cache=False).last_hidden_state
            rows = torch.arange(x.shape[0], device=device)
            pos = torch.as_tensor(data.plen[i:i + batch] - 1, device=device)
            scores += RD.decision_score(head(lm, h[rows, pos]), dec).tolist()
            mm = m[:, 1:]
            tot += float(chunked_ce(lm, h[:, :-1][mm], x[:, 1:][mm]))
        n += int(mm.sum())
    return {"s": scores, "target_loss": tot / max(1, n)}


@torch.no_grad()
def prompt_scores(net, tok, prompts: list[str], device: torch.device, batch: int = 16, autocast=None) -> list[float]:
    """s at the end of each prompt (one forward, left-aligned): the decision without generating."""
    dec = RD.decision_ids(tok)
    net.eval()
    lm = causal_lm(net)
    out = []
    for i in range(0, len(prompts), batch):
        enc = [np.asarray(tok.encode(p, add_special_tokens=False), dtype=np.int64) for p in prompts[i:i + batch]]
        x, _ = pad_batch(enc, [len(e) for e in enc], dec["pad"])
        x = x.to(device)
        with (autocast() if autocast else torch.autocast("cpu", enabled=False)):
            h = lm.model(input_ids=x, use_cache=False).last_hidden_state
            last = torch.as_tensor([len(e) - 1 for e in enc], device=device)
            out += RD.decision_score(head(lm, h[torch.arange(len(enc), device=device), last]), dec).tolist()
    return out


@torch.no_grad()
def heldout_bpb(net, tok, src_tok, val_dir: Path, subset: dict, device: torch.device, autocast=None,
                batch: int = 4) -> dict:
    """Bits per UTF-8 byte on the Phase 7 held-out subset (eval_forgetting.heldout_subset): each window
    is decoded with OUR tokenizer (skip_special_tokens, the same bytes as eval_forgetting.heldout_bpb),
    re-tokenized by Gemma after <bos>, and every text token is scored."""
    import hashlib
    net.eval()
    lm = causal_lm(net)
    out, tot_nats, tot_bytes = {}, 0.0, 0
    for src, spec in subset.items():
        path = val_dir / spec["shard"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise RuntimeError(f"{path} differs from the shard the subset was drawn from")
        arr = np.memmap(path, dtype=np.uint16, mode="r").reshape(-1, config.SEQ_LEN)
        texts = [src_tok.decode(w.tolist(), skip_special_tokens=True) for w in np.asarray(arr[spec["indices"]])]
        seqs = [np.asarray([tok.bos_token_id] + tok(t, add_special_tokens=False, split_special_tokens=True)["input_ids"],
                           dtype=np.int64) for t in texts]
        nats, nb = 0.0, sum(len(t.encode("utf-8")) for t in texts)
        for i in range(0, len(seqs), batch):
            x, m = pad_batch(seqs[i:i + batch], [1] * len(seqs[i:i + batch]), tok.pad_token_id)
            x, m = x.to(device), m.to(device)
            with (autocast() if autocast else torch.autocast("cpu", enabled=False)):
                h = lm.model(input_ids=x, use_cache=False).last_hidden_state
                mm = m[:, 1:]
                nats += float(chunked_ce(lm, h[:, :-1][mm], x[:, 1:][mm]))
        out[src] = {"windows": len(seqs), "bytes": nb, "nats": nats, "bpb": nats / (math.log(2) * nb)}
        tot_nats += nats
        tot_bytes += nb
    out["pooled_bpb"] = tot_nats / (math.log(2) * tot_bytes)
    return out


# ------------------------------------------------------------------------------ train
def step_micros(order: StreamOrder, train: GemmaData, plan: sft.Plan, step: int) -> list[np.ndarray]:
    """The step's rows (the same set as sft.micro_batch would take), sorted by length and split into
    micro-batches: less padding (9.2 probe: -12% tokens), same per-step loss up to summation order."""
    idx = order.windows(np.arange(step * plan.sft_per_step, (step + 1) * plan.sft_per_step, dtype=np.int64))
    idx = idx[np.argsort([len(train.seqs[i]) for i in idx], kind="stable")]
    return [idx[k * plan.sft_micro:(k + 1) * plan.sft_micro] for k in range(plan.accum)]


def better(ev: dict, best: dict) -> bool:
    """keep='select' (sft.train_one, D-27): best proxy among gate-passing evals, else lowest bits/byte."""
    if not best.get("gate_ok") and not ev["gate_ok"]:
        return ev["heldout_pooled_bpb"] < best.get("heldout_pooled_bpb", math.inf)
    return ev["gate_ok"] and (not best.get("gate_ok") or ev["select_score"] > best["select_score"])


def train_run(pm, init_state: dict, train: GemmaData, dev: GemmaData, dec: dict, device: torch.device, *,
              lr: float, epochs: int = G.epochs, n_evals: int, extra_eval, micro: int = G.micro_batch,
              seed: int = G.seed, budget_ok=lambda: True, log=print, grad_ckpt: bool = False) -> dict:
    """One LoRA run from `init_state`. `extra_eval()` must return select_score, gate_ok and
    heldout_pooled_bpb (bits/byte gate + copy2 proxy). Returns {'status', 'steps', 'best', 'best_state'
    (adapter only), 'plan', 'history'}."""
    from train import autocast
    load_adapter(pm, init_state)
    if grad_ckpt:
        pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        pm.enable_input_require_grads()
    plan = sft.make_plan(len(train), epochs, G.max_len, G.sft_rows_per_step, micro, sft_per_replay=0)
    order = StreamOrder(len(train), seed)
    eval_every = max(1, round(plan.total_steps / n_evals))
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(config.TRAIN.beta1, config.TRAIN.beta2), eps=1e-8,
                            weight_decay=G.weight_decay, **({"fused": True} if device.type == "cuda" else {}))
    best, best_state, history, status = {"step": 0}, None, [], "completed"
    log(f"[plan] steps {plan.total_steps} ({plan.steps_per_epoch}/epoch), {plan.sft_micro} x {plan.accum} rows/step, "
        f"eval every {eval_every}, lr {lr:g}, trainable {sum(p.numel() for p in params):,}")
    pm.train()
    t0, toks = time.time(), 0
    for step in range(plan.total_steps):
        cur = lr_at(step, plan.total_steps, plan.warmup_steps, lr, lr * S.min_lr_fraction)
        for g in opt.param_groups:
            g["lr"] = cur
        parts = [([train.seqs[i] for i in idx], train.plen[idx].tolist())
                 for idx in step_micros(order, train, plan, step)]
        n_tok = sum(int(len(s) - p) for seqs, ps in parts for s, p in zip(seqs, ps))
        acc = {"sft": 0.0, "dec_ce": 0.0, "dec_ok": 0, "n": 0}
        for seqs, ps in parts:
            x, m = pad_batch(seqs, ps, dec["pad"])
            x, m = x.to(device), m.to(device)
            with autocast(device):
                r = loss_sums(pm, x, m, dec)
            (r["sft"] / max(1, n_tok)).backward()
            acc["sft"] += float(r["sft"]); acc["dec_ce"] += float(r["dec_ce_sum"])
            acc["dec_ok"] += int(r["dec_correct"]); acc["n"] += x.shape[0]
            toks += x.numel()
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, G.grad_clip))
        opt.step()
        opt.zero_grad(set_to_none=True)
        done = step + 1
        l_sft = acc["sft"] / max(1, n_tok)
        if not (math.isfinite(l_sft) and math.isfinite(gnorm)):
            status = "nonfinite"
            log(f"[stop] non-finite at step {done}")
            break
        if done % 10 == 0 or done == plan.total_steps:
            log(f"step {done}/{plan.total_steps} lr {cur:.3g} l_sft {l_sft:.4f} dec_acc {acc['dec_ok'] / acc['n']:.3f} "
                f"gnorm {gnorm:.3g} tok/s {toks / max(1e-9, time.time() - t0):.0f}")
            t0, toks = time.time(), 0
            if not budget_ok():
                status = "budget_stop"
                log("[stop] budget")
                break
        if done % eval_every == 0 or done == plan.total_steps:
            tf = teacher_forced(pm, dev, dec, device, autocast=lambda: autocast(device))
            ev = {"step": done, "dev_target_loss": tf["target_loss"],
                  **{f"dev_{k}": v for k, v in E.decision_metrics(tf["s"], dev.refusal.tolist(), dev.buckets).items()},
                  **extra_eval()}
            pm.train()
            history.append(ev)
            log(f"[eval] step {done} " + " ".join(f"{k} {v:.4f}" for k, v in ev.items() if isinstance(v, float)))
            if better(ev, best):
                best, best_state = {**ev}, adapter_state(pm)
    if grad_ckpt:
        pm.gradient_checkpointing_disable()
    return {"status": status, "steps": plan.total_steps, "best": best, "best_state": best_state,
            "plan": plan.__dict__, "history": history}


# ------------------------------------------------------------------- instruct baseline
ZS_INSTRUCTIONS = (
    "Answer using only the passages. If they do not contain the answer, reply exactly:\n"
    "Source: none\nAnswer: The passages do not contain the answer.\n"
    "Otherwise reply exactly:\n"
    "Source: Passage k\nQuote: \"<a span copied word for word from passage k>\"\nAnswer: <short answer>\n"
    "For a task instead of a question, do the task from the passages and name the passage the same way.")


def _body(stored_prompt: str) -> str:
    """'Passage 1: ...\\nQuestion: ...' from a stored (SLM-spelled) prompt."""
    return stored_prompt[len(RD.SLM.start):].split(RD.SLM.turn)[0]


def instruct_prompt(row: dict, shots: list[dict]) -> str:
    """Few-shot prompt for gemma-2-2b-it in its chat template, the model turn prefilled with `Source:`
    so the decision is read from the same token as the fine-tuned models (docs/10-gemma.md §4, 9.3)."""
    ex = "\n\n".join(f"Example {i}:\n{_body(s['prompt'])}\nSource:{s['target'].replace(RD.SLM.eos, '')}"
                     for i, s in enumerate(shots, 1))
    user = f"{ZS_INSTRUCTIONS}\n\n{ex}\n\nNow:\n{_body(row['prompt'])}"
    return f"<bos><start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\nSource:"


def closed_book_prompt(question: str) -> str:
    return (f"<bos><start_of_turn>user\nAnswer the question in a few words.\nQuestion: {question}"
            f"<end_of_turn>\n<start_of_turn>model\n")


def pick_shots(train_rows: list[dict]) -> list[dict]:
    """Two fixed train rows: the shortest answerable A row with a quote, and the shortest near miss."""
    a = min((r for r in train_rows if r["bucket"] == "A" and r.get("quote")), key=lambda r: (len(r["prompt"]), r["id"]))
    b = min((r for r in train_rows if r["bucket"] == "B2"), key=lambda r: (len(r["prompt"]), r["id"]))
    return [a, b]
