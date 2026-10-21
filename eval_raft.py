"""RAFT evaluation helpers (docs/08-raft.md §4, §7.2, §8).

- `decision_ids`: the token ids the format relies on (verified against the tokenizer).
- `RaftConstraint` + `generate_batch`: constrained greedy decoding. The prompt ends with
  `Source:`; the first generated token is the decision. Refuse iff
  s = P(Ġnone) / (P(Ġnone) + P(ĠPass)) >= tau. After `ĠPass`, `age` is forced and the next
  token is restricted to the passage numbers present. Ids 0 and 2-6 are always suppressed.
- `teacher_forced`: one forward over prompt+target rows. Gives the decision score s and the
  target-token loss without generating (cheap; used every epoch).
- `parse_output`, `decision_metrics`, `auroc`, `proxy_metrics`: the deterministic metrics used
  to select runs in the pilot (§7.2). The judge-based metrics of §8.1 live with step 7.8.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter

import numpy as np
import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

import config
import model as M

SUPPRESS = (M.BOS_TOKEN_ID, M.PAD_TOKEN_ID, M.UNK_TOKEN_ID, 4, 5, 6)
GREEDY = GenerationConfig(do_sample=False, num_beams=1, min_new_tokens=0, max_new_tokens=config.SFT.max_new_tokens,
                          eos_token_id=M.EOS_TOKEN_ID, pad_token_id=M.PAD_TOKEN_ID, bos_token_id=M.BOS_TOKEN_ID)


def decision_ids(tok) -> dict:
    """{'none', 'pass', 'age', 'digits': {k: id}}; fails loudly if the tokenizer disagrees."""
    none = tok.encode(" none", add_special_tokens=False)
    passage = tok.encode(" Passage", add_special_tokens=False)
    digits = {k: tok.encode(f" {k}", add_special_tokens=False) for k in range(1, config.RAFT.k_passages + 1)}
    if len(none) != 1 or len(passage) != 2 or any(len(v) != 1 for v in digits.values()):
        raise RuntimeError(f"unexpected tokenisation: none={none} passage={passage} digits={digits}")
    return {"none": none[0], "pass": passage[0], "age": passage[1],
            "digits": {k: v[0] for k, v in digits.items()}}


def decision_score(logits: torch.Tensor, ids: dict) -> torch.Tensor:
    """s = P(none) / (P(none) + P(Pass)) from next-token logits (..., V)."""
    two = logits[..., [ids["none"], ids["pass"]]].float()
    return torch.softmax(two, dim=-1)[..., 0]


class RaftConstraint(LogitsProcessor):
    """Per-row constrained decoding for a left-padded batch (all prompts end with `Source:`)."""

    def __init__(self, ids: dict, prompt_width: int, ks: list[int], tau: float | None):
        self.ids, self.width, self.ks, self.tau = ids, prompt_width, ks, tau
        self.scores: list[float] = [float("nan")] * len(ks)
        self.choice: list[str | None] = [None] * len(ks)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        step = input_ids.shape[1] - self.width
        scores = scores.clone()
        scores[:, list(SUPPRESS)] = -math.inf
        if self.tau is None:                         # unconstrained: specials suppressed only
            return scores
        neg = torch.full_like(scores, -math.inf)
        for b in range(scores.shape[0]):
            if step == 0:
                s = float(decision_score(scores[b], self.ids))
                self.scores[b] = s
                self.choice[b] = "none" if s >= self.tau else "pass"
                keep = [self.ids["none"] if self.choice[b] == "none" else self.ids["pass"]]
            elif self.choice[b] == "pass" and step == 1:
                keep = [self.ids["age"]]
            elif self.choice[b] == "pass" and step == 2:
                keep = [self.ids["digits"][k] for k in range(1, max(1, self.ks[b]) + 1)]
            else:
                continue
            row = neg[b].clone()
            row[keep] = scores[b, keep]
            scores[b] = row
        return scores


_QUOTE_OPEN = re.compile(r'^ Passage (\d)\nQuote: "(.*)$', re.S)


class CopyConstraint(LogitsProcessor):
    """Copy-constrained quoting: while generation is inside `Quote: "...`, only tokens that keep
    the quote an exact substring of the cited passage (or that close the quote) are allowed.
    The answer/refuse decision is untouched; tokens are ranked by the model, best first."""

    def __init__(self, tok, prompt_width: int, passages: list[list[str]], vocab: list[str], top: int = 256,
                 max_quote_chars: int = 300):
        self.tok, self.width, self.passages, self.vocab, self.top = tok, prompt_width, passages, vocab, top
        self.max_quote_chars = max_quote_chars        # ~40 words, the target spec; then force the close
        self.fallbacks = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] - self.width < 3:
            return scores
        for b in range(scores.shape[0]):
            text = self.tok.decode(input_ids[b, self.width:].tolist(), skip_special_tokens=True)
            m = _QUOTE_OPEN.match(text)
            if not m or '"' in m.group(2):
                continue
            k, q = int(m.group(1)), m.group(2)
            ps = self.passages[b]
            if not 1 <= k <= len(ps):
                continue
            src = ps[k - 1]
            allowed = []
            must_close = len(q) >= self.max_quote_chars
            for i in torch.topk(scores[b], self.top).indices.tolist():
                t = self.vocab[i]
                closes = (t == '"' or t.startswith('"\n')) and q and q in src   # keep `"\nAnswer:` intact
                if closes or (not must_close and t and not t.startswith('"') and (q + t) in src):
                    allowed.append(i)
            if not allowed:                       # nothing in the top candidates continues the copy
                self.fallbacks += 1
                continue
            keep = scores[b, allowed].clone()
            scores[b] = -math.inf
            scores[b, allowed] = keep
        return scores


@torch.no_grad()
def generate_batch(net, tok, prompts: list[str], ks: list[int], device: torch.device,
                   tau: float | None = 0.5, max_new: int = config.SFT.max_new_tokens,
                   batch: int = 32, autocast=None, copy_passages: list[list[str]] | None = None) -> list[dict]:
    """Greedy constrained generation. Returns [{'s', 'choice', 'text'}] where text starts after
    `Source:` (prepend 'Source:' to parse)."""
    ids = decision_ids(tok)
    net.eval()
    out = []
    for i in range(0, len(prompts), batch):
        enc = [tok.encode(p, add_special_tokens=False) for p in prompts[i:i + batch]]
        width = max(len(e) for e in enc)
        x = torch.full((len(enc), width), M.PAD_TOKEN_ID, dtype=torch.long)
        mask = torch.zeros((len(enc), width), dtype=torch.long)
        for j, e in enumerate(enc):                  # left padding for generation
            x[j, width - len(e):] = torch.tensor(e)
            mask[j, width - len(e):] = 1
        proc = RaftConstraint(ids, width, ks[i:i + batch], tau)
        procs = [proc]
        if copy_passages is not None:
            if not hasattr(generate_batch, "_vocab"):
                generate_batch._vocab = [tok.decode([j]) for j in range(len(tok))]
            procs.append(CopyConstraint(tok, width, copy_passages[i:i + batch], generate_batch._vocab))
        ctx = autocast() if autocast else torch.autocast("cpu", enabled=False)
        with ctx:
            # An explicit config: the base model's saved one has min_new_tokens=16 and sampling,
            # which blocks <|eos|> after the 15-token refusal (found by the 7.5 smoke test).
            y = net.generate(input_ids=x.to(device), attention_mask=mask.to(device),
                             generation_config=GREEDY.__class__(**{**GREEDY.to_dict(), "max_new_tokens": max_new}),
                             logits_processor=LogitsProcessorList(procs))
        for j in range(len(enc)):
            gen = y[j, width:].tolist()
            if M.EOS_TOKEN_ID in gen:
                gen = gen[:gen.index(M.EOS_TOKEN_ID)]
            out.append({"s": proc.scores[j], "choice": proc.choice[j],
                        "text": tok.decode(gen, skip_special_tokens=False)})
    return out


@torch.no_grad()
def teacher_forced(net, rows_ids: list[list[int]], decision_pos: list[int], target_mask: list[np.ndarray],
                   device: torch.device, dec: dict, batch: int = 16, autocast=None) -> dict:
    """Decision scores and the mean target-token loss from one forward per row (right padding)."""
    net.eval()
    scores, tgt_sum, tgt_n = [], 0.0, 0
    for i in range(0, len(rows_ids), batch):
        part = rows_ids[i:i + batch]
        width = max(len(r) for r in part)
        x = torch.full((len(part), width), M.PAD_TOKEN_ID, dtype=torch.long)
        m = torch.zeros((len(part), width), dtype=torch.bool)
        for j, r in enumerate(part):
            x[j, :len(r)] = torch.tensor(r)
            m[j, :len(r)] = torch.from_numpy(np.asarray(target_mask[i + j][:len(r)], dtype=bool))
        x, m = x.to(device), m.to(device)
        ctx = autocast() if autocast else torch.autocast("cpu", enabled=False)
        with ctx:
            logits = net(input_ids=x, use_cache=False).logits
        pos = torch.tensor([decision_pos[i + j] - 1 for j in range(len(part))], device=device)
        scores += decision_score(logits[torch.arange(len(part), device=device), pos], dec).tolist()
        lg, tg, mm = logits[:, :-1].float(), x[:, 1:], m[:, 1:]
        tgt_sum += float(torch.nn.functional.cross_entropy(lg[mm], tg[mm], reduction="sum"))
        tgt_n += int(mm.sum())
    return {"s": scores, "target_loss": tgt_sum / max(1, tgt_n)}


# ------------------------------------------------------------------------------ parsing
_OUT = re.compile(r'^Source: (none|Passage (\d))\n(?:Quote: "(.*)"\n)?Answer: (.*)$', re.S)


def parse_output(text_after_source: str) -> dict:
    """Parse 'Source:' + generated text. valid=False when the format is broken."""
    m = _OUT.match("Source:" + text_after_source)
    if not m:
        return {"valid": False, "refusal": None, "k": None, "quote": None, "answer": None}
    refusal = m.group(1) == "none"
    answer = m.group(4).strip()
    valid = (answer == config.RAFT.refusal) if refusal else True
    return {"valid": valid, "refusal": refusal, "k": None if refusal else int(m.group(2)),
            "quote": m.group(3), "answer": answer}


# ------------------------------------------------------------------------------ metrics
def _norm_tokens(s: str) -> list[str]:
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in s.split() if w not in ("a", "an", "the")]


def token_f1(pred: str, ref: str) -> float:
    p, r = _norm_tokens(pred), _norm_tokens(ref)
    common = sum((Counter(p) & Counter(r)).values())
    if not p or not r or common == 0:
        return float(p == r)
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def auroc(scores: list[float], positive: list[bool]) -> float | None:
    """Probability a random positive outscores a random negative (ties count half)."""
    pos = [s for s, y in zip(scores, positive) if y]
    neg = [s for s, y in zip(scores, positive) if not y]
    if not pos or not neg:
        return None
    allv = np.array(pos + neg)
    ranks = np.empty(len(allv))
    order = allv.argsort(kind="mergesort")
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                        # average ranks for ties
        idx = allv == v
        ranks[idx] = ranks[idx].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def decision_metrics(scores: list[float], refuse_gold: list[bool], buckets: list[str],
                     tau: float = 0.5) -> dict:
    """Answer/refuse as a classifier at threshold tau, plus AUROC on A vs B1+B2."""
    pred = [s >= tau for s in scores]
    tp = sum(p and g for p, g in zip(pred, refuse_gold))
    fp = sum(p and not g for p, g in zip(pred, refuse_gold))
    fn = sum((not p) and g for p, g in zip(pred, refuse_gold))
    tn = sum((not p) and (not g) for p, g in zip(pred, refuse_gold))
    f1 = lambda a, b, c: 2 * a / (2 * a + b + c) if a else 0.0
    sel = [i for i, b in enumerate(buckets) if b in ("A", "B1", "B2")]
    return {"acc": (tp + tn) / max(1, len(pred)), "refusal_precision": tp / max(1, tp + fp),
            "refusal_recall": tp / max(1, tp + fn), "refusal_f1": f1(tp, fp, fn),
            "answer_f1": f1(tn, fn, fp), "macro_f1": (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2,
            "auroc": auroc([scores[i] for i in sel], [buckets[i] != "A" for i in sel])}


def tau_table(scores: list[float], refuse_gold: list[bool], buckets: list[str],
              grid: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))) -> list[dict]:
    """Decision metrics at every tau, plus over-refusal on A and refusal recall on B1+B2+B3."""
    rows = []
    for t in grid:
        a = [x >= t for x, b in zip(scores, buckets) if b == "A"]
        u = [x >= t for x, b in zip(scores, buckets) if b in ("B1", "B2", "B3")]
        rows.append({"tau": t, **decision_metrics(scores, refuse_gold, buckets, t),
                     "over_refusal_A": sum(a) / max(1, len(a)), "recall_B": sum(u) / max(1, len(u))})
    return rows


def choose_tau(table: list[dict], max_over_refusal: float = config.RAFT.max_over_refusal,
               min_recall: float = config.RAFT.min_refusal_recall) -> dict:
    """Pre-registered rule (docs/08-raft.md §13, 7.7): highest macro-F1 among taus meeting both
    targets; if none does, the highest macro-F1 overall, flagged `meets_targets: False`. Ties go to
    the lower tau (answer more)."""
    ok = [r for r in table if r["over_refusal_A"] <= max_over_refusal and r["recall_B"] >= min_recall]
    pool = ok or table
    best = max(pool, key=lambda r: (r["macro_f1"], -r["tau"]))
    return {**best, "meets_targets": bool(ok)}


def proxy_metrics(rows: list[dict], outputs: list[dict]) -> dict:
    """Deterministic §7.2 proxies from constrained generations on dev rows.

    Answer/refuse comes from the constrained decision (`choice`), not from parsing: a malformed
    output after `ĠPass` still asserted an answer, so it counts as a hallucination on an
    unanswerable item and can never count as grounded (fixed 2026-10-16; parsing alone let
    malformed outputs count as neither)."""
    a_ok = a_n = a_refused = unans_answered = unans_n = bad_quote = valid = 0
    for r, o in zip(rows, outputs):
        p = parse_output(o["text"])
        valid += p["valid"]
        choice = o.get("choice")
        refused = choice == "none" if choice else p["refusal"] is True
        answered = choice == "pass" if choice else p["refusal"] is False
        if r["bucket"] in ("B1", "B2", "B3"):
            unans_n += 1
            unans_answered += answered
        if r["bucket"] == "A":
            a_n += 1
            a_refused += refused
            quote_ok = (p["valid"] and not p["refusal"] and p["quote"] is not None and p["k"]
                        and 1 <= p["k"] <= len(r["passages"]) and p["quote"] in r["passages"][p["k"] - 1])
            if answered and not quote_ok:
                bad_quote += 1
            if (answered and quote_ok and p["k"] == r["k"]
                    and token_f1(p["answer"] or "", r["answer"] or "") >= 0.5):
                a_ok += 1
    n = max(1, len(rows))
    return {"proxy_grounded_acc": a_ok / max(1, a_n),
            "proxy_hallucination": (unans_answered + bad_quote) / n,
            "over_refusal_A": a_refused / max(1, a_n),
            "refusal_recall": 1 - unans_answered / max(1, unans_n),
            "format_validity": valid / n,
            "selection_score": a_ok / max(1, a_n) - (unans_answered + bad_quote) / n - a_refused / max(1, a_n)}
