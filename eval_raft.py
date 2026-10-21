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

import string
from collections import Counter

import numpy as np
import torch

import config
import model as M

import raft_decode as RD
from raft_decode import (_YES_NO, GREEDY, SUPPRESS, CopyConstraint, RaftConstraint, decision_ids,
                         decision_score, generate_batch, parse_output)

# re-exported for callers that use eval_raft.<name> (sft, raft_gpu_app, raft_pipeline, smoke_raft)
REEXPORTED = (_YES_NO, GREEDY, SUPPRESS, CopyConstraint, RaftConstraint, decision_ids, generate_batch)

# the standalone module's constants must match the project's (config.py and model.py)
assert (RD.BOS_ID, RD.EOS_ID, RD.PAD_ID, RD.UNK_ID) == (M.BOS_TOKEN_ID, M.EOS_TOKEN_ID, M.PAD_TOKEN_ID, M.UNK_TOKEN_ID)
assert (RD.K_PASSAGES, RD.MAX_NEW_TOKENS, RD.REFUSAL) == (config.RAFT.k_passages, config.SFT.max_new_tokens, config.RAFT.refusal)


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
            quote_ok = (p["valid"] and not p["refusal"] and p["quote"] and p["k"]
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
