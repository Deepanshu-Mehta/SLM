"""Phase 12 data: RLAIF prompts, the Gemini Flash constitution judge, preference pairs (docs/13-rlaif.md §3-§4).

- `build_prompts`: spare rows from the 7.8 scale stage that passed every Phase 7 check (same `accepted` rule,
  probe-cited and dev/test documents excluded) but were not selected for training, plus B3 rows built
  from them. Never dev/test, never a training row.
- `CONSTITUTION_*` + `judge_jobs`: one Gemini Flash call per distinct, format-valid sample (pointwise 1-5).
- `pairs`: within a prompt, every pair of valid samples whose scores differ by >= 2 (C(K,2) filtered),
  capped per prompt; length statistics of chosen vs rejected (length-bias guard).
- `agreement`: the judge's should_refuse against the construction label (B1/B2/B3 = refuse, A/C = answer).
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import config
import raft_decode as RD
from teacher import Job

LOCAL = Path(".smoke/rlaif")
REFUSE_BUCKETS = ("B1", "B2", "B3")
SCORE_GAP = 2
MAX_PAIRS_PER_PROMPT = 3
JUDGE_MAX_TOKENS = 384


# ----------------------------------------------------------------------------- prompts
def build_prompts(seed: int = config.RAFT.seed) -> list[dict]:
    """Spare verified scale rows (A, B2, C) + B3 rows made from spare A questions, in dataset_record shape."""
    from transformers import AutoTokenizer

    import raft_pipeline as RP
    st = RP.sdir("scale")
    rows, V = RP.jsonl(st / "assembled.jsonl"), RP._verdicts(st)
    dk = lambda pid: pid.rsplit("/", 1)[0]
    pdir = RP.PILOT_DATASETS
    probe_docs = {dk(p) for r in RP.jsonl(pdir / "probe.jsonl") for p in (r.get("pids") or [])}
    devtest_docs = {dk(p) for g in ("dev", "test") for r in RP.jsonl(pdir / f"{g}.jsonl") for p in (r.get("pids") or [])}
    used = {json.loads(ln)["id"] for g in ("train", "dev", "test", "probe")
            for ln in open(st / "datasets" / f"{g}.jsonl")}

    def ok(r: dict) -> bool:
        cited = r["bucket"] in ("A", "C") and r.get("k") and r.get("pids") and dk(r["pids"][int(r["k"]) - 1]) in probe_docs
        return (RP.accepted(r, V) and not cited and r["cid"] not in used
                and not ({dk(p) for p in (r.get("pids") or [])} & devtest_docs))
    spare = [r for r in rows if r["bucket"] in ("A", "B2", "C") and ok(r)]
    rng = random.Random(f"{seed}:rlaif")
    rng.shuffle(spare)
    others = defaultdict(list)
    for r in RP.jsonl(st / "sources.jsonl"):
        others[r["source"]].append(r)
    tok = AutoTokenizer.from_pretrained(".smoke/final/base")
    a = [r for r in spare if r["bucket"] == "A"]
    b3 = RP.make_b3(a, max(1, round(0.07 * len(spare))), others, tok, rng)
    out = [RP.dataset_record(r, "rlaif") for r in spare + b3]
    for r in out:
        r["k_samples"] = 8 if r["bucket"] == "B2" else 4     # near misses weighted (docs/13 §4)
    return out


# ------------------------------------------------------------------------- constitution
CONSTITUTION_SYSTEM = """You are grading one answer from a small model that must answer ONLY from the numbered passages it was
given, citing one passage and quoting it word for word, or refuse with "The passages do not contain the answer."
Apply these rules in order (this is the constitution):
1. Decision. Decide yourself whether the passages contain the answer. Refusing is right only if they do not.
   Answering a question the passages cannot answer is the worst error.
2. Support. Every factual claim in the answer must be stated in the cited passage, and the quote must be a
   meaningful span of that passage that backs the answer (a one- or two-word fragment does not back anything).
3. Correctness. When a reference answer is given, the answer must mean the same thing; missing the key fact
   or adding a wrong claim makes it incorrect.
4. Concision. Shorter is not worse and longer is not better; penalize added content that is not asked for.
Score 1-5:
5 = right decision; if it answered: correct, fully supported, quote backs it, concise.
4 = right decision and correct, with a minor flaw (weak quote, slight extra content).
3 = right decision but the answer is only partly correct or partly supported.
2 = refused although the passages contain the answer, OR answered with an unsupported or wrong answer.
1 = answered although the passages do not contain the answer.
Reply with JSON only: {"should_refuse": true|false, "refused": true|false, "supported": true|false,
"correct": true|false|null, "score": 1-5, "failed": "decision"|"support"|"correctness"|"concision"|"none"}"""


def constitution_messages(row: dict, output_text: str) -> tuple[str, str]:
    """(system, user). The reference is shown for A/C rows (reference-guided judging), never the bucket."""
    body = "\n".join(f"Passage {i}: {p}" for i, p in enumerate(row["passages"] or [], 1)) or "(no passages)"
    ask = row["question"] if row["bucket"] != "C" else row["task_text"]
    ref = f"\nReference answer: {row['answer']}" if row["bucket"] in ("A", "C") and row.get("answer") else ""
    return CONSTITUTION_SYSTEM, f"{body}\nQuestion or task: {ask}{ref}\nModel output:\nSource:{output_text}"


def valid_sample(row: dict, text: str) -> bool:
    """Parses, and an answered sample's quote is verbatim in the cited passage (never pair a syntax slip)."""
    p = RD.parse_output(text)
    if not p["valid"]:
        return False
    if p["refusal"]:
        return True
    ps = row["passages"] or []
    if not (p["k"] and 1 <= p["k"] <= len(ps)):
        return False
    return p["quote"] is None or (bool(p["quote"]) and p["quote"] in ps[p["k"] - 1])


def judge_jobs(rows: dict[str, dict], samples: dict[str, list[dict]], tag: str) -> list[Job]:
    """One job per distinct valid sample text per prompt. Key: '{row id}|{tag}|{sample index}'."""
    jobs = []
    for rid, ss in samples.items():
        seen = set()
        for i, s in enumerate(ss):
            if s["text"] in seen or not valid_sample(rows[rid], s["text"]):
                continue
            seen.add(s["text"])
            system, user = constitution_messages(rows[rid], s["text"])
            jobs.append(Job(f"{rid}|{tag}|{i}", system, user, JUDGE_MAX_TOKENS))
    return jobs


def scored(rows: dict[str, dict], samples: dict[str, list[dict]], verdicts: dict[str, dict], tag: str) -> dict[str, list[dict]]:
    """Per prompt: valid samples with their judge score (identical texts share the verdict)."""
    out = {}
    for rid, ss in samples.items():
        by_text = {}
        for i, s in enumerate(ss):
            v = verdicts.get(f"{rid}|{tag}|{i}")
            if v is not None and isinstance(v.get("score"), (int, float)):
                by_text[s["text"]] = v
        out[rid] = [{**s, "verdict": by_text[s["text"]]} for s in ss if s["text"] in by_text]
    return out


def pairs(rows: dict[str, dict], scored_samples: dict[str, list[dict]], tok, seed: int = 0) -> tuple[list[dict], dict]:
    """Chosen/rejected pairs (score gap >= SCORE_GAP, distinct texts, both valid), <= MAX_PAIRS_PER_PROMPT per
    prompt (largest gaps first), plus statistics including the length-bias check."""
    rng = random.Random(seed)
    out, kinds = [], Counter()
    for rid, ss in scored_samples.items():
        uniq = {}
        for s in ss:
            uniq.setdefault(s["text"], s)
        u = list(uniq.values())
        cand = []
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                a, b = u[i], u[j]
                gap = a["verdict"]["score"] - b["verdict"]["score"]
                if abs(gap) >= SCORE_GAP:
                    c, r = (a, b) if gap > 0 else (b, a)
                    cand.append((abs(gap), rng.random(), c, r))
        cand.sort(key=lambda t: (-t[0], t[1]))
        for gap, _, c, r in cand[:MAX_PAIRS_PER_PROMPT]:
            kind = f"{'refuse' if c['choice'] == 'none' else 'answer'}>{'refuse' if r['choice'] == 'none' else 'answer'}"
            kinds[kind] += 1
            out.append({"id": rid, "bucket": rows[rid]["bucket"], "chosen": c["text"], "rejected": r["text"],
                        "chosen_score": c["verdict"]["score"], "rejected_score": r["verdict"]["score"], "kind": kind})
    ln = lambda t: len(tok.encode(t, add_special_tokens=False))
    lc = [ln(p["chosen"]) for p in out]
    lr = [ln(p["rejected"]) for p in out]
    mc, mr = (sum(lc) / max(1, len(lc)), sum(lr) / max(1, len(lr)))
    aa = [p for p in out if p["kind"] == "answer>answer"]          # the length-bias test: like against like
    ac = sum(ln(p["chosen"]) for p in aa) / max(1, len(aa))
    ar = sum(ln(p["rejected"]) for p in aa) / max(1, len(aa))
    stats = {"pairs": len(out), "prompts_with_pairs": len({p["id"] for p in out}), "kinds": dict(kinds),
             "by_bucket": dict(Counter(p["bucket"] for p in out)),
             "mean_tokens_chosen": round(mc, 1), "mean_tokens_rejected": round(mr, 1),
             "answer_pairs_mean_tokens_chosen": round(ac, 1), "answer_pairs_mean_tokens_rejected": round(ar, 1),
             "length_ratio": round(ac / max(1e-9, ar), 3), "length_bias_flag": bool(aa) and ac > 1.15 * ar}
    return out, stats


def agreement(rows: dict[str, dict], verdicts: dict[str, dict]) -> dict:
    """Judge should_refuse vs the construction label, per bucket (stop rule: < 0.90 on B1 or B3)."""
    hit, n = Counter(), Counter()
    for key, v in verdicts.items():
        if not isinstance(v, dict) or "should_refuse" not in v:
            continue
        r = rows[key.split("|")[0]]
        want = r["bucket"] in REFUSE_BUCKETS
        n[r["bucket"]] += 1
        hit[r["bucket"]] += bool(v["should_refuse"]) == want
    return {b: round(hit[b] / n[b], 3) for b in n}


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()] if Path(path).exists() else []


_JSON = re.compile(r"\{.*\}", re.S)
