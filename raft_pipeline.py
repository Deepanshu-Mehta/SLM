"""Phase 7 stage pipeline (local CLI): sources -> generate -> candidates -> assemble -> check -> report.

Paid steps (anything calling Modal or an API) require --confirm and charge the stage's ledger
(`.smoke/raft/{stage}/ledger.json`, cap from config.RAFT_STAGE_CAPS_USD). Local files live in
`.smoke/raft/{stage}/`; Modal-side files in `/data/raft/{stage}/`. docs/08-raft.md §5.

Trial (step 7.3):
  python raft_pipeline.py sources    --stage trial --n 150 --confirm
  python raft_pipeline.py generate   --stage trial --writer deepseek --confirm
  python raft_pipeline.py generate   --stage trial --writer gemini   --confirm
  python raft_pipeline.py candidates --stage trial
  python raft_pipeline.py assemble   --stage trial --confirm
  python raft_pipeline.py check      --stage trial --checker ds_off --confirm   (and ds_on, gemini)
  python raft_pipeline.py report     --stage trial
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import config
import raft_data as R
import raft_prompts as P
from teacher import DeepSeek, GeminiBatch, Job, Ledger, load_env

T = config.TEACHER
LOCAL = Path(".smoke/raft")
STAGE_CAP = {"trial": "7.3-trial", "pilot": "7.4-pilot-data", "scale": "7.8-scale-data"}
TRIAL_PLAN = {"G1": 2 / 3, "G2": 1 / 6, "G3": 1 / 6}
GEMINI_MAX_TOKENS = 1_024              # Gemini counts thinking inside max_output_tokens
CHECKERS = ("ds_off", "ds_on", "gemini")


def sdir(stage: str) -> Path:
    d = LOCAL / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def ledger(stage: str) -> Ledger:
    return Ledger(sdir(stage) / "ledger.json", config.RAFT_STAGE_CAPS_USD[STAGE_CAP[stage]])


def jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()] \
        if Path(path).exists() else []


def write_jsonl(path: Path, rows) -> int:
    n = 0
    with Path(path).open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def need_confirm(args, what: str) -> None:
    if not args.confirm:
        raise SystemExit(f"{what} costs money. Re-run with --confirm once approved.")


def _modal():
    load_env()
    import modal

    import raft_app as A
    return modal, A


def modal_cost(results: list[dict]) -> float:
    from raft_app import CPU_RATE, MEM_RATE
    return sum(r["seconds"] * (r["cpu"] * CPU_RATE + r["mem_gib"] * MEM_RATE) for r in results)


def log_modal_cost(stage: str, what: str, usd: float) -> None:
    path = sdir(stage) / "modal_cost.json"
    d = json.loads(path.read_text()) if path.exists() else {}
    d[what] = d.get(what, 0.0) + usd
    path.write_text(json.dumps(d, indent=2))


# ------------------------------------------------------------------------- sources
def cmd_sources(args) -> None:
    need_confirm(args, "Selecting sources (Modal CPU)")
    modal, A = _modal()
    plan = TRIAL_PLAN if args.stage == "trial" else json.loads(args.plan)
    with modal.enable_output(), A.app.run():
        res = A.train_sources.remote(args.stage, args.n, plan, args.exclude.split(",") if args.exclude else [])
    log_modal_cost(args.stage, "sources", modal_cost([res]))
    n = write_jsonl(sdir(args.stage) / "sources.jsonl", res["rows"])
    print(f"{n} sources: {dict(Counter((r['source'], r['kind']) for r in res['rows']))}")


# ------------------------------------------------------------------------ generate
def gen_jobs(stage: str, writer: str) -> list[Job]:
    jobs = []
    for r in jsonl(sdir(stage) / "sources.jsonl"):
        system, user = P.gen_messages(r["kind"], r["source"], r["text"], r.get("task"))
        jobs.append(Job(f"{r['pid']}|{r['kind']}|{writer}", system, user, T.max_tokens_generate))
    return jobs


def cmd_generate(args) -> None:
    need_confirm(args, "Generation (API)")
    lg = ledger(args.stage)
    out = sdir(args.stage) / f"gen-{args.writer}.jsonl"
    jobs = gen_jobs(args.stage, args.writer)
    if args.writer == "deepseek":
        stats = DeepSeek(lg, allow_peak=args.allow_peak, workers=args.workers).run(jobs, out)
    else:
        stats = GeminiBatch(lg).run([Job(j.key, j.system, j.user, GEMINI_MAX_TOKENS) for j in jobs],
                                    out, f"raft-{args.stage}-gen")
    print(f"generate {args.writer}: {stats}; stage spend ${lg.spent:.4f} of ${lg.cap:.2f}")


# ---------------------------------------------------------------------- candidates
def cmd_candidates(args) -> None:
    """Mechanical checks (§5.2) on generator output; B1 pairs from accepted A questions."""
    st = sdir(args.stage)
    sources = {r["pid"]: r for r in jsonl(st / "sources.jsonl")}
    rng = random.Random(config.RAFT.seed)
    cands, funnel = [], defaultdict(Counter)
    seen_q: set[str] = set()
    for writer in ("deepseek", "gemini"):
        for g in jsonl(st / f"gen-{writer}.jsonl"):
            pid, kind, _ = g["key"].split("|")
            src = sources[pid]
            f = funnel[f"{writer}:{kind}"]
            f["calls"] += 1
            if not g["ok"]:
                f["call_failed:" + (g.get("error") or "?")[:40]] += 1
                continue
            base = {"writer": writer, "source": src["source"], "split": src["split"],
                    "source_pid": pid, "kind": kind}
            if kind == "G1":
                items = (g["json"] or {}).get("items") or []
                if not items:
                    f["empty"] += 1
                for i, it in enumerate(items[:2] if isinstance(items, list) else []):
                    f["items"] += 1
                    why, clean = R.check_g1_item(it if isinstance(it, dict) else {}, src["text"])
                    qn = R.canon(clean.get("question", "")).lower()
                    if not why and qn in seen_q:
                        why = "duplicate_question"
                    if why:
                        f["drop:" + why] += 1
                        continue
                    seen_q.add(qn)
                    f["passed"] += 1
                    cands.append({**base, **clean, "bucket": "A", "cid": f"{writer}:{pid}:{i}"})
            elif kind == "G2":
                f["items"] += 1
                why, clean = R.check_g2_item(g["json"] or {}, src["text"])
                if why:
                    f["drop:" + why] += 1
                    continue
                f["passed"] += 1
                cands.append({**base, **clean, "bucket": "B2", "cid": f"{writer}:{pid}:B2"})
            else:
                f["items"] += 1
                why, clean = R.check_g3_item(g["json"] or {}, src["text"], src["task"])
                if why:
                    f["drop:" + why] += 1
                    continue
                f["passed"] += 1
                cands.append({**base, **clean, "bucket": "C", "cid": f"{writer}:{pid}:C"})
    a_rows = [c for c in cands if c["bucket"] == "A"]
    for c in rng.sample(a_rows, len(a_rows) // 2):           # trial: pair half of A
        cands.append({**c, "bucket": "B1", "cid": c["cid"] + ":B1", "pair": c["cid"]})
    n = write_jsonl(st / "candidates.jsonl", cands)
    (st / "funnel.json").write_text(json.dumps({k: dict(v) for k, v in funnel.items()}, indent=2))
    print(f"{n} candidates: {dict(Counter((c['writer'], c['bucket']) for c in cands))}")
    for k, v in sorted(funnel.items()):
        print(f"  {k:<14} {dict(v)}")


# ------------------------------------------------------------------------ assemble
def cmd_assemble(args) -> None:
    need_confirm(args, "Assembly (Modal CPU)")
    modal, A = _modal()
    st = sdir(args.stage)
    vol = modal.Volume.from_name(config.VOLUME_NAME)
    with vol.batch_upload(force=True) as up:
        up.put_file(str(st / "candidates.jsonl"), f"raft/{args.stage}/candidates.jsonl")
    cands = jsonl(st / "candidates.jsonl")
    jobs = sorted({(c["split"], c["source"]) for c in cands})
    with modal.enable_output(), A.app.run():
        res = list(A.assemble_source.starmap([(args.stage, sp, so) for sp, so in jobs]))
    log_modal_cost(args.stage, "assemble", modal_cost(res))
    rows = []
    for sp, so in jobs:
        data = b"".join(vol.read_file(f"raft/{args.stage}/assembled-{sp}-{so}.jsonl"))
        rows += [json.loads(ln) for ln in data.decode().splitlines() if ln.strip()]
    write_jsonl(st / "assembled.jsonl", rows)
    for r in res:
        print(f"  {r['split']}-{r['source']}: {r['assembled']} assembled, drops {r['drops']}")
    print(f"{len(rows)} assembled: {dict(Counter((r['writer'], r['bucket']) for r in rows))}")


# --------------------------------------------------------------------------- check
def check_jobs(rows: list[dict], checker: str) -> list[Job]:
    jobs = []
    for r in rows:
        if r["bucket"] in ("A", "B1", "B2"):
            system, user = P.v1_messages(r["passages"], r["question"])
        elif r.get("task") in ("summarize", "yes_no_check") and checker != "ds_on":
            resp = None if r["task"] == "yes_no_check" else r["response"]
            system, user = P.v2_messages(r["k"], r["passages"][r["k"] - 1], r["task_text"], resp)
        else:
            continue                          # extract_list is model-free; V2 runs without thinking
        think = checker == "ds_on"
        mt = (GEMINI_MAX_TOKENS if checker == "gemini"
              else T.max_tokens_check_thinking if think else T.max_tokens_check)
        jobs.append(Job(f"{r['cid']}|{checker}", system, user, mt, think))
    return jobs


def cmd_check(args) -> None:
    need_confirm(args, "Checking (API)")
    st, lg = sdir(args.stage), ledger(args.stage)
    rows = jsonl(st / "assembled.jsonl")
    jobs = check_jobs(rows, args.checker)
    out = st / f"checks-{args.checker}.jsonl"
    if args.checker == "gemini":
        stats = GeminiBatch(lg).run(jobs, out, f"raft-{args.stage}-check")
    else:
        stats = DeepSeek(lg, allow_peak=args.allow_peak, workers=args.workers).run(jobs, out)
    print(f"check {args.checker}: {len(jobs)} jobs, {stats}; stage spend ${lg.spent:.4f} of ${lg.cap:.2f}")


# -------------------------------------------------------------------------- report
def accept(row: dict, v: dict | None) -> bool | None:
    """§5.4 acceptance for one checker verdict (None = no usable verdict)."""
    if not v or not isinstance(v, dict):
        return None
    b = row["bucket"]
    if b in ("B1", "B2"):
        return v.get("answerable") is False and not v.get("ambiguous")
    if b == "A":
        if v.get("answerable") is not True or v.get("ambiguous"):
            return False
        if sorted(v.get("support") or []) != [row["k"]]:
            return False
        cut = R.recut_quote(v.get("quote") or "", row["passages"][row["k"] - 1])
        if cut is None or not R.spans_overlap(cut[1], tuple(row["quote_span"])):
            return False
        return R.answers_agree(v.get("answer") or "", row["answer"])
    if row.get("task") == "extract_list":
        return True
    if row.get("task") == "yes_no_check":
        return (v.get("yes_no") or "").strip().rstrip(".").lower() == row["yes_no"].lower()
    return bool(v.get("supported")) and bool(v.get("on_task"))


def cmd_report(args) -> None:
    st = sdir(args.stage)
    rows = jsonl(st / "assembled.jsonl")
    verdicts = {c: {r["key"].split("|")[0]: r for r in jsonl(st / f"checks-{c}.jsonl")} for c in CHECKERS}

    def acc(row, checker):
        if row.get("task") == "extract_list":
            return True
        v = verdicts[checker].get(row["cid"])
        return accept(row, v["json"] if v and v["ok"] else None)

    def production(row):          # train rule: DeepSeek; thinking only if config enables it
        think = T.think_on_refusal_checks and row["bucket"] in ("B1", "B2")
        return acc(row, "ds_on" if think else "ds_off")

    def agree(a, b, subset):
        pairs = [(acc(r, a), acc(r, b)) for r in subset]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        return (sum(x == y for x, y in pairs) / len(pairs), len(pairs)) if pairs else (None, 0)

    funnel = json.loads((st / "funnel.json").read_text())
    rep: dict = {"stage": args.stage, "assembled": len(rows)}
    # 1. writer keep rate: generated items -> accepted by DeepSeek (production rule) AND Gemini
    keep = {}
    for w in ("deepseek", "gemini"):
        items = sum(v.get("items", 0) for k, v in funnel.items() if k.startswith(w))
        mine = [r for r in rows if r["writer"] == w and r["bucket"] != "B1"]
        both = sum(bool(production(r)) and bool(acc(r, "gemini")) for r in mine)
        keep[w] = {"items": items, "assembled": len(mine), "accepted_both": both,
                   "keep_rate": both / items if items else None}
    rep["writer_keep"] = keep
    dk, gk = keep["deepseek"]["keep_rate"] or 0, keep["gemini"]["keep_rate"] or 0
    rep["rule_writer"] = "keep DeepSeek" if gk < dk + 0.10 else "Gemini >= 10 points better: re-quote and ask"
    # 2. thinking on refusal labels
    refusal = [r for r in rows if r["bucket"] in ("B1", "B2")]
    on, n_on = agree("ds_on", "gemini", refusal)
    off, n_off = agree("ds_off", "gemini", refusal)
    rep["thinking"] = {"agree_on": on, "agree_off": off, "n": min(n_on, n_off)}
    rep["rule_thinking"] = ("thinking OFF everywhere" if on is not None and off is not None and off >= on - 0.02
                            else "keep thinking on refusal-label checks")
    # 3. parse rates
    rep["parse_rate"] = {c: (sum(v["ok"] for v in verdicts[c].values()) / len(verdicts[c])
                             if verdicts[c] else None) for c in CHECKERS}
    rep["thinking_leaks"] = 0          # a leak aborts the run in teacher.DeepSeek
    # 4. checker compatibility (production DeepSeek vs Gemini), all checked rows
    pairs = [(production(r), acc(r, "gemini")) for r in rows]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    comp = sum(x == y for x, y in pairs) / len(pairs) if pairs else None
    rep["checker_agreement"] = {"rate": comp, "n": len(pairs)}
    rep["rule_checkers"] = "ok" if comp is not None and comp >= 0.85 else "below 85%: inspect prompts"
    # 5. keep rates by bucket (production rule = train; both = dev/test rule)
    by = defaultdict(Counter)
    for r in rows:
        b = by[r["bucket"]]
        b["assembled"] += 1
        b["train_rule"] += bool(production(r))
        b["devtest_rule"] += bool(production(r)) and bool(acc(r, "gemini"))
    rep["by_bucket"] = {k: dict(v) for k, v in by.items()}
    rep["entity_present"] = {b: (sum(r["entity_present"] for r in rows if r["bucket"] == b)
                                 / max(1, sum(r["bucket"] == b for r in rows))) for b in ("B1", "B2")}
    # 6. measured cost per call and kind
    lg = json.loads((st / "ledger.json").read_text())
    rep["ledger"] = lg
    mc = json.loads((st / "modal_cost.json").read_text()) if (st / "modal_cost.json").exists() else {}
    rep["modal_cost_usd"] = mc
    rep["total_usd"] = lg["spent_usd"] + sum(mc.values())
    rep["funnel"] = funnel
    rep["spot_check"] = [{k: r.get(k) for k in ("bucket", "writer", "question", "task_text", "answer",
                                                 "quote", "response", "k")}
                         for r in random.Random(1).sample(rows, min(8, len(rows)))]
    (st / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep[k] for k in ("writer_keep", "rule_writer", "thinking", "rule_thinking",
                                          "parse_rate", "checker_agreement", "rule_checkers",
                                          "by_bucket", "entity_present", "total_usd")},
                     indent=1, default=str))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=("sources", "generate", "candidates", "assemble", "check", "report"))
    p.add_argument("--stage", default="trial", choices=tuple(STAGE_CAP))
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--plan", default="")
    p.add_argument("--exclude", default="")
    p.add_argument("--writer", choices=("deepseek", "gemini"), default="deepseek")
    p.add_argument("--checker", choices=CHECKERS, default="ds_off")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--allow-peak", action="store_true")
    p.add_argument("--confirm", action="store_true")
    args = p.parse_args()
    t0 = time.time()
    {"sources": cmd_sources, "generate": cmd_generate, "candidates": cmd_candidates,
     "assemble": cmd_assemble, "check": cmd_check, "report": cmd_report}[args.cmd](args)
    print(f"[{args.cmd}] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
