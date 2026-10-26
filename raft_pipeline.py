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

Pilot evaluation (step 7.7b), after raft_gpu_app.py::pilot_eval_check wrote .smoke/raft/pilot/eval/:
  python raft_pipeline.py judge --stage pilot-eval --confirm     (Gemini J1, both decodings)
  python raft_pipeline.py score --stage pilot-eval               (§8.1 headline metrics, $0)
"""

from __future__ import annotations

import argparse
import json
import os
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
STAGE_CAP = {"trial": "7.3-trial", "pilot": "7.4-pilot-data", "pilot-eval": "7.7-judge", "scale": "7.8-scale-data",
             "gemma": "9-judge",              # Phase 9 (docs/10-gemma.md): J1 on Gemma outputs, local files in .smoke/gemma
             "slm500": "10-judge",            # Phase 10 (docs/11-slm500m-raft.md): local files in .smoke/slm500m
             "gemma3": "11-judge"}            # Phase 11 (docs/12-gemma3-270m.md): local files in .smoke/gemma3
GEMMA_LOCAL = Path(".smoke/gemma")
SLM500_LOCAL = Path(".smoke/slm500m")
GEMMA3_LOCAL = Path(".smoke/gemma3")
TRIAL_PLAN = {"G1": 2 / 3, "G2": 1 / 6, "G3": 1 / 6}
GEMINI_MAX_TOKENS = 1_024              # Gemini counts thinking inside max_output_tokens
GEMINI_CHECK_MAX_TOKENS = 384          # trial 7.3: check output+thoughts median 58, max 202 tokens
GEMINI_CHUNK = int(os.environ.get("RAFT_GEMINI_CHUNK", 800))   # requests per batch: each worst-case reservation settles first
CHECKERS = ("ds_off", "ds_on", "gemini", "ds_v3")
DEVTEST_POOL = Path(".smoke/reports/source_pool_v2.jsonl")   # leak-scanned val sources (7.2 fix)

# Pilot (7.4) quotas: train 2,400 = enough for 15/25/35% refusal mixes of 2,000 (§5);
# dev and test per §8.1; probe = refusal items about pretrained-on documents (§8.1).
PILOT_QUOTAS = {
    "train": {"A": 1247, "B1": 420, "B2": 210, "B3": 70, "C": 453},
    "dev": {"A": 275, "B1": 75, "B2": 37, "B3": 13, "C": 100},
    "test": {"A": 500, "B1": 150, "B2": 100, "B3": 50, "C": 200},
    "probe": {"B1": 100},
}
# Generator calls per group, from trial 7.3 yields (E-22) with buffers (G1 x1.25 for the new
# vague-reference rule, others x1.15); B1 candidates paired from A candidates.
PILOT_CALLS = {"train": {"G1": 1300, "G2": 390, "G3": 620}, "dev": {"G1": 305, "G2": 77, "G3": 137},
               "test": {"G1": 555, "G2": 206, "G3": 274}, "probe": {"G1": 100}}
PILOT_B1_CANDIDATES = {"train": 520, "dev": 120, "test": 240}   # dev/test pairs need both twins accepted by both checkers (~0.67)
TASKS = ("summarize", "extract_list", "yes_no_check")
# Scale (7.8, D-27): 20K train = the pilot's 2,400 train rows + these quotas (docs/08-raft.md §13 7.8).
SCALE_QUOTAS = {"train": {"A": 8353, "B1": 3830, "B2": 2000, "B3": 650, "C": 3147}}
SCALE_CALLS = {"G1": 9300, "G2": 3600, "G3": 3800}       # pilot train keep rates (E-23) + ~10% margin
SCALE_B1_CANDIDATES = 4400                                 # B1 keep 0.88 in the pilot
PILOT_DATASETS = LOCAL / "pilot" / "datasets"
AUDIT_N = 1000                                             # 5% of the scale train rows, Gemini (§13 7.8)
AUDIT_MAX_DISAGREE = 0.15


def sdir(stage: str) -> Path:
    d = {"gemma": GEMMA_LOCAL, "slm500": SLM500_LOCAL, "gemma3": GEMMA3_LOCAL}.get(stage, LOCAL / stage)
    d.mkdir(parents=True, exist_ok=True)
    return d


def ledger(stage: str) -> Ledger:
    caps = {**config.RAFT_STAGE_CAPS_USD, **config.GEMMA_STAGE_CAPS_USD, **config.SLM500_STAGE_CAPS_USD,
            **config.GEMMA3_STAGE_CAPS_USD}
    return Ledger(sdir(stage) / "ledger.json", caps[STAGE_CAP[stage]])


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
    if args.stage == "scale":
        n = sum(SCALE_CALLS.values())
        plan = {k: v / n for k, v in SCALE_CALLS.items()}
        with modal.enable_output(), A.app.run(detach=True):     # a dropped client must not stop it (brief)
            res = A.train_sources.remote("scale", n, plan, ["trial", "pilot"], {"sec": 4})   # E-33
        log_modal_cost("scale", "sources", modal_cost([res]))
        rows, seen = res["rows"], Counter()
        for r in rows:
            r["group"] = "train"
            if r["kind"] == "G1":
                rot = P.TYPE_ROTATION[r["source"]]
                r["type_hint"] = rot[seen[r["source"]] % len(rot)]
                seen[r["source"]] += 1
        n = write_jsonl(sdir("scale") / "sources.jsonl", rows)
        print(f"{n} sources: {dict(Counter((r['source'], r['kind']) for r in rows))}")
        return
    if args.stage != "pilot":
        plan = TRIAL_PLAN if args.stage == "trial" else json.loads(args.plan)
        with modal.enable_output(), A.app.run():
            res = A.train_sources.remote(args.stage, args.n, plan,
                                         args.exclude.split(",") if args.exclude else [])
        log_modal_cost(args.stage, "sources", modal_cost([res]))
        n = write_jsonl(sdir(args.stage) / "sources.jsonl", res["rows"])
        print(f"{n} sources: {dict(Counter((r['source'], r['kind']) for r in res['rows']))}")
        return
    c = PILOT_CALLS
    n_train = sum(c["train"].values()) + c["probe"]["G1"]
    plan = {"G1": (c["train"]["G1"] + c["probe"]["G1"]) / n_train,
            "G2": c["train"]["G2"] / n_train, "G3": c["train"]["G3"] / n_train}
    with modal.enable_output(), A.app.run():
        res = A.train_sources.remote("pilot", n_train, plan, ["trial"])
    log_modal_cost("pilot", "sources", modal_cost([res]))
    rows = res["rows"]
    probe = {r["pid"] for r in [r for r in rows if r["kind"] == "G1"][:c["probe"]["G1"]]}
    for r in rows:
        r["group"] = "probe" if r["pid"] in probe else "train"
    rng = random.Random(config.RAFT.seed)
    pool = jsonl(DEVTEST_POOL)
    for grp in ("dev", "test"):
        cand = [r for r in pool if r["devtest"] == grp]
        rng.shuffle(cand)
        i = 0
        for kind, n in c[grp].items():
            if i + n > len(cand):
                raise SystemExit(f"{grp}: needs {i + n} sources, pool has {len(cand)}")
            for j, r in enumerate(cand[i:i + n]):
                rows.append({"pid": r["pid"], "doc": r["doc"], "source": r["pid"].split("/")[0],
                             "split": "val", "elig": r["elig"], "text": r["text"], "kind": kind,
                             "group": grp, **({"task": TASKS[j % 3]} if kind == "G3" else {})})
            i += n
    seen = Counter()
    for r in rows:                     # rotate the requested G1 type within each source
        if r["kind"] == "G1":
            rot = P.TYPE_ROTATION[r["source"]]
            r["type_hint"] = rot[seen[r["source"]] % len(rot)]
            seen[r["source"]] += 1
    n = write_jsonl(sdir("pilot") / "sources.jsonl", rows)
    print(f"{n} sources: {dict(Counter((r['group'], r['kind']) for r in rows))}")
    print(f"  by source: {dict(Counter((r['group'], r['source']) for r in rows))}")


# ------------------------------------------------------------------------ generate
def gen_jobs(stage: str, writer: str) -> list[Job]:
    jobs = []
    for r in jsonl(sdir(stage) / "sources.jsonl"):
        system, user = P.gen_messages(r["kind"], r["source"], r["text"], r.get("task"),
                                      r.get("type_hint"))
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
                    "source_pid": pid, "kind": kind, "group": src.get("group", "train"),
                    "type_hint": src.get("type_hint")}
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
    if args.stage == "trial":
        a_rows = [c for c in cands if c["bucket"] == "A"]
        for c in rng.sample(a_rows, len(a_rows) // 2):       # trial: pair half of A
            cands.append({**c, "bucket": "B1", "cid": c["cid"] + ":B1", "pair": c["cid"]})
    else:
        probe_a = [c for c in cands if c["bucket"] == "A" and c["group"] == "probe"]
        cands = [c for c in cands if c["group"] != "probe"]
        pairs = [{**c, "bucket": "B1", "cid": c["cid"] + ":B1", "pair": c["cid"]} for c in probe_a]
        b1_targets = {"train": SCALE_B1_CANDIDATES} if args.stage == "scale" else PILOT_B1_CANDIDATES
        for grp, n in b1_targets.items():
            a_rows = [c for c in cands if c["bucket"] == "A" and c["group"] == grp]
            pairs += [{**c, "bucket": "B1", "cid": c["cid"] + ":B1", "pair": c["cid"]}
                      for c in rng.sample(a_rows, min(n, len(a_rows)))]
        cands += pairs
    n = write_jsonl(st / "candidates.jsonl", cands)
    (st / "funnel.json").write_text(json.dumps({k: dict(v) for k, v in funnel.items()}, indent=2))
    print(f"{n} candidates: {dict(Counter((c['group'], c['bucket']) for c in cands))}")
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
    with modal.enable_output(), A.app.run(detach=args.stage == "scale"):
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
    if checker in ("ds_v3", "gemini_v3"):                 # near-miss check, B2 only (7.8)
        for r in rows:
            if r["bucket"] == "B2":
                system, user = P.v3_messages(r["passages"], r["question"])
                mt = GEMINI_CHECK_MAX_TOKENS if checker == "gemini_v3" else T.max_tokens_check
                jobs.append(Job(f"{r['cid']}|{checker}", system, user, mt))
        return jobs
    for r in rows:
        if r["bucket"] in ("A", "B1", "B2"):
            system, user = P.v1_messages(r["passages"], r["question"])
        elif r.get("task") in ("summarize", "yes_no_check") and checker != "ds_on":
            resp = None if r["task"] == "yes_no_check" else r["response"]
            system, user = P.v2_messages(r["k"], r["passages"][r["k"] - 1], r["task_text"], resp)
        else:
            continue                          # extract_list is model-free; V2 runs without thinking
        think = checker == "ds_on"
        mt = (GEMINI_CHECK_MAX_TOKENS if checker == "gemini"
              else T.max_tokens_check_thinking if think else T.max_tokens_check)
        jobs.append(Job(f"{r['cid']}|{checker}", system, user, mt, think))
    return jobs


def cmd_check(args) -> None:
    need_confirm(args, "Checking (API)")
    st, lg = sdir(args.stage), ledger(args.stage)
    rows = jsonl(st / "assembled.jsonl")
    if args.checker == "ds_v3":
        if args.stage != "scale":
            raise SystemExit("the V3 near-miss check is a 7.8 (scale) step")
        rows = [r for r in rows if r["bucket"] == "B2"] + pilot_train_b2()
    if args.stage != "trial":
        if args.checker == "ds_on":
            raise SystemExit("thinking checks are off after the trial (D-19)")
        if args.checker == "gemini":       # Gemini co-labels only what defines reported numbers
            rows = [r for r in rows if r.get("group") in ("dev", "test", "probe")]
    jobs = check_jobs(rows, args.checker)
    out = st / f"checks-{args.checker}.jsonl"
    if args.checker == "gemini":
        gb, stats = GeminiBatch(lg), Counter()
        for i in range(0, len(jobs), GEMINI_CHUNK):
            part = gb.run(jobs[i:i + GEMINI_CHUNK], out, f"raft-{args.stage}-check-{i // GEMINI_CHUNK}")
            stats.update({k: v for k, v in part.items() if isinstance(v, (int, float))})
            print(f"  chunk {i // GEMINI_CHUNK}: {part}; stage spend ${lg.spent:.4f}", flush=True)
        stats = dict(stats)
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


# -------------------------------------------------------------------------- select
def _verdicts(st: Path) -> dict[str, dict]:
    return {c: {r["key"].split("|")[0]: (r["json"] if r["ok"] else None)
                for r in jsonl(st / f"checks-{c}.jsonl")} for c in ("ds_off", "gemini", "ds_v3")}


def v3_ok(v: dict | None) -> bool:
    """V3 near-miss check: keep a B2 only if a careful reader could NOT give a plausible answer."""
    return isinstance(v, dict) and v.get("plausible_answer") is False


def pilot_train_b2() -> list[dict]:
    """The pilot's train B2 rows (selected, never reviewed: E-25) in candidate shape, for V3."""
    return [{"cid": r["id"], "bucket": "B2", "passages": r["passages"], "question": r["question"]}
            for r in jsonl(PILOT_DATASETS / "train.jsonl") if r["bucket"] == "B2"]


def accepted(row: dict, V: dict) -> bool:
    """Train: DeepSeek (no thinking). Dev/test/probe: DeepSeek AND Gemini (§5.4, D-19)."""
    if row.get("task") == "extract_list":
        return True                                   # model-free (§5.2 check 9)
    ok = bool(accept(row, V["ds_off"].get(row["cid"])))
    if row["bucket"] == "B2" and V.get("ds_v3"):          # 7.8: near misses also pass V3
        ok = ok and v3_ok(V["ds_v3"].get(row["cid"]))
    if row["group"] in ("dev", "test", "probe"):
        ok = ok and bool(accept(row, V["gemini"].get(row["cid"])))
    return ok


def make_b3(questions: list[dict], n: int, others: dict[str, list[dict]], tok, rng) -> list[dict]:
    """B3: a question with no passages (even i) or 4 passages from other sources (odd i)."""
    out = []
    for i, q in enumerate(questions[:n]):
        pool = [r for src, rs in others.items() if src != q["source"] for r in rs]
        if i % 2 == 0 or len(pool) < 4:
            ps, pids = [], []
        else:
            pick = rng.sample(pool, 4)
            ps, pids = [r["text"] for r in pick], [r["pid"] for r in pick]
        row = {k: q[k] for k in ("writer", "source", "split", "group", "question", "source_pid")}
        row.update({"bucket": "B3", "cid": q["cid"] + ":B3", "passages": ps, "pids": pids, "k": None,
                    "label": "unanswerable", "entity_present": False, "n_same_doc": 0})
        prompt, target = R.render_example(row)
        row["n_tokens"] = len(tok.encode(prompt + target, add_special_tokens=False))
        if row["n_tokens"] <= config.SEQ_LEN:
            out.append(row)
    return out


def cmd_select(args) -> None:
    """Accept, fill the exact quotas, build B3, write datasets + spot-check + report ($0)."""
    from transformers import AutoTokenizer

    st = sdir(args.stage)
    tok = AutoTokenizer.from_pretrained(".smoke/final/base")
    rows, V = jsonl(st / "assembled.jsonl"), _verdicts(st)
    sources = jsonl(st / "sources.jsonl")
    rng = random.Random(config.RAFT.seed)
    acc = [r for r in rows if accepted(r, V)]
    keep = defaultdict(Counter)
    for r in rows:
        keep[(r["group"], r["bucket"])]["assembled"] += 1
    for r in acc:
        keep[(r["group"], r["bucket"])]["accepted"] += 1
    datasets, shortfall, swapped = {}, {}, []
    excl = json.loads((st / "exclusions.json").read_text()) if (st / "exclusions.json").exists() else {}
    for grp, quota in PILOT_QUOTAS.items():
        by = defaultdict(list)
        for r in acc:
            if r["group"] == grp:
                by[r["bucket"]].append(r)
        for b in by:
            rng.shuffle(by[b])
        if grp == "probe":
            chosen = by["B1"][:quota["B1"]]
        else:
            b1_by_pair = {r["pair"]: r for r in by["B1"]}
            paired = [r for r in by["A"] if r["cid"] in b1_by_pair]
            nb1 = min(quota["B1"], len(paired))
            if grp == "train":                       # pairs not required for training
                a_sel = by["A"][:quota["A"]]
                b1_sel = by["B1"][:quota["B1"]]
            else:                                    # dev/test: keep each B1 with its A twin
                rest = [r for r in by["A"] if r["cid"] not in b1_by_pair] + paired[nb1:]
                a_sel = paired[:nb1] + rest[:quota["A"] - nb1]
                b1_sel = [b1_by_pair[a["cid"]] for a in paired[:nb1]]
            used = {r["cid"] for r in a_sel} | {r["pair"] for r in b1_sel}
            spare = [r for r in by["A"] if r["cid"] not in used] + a_sel
            split = "train" if grp == "train" else "val"
            others = defaultdict(list)
            for r in sources:
                if r["split"] == split and r.get("group") != "probe":
                    others[r["source"]].append(r)
            b3 = make_b3(spare, quota["B3"], others, tok, rng)
            chosen = a_sel + b1_sel + by["B2"][:quota["B2"]] + by["C"][:quota["C"]] + b3
        # Reviewed exclusions (exclusions.json: id -> reason): swap each for the next unused
        # accepted row of the same bucket, so every other selection stays identical.
        in_use = {r["cid"] for r in chosen}
        for i, r in enumerate(chosen):
            if r["cid"] in excl:
                if r["bucket"] == "B1" and grp in ("dev", "test"):
                    raise SystemExit(f"{r['cid']}: excluding a paired B1 needs its A twin handled too")
                spare = next((x for x in by[r["bucket"]] if x["cid"] not in in_use and x["cid"] not in excl), None)
                chosen[i] = spare                    # None = no reviewed spare left: an honest shortfall
                if spare is not None:
                    in_use.add(spare["cid"])
                swapped.append({"group": grp, "out": r["cid"], "in": spare and spare["cid"], "reason": excl[r["cid"]]})
        chosen = [r for r in chosen if r is not None]
        got = Counter(r["bucket"] for r in chosen)
        shortfall[grp] = {b: q - got[b] for b, q in quota.items() if got[b] < q}
        rng.shuffle(chosen)
        datasets[grp] = chosen
    out_dir = st / "datasets"
    out_dir.mkdir(exist_ok=True)
    for grp, chosen in datasets.items():
        recs = []
        for r in chosen:
            prompt, target = R.render_example(r)
            recs.append({"id": r["cid"], "group": grp, "bucket": r["bucket"], "source": r["source"],
                         "prompt": prompt, "target": target, "k": r.get("k"),
                         "question": r.get("question"), "task": r.get("task"),
                         "task_text": r.get("task_text"), "answer": r.get("answer") or r.get("response"),
                         "quote": r.get("quote"), "passages": r.get("passages"), "pids": r.get("pids"),
                         "pair": r.get("pair"), "type": r.get("type"), "label": r.get("label"),
                         "entity_present": r.get("entity_present"), "n_tokens": r.get("n_tokens")})
        write_jsonl(out_dir / f"{grp}.jsonl", recs)
    # spot-check file for the user: 50 dev/test items, stratified by bucket
    picks = []
    for b, n in (("A", 20), ("B1", 10), ("B2", 8), ("B3", 4), ("C", 8)):
        pool = [r for g in ("dev", "test") for r in datasets[g] if r["bucket"] == b]
        picks += rng.sample(pool, min(n, len(pool)))
    md = ["# Phase 7 pilot: 50 dev/test items to spot-check", "",
          "For each item: is the target right given the passages? Note any id you disagree with.", ""]
    for i, r in enumerate(picks, 1):
        prompt, target = R.render_example(r)
        md += [f"## {i}. `{r['cid']}` ({r['bucket']}, {r['source']})", "",
               f"**{'Task' if r['bucket'] == 'C' else 'Question'}:** {r.get('task_text') or r['question']}", ""]
        for j, ptxt in enumerate(r["passages"] or [], 1):
            mark = " ← cited" if r.get("k") == j and r["bucket"] in ("A", "C") else ""
            body = ptxt if (r.get("k") == j or r["bucket"] in ("B1", "B2")) else ptxt[:240] + " …"
            md += [f"> **Passage {j}{mark}:** {body}", ">"]
        if not r["passages"]:
            md += ["> *(no passages)*", ">"]
        md += ["", f"**Target:** `Source:{target.replace('<|eos|>', '')}`", ""]
    if not (st / "spot_check.md").exists():       # keep the file the user already reviewed
        (st / "spot_check.md").write_text("\n".join(md))
    # report
    rep = {"keep": {f"{g}:{b}": dict(v) for (g, b), v in sorted(keep.items())},
           "composition": {g: dict(Counter(r["bucket"] for r in d)) for g, d in datasets.items()},
           "shortfall": shortfall,
           "source_mix": {g: dict(Counter(r["source"] for r in d)) for g, d in datasets.items()},
           "oracle_position": {g: dict(sorted(Counter(r["k"] for r in d if r["bucket"] in ("A", "C")).items()))
                               for g, d in datasets.items() if g != "probe"},
           "entity_present_B1": {g: round(sum(r["entity_present"] for r in d if r["bucket"] == "B1")
                                          / max(1, sum(r["bucket"] == "B1" for r in d)), 3) for g, d in datasets.items()},
           "a_types": {g: dict(Counter(r.get("type") for r in d if r["bucket"] == "A")) for g, d in datasets.items()},
           "tokens_max": {g: max((r["n_tokens"] for r in d), default=0) for g, d in datasets.items()},
           "test_pairs": sum(1 for r in datasets["test"] if r["bucket"] == "B1"),
           "exclusions_applied": swapped,
           "funnel": json.loads((st / "funnel.json").read_text())}
    lg = json.loads((st / "ledger.json").read_text())
    mc = json.loads((st / "modal_cost.json").read_text()) if (st / "modal_cost.json").exists() else {}
    rep["spend"] = {"api_usd": lg["spent_usd"], "modal_usd": round(sum(mc.values()), 4),
                    "total_usd": round(lg["spent_usd"] + sum(mc.values()), 4), "cap_usd": lg["cap_usd"]}
    (st / "select_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: rep[k] for k in ("composition", "shortfall", "source_mix", "oracle_position",
                                          "entity_present_B1", "a_types", "tokens_max", "test_pairs",
                                          "spend")}, indent=1))
    print(f"datasets -> {out_dir}; spot-check -> {st / 'spot_check.md'}")


def dataset_record(r: dict, grp: str) -> dict:
    prompt, target = R.render_example(r)
    return {"id": r["cid"], "group": grp, "bucket": r["bucket"], "source": r["source"],
            "prompt": prompt, "target": target, "k": r.get("k"),
            "question": r.get("question"), "task": r.get("task"),
            "task_text": r.get("task_text"), "answer": r.get("answer") or r.get("response"),
            "quote": r.get("quote"), "passages": r.get("passages"), "pids": r.get("pids"),
            "pair": r.get("pair"), "type": r.get("type"), "label": r.get("label"),
            "entity_present": r.get("entity_present"), "n_tokens": r.get("n_tokens")}


def cmd_select_scale(args) -> None:
    """7.8: scale quotas from accepted scale rows + the pilot train rows (pilot B2 must pass V3);
    dev/test/probe are copied unchanged from the frozen pilot sets ($0)."""
    from transformers import AutoTokenizer

    st = sdir("scale")
    tok = AutoTokenizer.from_pretrained(".smoke/final/base")
    rows, V = jsonl(st / "assembled.jsonl"), _verdicts(st)
    rng = random.Random(config.RAFT.seed)
    dk = lambda pid: pid.rsplit("/", 1)[0]
    probe_docs = {dk(p) for r in jsonl(PILOT_DATASETS / "probe.jsonl") for p in (r.get("pids") or [])}

    def cites_probe(r: dict) -> bool:     # E-34: the loss covers the target, which quotes the cited passage
        return (r["bucket"] in ("A", "C") and bool(r.get("k")) and bool(r.get("pids"))
                and dk(r["pids"][int(r["k"]) - 1]) in probe_docs)

    acc = [r for r in rows if accepted(r, V) and not cites_probe(r)]
    by = defaultdict(list)
    for r in acc:
        by[r["bucket"]].append(r)
    for b in by:
        rng.shuffle(by[b])
    pilot = jsonl(PILOT_DATASETS / "train.jsonl")
    v3_drop = [r["id"] for r in pilot if r["bucket"] == "B2" and not v3_ok(V["ds_v3"].get(r["id"]))]
    probe_drop = [r["id"] for r in pilot if cites_probe(r)]
    gone = set(v3_drop) | set(probe_drop)
    pilot_kept = [r for r in pilot if r["id"] not in gone]
    lost = Counter(r["bucket"] for r in pilot if r["id"] in set(probe_drop))
    q = {b: n + lost[b] for b, n in SCALE_QUOTAS["train"].items()}     # replace pilot rows dropped for the probe
    others = defaultdict(list)
    for r in jsonl(st / "sources.jsonl"):
        others[r["source"]].append(r)
    a_sel = by["A"][:q["A"]]
    b3 = make_b3(by["A"][q["A"]:] + a_sel, q["B3"], others, tok, rng)
    chosen = a_sel + by["B1"][:q["B1"]] + by["B2"][:q["B2"]] + by["C"][:q["C"]] + b3
    got = Counter(r["bucket"] for r in chosen)
    train = pilot_kept + [dataset_record(r, "train") for r in chosen]
    rng.shuffle(train)
    devtest_docs = {dk(pid) for g in ("dev", "test") for r in jsonl(PILOT_DATASETS / f"{g}.jsonl")
                    for pid in (r.get("pids") or [])}
    train_docs = {dk(pid) for r in train for pid in (r.get("pids") or [])}
    out_dir = st / "datasets"
    out_dir.mkdir(exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train)
    for g in ("dev", "test", "probe"):
        write_jsonl(out_dir / f"{g}.jsonl", jsonl(PILOT_DATASETS / f"{g}.jsonl"))
    lg = json.loads((st / "ledger.json").read_text())
    mc = json.loads((st / "modal_cost.json").read_text()) if (st / "modal_cost.json").exists() else {}
    rep = {"accepted": {b: len(v) for b, v in by.items()},
           "assembled": dict(Counter(r["bucket"] for r in rows)),
           "scale_selected": dict(got), "shortfall": {b: n - got[b] for b, n in q.items() if got[b] < n},
           "pilot_b2_v3_dropped": len(v3_drop), "pilot_probe_cited_dropped": len(probe_drop),
           "pilot_rows_kept": len(pilot_kept), "quotas_used": q,
           "scale_probe_cited_excluded": sum(1 for r in rows if accepted(r, V) and cites_probe(r)),
           "train_total": len(train), "train_buckets": dict(Counter(r["bucket"] for r in train)),
           "train_sources": dict(Counter(r["source"] for r in train)),
           "a_types": dict(Counter(r.get("type") for r in train if r["bucket"] == "A")),
           "tokens_max": max(r["n_tokens"] for r in train),
           "doc_overlap_with_devtest": len(train_docs & devtest_docs),
           "probe_docs_cited_in_train": sum(cites_probe(r) for r in train),
           "probe_docs_as_context_only": len(train_docs & probe_docs),
           "v3_on_scale_b2": {"checked": sum(1 for r in rows if r["bucket"] == "B2" and r["cid"] in V["ds_v3"]),
                              "pass": sum(1 for r in rows if r["bucket"] == "B2" and v3_ok(V["ds_v3"].get(r["cid"])))},
           "spend": {"api_usd": lg["spent_usd"], "modal_usd": round(sum(mc.values()), 4), "cap_usd": lg["cap_usd"]},
           "funnel": json.loads((st / "funnel.json").read_text())}
    (st / "select_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "funnel"}, indent=1))
    if rep["doc_overlap_with_devtest"] or rep["probe_docs_cited_in_train"]:
        raise SystemExit("train shares documents with dev/test, or quotes a probe document: stop")


def audit_rows() -> list[dict]:
    """5% audit sample: AUDIT_N random scale rows selected into the 20K train set (not the pilot's)."""
    st = sdir("scale")
    ids = {r["id"] for r in jsonl(st / "datasets" / "train.jsonl")}
    rows = [r for r in jsonl(st / "assembled.jsonl") if r["cid"] in ids]
    return random.Random(config.RAFT.seed).sample(rows, min(AUDIT_N, len(rows)))


def cmd_audit(args) -> None:
    """7.8: Gemini re-checks a 5% sample of the scale train rows (V1/V2 and V3 for B2) and reports
    per-bucket disagreement with the DeepSeek-accepted label. > 15% in any bucket: stop."""
    st = sdir("scale")
    rows = audit_rows()
    out = st / "audit.jsonl"
    if args.confirm:
        lg = ledger("scale")
        jobs = [Job(j.key.replace("|gemini", "|audit"), j.system, j.user, j.max_tokens)
                for j in check_jobs(rows, "gemini")] + check_jobs(rows, "gemini_v3")
        GeminiBatch(lg).run(jobs, out, "raft-scale-audit")
        print(f"audit: {len(jobs)} Gemini calls; stage spend ${lg.spent:.4f} of ${lg.cap:.2f}")
    v = {r["key"]: (r["json"] if r["ok"] else None) for r in jsonl(out)}
    dis, n = Counter(), Counter()
    for r in rows:
        if r.get("task") == "extract_list":
            continue
        ok = accept(r, v.get(f"{r['cid']}|audit"))
        if r["bucket"] == "B2":
            ok = bool(ok) and v3_ok(v.get(f"{r['cid']}|gemini_v3"))
        n[r["bucket"]] += 1
        dis[r["bucket"]] += not ok
    rep = {b: {"n": n[b], "disagree": dis[b], "rate": round(dis[b] / max(1, n[b]), 3)} for b in sorted(n)}
    bad = [b for b, x in rep.items() if x["rate"] > AUDIT_MAX_DISAGREE]
    (st / "audit_report.json").write_text(json.dumps({"by_bucket": rep, "over_limit": bad}, indent=2))
    print(json.dumps(rep, indent=1))
    print("audit: OK" if not bad else f"audit: buckets over {AUDIT_MAX_DISAGREE:.0%}: {bad} -> stop and report")


CEILING_N = 200


def ceiling_rows() -> list[dict]:
    """200 random test items for the teacher ceiling (§8.3), every bucket, fixed seed."""
    return random.Random(config.RAFT.seed).sample(jsonl(LOCAL / "scale" / "datasets" / "test.jsonl"), CEILING_N)


def ceiling_text(j: dict | None) -> tuple[str, str]:
    """DeepSeek's JSON answer rendered in the student's output format -> (choice, text after `Source:`)."""
    src = (j or {}).get("source")
    if j is None or src in (None, "none", 0, "0"):
        return "none", " none\nAnswer: " + config.RAFT.refusal
    k = int(str(src).strip().split()[-1]) if str(src).strip().split()[-1].isdigit() else 0
    quote = f'\nQuote: "{j["quote"]}"' if j.get("quote") else ""
    return "pass", f" Passage {k}{quote}\nAnswer: {j.get('answer') or ''}"


def cmd_ceiling(args) -> None:
    """7.8 teacher ceiling (§8.3): DeepSeek answers 200 test items in the student's format; the
    result is written as test-set generations named `ceiling` for J1 + score. Inflated by
    construction (test items were kept only where DeepSeek's check agreed): an upper bound."""
    need_confirm(args, "Teacher ceiling (DeepSeek)")
    st, lg = sdir("scale"), ledger("scale")
    rows = ceiling_rows()
    jobs = []
    for r in rows:
        q = r["question"] if r["bucket"] != "C" else r["task_text"]
        user = (P.passages_block(r["passages"]) if r["passages"] else "(no passages)") + f"\nQuestion: {q}"
        jobs.append(Job(f"{r['id']}|ceiling", P.CEIL_SYSTEM, user, T.max_tokens_check))
    out = st / "ceiling-raw.jsonl"
    stats = DeepSeek(lg, allow_peak=args.allow_peak, workers=args.workers).run(jobs, out)
    got = {x["key"].split("|")[0]: (x["json"] if x["ok"] else None) for x in jsonl(out)}
    gens = []
    for r in rows:
        choice, text = ceiling_text(got.get(r["id"]))
        gens.append({"id": r["id"], "bucket": r["bucket"], "s": None, "choice": choice, "text": text})
    write_jsonl(st / "eval" / "ceiling_copy2.jsonl", gens)     # picked up by judge/score as a scored set
    print(f"ceiling: {stats}; stage spend ${lg.spent:.4f} of ${lg.cap:.2f}")


def cmd_upload(args) -> None:
    """Upload the selected datasets to the Volume (raft/datasets/{stage}/)."""
    modal, _ = _modal()
    vol = modal.Volume.from_name(config.VOLUME_NAME)
    st = sdir(args.stage)
    with vol.batch_upload(force=True) as up:
        for f in sorted((st / "datasets").glob("*.jsonl")):
            up.put_file(str(f), f"raft/datasets/{args.stage}/{f.name}")
            print(f"  {f.name} -> raft/datasets/{args.stage}/{f.name}")


# --------------------------------------------------------------------------- 7.7 J1
EVAL_DIR = LOCAL / "pilot" / "eval"
MODES = ("plain", "copy", "copy2")        # copy2: the 7.7 format-fixed CopyConstraint (optional file)


def _parsed(gen: dict) -> dict:
    import eval_raft as E
    return E.parse_output(gen["text"])


def judge_jobs(rows: dict[str, dict], gens: dict[str, list[dict]]) -> list[Job]:
    """One J1 call per answered, parseable A or C output (unanswerable items that were answered are
    hallucinations whatever the answer, so they are not judged). Key: '{id}|{mode}'."""
    jobs = []
    for mode, gl in gens.items():
        for g in gl:
            r, p = rows[g["id"]], _parsed(g)
            if r["bucket"] not in ("A", "C") or g["choice"] != "pass" or not p["valid"] or p["refusal"]:
                continue
            question = r["question"] if r["bucket"] == "A" else r["task_text"]
            system, user = P.j1_messages(r["passages"] or [], question, r["answer"], p["answer"])
            jobs.append(Job(f"{g['id']}|{mode}", system, user, GEMINI_CHECK_MAX_TOKENS))
    return jobs


def score_eval(rows: dict[str, dict], gens: list[dict], verdicts: dict[str, dict], mode: str) -> dict:
    """§8.1 headline metrics for one decoding mode from generations + J1 verdicts. A J1 verdict
    missing for a judged item counts against the model (not supported, not correct)."""
    import eval_raft as E
    n = len(gens)
    c = Counter()
    correct_A, refused = {}, {}
    proxy_vs_j1 = Counter()
    for g in gens:
        r, p = rows[g["id"]], _parsed(g)
        b, answered = r["bucket"], g["choice"] == "pass"
        c["valid"] += p["valid"]
        c[f"n_{b}"] += 1
        refused[g["id"]] = not answered
        if b in ("B1", "B2", "B3"):
            c["n_B"] += 1
            c["refused_B"] += not answered
            c[f"refused_{b}"] += not answered
            c["halluc"] += answered
            continue
        if not answered:
            c[f"refused_{b}"] += 1
            continue
        v = verdicts.get(f"{g['id']}|{mode}") or {}
        ok_parse = p["valid"] and not p["refusal"]
        k_ok = ok_parse and p["k"] and 1 <= p["k"] <= len(r["passages"] or [])
        has_quote = ok_parse and p["quote"] is not None      # an empty quote counts as a non-verbatim quote
        verbatim = bool(has_quote and p["quote"] and k_ok and p["quote"] in r["passages"][p["k"] - 1])
        if has_quote:
            c["quoted"] += 1
            c["verbatim"] += verbatim
        supported = bool(v.get("supported")) if ok_parse else False
        if not ok_parse or not supported or (has_quote and not verbatim):
            c["halluc"] += 1
        if b == "A":
            c["answered_A"] += 1
            cites = bool(k_ok and p["k"] == int(r["k"]))
            c["cites_oracle"] += cites
            good = bool(ok_parse and v.get("correct") and cites)
            c["grounded"] += good
            correct_A[g["id"]] = good
            proxy = bool(cites and verbatim and E.token_f1(p["answer"] or "", r["answer"] or "") >= 0.5)
            proxy_vs_j1[f"proxy_{int(proxy)}_j1_{int(good)}"] += 1
    in_set = {g["id"] for g in gens}
    pairs = [r for r in rows.values() if r["bucket"] == "B1" and r["id"] in in_set and r.get("pair") in in_set]
    paired = [correct_A.get(r["pair"], False) and refused.get(r["id"], False) for r in pairs]
    nA, nB = max(1, c["n_A"]), max(1, c["n_B"])
    return {"mode": mode, "n": n,
            "hallucination": c["halluc"] / n,
            "refusal_recall": c["refused_B"] / nB,
            **{f"refusal_recall_{b}": c[f"refused_{b}"] / max(1, c[f"n_{b}"]) for b in ("B1", "B2", "B3")},
            "over_refusal_A": c["refused_A"] / nA,
            "grounded_accuracy": c["grounded"] / nA,
            "citation_accuracy": c["cites_oracle"] / max(1, c["answered_A"]),
            "quote_fidelity": c["verbatim"] / max(1, c["quoted"]),
            "format_validity": c["valid"] / n,
            "paired_consistency": sum(paired) / max(1, len(paired)), "n_pairs": len(pairs),
            "proxy_vs_j1_on_answered_A": dict(proxy_vs_j1)}


def _eval_inputs(stage: str = "pilot-eval") -> tuple[dict[str, dict], dict[str, list[dict]]]:
    if stage in ("gemma", "slm500", "gemma3"):   # Phases 9-11: every copy2 output and every test output, keyed '{step}/{file}'
        rows = {r["id"]: r for g in ("dev", "test") for r in jsonl(LOCAL / "scale" / "datasets" / f"{g}.jsonl")}
        root, dirs = {"gemma": (GEMMA_LOCAL, ("baselines", "pilot", "full")), "slm500": (SLM500_LOCAL, ("pilot", "full", "control")),
                      "gemma3": (GEMMA3_LOCAL, ("pilot", "full", "lora_pilot", "lora_full"))}[stage]
        gens = {f"{f.parent.name}/{f.stem}": jsonl(f) for d in dirs
                for f in sorted((root / d).glob("*.jsonl"))
                if f.stem.endswith("_copy2") or f.stem.startswith("test_")}
        if not gens:
            raise SystemExit(f"no generations under {root}")
        return rows, gens
    if stage == "scale":                    # 7.8: dev copy2 per curve point + test plain/copy2 of the best run
        d = LOCAL / "scale" / "eval"
        rows = {r["id"]: r for g in ("dev", "test") for r in jsonl(LOCAL / "scale" / "datasets" / f"{g}.jsonl")}
        gens = {f.stem: jsonl(f) for f in sorted(d.glob("*.jsonl"))
                if f.stem.startswith("test_") or f.stem.endswith(("_copy2", "_copy3"))}
        if not gens:
            raise SystemExit(f"no generations in {d}: run raft_gpu_app.py::full_check first")
        return rows, gens
    rows = {r["id"]: r for r in jsonl(LOCAL / "pilot" / "datasets" / "dev.jsonl")}
    gens = {m: jsonl(EVAL_DIR / f"dev_gen_{m}.jsonl") for m in MODES if (EVAL_DIR / f"dev_gen_{m}.jsonl").exists()}
    if not (gens.get("plain") and gens.get("copy")):
        raise SystemExit(f"missing generations in {EVAL_DIR}: run raft_gpu_app.py::pilot_eval_check first")
    return rows, gens


def _reused(gens: dict[str, list[dict]]) -> dict[str, str]:
    """Outputs identical to an already-judged decoding are not re-judged: 'copy2' vs 'copy' (7.7),
    '*_copy3' vs '*_copy2' (7.8b). Returns new key -> judged key."""
    out = {}
    for new, base in [("copy2", "copy")] + [(k, k[:-1] + "2") for k in gens if k.endswith("_copy3")]:
        old = {g["id"]: g["text"] for g in gens.get(base, [])}
        out.update({f"{g['id']}|{new}": f"{g['id']}|{base}" for g in gens.get(new, []) if old.get(g["id"]) == g["text"]})
    return out


def cmd_judge(args) -> None:
    need_confirm(args, "J1 judging (Gemini)")
    st, lg = sdir(args.stage), ledger(args.stage)
    rows, gens = _eval_inputs(args.stage)
    jobs = [j for j in judge_jobs(rows, gens) if j.key not in _reused(gens)]
    if args.stage == "gemma":
        jobs += closed_book_jobs(rows)
    if args.stage == "gemma3":
        jobs += closed_book_jobs(rows, GEMMA3_LOCAL / "probe" / "closed_book_base.jsonl")
    out = st / "j1.jsonl"
    done = {r["key"] for r in jsonl(out) if r["ok"]}       # Phase 9 judges in rounds (pilot, then full)
    jobs = [j for j in jobs if j.key not in done]
    gb = GeminiBatch(lg)
    for i in range(0, len(jobs), GEMINI_CHUNK):
        part = gb.run(jobs[i:i + GEMINI_CHUNK], out, f"raft-{args.stage}-j1-{i // GEMINI_CHUNK}")
        print(f"  chunk {i // GEMINI_CHUNK}: {part}; stage spend ${lg.spent:.4f} of ${lg.cap:.2f}", flush=True)
    failed = [r["key"] for r in jsonl(out) if not r["ok"]]
    if failed:                              # retry failures once (§13 7.7 validation)
        retry = [j for j in jobs if j.key in set(failed)]
        kept = [r for r in jsonl(out) if r["ok"]]
        write_jsonl(out, kept)
        part = gb.run(retry, out, f"raft-{args.stage}-j1-retry")
        print(f"  retry {len(retry)}: {part}")
    print(f"J1: {len(jobs)} jobs; {sum(r['ok'] for r in jsonl(out))} verdicts; spend ${lg.spent:.4f}")


CLOSED_BOOK = GEMMA_LOCAL / "baselines" / "closed_book_test_A.jsonl"


def closed_book_jobs(rows: dict[str, dict], path: Path | None = None) -> list[Job]:
    """Phase 9 closed-book probe: gemma-2-2b-it answered the test A questions without passages; J1
    grades `correct` against the reference (with the passages shown to the judge, as always)."""
    jobs = []
    for g in jsonl(path or CLOSED_BOOK):
        r = rows[g["id"]]
        system, user = P.j1_messages(r["passages"] or [], r["question"], r["answer"], g["answer"])
        jobs.append(Job(f"{g['id']}|closed_book", system, user, GEMINI_CHECK_MAX_TOKENS))
    return jobs


def cmd_score(args) -> None:
    rows, gens = _eval_inputs(args.stage)
    verdicts = {r["key"]: r["json"] for r in jsonl(sdir(args.stage) / "j1.jsonl") if r["ok"]}
    for k2, k1 in _reused(gens).items():                  # identical copy2 text: same verdict
        if k1 in verdicts:
            verdicts[k2] = verdicts[k1]
    judged = {j.key for j in judge_jobs(rows, gens)}
    print(f"J1 verdicts for {len(judged & set(verdicts))} of {len(judged)} judged outputs")
    rep = {m: score_eval(rows, gens[m], verdicts, m) for m in gens}
    for m in gens:                          # missing verdicts count as wrong: record coverage so partial sets are never read
        need = [j.key for j in judge_jobs(rows, {m: gens[m]})]
        rep[m]["j1_coverage"] = sum(k in verdicts for k in need) / max(1, len(need))
    cb_path = {"gemma": CLOSED_BOOK, "gemma3": GEMMA3_LOCAL / "probe" / "closed_book_base.jsonl"}.get(args.stage)
    if cb_path is not None and cb_path.exists():     # grounded accuracy without the closed-book-known items
        known = {k.split("|")[0] for k, v in verdicts.items() if k.endswith("|closed_book") and v.get("correct")}
        n_cb = sum(k.endswith("|closed_book") for k in verdicts)
        yn = {i for i in {k.split("|")[0] for k in verdicts if k.endswith("|closed_book")}
              if rows[i]["answer"].strip().rstrip(".").lower() in ("yes", "no")}
        rep["closed_book"] = {"judged": n_cb, "correct": len(known), "rate": len(known) / max(1, n_cb),
                              "yes_no_items": len(yn), "correct_excluding_yes_no": len(known - yn),
                              "rate_excluding_yes_no": len(known - yn) / max(1, n_cb - len(yn))}
        for m in [m for m in gens if m.split("/")[-1].startswith("test_")]:
            unknown = [g for g in gens[m] if not (rows[g["id"]]["bucket"] == "A" and g["id"] in known)]
            rep[m]["grounded_accuracy_closed_book_unknown"] = score_eval(rows, unknown, verdicts, m)["grounded_accuracy"]
        print(f"closed book: {len(known)} of {n_cb} test A questions answered correctly without passages")
    (sdir(args.stage) / "headline.json").write_text(json.dumps(rep, indent=2))
    targets = {"hallucination": ("<=", 0.15), "refusal_recall": (">=", 0.80), "over_refusal_A": ("<=", 0.20),
               "grounded_accuracy": (">=", 0.60), "citation_accuracy": (">=", 0.80),
               "quote_fidelity": (">=", 0.90), "format_validity": (">=", 0.99)}
    for m in gens:
        if m not in rep:
            continue
        cov = rep[m]["j1_coverage"]
        print(f"[{m}]" + ("" if cov == 1 else f"  ** INCOMPLETE: J1 verdicts for {cov:.0%} of judged outputs; numbers below are NOT valid **"))
        for k, (op, t) in targets.items():
            v = rep[m][k]
            ok = v <= t if op == "<=" else v >= t
            print(f"  {k:20s} {v:.3f}  target {op} {t}  {'ok' if ok else 'MISS'}")
        print(f"  paired consistency   {rep[m]['paired_consistency']:.3f} ({rep[m]['n_pairs']} pairs); "
              f"proxy vs J1 {rep[m]['proxy_vs_j1_on_answered_A']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=("sources", "generate", "candidates", "assemble", "check", "report",
                                   "select", "upload", "judge", "score", "audit", "ceiling"))
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
     "assemble": cmd_assemble, "check": cmd_check, "report": cmd_report,
     "select": cmd_select_scale if args.stage == "scale" else cmd_select, "upload": cmd_upload,
     "judge": cmd_judge, "score": cmd_score, "audit": cmd_audit, "ceiling": cmd_ceiling}[args.cmd](args)
    print(f"[{args.cmd}] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
