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
GEMINI_CHECK_MAX_TOKENS = 384          # trial 7.3: check output+thoughts median 58, max 202 tokens
GEMINI_CHUNK = 800                     # requests per batch: each worst-case reservation settles first
CHECKERS = ("ds_off", "ds_on", "gemini")
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
        for grp, n in PILOT_B1_CANDIDATES.items():
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
        mt = (GEMINI_CHECK_MAX_TOKENS if checker == "gemini"
              else T.max_tokens_check_thinking if think else T.max_tokens_check)
        jobs.append(Job(f"{r['cid']}|{checker}", system, user, mt, think))
    return jobs


def cmd_check(args) -> None:
    need_confirm(args, "Checking (API)")
    st, lg = sdir(args.stage), ledger(args.stage)
    rows = jsonl(st / "assembled.jsonl")
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
                for r in jsonl(st / f"checks-{c}.jsonl")} for c in ("ds_off", "gemini")}


def accepted(row: dict, V: dict) -> bool:
    """Train: DeepSeek (no thinking). Dev/test/probe: DeepSeek AND Gemini (§5.4, D-19)."""
    if row.get("task") == "extract_list":
        return True                                   # model-free (§5.2 check 9)
    ok = bool(accept(row, V["ds_off"].get(row["cid"])))
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
    datasets, shortfall = {}, {}
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


def cmd_upload(args) -> None:
    """Upload the selected datasets to the Volume (raft/datasets/{stage}/)."""
    modal, _ = _modal()
    vol = modal.Volume.from_name(config.VOLUME_NAME)
    st = sdir(args.stage)
    with vol.batch_upload(force=True) as up:
        for f in sorted((st / "datasets").glob("*.jsonl")):
            up.put_file(str(f), f"raft/datasets/{args.stage}/{f.name}")
            print(f"  {f.name} -> raft/datasets/{args.stage}/{f.name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=("sources", "generate", "candidates", "assemble", "check", "report",
                                   "select", "upload"))
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
     "select": cmd_select, "upload": cmd_upload}[args.cmd](args)
    print(f"[{args.cmd}] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
