"""Phase 12 local steps: judging, headroom, labels, pairs (docs/13-rlaif.md). Paid steps need --confirm.

    python rlaif_pipeline.py fetch     --track slm125 --what headroom   # Volume -> .smoke/rlaif/<track>/
    python rlaif_pipeline.py headroom-judge --confirm                   # 12.1: J1 on distinct valid answered A samples
    python rlaif_pipeline.py headroom-score                             # 12.1: best-of-8 vs greedy; go/no-go per track
    python rlaif_pipeline.py label     --track slm125 --confirm         # 12.4: constitution scores (Gemini Flash batch)
    python rlaif_pipeline.py pairs     --track slm125                   # 12.4: pairs, length check, judge agreement; uploads
    python rlaif_pipeline.py dpo-judge --track slm125 --confirm         # 12.5: J1 on each DPO arm's dev copy2 outputs
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import config
import raft_decode as RD
import raft_pipeline as RP
import raft_prompts as P
import rlaif_data as D
from teacher import GeminiBatch, Job, Ledger, load_env

LOCAL = D.LOCAL
SCALE = Path(".smoke/raft/scale/datasets")
GREEDY_DEV = {"slm125": (".smoke/raft/scale/headline.json", "dev_n20000_copy2"),
              "gemma3": (".smoke/gemma3/headline.json", "full/dev_n20000-lr2e-05_copy2")}
CHUNK = int(os.environ.get("RAFT_GEMINI_CHUNK", 300))


def ledger(stage: str) -> Ledger:
    return Ledger(LOCAL / f"ledger_{stage}.json", config.RLAIF_STAGE_CAPS_USD[stage])


def rows_dev() -> dict[str, dict]:
    return {r["id"]: r for r in RP.jsonl(SCALE / "dev.jsonl")}


def rows_prompts() -> dict[str, dict]:
    return {r["id"]: r for r in RP.jsonl(LOCAL / "prompts.jsonl")}


def samples(track: str, name: str) -> dict[str, list[dict]]:
    return {r["id"]: r["samples"] for r in RP.jsonl(LOCAL / track / f"{name}.jsonl")}


def run_batch(jobs: list[Job], out: Path, stage: str, name: str) -> None:
    lg, gb = ledger(stage), None
    done = {r["key"] for r in RP.jsonl(out) if r["ok"]}
    jobs = [j for j in jobs if j.key not in done]
    print(f"{len(jobs)} jobs to judge ({len(done)} already)")
    gb = GeminiBatch(lg)
    for i in range(0, len(jobs), CHUNK):
        part = gb.run(jobs[i:i + CHUNK], out, f"rlaif-{name}-{i // CHUNK}")
        print(f"  chunk {i // CHUNK}: {part}; stage spend ${lg.spent:.4f} of ${lg.cap:.2f}", flush=True)


def need_confirm(a) -> None:
    if not a.confirm:
        raise SystemExit("This step costs money. Re-run with --confirm once approved.")


# --------------------------------------------------------------------------------- 12.1
def headroom_jobs(track: str) -> list[Job]:
    """J1 (the evaluation judge's rubric) on each distinct valid answered sample of a dev A row."""
    rows, ss = rows_dev(), samples(track, "headroom")
    jobs = []
    for rid, lst in ss.items():
        r = rows[rid]
        if r["bucket"] != "A":
            continue
        seen = set()
        for i, s in enumerate(lst):
            p = RD.parse_output(s["text"])
            if s["choice"] != "pass" or not p["valid"] or p["refusal"] or s["text"] in seen:
                continue
            seen.add(s["text"])
            system, user = P.j1_messages(r["passages"] or [], r["question"], r["answer"], p["answer"])
            jobs.append(Job(f"{rid}|{track}|{i}", system, user, RP.GEMINI_CHECK_MAX_TOKENS))
    return jobs


def cmd_headroom_judge(a) -> None:
    need_confirm(a)
    load_env()
    jobs = [j for t in ("slm125", "gemma3") for j in headroom_jobs(t)]
    run_batch(jobs, LOCAL / "j1_headroom.jsonl", "12.1-judge", "headroom")


def cmd_headroom_score(a) -> None:
    rows = rows_dev()
    V = {r["key"]: r["json"] for r in RP.jsonl(LOCAL / "j1_headroom.jsonl") if r["ok"]}
    rep = {}
    for track in ("slm125", "gemma3"):
        ss = samples(track, "headroom")
        hpath, hkey = GREEDY_DEV[track]
        greedy = json.loads(Path(hpath).read_text())[hkey]
        by_text = {}
        need = got = 0
        for rid, lst in ss.items():
            for i, s in enumerate(lst):
                k = f"{rid}|{track}|{i}"
                if k in V:
                    by_text[(rid, s["text"])] = V[k]
        best, a_n, b2_any, b2_n, valid, n = 0, 0, 0, 0, 0, 0
        for rid, lst in ss.items():
            r = rows[rid]
            for s in lst:
                n += 1
                valid += RD.parse_output(s["text"])["valid"]
            if r["bucket"] == "A":
                a_n += 1
                ok = False
                for s in lst:
                    p = RD.parse_output(s["text"])
                    if s["choice"] != "pass" or not p["valid"] or p["refusal"]:
                        continue
                    need += 1
                    v = by_text.get((rid, s["text"]))
                    got += v is not None
                    ok |= bool(v and v.get("correct") and p["k"] == int(r["k"]))
                best += ok
            if r["bucket"] == "B2":
                b2_n += 1
                b2_any += any(s["choice"] == "none" for s in lst)
        bo8, b2 = best / max(1, a_n), b2_any / max(1, b2_n)
        g_ground, g_b2 = greedy["grounded_accuracy"], greedy["refusal_recall_B2"]
        go = bo8 >= g_ground + config.RLAIF.go_grounded_gain or b2 >= g_b2 + config.RLAIF.go_b2_gain
        rep[track] = {"format_validity": valid / max(1, n), "samples": n, "j1_coverage": got / max(1, need),
                      "best_of_8_grounded": bo8, "greedy_grounded": g_ground, "b2_any_refusal": b2, "greedy_b2_recall": g_b2,
                      "go": go}
        print(f"{track}: format {valid / max(1, n):.4f} | best-of-8 grounded {bo8:.3f} vs greedy {g_ground:.3f} "
              f"({bo8 - g_ground:+.3f}) | B2 any-refusal {b2:.3f} vs greedy {g_b2:.3f} ({b2 - g_b2:+.3f}) | "
              f"J1 coverage {got / max(1, need):.3f} | GO={go}")
    (LOCAL / "headroom.json").write_text(json.dumps(rep, indent=2))


# --------------------------------------------------------------------------------- 12.4
def cmd_label(a) -> None:
    need_confirm(a)
    load_env()
    jobs = D.judge_jobs(rows_prompts(), samples(a.track, "samples_r1"), "r1")
    run_batch(jobs, LOCAL / a.track / "labels_r1.jsonl", "12.4-labels", f"label-{a.track}")


def cmd_pairs(a) -> None:
    from transformers import AutoTokenizer
    rows = rows_prompts()
    V = {r["key"]: r["json"] for r in RP.jsonl(LOCAL / a.track / "labels_r1.jsonl") if r["ok"]}
    ss = samples(a.track, "samples_r1")
    jobs = D.judge_jobs(rows, ss, "r1")
    cov = sum(j.key in V for j in jobs) / max(1, len(jobs))
    if cov < 0.99:                          # a missing label only drops that sample from pairing (no score bias)
        raise SystemExit(f"labels cover {cov:.1%} of the judged samples: finish `label` first")
    print(f"label coverage {cov:.2%}")
    tok = AutoTokenizer.from_pretrained(".smoke/final/base")
    prs, stats = D.pairs(rows, D.scored(rows, ss, V, "r1"), tok)
    agree = D.agreement(rows, V)
    stats["judge_label_agreement"] = agree
    stats["score_hist"] = dict(Counter(v.get("score") for v in V.values() if isinstance(v, dict)))
    D.write_jsonl(LOCAL / a.track / "pairs_r1.jsonl", prs)
    (LOCAL / a.track / "pairs_r1_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    if any(agree.get(b, 1) < 0.90 for b in ("B3",)) or stats["length_bias_flag"]:
        raise SystemExit("STOP (pre-registered): judge-label agreement < 0.90 on B3 or a length bias in answer pairs")
    subprocess.run([".venv/bin/modal", "volume", "put", "--force", config.VOLUME_NAME,
                    str(LOCAL / a.track / "pairs_r1.jsonl"), f"rlaif/{a.track}/pairs_r1.jsonl"], check=True)
    print("uploaded")


# --------------------------------------------------------------------------------- 12.5
def cmd_dpo_judge(a) -> None:
    need_confirm(a)
    load_env()
    rows = rows_dev()
    d = LOCAL / a.track / "dpo"
    gens = {f.stem: RP.jsonl(f) for f in sorted(d.glob("dev_*_copy2.jsonl"))}
    jobs = RP.judge_jobs(rows, gens)
    run_batch(jobs, d / "j1.jsonl", "12.5-judge", f"dpo-{a.track}")
    V = {r["key"]: r["json"] for r in RP.jsonl(d / "j1.jsonl") if r["ok"]}
    for m, g in gens.items():
        need = [j.key for j in RP.judge_jobs(rows, {m: g})]
        s = RP.score_eval(rows, g, V, m)
        print(m, "coverage", round(sum(k in V for k in need) / max(1, len(need)), 3),
              {k: round(s[k], 3) for k in ("grounded_accuracy", "hallucination", "refusal_recall", "refusal_recall_B2",
                                            "over_refusal_A", "citation_accuracy", "quote_fidelity", "format_validity")})


def cmd_fetch(a) -> None:
    names = {"headroom": ["headroom.jsonl"], "samples": ["samples_r1.jsonl"]}[a.what]
    for n in names:
        dst = LOCAL / a.track / n
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([".venv/bin/modal", "volume", "get", "--force", config.VOLUME_NAME, f"rlaif/{a.track}/{n}", str(dst)], check=True)
        print("fetched", dst, sum(1 for _ in open(dst)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("fetch", "headroom-judge", "headroom-score", "label", "pairs", "dpo-judge"))
    p.add_argument("--track", choices=("slm125", "gemma3"), default="slm125")
    p.add_argument("--what", choices=("headroom", "samples"), default="headroom")
    p.add_argument("--confirm", action="store_true")
    a = p.parse_args()
    {"fetch": cmd_fetch, "headroom-judge": cmd_headroom_judge, "headroom-score": cmd_headroom_score, "label": cmd_label,
     "pairs": cmd_pairs, "dpo-judge": cmd_dpo_judge}[a.cmd](a)


if __name__ == "__main__":
    main()
