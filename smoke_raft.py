"""$0 local checks for Phase 7 code (docs/08-raft.md §5.1). Exits non-zero on any failure.

Data checks run on real cleaned documents from `.smoke/reports/cleaning_examples.json`.
"""

from __future__ import annotations

import json
from collections import Counter
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

import config
import raft_data as R

TOKENIZER_DIR = ".smoke/final/base"
EXAMPLES = Path(".smoke/reports/cleaning_examples.json")
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def real_documents() -> dict[str, list[str]]:
    """Kept documents rebuilt from their kept lines (the corpus stores one doc per line)."""
    data = json.loads(EXAMPLES.read_text())
    return {src: [" ".join(ln["text"] for ln in ex["lines"] if ln["removed_by"] is None)
                  for ex in rows if ex["kept"]]
            for src, rows in data.items()}


def data_checks(tok) -> None:
    print("raft_data")
    # 1. split rules match Phase 4 exactly
    import modal_app
    rng = random.Random(0)
    samples = ["".join(rng.choice("abc xyz.") for _ in range(rng.randint(1, 200)))
               for _ in range(5_000)]
    check("is_val_doc == modal_app._is_val_doc on 5,000 strings",
          all(R.is_val_doc(s) == modal_app._is_val_doc(s) for s in samples))
    dev = sum(R.is_dev_doc(s) for s in samples) / len(samples)
    check("is_dev_doc selects ~1/3", 0.30 < dev < 0.37, f"{dev:.3f}")

    # 2. sentence splitting keeps legal abbreviations together
    text = ("See Smith v. Jones, 123 F.3d 456 (9th Cir. 1999). The court held that the U.S. "
            "Department of Labor erred. Id. at 460. Mr. Brown, Inc. objected. It failed.")
    sents = [text[a:b].strip() for a, b in R.sentence_spans(text)]
    check("sentence split respects v. / Cir. / U.S. / Id. / Mr. / Inc.",
          sents == ["See Smith v. Jones, 123 F.3d 456 (9th Cir. 1999).",
                    "The court held that the U.S. Department of Labor erred.", "Id. at 460.",
                    "Mr. Brown, Inc. objected.", "It failed."], " | ".join(sents))

    # 3. chunking on real documents
    docs = real_documents()
    lengths, bad_order, n_docs = [], 0, 0
    for src, texts in docs.items():
        for doc in texts:
            n_docs += 1
            passages = R.chunk_document(doc, tok)
            pos = 0
            for p, n in passages:
                at = doc.find(p, pos)
                if at < 0:
                    bad_order += 1
                pos = max(pos, at)
                lengths.append(len(tok.encode(p, add_special_tokens=False)))
    lo, hi = min(lengths), max(lengths)
    check(f"chunking: {len(lengths):,} passages from {n_docs} real docs are in-order substrings",
          bad_order == 0, f"{bad_order} out of order")
    # Offset-based counts differ from standalone encoding by -1..+4 tokens at passage edges
    # (measured on 6,345 real passages); the hard 1,024 limit is enforced per formatted example.
    check("chunking: standalone token counts within [58, 164]", lo >= 58 and hi <= 164,
          f"min {lo}, max {hi}, mean {sum(lengths) / len(lengths):.0f}")

    # 4. quality tags
    cites = ("Brown v. Board, 347 U.S. 483 (1954); Roe v. Wade, 410 U.S. 113 (1973); "
             "Miranda v. Arizona, 384 U.S. 436 (1966); Smith v. Jones, 123 F.3d 456 (1999).")
    table = "2019 2018 2017 $ 4,210 $ 3,118 $ 2,904 1,203 1,118 1,002 (412) (390) (377) 8.1% 7.4%"
    boiler = ("Indicate by check mark whether the registrant (1) has filed all reports required "
              "to be filed by Section 13 or 15(d) of the Securities Exchange Act of 1934.")
    prose = ("The court held that the statute of limitations began to run when the plaintiff "
             "discovered the injury, and therefore the claim was timely filed in the district "
             "court before the deadline set by the legislature for actions of this kind.")
    check("tags: citation list", "citations" in R.passage_tags(cites, "case-law"), str(R.passage_tags(cites, "case-law")))
    check("tags: numeric table", {"numeric", "low_alpha"} & set(R.passage_tags(table, "sec")),
          str(R.passage_tags(table, "sec")))
    check("tags: SEC cover-page boilerplate", "boilerplate" in R.passage_tags(boiler, "sec"))
    check("tags: clean prose has none", R.passage_tags(prose, "case-law") == [], str(R.passage_tags(prose, "case-law")))
    check("eligibility: SEC table kept as 'numeric'",
          R.source_eligibility(R.passage_tags(table, "sec"), "sec") == "numeric")
    check("eligibility: case-law citation list is not a source",
          R.source_eligibility(R.passage_tags(cites, "case-law"), "case-law") == "")

    # 5. tag rates on real passages (informational, sanity bounds)
    for src, texts in docs.items():
        ps = [p for d in texts for p, _ in R.chunk_document(d, tok)]
        elig = [R.source_eligibility(R.passage_tags(p, src), src) for p in ps]
        clean = sum(e == "clean" for e in elig) / max(1, len(ps))
        print(f"    {src:<12} {len(ps):>5} passages, {clean:.0%} clean, "
              f"{sum(e == 'numeric' for e in elig) / max(1, len(ps)):.0%} numeric-eligible")
        check(f"{src}: a majority of real passages are clean sources", clean > 0.5, f"{clean:.0%}")


def pipeline_checks(tok) -> None:
    """Step 7.2's core functions on a tiny corpus of real documents ($0, local)."""
    import tempfile

    import raft_app as A

    print("raft_app (7.2 core)")
    docs = [d for texts in real_documents().values() for d in texts]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus = tmp / "corpus-000.txt"
        corpus.write_text("".join(d + "\n" for d in docs))
        res = A.build_passages(corpus, "sec", 1.0, tok, tmp / "passages")
        check("pass 1: every doc assigned to exactly one split",
              res["train_docs"] + res["val_docs"] == res["docs"] == len(docs),
              f"{res['docs']} docs -> {res['train_docs']} train / {res['val_docs']} val")
        train_rows = A._jsonl(tmp / "passages" / "train" / "sec" / "corpus-000.jsonl")
        check("pass 1: passage rows written with tags and eligibility",
              len(train_rows) == res["train_passages"] > 0
              and all({"pid", "doc", "tags", "elig", "text"} <= set(r) for r in train_rows))

        # leakage scan: passages copied from train docs must score ~1.0, novel text 0.0
        rng = random.Random(1)
        leaked = rng.sample([r for r in train_rows if r["tokens"] >= 100], 20)
        novel = {"text": " ".join(rng.choice(["alpha", "beta", "gamma", "delta", "omega",
                                              "sigma", "kappa", "zeta"]) for _ in range(120))}
        cands = leaked + [novel]
        fr = A.leak_fractions(cands, [A.leak_scan(corpus, cands)])
        check("leak scan: passages copied from train docs flagged (fraction > 0.9)",
              min(fr[:-1]) > 0.9, f"min {min(fr[:-1]):.2f}")
        check("leak scan: novel text not flagged", fr[-1] == 0.0, f"{fr[-1]:.2f}")

        # dev/test selection: mix, dev:test ratio and per-document cap
        # real documents are distinct across sources, so the fake rows get per-source doc ids
        fake_val = [{**r, "pid": f"{src}/{r['pid']}", "doc": f"{src}:{r['doc']}",
                     "devtest": rng.choice(["dev", "test", "test"])}
                    for src in ("case-law", "sec", "fineweb-edu") for r in train_rows if r["elig"]]
        picked = A.select_candidates(fake_val, n_total=300)
        by_src = Counter(r["pid"].split("/")[0] for r in picked)
        per_doc = Counter(r["doc"] for r in picked)
        check("select: 40/40/20 source mix", by_src == {"case-law": 120, "sec": 120, "fineweb-edu": 60},
              str(dict(by_src)))
        check("select: at most 2 sources per document", max(per_doc.values()) <= 2)
        check("select: SEC numeric passages <= 10% of SEC picks",
              sum(r["elig"] == "numeric" for r in picked if r["pid"].startswith("sec/")) <= 12)

        # BM25 sanity probe
        bm = A.build_bm25(train_rows, None, probes=300)
        check("bm25: a passage's longest sentence retrieves it top-1 (>= 95%)",
              bm["top1_hit_rate"] >= 0.95, f"{bm['top1_hit_rate']:.1%} of {bm['probes']}")

        # source selection + context assembly (§4.1 step 4)
        import bm25s
        srcs = A.select_train_sources({"case-law": [], "sec": train_rows, "fineweb-edu": []}, 50,
                                      {"G1": 0.6, "G2": 0.2, "G3": 0.2})
        check("sources: one per document, documents with >= 4 chunks",
              len({r["doc"] for r in srcs}) == len(srcs) > 0
              and all(r["pid"] in {t["pid"] for t in train_rows} for r in srcs),
              f"{len(srcs)} SEC sources")
        ret = bm25s.BM25()
        ret.index(bm25s.tokenize([r["text"] for r in train_rows], stopwords="en", show_progress=False),
                  show_progress=False)
        builder = A.ContextBuilder(train_rows, ret, [r["pid"] for r in train_rows], tok)
        by_pid = {r["pid"]: r for r in train_rows}
        built, problems = 0, []
        for i, s in enumerate(srcs[:24]):
            o = by_pid[s["pid"]]
            base = {"cid": f"t{i}", "source": "sec", "split": "train", "source_pid": o["pid"]}
            first = " ".join(o["text"].split()[:12])
            for bucket, extra in (("A", {"question": f"What does the filing say about {first}?",
                                         "answer": "x", "quote": first}),
                                  ("B1", {"question": f"What does the filing say about {first}?"}),
                                  ("C", {"instruction": "Summarize the passage in one sentence.",
                                         "task": "summarize", "response": "A summary."})):
                row, why = builder.build({**base, **extra, "bucket": bucket, "cid": f"t{i}{bucket}"})
                if row is None:
                    continue
                built += 1
                ps = [by_pid[p] for p in row["pids"]]
                others = [p for p in ps if p["pid"] != o["pid"]]
                same = [p for p in others if p["doc"] == o["doc"]]
                if len(ps) != 4:
                    problems.append(f"{bucket}: {len(ps)} passages")
                if bucket == "B1" and o["pid"] in row["pids"]:
                    problems.append("B1 contains its oracle")
                if bucket != "B1" and row["pids"][row["k"] - 1] != o["pid"]:
                    problems.append(f"{bucket}: oracle not at k")
                if any(abs(p["chunk"] - o["chunk"]) < 2 for p in same):
                    problems.append(f"{bucket}: adjacent chunk used")
                if len(same) > 2 or len({p["doc"] for p in others if p["doc"] != o["doc"]}) != len(others) - len(same):
                    problems.append(f"{bucket}: document rule broken")
                if any(R.jaccard(builder.shingles(p), builder.shingles(o)) > 0.5 for p in others):
                    problems.append(f"{bucket}: near-duplicate distractor")
                if bucket == "C" and f"Passage {row['k']}" not in row["task_text"]:
                    problems.append("C: task does not name its passage")
                if row["n_tokens"] > config.SEQ_LEN:
                    problems.append("too long")
        again, _ = builder.build({"cid": "t0A", "source": "sec", "split": "train", "bucket": "A",
                                  "source_pid": srcs[0]["pid"], "answer": "x",
                                  "quote": " ".join(by_pid[srcs[0]["pid"]]["text"].split()[:12]),
                                  "question": "What does the filing say about "
                                              + " ".join(by_pid[srcs[0]["pid"]]["text"].split()[:12]) + "?"})
        first_a, _ = builder.build({**{"cid": "t0A", "source": "sec", "split": "train", "bucket": "A",
                                       "source_pid": srcs[0]["pid"]},
                                    "answer": "x", "quote": " ".join(by_pid[srcs[0]["pid"]]["text"].split()[:12]),
                                    "question": "What does the filing say about "
                                                + " ".join(by_pid[srcs[0]["pid"]]["text"].split()[:12]) + "?"})
        check(f"assembly: {built} contexts obey the oracle, adjacency, document and near-dup rules",
              built >= 40 and not problems, "; ".join(problems[:4]) or f"{built} built")
        check("assembly: deterministic per candidate id",
              again is not None and first_a is not None and again["pids"] == first_a["pids"])


def teacher_output_checks(tok) -> None:
    """§5.2 mechanical checks and the §4 student format."""
    print("teacher-output checks + student format")
    passage = ("Acme Corp. reported net revenues of $4.2 million for fiscal 2019, compared to "
               "$3.1 million in 2018. The Court’s “holding” — see Smith v. Jones "
               "— was that the claim was timely. The Federal Deposit Insurance Corporation of "
               "the United States was not a party to the merger agreement.")
    cut = R.recut_quote('The Court\'s "holding" - see Smith v. Jones', passage)
    check("quote: straight-quote copy re-cut to the exact curly-quote span",
          cut is not None and cut[0] == "The Court’s “holding” — see Smith v. Jones"
          and passage[cut[1][0]:cut[1][1]] == cut[0], repr(cut))
    check("quote: elided quote rejected", R.recut_quote("Acme Corp. ... fiscal 2019", passage) is None)
    good = {"question": "What net revenues did Acme Corp. report for fiscal 2019?",
            "answer": "$4.2 million.", "quote": "net revenues of $4.2 million for fiscal 2019",
            "type": "number_or_date"}
    check("G1: valid item passes", R.check_g1_item(good, passage)[0] == "", R.check_g1_item(good, passage)[0])
    check("G1: number not in passage rejected",
          R.check_g1_item({**good, "answer": "$5.2 million."}, passage)[0] == "answer_number_not_in_passage")
    check("G1: 'No.' and a leading 'The' are not treated as names",
          R.check_g1_item({**good, "question": "Was the Federal Deposit Insurance Corporation a party to the agreement?",
                           "answer": "No.", "quote": "was not a party to the merger agreement",
                           "type": "yes_no"}, passage)[0] == "")
    check("G1: hedging answer rejected",
          R.check_g1_item({**good, "answer": "The amount is not stated."}, passage)[0] == "answer_hedges")
    check("G1: question copying 6+ passage words rejected",
          R.check_question("Did they report net revenues of $4.2 million for fiscal 2019?", passage)
          == "question_copies_passage")
    check("G1: a long name span is not counted as copying",
          R.check_question("Was the Federal Deposit Insurance Corporation Of The United States involved?",
                           passage) == "")
    check("G3: extract_list items must be verbatim",
          R.check_g3_item({"instruction": "List the parties named in the passage.",
                           "response": "Acme Corp.; Smith; Jones"}, passage, "extract_list")[0] == ""
          and R.check_g3_item({"instruction": "List the parties named in the passage.",
                               "response": "Acme Corp.; Wayne Enterprises"}, passage,
                              "extract_list")[0] == "list_item_not_verbatim")
    yn = R.check_g3_item({"instruction": "Does the passage state that Acme reported $5 million?",
                          "response": "No. net revenues of $4.2 million for fiscal 2019"},
                         passage, "yes_no_check")
    check("G3: yes/no keeps 'No.' and re-cuts its quote",
          yn[0] == "" and yn[1]["yes_no"] == "No" and yn[1]["quote"].startswith("net revenues"), str(yn))
    check("answers_agree: symmetric containment, numbers, different facts",
          R.answers_agree("It reported $4.2 million.", "$4.2 million")
          and R.answers_agree("The claim was timely filed", "the claim was timely")
          and R.answers_agree("Twenty-two additional charges were nol prossed on February 23, 1970.",
                              "twenty-two")
          and R.answers_agree('Disselkamp claimed Bischoff and Shilling used Patton as a "rubber stamp."',
                              "Bischoff and Shilling")
          and R.answers_agree("the Requisite Lenders' instruction",
                              "at any time that the Requisite Lenders have instructed Agent to act")
          and not R.answers_agree("$3.1 million", "$4.2 million")
          and not R.answers_agree("The claim was dismissed as untimely.", "timely filed claim"))

    # student format: decision token first, labels mask the prompt, segment == whole string
    prompt = R.format_prompt(["Alpha beta gamma.", "Delta epsilon."], question="What is alpha?")
    for k, tgt, first in ((None, R.format_target(None), 5494),
                          (1, R.format_target(1, "Beta.", quote="Alpha beta"), 12175)):
        ids, labels = R.encode_example(tok, prompt, tgt)
        whole = tok.encode(prompt + tgt, add_special_tokens=False)
        n_prompt = sum(lab == -100 for lab in labels)
        check(f"format ({'refusal' if k is None else 'answer'}): first target token is "
              f"{tok.convert_ids_to_tokens(first)}, prompt masked, == whole-string encoding",
              ids == whole and ids[n_prompt] == first and labels[n_prompt] == first
              and ids[0] == 1 and ids[-1] == 1, f"first target {tok.convert_ids_to_tokens(ids[n_prompt])}")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    assert config.RAFT.k_passages == 4
    data_checks(tok)
    pipeline_checks(tok)
    teacher_output_checks(tok)
    pipeline_dry_run(tok)
    print(f"\n{'ALL CHECKS PASS' if not failures else f'{len(failures)} FAILED: {failures}'}")
    sys.exit(1 if failures else 0)



def pipeline_dry_run(tok) -> None:
    """candidates -> assemble -> report with mocked generator and checker outputs ($0)."""
    import argparse
    import tempfile

    import bm25s

    import raft_app as A
    import raft_pipeline as RP

    print("raft_pipeline dry run (mocked APIs)")
    docs = [d for texts in real_documents().values() for d in texts]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "c.txt").write_text("".join(d + "\n" for d in docs))
        A.build_passages(tmp / "c.txt", "sec", 1.0, tok, tmp / "p")
        pool = A._jsonl(tmp / "p" / "train" / "sec" / "c.jsonl")
        srcs = A.select_train_sources({"case-law": [], "sec": pool, "fineweb-edu": []}, 30,
                                      {"G1": 0.6, "G2": 0.2, "G3": 0.2})
        RP.LOCAL = tmp / "raft"
        st = RP.sdir("trial")
        RP.write_jsonl(st / "sources.jsonl", srcs)
        g1_first = next(j for j, s in enumerate(srcs) if s["kind"] == "G1")
        for w in ("deepseek", "gemini"):
            rows = []
            for j, s in enumerate(srcs):
                words = s["text"].split()
                key = f"{s['pid']}|{s['kind']}|{w}"
                # unique per source and writer, except one planted duplicate (gemini's first G1)
                qid = f"deepseek number {j}" if (w == "gemini" and j == g1_first) else f"{w} number {j}"
                if s["kind"] == "G1":
                    js = {"items": [{"question": f"According to filing {qid}, what is stated at the outset?",
                                     "answer": " ".join(words[2:5]), "quote": " ".join(words[:10]),
                                     "type": "fact"}]}
                elif s["kind"] == "G2":
                    js = {"question": "What was the total revenue of the Company in fiscal 1987?",
                          "missing_detail": "revenue"}
                else:
                    js = {"instruction": "Summarize the passage in one sentence.",
                          "response": "It describes the business."}
                rows.append({"key": key, "ok": True, "json": js})
            RP.write_jsonl(st / f"gen-{w}.jsonl", rows)
        RP.cmd_candidates(argparse.Namespace(stage="trial"))
        cands = RP.jsonl(st / "candidates.jsonl")
        ret = bm25s.BM25()
        ret.index(bm25s.tokenize([r["text"] for r in pool], stopwords="en", show_progress=False),
                  show_progress=False)
        builder = A.ContextBuilder(pool, ret, [r["pid"] for r in pool], tok)
        assembled = [row for row, _ in (builder.build(c) for c in cands) if row]
        RP.write_jsonl(st / "assembled.jsonl", assembled)
        flip = {r["cid"] for r in assembled if r["bucket"] == "B1"}
        flip = set(sorted(flip)[:1])                  # ds_off wrongly says one B1 is answerable
        for ck in RP.CHECKERS:
            out = []
            for r in assembled:
                if r["bucket"] == "A":
                    v = {"answerable": True, "support": [r["k"]], "answer": r["answer"],
                         "quote": r["quote"], "ambiguous": False}
                elif r["bucket"] in ("B1", "B2"):
                    v = {"answerable": ck == "ds_off" and r["cid"] in flip, "support": [], "answer": None,
                         "quote": None, "ambiguous": False}
                else:
                    v = {"yes_no": None, "supported": True, "on_task": True}
                out.append({"key": f"{r['cid']}|{ck}", "ok": True, "json": v})
            RP.write_jsonl(st / f"checks-{ck}.jsonl", out)
        (st / "ledger.json").write_text(json.dumps({"spent_usd": 0.0, "calls": {}, "tokens": {}}))
        RP.cmd_report(argparse.Namespace(stage="trial"))
        rep = json.loads((st / "report.json").read_text())
        nb1 = sum(r["bucket"] in ("B1", "B2") for r in assembled)
        check("dry run: buckets A/B1/B2/C all assembled",
              {r["bucket"] for r in assembled} == {"A", "B1", "B2", "C"}, str(Counter(r["bucket"] for r in assembled)))
        check("dry run: perfect mock checkers -> DeepSeek keep rate equals assembled/items",
              rep["writer_keep"]["deepseek"]["accepted_both"] == rep["writer_keep"]["deepseek"]["assembled"])
        check("dry run: thinking agreement reflects the one flipped refusal label",
              rep["thinking"]["agree_on"] == 1.0 and abs(rep["thinking"]["agree_off"] - (1 - 1 / nb1)) < 1e-9,
              str(rep["thinking"]))
        dups = sum(v.get("drop:duplicate_question", 0) for v in rep["funnel"].values())
        check("dry run: the one planted duplicate question is dropped, nothing else",
              dups == 1 and rep["funnel"]["gemini:G1"].get("drop:duplicate_question") == 1, f"{dups} dropped")


if __name__ == "__main__":
    main()
