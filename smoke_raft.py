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


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    assert config.RAFT.k_passages == 4
    data_checks(tok)
    pipeline_checks(tok)
    print(f"\n{'ALL CHECKS PASS' if not failures else f'{len(failures)} FAILED: {failures}'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
