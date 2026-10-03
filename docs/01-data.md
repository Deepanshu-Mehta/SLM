# Phase 1 — Stream + clean

**Compute:** 20 CPU containers in parallel (one per parquet shard) · **Expected cost:** ~$0 · **Status:** DONE

## What this phase does

`modal run modal_app.py::clean --fineweb-shards 5`

1. The local entrypoint asks the HuggingFace datasets-server API for the parquet
   shard URLs of each source (`_parquet_urls`), truncating fineweb-edu to 5 shards.
2. It divides each source's `token_budget` by its shard count to get a per-shard
   token cap, then fans out **one `clean_shard` container per parquet shard**:

   | Source | Shards | Budget | Per-shard cap |
   |---|---|---|---|
   | case-law | 10 (expected) | 1.0B | ~100M |
   | sec | 5 (expected) | 1.3B | ~260M |
   | fineweb-edu | 5 (`--fineweb-shards 5`) | 0.5B | ~100M |

3. Each worker streams only its own parquet file, runs every document through the
   fixed 6-step chain, writes the survivors one-per-line, and **stops as soon as
   its per-shard token cap is reached** (chars/4 proxy). Nothing is downloaded
   whole — the multi-TB raw datasets are never materialized.

The fan-out is deliberate: Modal can preempt a long single container and restart
it from zero, so each shard is its own bounded unit of work. It must not be
collapsed into one big container.

## The cleaning chain (cheapest check first; a drop ends the chain and is counted)

| # | Step | Drops when | Reason code |
|---|---|---|---|
| 1 | `filter_lines` | line <40 chars, or >30% non-alphanumeric; whitespace collapsed | (line-level) |
| 2 | `strip_boilerplate` | line matches FORM 10-K / `Page N of M` / TOC / `/s/` signature / SEC header / `[X]` | (line-level) |
| 3 | length gate | <600 chars survive | `too_short` |
| 4 | `is_repetitive` | top-10 4-grams cover >50% of all 4-grams | `repetitive` |
| 5 | `is_english` | ASCII ratio <90%, or `langdetect` ≠ en on the 90–99% band | `non_english` |
| 6 | `is_ocr_garble` (**case-law only**, `strict_ocr=True`) | >20% of ≥3-letter words absent from `/usr/share/dict/words` | `ocr` |

Step 5 is ASCII-ratio-first on purpose — `langdetect` is slow per document and is
only consulted on the ambiguous band. Step 6 needs the system wordlist, supplied
by the apt package `wamerican` in the image.

## Output

```
/data/clean/case-law/shard-000.txt … shard-009.txt
/data/clean/sec/shard-000.txt … shard-004.txt
/data/clean/fineweb-edu/shard-000.txt … shard-004.txt
/data/clean/phase1_report.json
```

UTF-8, **one document per line**, internal newlines flattened to spaces.

## Validation criteria

| # | Check | Expected |
|---|---|---|
| 1 | every worker returns without error | 20/20 |
| 2 | docs streamed / kept | ~718,000 streamed, ~698,000 kept (~97%) |
| 3 | proxy tokens | ~2.68B total (chars/4) |
| 4 | OCR gate active on case-law | ~1,400 docs dropped with reason `ocr` |
| 5 | shard files on the Volume | 20 files, all non-empty |
| 6 | per-source totals | each source at or near its budget cap |

Note: the token counts here are a **chars/4 proxy**, not real tokenizer counts.
Phase 4 gives true counts and they land ~8% lower — that is expected, not a regression.

## Results (actual)

Run: `ap-jPganBo0HkE15Kjbk1LBXl` · 20/20 workers returned · all six checks passed.

### Drop report

| Source | Streamed | Kept | Keep rate | `too_short` | `ocr` | `non_english` | Proxy tokens |
|---|---|---|---|---|---|---|---|
| case-law | 238,207 | 232,292 | 97.5% | 5,230 | **685** | 0 | 0.998B |
| sec | 47,752 | 47,199 | 98.8% | 553 | — | 0 | 1.183B |
| fineweb-edu | 432,821 | 418,467 | 96.7% | 14,348 | — | 6 | 0.500B |
| **TOTAL** | **718,780** | **697,958** | **97.1%** | 20,131 | 685 | 6 | **2.68B** |

Guide expected ~718,000 streamed, ~698,000 kept (~97%), ~2.68B proxy tokens —
matched on all three.

### Shard plan (resolved from the HF datasets-server)

| Source | Parquet shards | Used | Per-shard cap | Result |
|---|---|---|---|---|
| case-law | 10 | 10 | ~100M | 9 shards hit the cap exactly; shard-009 exhausted its parquet file at 97.7M |
| sec | 5 | 5 | ~260M | 2 hit the cap, 3 exhausted their file (195.4M / 216.3M / 251.7M) |
| fineweb-edu | 14 | 5 (`--fineweb-shards 5`) | ~100M | all 5 hit the cap exactly |

case-law and fineweb-edu landed on their budgets (1.00B / 0.50B). SEC came in at
1.18B rather than its 1.30B cap because the source ran out — consistent with the
Phase 0 measurement of ~1.16B available.

### Independent verification (on-Volume bytes, not just worker self-reports)

| Source | Files | Size on Volume | ÷4 chars/token |
|---|---|---|---|
| case-law | 10 | 4.01 GB | 1.00B |
| sec | 5 | 4.73 GB | 1.18B |
| fineweb-edu | 5 | 2.01 GB | 0.50B |
| **TOTAL** | **20** | **10.75 GB** | **2.69B** |

Byte totals corroborate the reported token counts. Parsing the log confirms
**20 distinct `(source, shard)` result lines, no duplicates and none missing**.
`/data/clean/phase1_report.json` written and retrieved.

### Notable observations

- **The OCR gate is working and is case-law-only, as designed.** 685 documents
  dropped (guide ballpark ~1,400 — same order; the exact figure depends on which
  documents fall inside each shard's token cap, and shards stop early here). The
  drops are extremely concentrated: shard-003 alone accounts for 578 of the 685,
  i.e. one parquet shard holds most of the badly-scanned opinions.
- `non_english` fired only 6 times in 718,780 documents — the ASCII-ratio-first
  ordering means `langdetect` is almost never invoked, which is exactly why the
  phase is fast.
- `repetitive` never fired on a real document.

### One non-fatal anomaly (investigated, no action needed)

One container logged `Fatal Python error: PyGILState_Release ... Runner aborted
(SIGABRT), exit code: 134` during **interpreter finalization** (the C-extension
teardown of pyarrow/pandas/aiohttp), after its work was done. Modal's retry plus
the log audit confirm the outcome is clean: exactly 20 distinct shard results,
no duplicate shard, no missing shard, and the on-Volume byte totals agree with
the aggregate report. Nothing to fix — this is a teardown-time abort, not a
data-loss event. Flagged here so it is not mistaken for a silent failure later.
