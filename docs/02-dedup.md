# Phase 2 — Dedup + contamination strip

**Compute:** CPU, three fanned-out stages (10 + 1 + 20 containers) · **Expected cost:** ~$0 · **Status:** DONE

## What this phase does

`modal run modal_app.py::dedup`

### Stage 1 — MinHash signatures (10 containers, case-law only)

`minhash_shard` reads each `/data/clean/case-law/shard-XXX.txt`, and per document:
`normalize → words → 5-word shingles → MinHash(num_perm=32)`. Signatures are
stacked and saved as `/data/tmp/minhash_sigs/shard-XXX.txt.npz` (`sigs`, `idxs`).

Only case-law gets near-dedup: it is the dominant, most redundant source (court
opinions quote each other and republish orders verbatim). SEC and fineweb-edu
pass through with exact-hash dedup only.

### Stage 2 — Near-duplicate set (1 container, LSH)

`build_near_dups` loads every signature, streams them through a single
`MinHashLSH(threshold=0.8)`: the first document of a cluster is inserted and kept,
any later document that the LSH query matches is recorded as a near-duplicate.
Result: `/data/tmp/near_dups.json` — `{shard_basename: [line indices]}`.

### Stage 3 — Write the final corpus (20 containers, one per shard)

`write_corpus_shard` applies three drops in order per line:

| Drop | Applies to | Test |
|---|---|---|
| `near_dup` | case-law | line index is in that shard's near-dup list |
| `exact_dup` | all sources | `blake2b(normalize(text), 16)` already seen **within this shard** |
| `contaminated` | case-law, sec (`DECONTAM_SOURCES`) | the doc shares **any 13-gram** with the eval benchmark sets |

The contamination set is built by `_build_contamination_ngrams()` from
`casehold/casehold` and the LexGLUE `case_hold` config — every 13-word n-gram of
the benchmark text. Any training document sharing even one such n-gram is dropped,
so held-out evaluation stays honest. Per GUIDE §8.7, `casehold/casehold` parquet
may fail to resolve; LexGLUE `case_hold` covers the same benchmark, so one failure
is acceptable — **both** failing (0 n-grams) is a hard error and must stop the phase.

## Output

```
/data/corpus/case-law/shard-000.txt … shard-009.txt
/data/corpus/sec/shard-000.txt … shard-004.txt
/data/corpus/fineweb-edu/shard-000.txt … shard-004.txt
/data/corpus/phase2_report.json
```

Same format as Phase 1: UTF-8, one document per line.

## Validation criteria

| # | Check | Expected |
|---|---|---|
| 1 | all three stages complete, no worker error | 10 + 1 + 20 |
| 2 | decontamination n-gram set | **non-zero** (guide: eval 13-grams loaded) |
| 3 | case-law contaminated drops | ~24,000 docs |
| 4 | case-law near-dups | ~1,600 docs |
| 5 | SEC exact-dups | ~2,000 docs |
| 6 | corpus size | ~670,000 docs, ~2.40B proxy tokens |
| 7 | `phase2_report.json` written to the Volume | present |

## Results (actual)

Run: `ap-9dXf69BrlnveIP5b5e2gZX` (detached) · 10 + 1 + 20 workers · all 7 checks passed.

### Phase 2 report

| Source | Kept docs | `near_dup` | `exact_dup` | `contaminated` | Proxy tokens |
|---|---|---|---|---|---|
| case-law | 206,684 | **1,606** | 0 | **24,002** | 0.81B |
| sec | 45,035 | — | **1,989** | 175 | 1.09B |
| fineweb-edu | 418,405 | — | 62 | — | 0.50B |
| **TOTAL** | **670,124** | 1,606 | 2,051 | 24,177 | **2.40B** |

Against the guide's expectations — ~24,000 contaminated, ~1,600 near-dups,
~2,000 SEC exact-dups, ~670,000 docs, ~2.40B proxy tokens — **all five matched.**

### Decontamination actually ran (the check that matters)

`[decontam] 480,908 eval 13-grams loaded`, printed identically by all **15**
decontam workers (case-law 10 + sec 5 = `DECONTAM_SOURCES`). Non-zero, so the
24,177 contamination drops are real removals, not a silent no-op.

`casehold/casehold` failed to load in all 15 workers
(`Empty 'data_files': '[]'`) — **this is the documented GUIDE §8.7 behaviour**,
not a new fault. The LexGLUE `case_hold` config supplied the full 480,908-gram
set and covers the same CaseHOLD benchmark, so decontamination is intact.

Worth noting the asymmetry: 11.6% of case-law was dropped as CaseHOLD-contaminated
versus 0.4% of SEC — expected, since CaseHOLD is built from court opinions.
fineweb-edu is not in `DECONTAM_SOURCES` and was not scanned.

### Independent verification (on-Volume bytes)

| Source | Files | Size | ÷4 chars/token |
|---|---|---|---|
| case-law | 10 | 3.24 GB | 0.81B |
| sec | 5 | 4.36 GB | 1.09B |
| fineweb-edu | 5 | 2.01 GB | 0.50B |
| **TOTAL** | **20** | **9.61 GB** | **2.40B** |

Byte totals corroborate the report. 20 corpus shards + `phase2_report.json`
present. Phase 1 → Phase 2 shrank the corpus 10.75 GB → 9.61 GB (−10.6%).

## Infrastructure failures hit on the way (diagnosed, fixed, not silently skipped)

Two aborted attempts preceded the successful run. Neither was a pipeline fault:

1. **Attempt 1** — local `ConnectionError: [Errno 8] nodename nor servname
   provided, or not known`, raised inside the Modal client's `check_lost_inputs`
   polling loop. A local DNS resolution failure.
2. **Attempt 2** — surfaced locally as a bare `RemoteError`. The client-side
   traceback showed only Modal plumbing; the *real* cause was visible only in the
   container logs: `Stopping app - local client disconnected` followed by
   `Runner terminated` ×7. The containers were killed **because** the local client
   dropped, not because of any error in the work.

**Root cause:** an unstable local network, combined with plain `modal run` tying
the ephemeral app's lifetime to the local client.

**Fix:** `modal run --detach` — Modal's own documented remedy, which keeps the app
alive server-side across client disconnects. **No source file was modified.**
The guide's `--no-compute-sigs` flag gives a cheap resume path as well, since
MinHash signatures persist to the Volume independently of the client.

**Going forward:** every long-running phase is launched detached.
