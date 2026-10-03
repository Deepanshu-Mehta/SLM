# Phase 4 — Tokenize, pack into 1024-token uint16 windows, split 99/1

**Compute:** 14 CPU containers (8 vCPU, 16 GB each) · **Expected cost:** ~$0 · **Status:** DONE — **v2 (document-level split) is current**; v1 below is the historical record

## What this phase does

`modal run modal_app.py::tokenize`

14 workers — case-law 4, sec 6, fineweb-edu 4 — each taking a **strided slice** of
its source's corpus lines (`line_index % num_shards == shard_index`), so the work
splits evenly without any coordination and no document is processed twice.

Per worker:

1. Batch-encode 1,000 documents at a time (`add_special_tokens=False`).
2. Append the **`<|eos|>` id after every document** — this is the only document
   boundary marker in the stream.
3. Concatenate into one flat id stream and slice it into **fixed 1024-token
   windows** (= `config.SEQ_LEN`, the model's context length). No padding inside a
   window; a window may span a document boundary, which is intended for packing
   efficiency.
4. Route **every 100th window** (`win_count % 100 == 0`) to val, the rest to train.
   This is deterministic and by-window, so no validation window can leak into
   training.
5. Write raw `uint16` little-endian bytes — a 16,384 vocab fits in 16 bits, halving
   disk versus `int32`.

## Data format (the concrete answer to "what format is the training data?")

| Property | Value |
|---|---|
| container | headerless `.bin`, raw `numpy` `uint16` |
| window size | 1024 token ids = 2048 bytes |
| separator | `<|eos|>` id appended after each document |
| padding | none |
| split | every 100th window → val (99% / 1%) |
| manifest | `/data/tokens/index.json` — `seq_len`, `dtype`, total + per-shard window and token counts |

```
/data/tokens/train/case-law-000.bin … fineweb-edu-003.bin   (14 files)
/data/tokens/val/case-law-000.bin   … fineweb-edu-003.bin   (14 files)
/data/tokens/index.json
```

## Validation criteria

| # | Check | Expected |
|---|---|---|
| 1 | all 14 workers return | 14/14 |
| 2 | index totals | `train≈2.19B tok (~2.14M win), val≈22.1M tok (~21.6K win)` |
| 3 | val fraction | ≈1.0% of all windows |
| 4 | every `.bin` file size | exact multiple of 2048 bytes |
| 5 | realized source mix | case-law ~863M, sec ~861M, fineweb-edu ~465M (≈40/40/20) |
| 6 | round-trip spot check | decode the first window of a train shard → readable domain text |

Real tokenizer counts are expected to land ~8% **below** the Phase 1/2 chars/4
proxy. That is the documented, expected relationship, not a regression.

## Results (actual)

Run: `ap-zx2QuWmUlxvrBVkav2nnOI` (detached) · 14 workers · verification run
`ap-UDbfklg4I7AFbccVcdqPO6`.

### Final index

```
index: train=2.04B tok (1,991,282 win), val=20.6M tok (20,119 win)
seq_len=1024  dtype=uint16
```

| Check | Result |
|---|---|
| 1. all 14 workers returned | **14/14 recorded in `index.json`**, no duplicates, no gaps ✓ |
| 2. index totals | train 2.039B / val 20.6M — **below the 2.19B expectation, see below** |
| 3. val fraction | **1.000%** exactly (20,119 of 2,011,401 windows) ✓ |
| 4. `.bin` window alignment | **0 violations** across all 28 files ✓ |
| 5. realized mix | case-law 35.1% / sec 42.2% / fineweb-edu 22.8% (**77.3% legal**) |
| 6. decode round-trip | genuine domain text in all three sources ✓ |

`index.json` totals and raw bytes on disk agree **exactly**:
2,039,072,768 train tokens = 1,991,282 × 1024, and 20,601,856 val = 20,119 × 1024.

### Decoded first windows (proof the packing is real, not garbage)

- **case-law** — *"respect we owe to States as sovereigns and to the people of those States … Those denials followed this Court's decision in"*
- **sec** — *"units in 1993. The Company's share of the market in 1993 was 30 percent, a major factor of which was sales to Ford Motor Company…"*
- **fineweb-edu** — *"…alination operating costs. Their findings appear in the current issue of the Journal of Materials Chemistry. Reverse-osmosis (RO) desalination…"*

## Deviation: 2.04B tokens, not the guide's 2.19B (investigated, explained, not data loss)

The shortfall is **entirely in case-law** and is a tokenizer-efficiency effect:

| Source | Realized (train+val) | Guide expected | Delta |
|---|---|---|---|
| sec | 868.7M | ~861M | **+1%** ✓ |
| fineweb-edu | 468.9M | ~465M | **+1%** ✓ |
| case-law | 722.1M | ~863M | **−16%** |
| **TOTAL** | **2.060B** | ~2.21B | −7% |

Two sources land within 1% of the guide, which rules out systemic loss.
The evidence that case-law is efficiency, not missing documents:

1. **The corpus is the right size.** Phase 2 produced 206,684 case-law documents
   in 3.24 GB — matching the guide's own Phase 2 figures. Nothing was dropped
   between Phase 2 and Phase 4.
2. **All 14 tokenize shards are present**, none duplicated, none missing.
3. **Measured chars/token** (500 docs/source, real tokenizer):
   case-law **4.127**, sec **4.980**, fineweb-edu **4.230**, overall **4.844**.
   For the guide's run to have produced 863M tokens from the same 3.24 GB of
   case-law, its tokenizer would have had to encode at **3.76 chars/token** —
   about 10% less efficient than ours.

So the same text simply encodes to fewer tokens. Our fresh BPE allocated its
16,384 merges slightly differently (it was trained on our corpus, whose case-law
shards stop at their own token caps and whose OCR gate dropped 685 rather than
~1,400 documents), and the result is a **more** efficient vocabulary on legal
text — a better tokenizer, not a smaller corpus.

GUIDE §8.6 already documents that real counts land ~8% below the chars/4 proxy;
ours land 14% below (2.40B proxy → 2.06B real), for the same reason.

**Deliberately not "fixed."** The obvious way to reach 2.19B would be raising the
fineweb-edu budget, but the data plan (case-law 1.0B / sec 1.3B / fineweb-edu 0.5B
proxy) is fixed by specification, and inflating the web share to hit a token
target would dilute the legal ratio the whole project is built around. The corpus
is reported as it is: **2.039B train + 20.6M val tokens, 77.3% legal.**

### Preemption during the run (contained by design)

Two containers were preempted mid-run; Modal restarted each with the same input
(`Container terminated due to preemption. Your Function will be restarted with
the same input`). Only those shards restarted — the other workers were unaffected,
and the final index shows all 14 shards exactly once. This is precisely the
failure GUIDE §8.5 warns about and the reason the per-shard fan-out must not be
collapsed into one container. One preempted container's stdout line was lost from
the local log (13 lines for 14 shards), which is why completeness was confirmed
from `index.json` rather than from log lines.


---

# v2 rebuild — document-level validation split

**Approved:** 2026-10-03 · **Compute:** same 14 CPU workers · **Cost:** ~$0 · **Status:** DONE

## Why v1 is being replaced

v1 (above) followed the GUIDE: every 100th **window** went to val. Documents span
windows, so the rest of each val document stayed in train — an SEC val window has
~18 sibling windows of the same filing in training (case-law ~2–3, fineweb ~0).
With 4 training epochs over those siblings, SEC val loss would measure recall of
specific filings rather than generalisation, and a val set that shares documents
with train tends to *fall with* train loss — hiding the very overfitting that 4
epochs risks. v1's numbers above are kept as the historical record.

## What changes (and what doesn't)

Only the routing inside `tokenize_shard` changes. Tokenizer, corpus, packing
(EOS after each document, 1024-token `uint16` windows, no padding), worker layout
and file names are all unchanged; v2 files overwrite v1 files in place.

- Each **document** goes wholly to val when
  `blake2b(utf-8 text, 8 bytes) % 100 == 0` (`config.VAL_ONE_IN_N_DOCS = 100`),
  otherwise wholly to train.
- Train and val pack into **separate** buffers, so a window never mixes the two.
- The hash is a pure function of the document text: stable across processes and
  workers. Python's `hash()` is deliberately **not** used — it is salted per process.
- Side benefit: an exact duplicate that survived per-shard dedup (identical text in
  two shards) hashes identically, so both copies land on the **same** side. It
  cannot leak across the split. (Near-duplicates in SEC/fineweb still can — they
  were never near-deduped. Known, small.)
- `index.json` gains `"split": "document"` and train/val document counts.

## Validation criteria (v2)

| # | Check | Expected |
|---|---|---|
| 1 | 14/14 workers, no duplicate or missing shard | in `index.json` |
| 2 | val document share | ≈1.0% of documents per source |
| 3 | val token share | ≈1% (noisier than v1: SEC filings average ~19K tokens) |
| 4 | total tokens | ≈ v1 total (2.06B), minus ≤ 2×1023 tail tokens per shard |
| 5 | `.bin` window alignment | 0 violations (`verify_tokens.py`) |
| 6 | doc counts reconcile | Σ val_docs + Σ train_docs = 670,124 corpus docs |
| 7 | preflight invariants + **new BPB constant** | `verify_model.py` passes; BPB factor re-measured on the new val split |
| 8 | `pretrain_steps()` | re-derived from the new index (≈15.5K) |

## Results (v2)

Run `ap-*` (detached, 14/14 workers, no preemptions) · verified by
`verify_tokens.py` (`ap-PF7IXNifwm44tiLP9cUdRd`) and `verify_model.py`
(`ap-332ROS5YsJCefQATzAq5S1`). **All 8 checks passed.**

```
index: train=2.04B tok (1,992,274 win), val=19.6M tok (19,119 win)   split=document
```

| # | Check | Result |
|---|---|---|
| 1 | shards | 14/14, no duplicates, none missing ✓ |
| 2 | val document share | **0.991%** (6,638 of 670,124) ✓ |
| 3 | val token share | **0.951%** ✓ |
| 4 | total vs v1 | 2,059,666,432 vs 2,059,674,624 = **−8,192 tokens** (8 windows: each shard now has two partial tail buffers instead of one) ✓ |
| 5 | window alignment | **0 violations**; disk = index exactly ✓ |
| 6 | doc reconciliation | 663,486 train + 6,638 val = **670,124** = corpus — every document routed exactly once ✓ |
| 7 | preflight invariants | ALL HOLD; `<\|bos\|>` still 0×; **new BPB factor 0.308978** ✓ |
| 8 | `pretrain_steps()` | **15,564** (v1: 15,556), re-derived from the index automatically ✓ |

### Per source

| Source | Train tokens | Train share | Val tokens | Val share | Val docs / all docs |
|---|---|---|---|---|---|
| case-law | 715.3M | 35.1% | 6.79M | 34.7% | 1,992 / 206,684 = 0.964% |
| sec | 860.6M | 42.2% | 8.12M | 41.5% | 428 / 45,035 = 0.950% |
| fineweb-edu | 464.2M | 22.8% | 4.66M | 23.8% | 4,218 / 418,405 = 1.008% |

The val mix mirrors the train mix to within ~1 point per source, so val loss stays
representative of the training distribution while now being genuinely held out at
the document level. SEC val is 428 complete filings (8.1M tokens) — small in count,
but each is a whole unseen document, which is the point.

### What else changed in the data (and why it is fine)

- **Bits-per-byte factor:** val bytes/token 4.669241 → `BPB = loss × 0.308978`
  (v1: 0.309145). Per source: case-law 4.4796, sec 5.0437, fineweb 4.2929 bytes/token.
- **Train windows now start at document boundaries** where v1's did not: decoded
  first windows read "The petition filed in this Court…", "Cummins Engine Company,
  Inc.…", "For all the love, romance and scandal in Jane Austen's books…".
- **Routing is cross-process stable.** Local test before launch: 200,000 synthetic
  docs → 1.0095% to val with an identical split fingerprint under three different
  `PYTHONHASHSEED` values (while Python's own `hash()` changed every time).

### Code change (the only edit to the verbatim Phase 4 logic)

`modal_app.py`: new `_is_val_doc()` (blake2b of the document text,
`% config.VAL_ONE_IN_N_DOCS`); `tokenize_shard` routes whole documents into separate
`train_buf` / `val_buf` and packs each independently; shard results and `index.json`
record `train_docs` / `val_docs` and `"split": "document"`. `config.py`:
`VAL_EVERY_N_WINDOWS` → `VAL_ONE_IN_N_DOCS`. +55/−20 lines.
