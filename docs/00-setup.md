# Phase 0 — Setup + smoke test + measure

**Compute:** CPU only (Modal `debian_slim` py3.12) · **Expected cost:** ~$0 · **Status:** DONE

## What this phase does

1. Install the `modal` CLI locally and authenticate from `.env.local`.
2. Create the persistent Volume `slm-125m` (mounted at `/data`) and the Modal
   secret `huggingface-token` (needed only to *push* the model in Phase 6 — all
   three datasets are ungated, so reading data needs no token).
3. Write the four source files **verbatim from `GUIDE.md` §5** — no refactoring,
   no re-derivation:
   - `config.py` — single source of truth (model geometry, data mix, budgets, paths, thresholds)
   - `cleaning.py` — the deterministic 6-step cleaning chain (pure functions)
   - `dedup.py` — hashing / n-gram / shingle helpers for Phase 2
   - `modal_app.py` — the Modal App: two images, one Volume, one function per phase
4. Prove the read path end to end: stream **10 documents per source**, run each
   through the cleaner, print before/after (`modal run modal_app.py`).
5. Measure the **true token yield** of each source (`modal run modal_app.py::measure`,
   2,000 docs/source), which is what justifies the legal-first mix over 70/20/10.

## Why the mix is legal-first (and not 70/20/10)

case-law holds only ~282,390 documents and SEC ~48,543. Together they cap at
roughly 2B clean tokens, so "70% case-law of a 10B-token corpus" is physically
impossible with these sources. The plan instead takes **all** of both legal
sources (budget caps 1.0B and 1.3B proxy tokens) and adds a 0.5B-token slice of
fineweb-edu for fluency — realized ≈40/40/20, ≈2.19B training tokens. Phase 0's
`measure` step is the evidence for that decision.

## Commands

```bash
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
modal volume create slm-125m
modal secret create huggingface-token HUGGINGFACE_TOKEN=... HF_TOKEN=...
python3 config.py
modal run modal_app.py            # smoke test, 10 docs/source
modal run modal_app.py::measure   # yield projection, 2000 docs/source
```

## Validation criteria

| # | Check | Expected |
|---|---|---|
| 1 | `python3 config.py` | `slm-125m` / `model: 125,847,552 params (~125.8M) \| vocab 16384 \| 12L/768d/12h kv=12` |
| 2 | local imports of `config`, `cleaning`, `dedup` | no error |
| 3 | `modal profile current` + volume/secret exist | authenticated |
| 4 | smoke test | ~9–10 of 10 docs kept per source; excerpts are real text from the right field |
| 5 | `measure` | case-law ~0.8–1.0B, sec ~1.1–1.2B, fineweb ~11B est. clean tokens |

## Results (actual)

All five validation checks passed.

**1. `python3 config.py`** — exact expected output:
```
slm-125m
model: 125,847,552 params (~125.8M) | vocab 16384 | 12L/768d/12h kv=12
target tokens: 2.5B (~20 tok/param)
stages: setup -> clean -> dedup -> tokenizer -> tokenize -> pretrain -> deploy
```

> **Later correction (Phase 5, 2026-10-03):** the GUIDE's `approx_params()`
> omitted the final RMSNorm. It now returns **125,848,320**, matching the real
> `LlamaForCausalLM`. The 125,847,552 above is left as the true record of what
> the verbatim file printed during Phase 0.

**2. Local imports** — `config`, `cleaning`, `dedup` import and execute; all four
files pass `py_compile`. Files were extracted programmatically from the GUIDE.md
fences, so they are byte-for-byte the specified source (167 / 148 / 34 / 560 lines).

**3. Infrastructure** — Modal CLI 1.5.5 in a project-local `.venv` (Python 3.12),
profile `default` authenticated as `mdeepanshu2706`. Volume `slm-125m` created
2026-09-21 22:31 IST. Secret `huggingface-token` created (keys `HUGGINGFACE_TOKEN`
and `HF_TOKEN`). `.env.local` is git-ignored and was never echoed or committed.

**4. Smoke test** (`modal run modal_app.py::main`):

| Source | Kept | Drop reasons | Field check |
|---|---|---|---|
| case-law | **9 / 10** | `too_short: 1` | `document` → US court opinions, visible OCR noise (`Bx parte`, `IT 18 ORDERED`) as expected |
| sec | **10 / 10** | — | `text` → 10-K business descriptions (Cummins, Weyerhaeuser, IBM Credit, Norfolk Southern) |
| fineweb-edu | **10 / 10** | — | `text` → general educational web prose |

Matches the guide's "about 9 or 10 of 10 docs per source".

**5. Yield measurement** (`modal run modal_app.py::measure`, 2,000 docs/source):

| Source | Keep rate | Avg clean chars/doc | Rows | Est. clean tokens | Guide expected |
|---|---|---|---|---|---|
| case-law | 76% | 11,455 | 282,390 | **0.81B** | ~0.8B ✓ |
| sec | 98% | 95,371 | 48,543 | **1.16B** | ~1.1B ✓ |
| fineweb-edu | 96% | 4,827 | 9,670,000 | **11.67B** | ~11B ✓ |
| | | | | TOTAL 13.64B | |

**This is the evidence for the legal-first mix.** The two legal sources together
yield ~1.97B clean tokens — so a 10B-token corpus at 70% case-law (7B) is
impossible; case-law does not contain 7B tokens. Budgets stay as configured:
case-law 1.0B, sec 1.3B, fineweb-edu 0.5B.

## Deviation from the guide (one, minimal)

`modal run modal_app.py` with no function named fails on Modal CLI 1.5.5 —
when a file defines several local entrypoints the CLI now requires an explicit
one instead of defaulting to `main`. Fix: invoke `modal run modal_app.py::main`.
**No source file was changed**; this is a CLI invocation change only.
