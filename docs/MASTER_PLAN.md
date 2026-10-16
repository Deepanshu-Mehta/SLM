# MASTER PLAN — 125M Legal/Financial SLM, from scratch to deployment

Status legend: `PENDING` / `RUNNING` / `DONE` / `BLOCKED`
Last updated: 2026-10-16

| Phase | Name | Compute | Cost | Status |
|-------|------|---------|------|--------|
| 0 | Setup + smoke test + measure | CPU (Modal) | ~$0 | **DONE** |
| 1 | Stream + clean | CPU fan-out ×20 | ~$0 | **DONE** |
| 2 | Dedup + decontaminate | CPU fan-out ×20 | ~$0 | **DONE** |
| 3 | Train 16K byte-level BPE tokenizer | CPU ×1 (8 vCPU) | ~$0 | **DONE** |
| 4 | Tokenize + pack + 99/1 split | CPU fan-out ×14 | ~$0 | **DONE** |
| 5 | Pretrain 125M (4 epochs) | **GPU H100×8** | **≤ $25.32 actual** of $50 cap | **DONE** — val 2.127 / ppl 8.39 / **0.657 bits/byte** |
| 6 | Deploy (HF repo + inference API + web frontend) | CPU | ~$0 (idle = $0) | **DONE**: model, API and website live |
| 7 | Grounded SFT / RAFT (answer only from passages, refuse otherwise) | APIs + **GPU H100×1** | ≈ $20 est., $31 staged caps | **PLANNED**: `docs/08-raft.md` (revised 10-16); awaiting approval |

---

## 1. Project objective

Build, from zero, a **125M-parameter Llama-style decoder-only language model**
specialized on US legal + SEC-financial text, pretrained on a corpus we construct
ourselves (fresh tokenizer, fresh weights), then deploy it for public inference.

No SFT, no alignment, no RAG in this build. The deliverable is a **base completer**.
(Phases 0–6. Post-training was added later as Phase 7, grounded SFT / RAFT: see `docs/08-raft.md`.)

### Final architecture (config.ModelConfig — maps 1:1 to `transformers.LlamaConfig`)

| Field | Value |
|---|---|
| parameters | **125,848,320** (~125.8M), tied embeddings — exact `LlamaForCausalLM` count |
| layers | 12 |
| hidden size | 768 |
| attention heads | 12 (head dim 64), MHA (`num_key_value_heads == num_attention_heads`) |
| MLP | SwiGLU, intermediate 3072, `hidden_act="silu"` |
| normalization | RMSNorm (pre-norm), eps 1e-5 |
| positions | RoPE, theta 10000 |
| context length | 1024 |
| vocab | 16,384 (byte-level BPE) |
| attention bias | none |

### Final corpus (fixed by measurement — NOT 70/20/10, NOT 10B tokens)

Per `GUIDE.md §2.2` this is the authoritative data plan for Phases 0–4:

| Source | HF id | split / config | field | token_budget (proxy) | expected realized |
|---|---|---|---|---|---|
| case-law | `HFforLegal/case-law` | `us` | `document` | 1.00B | ~863M (~39%) |
| sec | `PleIAs/SEC` | `train` | `text` | 1.30B | ~861M (~39%) |
| fineweb-edu | `HuggingFaceFW/fineweb-edu` | `train` / `sample-10BT` | `text` | 0.50B | ~465M (~21%) |

Expected final: **≈2.19B train tokens + ≈22.1M val tokens**, realized mix ≈ 40/40/20.
Rationale: the two legal sources physically cap at ~2B clean tokens (282,390 and
48,543 documents respectively), so "70% case-law of 10B" is impossible. More
"tokens seen" during pretraining comes from **multiple epochs** over this fixed
corpus, not from collecting more unique tokens.

---

## 2. Phase breakdown

### Phase 0 — Setup + smoke test + measure (CPU, ~$0)

**Does:** installs tooling, authenticates Modal, creates the Volume and HF secret,
writes the four source files **verbatim from GUIDE.md §5**, then proves the whole
read path works end to end on 10 documents per source, and measures the true token
yield of each source.

**Creates:**
- `.venv/` (local Python 3.12/3.13 venv holding the `modal` CLI — git-ignored)
- `.gitignore` (`.env.local`, `.venv/`, `__pycache__/`, `*.bin`, `index.json`)
- `config.py` — verbatim, single source of truth
- `cleaning.py` — verbatim, 6-step deterministic clean chain
- `dedup.py` — verbatim, hashing/shingle helpers
- `modal_app.py` — verbatim, the Modal App (image, Volume, one function per phase)
- `docs/00-setup.md`
- Modal Volume `slm-125m`; Modal secret `huggingface-token`

**Validation:**
1. `python3 config.py` prints `slm-125m` and `model: 125,847,552 params (~125.8M) | vocab 16384 | 12L/768d/12h kv=12`
   *(the verbatim GUIDE value; Phase 5 corrected `approx_params()` to 125,848,320 by
   counting the final RMSNorm)*
2. `python3 -c "import config, cleaning, dedup"` — imports clean
3. `modal profile current` succeeds
4. `modal run modal_app.py` (smoke test) — expect ~9–10 of 10 docs kept per source, sane excerpts, right text field per source
5. `modal run modal_app.py::measure` — expect case-law ~0.8–1.0B, sec ~1.1–1.2B, fineweb ~11B est. clean tokens

**Done when:** all five checks pass and the measured yields confirm the legal-first mix.

**ACTUAL (2026-09-21):** DONE. Smoke 9/10, 10/10, 10/10. Measured yields case-law
0.81B / sec 1.16B / fineweb-edu 11.67B clean tokens — legal sources cap at 1.97B,
confirming the legal-first mix. See `docs/00-setup.md`.

---

### Phase 1 — Stream + clean (CPU fan-out, ~$0, minutes)

**Does:** `modal run modal_app.py::clean --fineweb-shards 5`. Resolves the parquet
shard URLs for each source via the HF datasets-server API, then launches **one
container per parquet shard** (case-law 10 + sec 5 + fineweb-edu 5 = 20 workers).
Each worker streams its shard, runs every document through the 6-step chain, and
stops at its per-shard token cap (`token_budget // n_shards`, chars/4 proxy).

The cleaning chain (cheapest check first; a drop ends the chain and is counted):
1. `filter_lines` — drop lines <40 chars or >30% non-alphanumeric, collapse whitespace
2. `strip_boilerplate` — drop lines matching FORM 10-K / `Page N of M` / SEC headers / `/s/` / TOC / All rights reserved
3. length gate — drop doc if <600 chars survive → `too_short`
4. `is_repetitive` — drop if top-10 4-grams cover >50% of 4-grams → `repetitive`
5. `is_english` — ASCII-ratio first, `langdetect` only on the ambiguous 90–99% band → `non_english`
6. `is_ocr_garble` — **case-law only** (`strict_ocr=True`): drop if >20% of ≥3-letter words are not in `/usr/share/dict/words` → `ocr`

**Writes:** `/data/clean/<source>/shard-XXX.txt` (one document per line, newlines
flattened to spaces) + `/data/clean/phase1_report.json`

**Validation:** drop report per source. Expect ~718,000 docs streamed, ~698,000
kept (~97%), ~2.68B proxy tokens; OCR gate drops ~1,400 case-law docs.
Optional: `modal run modal_app.py::ocr` to re-confirm the 0.20 threshold.

**Done when:** 20 shard files exist on the Volume, the proxy total is in the
2.4–2.9B range, and no worker errored.

**ACTUAL (2026-09-21):** DONE. 20/20 workers. 718,780 streamed → 697,958 kept
(97.1%), **2.68B proxy tokens**, 10.75 GB on the Volume. case-law 1.00B / sec
1.18B / fineweb-edu 0.50B. OCR gate dropped 685 case-law docs. One container
SIGABRT'd at interpreter teardown with no data impact (audited: 20 distinct
shards, no dupes, no gaps). See `docs/01-data.md`.

---

### Phase 2 — Dedup + decontaminate (CPU fan-out, ~$0, ~6 min)

**Does:** `modal run modal_app.py::dedup` — three stages:
1. `minhash_shard` ×10 (case-law only) — 5-word shingles → 32-perm MinHash → `/data/tmp/minhash_sigs/*.npz`
2. `build_near_dups` ×1 — one LSH pass at Jaccard 0.8 → `/data/tmp/near_dups.json`
3. `write_corpus_shard` ×20 — per shard: drop near-dups (case-law), drop exact
   blake2b duplicates, drop docs sharing any **13-gram** with the CaseHOLD /
   LexGLUE `case_hold` eval sets (case-law + sec only)

**Writes:** `/data/corpus/<source>/shard-XXX.txt` + `/data/corpus/phase2_report.json`

**Validation:** expect ~24,000 case-law docs dropped as contaminated, ~1,600
near-dups, ~2,000 SEC exact-dups; corpus ~670,000 docs, ~2.40B proxy tokens.
The decontam 13-gram set must print a non-zero count — a zero there means the
eval sets failed to load and decontamination silently did nothing (hard fail).

**Done when:** 20 corpus shards exist, report written, contamination set non-empty.

**ACTUAL (2026-09-21):** DONE. 670,124 docs kept, **2.40B proxy tokens**, 9.61 GB.
Drops: 24,177 contaminated / 2,051 exact-dup / 1,606 near-dup. Decontam set
**480,908 eval 13-grams** (non-zero — verified). `casehold/casehold` unresolvable
as documented in GUIDE §8.7; LexGLUE `case_hold` covered it. Required two retries
after local-network client disconnects; fixed with `modal run --detach`.
See `docs/02-dedup.md`.

---

### Phase 3 — Train the tokenizer (CPU ×1, ~$0, ~4 min)

**Does:** `modal run modal_app.py::tokenizer`. Trains a fresh **16,384-token
byte-level BPE** over every line of `/data/corpus/*/*.txt`, with specials
`<|bos|> <|eos|> <|pad|> <|unk|>` + chat tokens `<|user|> <|assistant|> <|system|>`
reserved at train time, and the full byte alphabet as `initial_alphabet` (⇒ no OOV
ever). Saved as a HF `PreTrainedTokenizerFast`.

**Writes:** `/data/tokenizer/` (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`)

**Validation:** `vocab_size=16384`, and two `roundtrip=True` lines on a legal and
a financial sentence.

**Known future gotcha (recorded now, acted on only if we export GGUF in Phase 6):**
`llama.cpp`'s `convert_hf_to_gguf.py` rejects a fresh byte-level BPE
("BPE pre-tokenizer was not recognized") and the hash changes on every retrain →
patch the converter to map our unknown byte-level BPE to the `gpt-2` pre-tokenizer,
idempotently.

**Done when:** tokenizer dir exists on the Volume and round-trips.

**ACTUAL (2026-09-21):** DONE. `vocab_size=16384`, both probe sentences
`roundtrip=True`. Measured **5.2–5.4 chars/token** on domain text (vs the 4.0
proxy), confirming the fresh BPE specialized on legal/financial strings.
See `docs/03-tokenizer.md`.

---

### Phase 4 — Tokenize + pack + split 99/1 (CPU fan-out ×14, ~$0, ~10 min)

**Does:** `modal run modal_app.py::tokenize`. 14 workers (case-law 4, sec 6,
fineweb-edu 4), each taking a strided slice of the corpus lines (`idx % num_shards
== shard_index`), batch-encoding 1,000 docs at a time, appending `<|eos|>` after
each document, packing the flat id stream into fixed **1024-token `uint16`
windows** with no intra-window padding, and routing **every 100th window** to val.

**Writes:** `/data/tokens/train/<source>-XXX.bin`, `/data/tokens/val/<source>-XXX.bin`,
`/data/tokens/index.json` (seq_len, dtype, per-shard and total window/token counts)

**Validation:** expect `index: train=2.19B tok (~2.14M win), val=22.1M tok (~21.6K win)`.
Additional local checks: every `.bin` size is a multiple of `1024 × 2` bytes;
val/(train+val) ≈ 1.0%; `index.json` totals match the sum of file sizes on disk.

**Done when:** `index.json` exists, totals are within ~10% of 2.19B, and the
byte-size invariant holds for every shard.

**v2 REBUILD (2026-10-03, current):** document-level split approved and run. Each
document goes wholly to val when `blake2b(text) % 100 == 0`. **2,040,088,576 train +
19,577,856 val tokens**; 6,638 val docs (0.991%); 663,486 + 6,638 = 670,124 docs =
corpus exactly; 0 alignment violations; BPB factor 0.308978; 15,564 steps.
`modal_app.py` (+55/−20) and `config.py` (`VAL_ONE_IN_N_DOCS`) changed accordingly.

**v1 (2026-09-22, superseded):** DONE. **2,039,072,768 train tokens** (1,991,282 windows)
+ **20,601,856 val** (20,119 windows), val fraction exactly 1.000%, 0 window-alignment
violations, 14/14 shards present. Realized mix case-law 35.1% / sec 42.2% /
fineweb-edu 22.8% (77.3% legal). Came in 7% under the guide's 2.19B: SEC and
fineweb matched within 1%, the whole gap is case-law encoding at 4.127 chars/token
versus the ~3.76 the guide's run implies — a more efficient tokenizer on the same
3.24 GB of text, not missing data. Two preemptions absorbed by the per-shard
fan-out. See `docs/04-tokenize.md`.

---

### Phase 5 — Pretrain the 125M model (GPU — **HARD APPROVAL GATE**)

**Nothing in this phase spends GPU money until explicit human approval.** Full
gate report and the 3-step execution plan: `docs/05-pretrain.md`.

**Approved parameters (2026-10-03):** 4 epochs, `--max-usd 50`, 8 × H100,
HF repo `Deepanshu027/slm-125m-base` (requested as `mdeepanshu/…`, a namespace that does not exist on HF).

**Run size:** 4 × 2,040,088,576 = **8.160B tokens seen**, **15,564 optimizer steps**
at 524,288 tokens/step, **381 warmup steps**, 64.8 tokens/param. One continuous
cosine schedule over all 15,564 steps — never restarted.

**Cost:** ≈ **$7.86 / MFU** (4 epochs, incl. 1.15 overhead). At the reference repo's
measured 16.2% MFU that is **$48.52 — a 3% margin under the $50 cap**; the run
needs MFU ≥ 15.7% to fit. The calibration step exists to measure this before the
full run is launched.

**Three steps:** (1) local smoke test on the M2 Pro, $0 → (2) 50-step calibration
on 1 × H100, < $1 → (3) full 4-epoch run on 8 × H100.

**Will create (modular + typed, per the Phase 5/6 rule):**
- `model.py` — **written**: `LlamaForCausalLM` from `config.MODEL`, explicit token ids, `PROMPT_PREFIX_TOKEN_ID = EOS`
- `data.py` — memory-mapped `uint16` windows, per-epoch deterministic shuffle, resumable position, disjoint per-rank slices
- `train.py` — AdamW (β 0.9/0.95, wd 0.1, fused), cosine + warmup, grad-clip 1.0,
  **bf16 autocast with fp32 master weights** (never `model.to(bfloat16)`), DDP,
  atomic checkpoint/resume, loss + perplexity + **bits-per-byte**, `--max-usd` guard
- `pretrain_app.py` — the GPU Modal function (`H100:8`), Volume mount, budget guard

**Checkpoint/resume:** atomic `/data/checkpoints/ckpt.pt` (model + optimizer +
scheduler + step + data position + RNG) every 500 steps; final HF safetensors at
`/data/checkpoints/base/`; `/data/checkpoints/metrics.jsonl`. The LR is a pure
function of the global step, so resuming after preemption continues the *same*
schedule with no re-warm.

**Validation:** loss decreasing, bounded grad-norm, val perplexity **and** BPB at
real milestones (BPB is tokenizer-independent, so comparable across tokenizers);
real samples from legal and financial prefixes, prompted with `<|eos|>`; a
demonstrated resume; `/data/checkpoints/base/` loads with `from_pretrained`.

---

### Phase 6 — Deploy for inference (CPU, ~$0)

**Does:**
1. Push `/data/checkpoints/base/` (config.json + model.safetensors +
   generation_config.json + the three tokenizer files) to the HF repo named by
   `config.HF_REPO`, using the write token from the Modal secret.
2. Stand up an inference API: `POST /generate {prompt, max_new_tokens, temperature,
   top_p, top_k} → {generated}`, CORS open. Default: a HF **Space** (Docker SDK,
   FastAPI, **Python 3.12 pinned**, CPU, zero cost, ~20–40s cold start).
   Alternative on file: a Modal `@modal.asgi_app` endpoint mounting the Volume
   (warmer, but bills while running).
3. A small static frontend on Vercel: prompt box, sampling controls, streamed
   completion, a "what this is" panel, and a "waking the model" state with auto-retry.

**Generation gotcha to bake in:** a base model under greedy decoding can emit EOS
after ~1 token → enforce `min_new_tokens`, apply `top_k`, and suppress rare
non-ASCII vocab tokens to avoid unicode garbage.

**Validation:** the HF repo loads with `AutoModelForCausalLM.from_pretrained`; the
endpoint returns a non-trivial completion for a legal prefix; the Vercel site
completes a prompt end to end from a cold start.

**Deliverable:** `docs/06-deploy.md` + live HF repo URL + Space URL + Vercel URL.

---

## 3. Files created or modified

| File | Phase | Verbatim from GUIDE? |
|---|---|---|
| `config.py` | 0, **amended 5** | verbatim §5.1 through Phase 4; **amended for Phase 5** (see below) |
| `cleaning.py` | 0 | **YES — verbatim §5.2** |
| `dedup.py` | 0 | **YES — verbatim §5.3** |
| `modal_app.py` | 0, **amended 4-v2** | verbatim §5.4 through Phase 4 v1; `tokenize_shard` changed to a document-level split (v2) |
| `.gitignore` | 0 | no |
| `docs/MASTER_PLAN.md` | 0 | no |
| `docs/00-setup.md` … `docs/06-deploy.md` | per phase | no |
| `model.py` | 5 | no — authored: `LlamaForCausalLM` factory, explicit token ids, EOS prompt prefix |
| `verify_tokens.py` | 4 | no — read-only Phase 4 verifier (window alignment, decode, chars/token) |
| `verify_model.py` | 5 | no — read-only Phase 5 preflight (param count, token ids, BOS occurrences, BPB ratio) |
| `data.py` | 5 | no — window store, seeded per-epoch order, step-derived layout |
| `train.py` | 5 | no — training loop (fp32 master + bf16 autocast, DDP, guard, atomic ckpt, BPB) |
| `pretrain_app.py` | 5 | no — Modal GPU launcher; every entrypoint requires `--confirm` |
| `smoke_local.py` | 5 | no — Step 1 harness: 31 assertions incl. bit-exact resume and DDP parity |
| `serve/` (Dockerfile, app.py), `web/` | 6 | no — authored |

`config.py` stays the single source of truth; Phase 5/6 code imports geometry,
paths and hyperparameters from it rather than restating them.

**Phase 5 amendments to `config.py`** (the only departure from the verbatim file;
Phases 0–4 ran against the unmodified version):

| Change | Why |
|---|---|
| `approx_params()` adds the final RMSNorm (+768) | now equals the real model's 125,848,320 |
| `PRETRAIN_EPOCHS = 4` | training length is epochs × corpus, not a token target |
| `BUDGET_CAP_USD = 40.0 → 50.0` | approved for the 4-epoch run |
| `GPU_RATE_PER_SEC`, `COST_OVERHEAD_FACTOR = 1.15` | live spend projection for the `--max-usd` guard |
| `pretrain_steps()`, `projected_cost_usd()` | the one place step count and cost are derived |
| `TARGET_TOKENS` commented as a Chinchilla *reference* | removes a second definition of training length |
| `HF_REPO = "Deepanshu027/slm-125m-base"`, `API_URL`, `SITE_URL` | Phase 6 targets: the live model repo, API and website |

---

## 4. Data and artifact flow

```
HuggingFace (streamed parquet, never fully downloaded)
  HFforLegal/case-law[us]  PleIAs/SEC[train]  HuggingFaceFW/fineweb-edu[sample-10BT]
        │                        │                       │
        └──── Phase 1: clean_shard ×20 (6-step chain, per-shard token cap) ────┐
                                                                               ▼
                       /data/clean/<source>/shard-XXX.txt   (1 doc per line, UTF-8)
                                                                               │
        Phase 2: minhash_shard ×10 → build_near_dups ×1 → write_corpus_shard ×20
                 (near-dup @ Jaccard 0.8 · exact blake2b · 13-gram decontam)
                                                                               ▼
                       /data/corpus/<source>/shard-XXX.txt  + phase2_report.json
                                                                               │
                         Phase 3: train_tokenizer ×1 (16K byte-level BPE)
                                                                               ▼
                       /data/tokenizer/{tokenizer.json, *_config.json, *_map.json}
                                                                               │
        Phase 4: tokenize_shard ×14 (batch encode → +<|eos|> → pack 1024 uint16)
                       every 100th window → val                                ▼
        /data/tokens/train/*.bin  /data/tokens/val/*.bin  /data/tokens/index.json
                                                                               │
                    Phase 5: train.py on H100×8 (DDP, bf16 autocast, cosine+warmup)
                                                                               ▼
        /data/checkpoints/ckpt.pt (resumable)   /data/checkpoints/base/ (safetensors)
                                                 /data/checkpoints/metrics.jsonl
                                                                               │
                      Phase 6: push to HF repo → inference API → Vercel site
```

Every durable artifact lives on the one Modal Volume `slm-125m` mounted at `/data`.
Each worker calls `volume.commit()` before returning.

---

## 5. Modal compute layout

- **One App**: `slm-125m` (`config.PROJECT`).
- **One Volume**: `slm-125m` at `/data`, `create_if_missing=True`.
- **Two images**, both built pip/apt-first and only then `add_local_python_source`
  (Modal requires this order, or the build errors):
  - `cpu_image` = debian_slim py3.12 + apt `wamerican` (for `/usr/share/dict/words`)
    + `datasets==3.6.0`, `huggingface_hub==0.34.4`, `langdetect==1.0.9`,
    `pyarrow==17.0.0`, `datasketch==1.6.5`
  - `ml_image` = the same base + `transformers==4.46.3`
- **Fan-out is deliberate and must be preserved.** Modal can preempt a long single
  container and restart it from zero; one worker per shard bounds that blast radius.
  Do **not** collapse Phases 1/2/4 into one big container.

| Function | Image | Resources | Timeout | Parallelism |
|---|---|---|---|---|
| `smoke_test` | cpu | default | 15 min | 1 |
| `measure_sources` | cpu | default | 20 min | 1 |
| `clean_shard` | cpu | default | 60 min | 20 |
| `minhash_shard` | cpu | 4 vCPU, 4 GB | 20 min | 10 |
| `build_near_dups` | cpu | 8 GB | 20 min | 1 |
| `write_corpus_shard` | cpu | 4 vCPU, 8 GB | 30 min | 20 |
| `train_tokenizer` | ml | 8 vCPU, 16 GB | 40 min | 1 |
| `tokenize_shard` | ml | 8 vCPU, 16 GB | 40 min | 14 |
| `ocr_sample` | cpu | default | 15 min | 1 (optional) |
| Phase 5 trainer | gpu (Phase 5) | **H100×8** | TBD at gate | 1 job / 8 ranks |

---

## 6. Cost controls and the GPU approval gate

- Phases 0–4 are CPU-only and cost well under **$1** in total (reference run: $0.18).
- Spend is checked with
  `modal billing report --start <date> --json | python3 -c "import sys,json; print(sum(float(r['cost']) for r in json.load(sys.stdin)))"`.
- **HARD STOP before Phase 5.** No GPU function is created, deployed or invoked,
  and no H100 is requested, until explicit human approval. The gate report states
  estimated cost, `--max-usd`, duration, GPU count, tokens/steps, and the
  checkpoint/resume strategy.
- `config.BUDGET_CAP_USD = 40.0` is the configured ceiling; the trainer enforces
  `--max-usd` and stops itself (after checkpointing) when projected spend exceeds it.
- Any operation that could create meaningful cost beyond the configured budget is
  likewise gated.

---

## 7. Dependencies and known gotchas

1. **Data plan**: legal-first, **not** 70/20/10 and **not** 10B tokens. GUIDE.md §2
   wins over INITIAL.md's older table for Phases 0–4.
2. **Modal image order**: every `pip_install` / `apt_install` must precede
   `add_local_python_source`, or the image build fails.
3. **`langdetect` is slow per document.** `is_english` is ASCII-ratio first and only
   calls `langdetect` on the ambiguous 90–99% band. Keep that ordering.
4. **OCR gate needs the system wordlist** — supplied by the apt package `wamerican`.
5. **Preemption**: keep every heavy step fanned out one worker per shard.
6. **Proxy vs real token counts**: Phases 1–2 report a chars/4 proxy; only Phase 4
   gives true counts, and they land ~8% below the proxy. Do not treat the proxy as
   a regression when Phase 4 comes in lower.
7. **`casehold/casehold` parquet may not resolve**; the LexGLUE `case_hold` config
   covers the same benchmark, so a single failure there is expected and fine — but
   both failing (0 n-grams) is a hard error.
8. **Secrets**: `.env.local` is git-ignored and never committed; the HF token
   reaches containers only through the Modal secret `huggingface-token`.
9. **Local toolchain**: the `modal` CLI lives in a project-local `.venv` (the system
   Python 3.13 is externally managed). Modal *containers* run Python 3.12.
10. **HF Space** (Phase 6): pin Python 3.12 in the Dockerfile; free CPU Spaces sleep
    and cold-start in ~20–40s.
11. **`dedup.word_ngrams` uses Python's `hash()`**, which is salted per process
    (`PYTHONHASHSEED`). Decontamination is correct only because the benchmark
    13-gram set and each document's 13-grams are hashed in the *same* process —
    which `write_corpus_shard` guarantees. **Never cache that set to disk** or
    share it across workers; the hashes would silently stop matching.
12. **Prompt with `<|eos|>`, not `<|bos|>`.** Phase 4 never inserts BOS (only an
    EOS after each document), so id 0 is never an input token during training and
    its input embedding is untrained. Every document in training starts right after
    an EOS, so that is the correct prompt prefix (`model.PROMPT_PREFIX_TOKEN_ID`).
13. **Exact dedup is per-shard only** — identical documents in *different* SEC or
    fineweb shards survive. Known and accepted.
14. **Base-model decoding** (Phase 6): enforce `min_new_tokens` + `top_k` and
    suppress rare non-ASCII tokens, or greedy decoding emits EOS almost immediately.

---

## 8. Definition of done (whole project)

1. `/data/tokens/index.json` reports **2.040B train tokens and 19.6M val tokens** in
   `uint16` 1024-token windows with a **document-level** 99/1 split, from a corpus that
   is 35/42/23 case-law/SEC/web (77.3% legal). ✅ ACHIEVED — see `docs/04-tokenize.md`
   for why the total is 7% under the guide's 2.19B projection (tokenizer efficiency,
   not data loss) and for the v2 split rebuild.
2. `/data/tokenizer/` holds a 16,384-token byte-level BPE that round-trips.
3. ✅ ACHIEVED — a 125,848,320-parameter Llama-style model pretrained for 4 epochs
   (8.16B tokens), final held-out val 2.1269 nats / ppl 8.39 / **0.6572 bits/byte**,
   improving at every eval. Original criterion: a 125,848,320-parameter Llama-style model has been pretrained to convergence
   under the approved budget, with a monotone-ish loss curve and a reported
   held-out **validation perplexity**, resumable from `/data/checkpoints/ckpt.pt`.
4. ✅ ACHIEVED — `/data/checkpoints/base/` is a standard HF model directory that loads with
   `AutoModelForCausalLM.from_pretrained` (verified independently; `architectures` added).
5. ✅ ACHIEVED — weights published at `Deepanshu027/slm-125m-base` (public, hash-verified from the Hub).
6. ✅ ACHIEVED — `https://mdeepanshu2706--slm-125m-serve-api.modal.run` answers `POST /generate`
   with coherent legal/SEC completions (Modal CPU, scale-to-zero, verified to idle at 0 containers).
7. ✅ ACHIEVED — https://slm-125m-legal-sandy.vercel.app calls the endpoint and streams completions, with a
   visible "waking the model" state on cold start (verified live: 9.6 s to first token, then 13.9 tok/s).
8. `docs/00-setup.md` … `docs/06-deploy.md` each record what ran and the **actual**
   measured results, and this master plan's status table is all `DONE`.
9. No secret was ever committed; total spend stayed within `BUDGET_CAP_USD`.

---

## 9. Execution log

| When | Phase | Event |
|---|---|---|
| 2026-09-21 | — | Master plan written. Proceeding to Phase 0 without further approval, per execution rules. |
| 2026-09-21 | 0 | DONE. Volume + secret created; 4 files written verbatim; smoke test 29/30 docs kept; yields measured (0.81B / 1.16B / 11.67B). One CLI-only deviation: `modal run modal_app.py::main`. |
| 2026-09-21 | 1 | DONE. 20 clean workers, 697,958 docs kept of 718,780 (97.1%), 2.68B proxy tokens / 10.75 GB on the Volume. |
| 2026-09-21 | 2 | Started. |
| 2026-09-21 | 2 | DONE after 2 network-related retries. 670,124 docs / 2.40B proxy tokens. Adopted `modal run --detach` for all long runs. |
| 2026-09-21 | 3 | Started. |
| 2026-09-21 | 3 | DONE. 16,384-token byte-level BPE trained, round-trips verified, 5.2-5.4 chars/token on domain text. |
| 2026-09-21 | 4 | Started. |
| 2026-09-22 | 4 | DONE. 2.039B train + 20.6M val tokens, 1.000% val split, 0 alignment violations. 7% under the guide's projection — investigated and attributed to tokenizer efficiency on case-law (verified with `verify_tokens.py`). |
| 2026-09-22 | 5 | ⛔ STOPPED AT GPU APPROVAL GATE. Gate report in `docs/05-pretrain.md`. Phases 0-4 total spend: $0.00. |
| 2026-10-03 | 5 | Plan revised: **4 epochs, $50 cap, 8×H100, HF `mdeepanshu/slm-125m-base`**. config.py amended (exact `approx_params()` = 125,848,320; `pretrain_steps()` = 15,556; GPU rates verified at $0.001097/s). `verify_model.py` preflight: ALL INVARIANTS HOLD; `<|bos|>` occurs 0× in 2.06B packed tokens → prompts use `<|eos|>`; BPB = loss × 0.309145. Cost ≈ $7.855/MFU → $48.49 at the reference 16.2% MFU (3% margin). Recommended perf stack: SDPA + fused AdamW + TF32, torch.compile A/B in calibration. Recommended doc-level val split (awaiting yes/no). |
| 2026-10-03 | 4 | **v2 rebuild — document-level val split** (approved). 2,040,088,576 train + 19,577,856 val tokens, 6,638 val docs (0.991%), every doc on exactly one side. Both verifiers pass; BPB factor 0.308978; steps 15,564; 4-epoch cost $48.52 @ 16.2% MFU. |
| 2026-10-03 | 5 | **Step 1 DONE ($0).** data.py / train.py / pretrain_app.py written. `smoke_local.py` 31/31 on M2 Pro CPU: loss 9.90→7.87 in 20 steps; SIGKILL+resume bit-identical to an uninterrupted run; 2-proc DDP identical to single-process; HF export bos0/eos1/pad2, 100% argmax agreement. Fixed: gloo-incompatible AVG reduce; smoke warmup override. Step 2 awaits approval. |
| 2026-10-03 | 5 | **Step 2 DONE (≤ $0.59, 1×H100, 7m46s).** Eager 15.2% MFU → 4 epochs $51.67 (over cap); **torch.compile 32.4% MFU (2.13×) → $24.28**, numerically identical losses. CUDA resume, bf16 autocast, fused AdamW, NCCL, export all exercised; no warnings. Verdict PROCEED. Step 3 projection ~$30–39 at `--max-usd 49`; awaiting approval. |
| 2026-10-03 | 5 | Final pre-launch review: pyflakes clean, Volume clean, collectives rank-consistent; budget-stop branch exercised locally (checkpoints then exits 3, ledger persists across launches). Fixed host resources for 8 ranks (cpu 16, 96 GiB, 4 compile threads/rank). **Step 3 approved and launched**: 8×H100, `--max-usd 49 --compile`. |
| 2026-10-03 | 5 | **DONE.** 8×H100, 15,564 steps, 44.9 min, **≤ $24.78** (Phase 5 total ≤ $25.32 of $50). 3.36M tok/s, 32.1% MFU, ~99% scaling. Final held-out val **2.1269 / ppl 8.39 / 0.6572 bits/byte**, improving at all 16 evals; per-epoch gains halve each pass. Artifact verified independently; missing `architectures` in config.json fixed in code and on the Volume. Next: Phase 6 (deploy). |
| 2026-10-03 | 6 | Model published: **Deepanshu027/slm-125m-base** (public; the `mdeepanshu` namespace does not exist, and the token is Deepanshu027's). Weights hash-verified from the Hub. FastAPI server written and tested locally. HF Docker Space **blocked by 402**: free Docker Spaces now need PRO. Website plan written (`docs/07-website-plan.md`). No Modal resources running. |
| 2026-10-03 | 6 | Hosting decided: **Modal CPU, scale-to-zero** (HF Docker Spaces need PRO). Live at `mdeepanshu2706--slm-125m-serve-api.modal.run`: cold start 11.9 s, 11.8–17.5 tok/s, every endpoint + CORS + 429 rate limit verified from the public URL; **scaled to 0 containers 218 s after the last request**. Fixed a crash-loop (`config` missing from the image). License left unset by decision. Next: build the website per `docs/07-website-plan.md`. |
| 2026-10-03 | 6 | **Website built** (Next.js 16, untracked by request): playground plus a 10-chapter build explainer, every number from `site.json`. Browser-tested at 1440px and 390px, light and dark; 0 console errors. Found: Phase 2 LSH = 3×10 bands (catches 29% at J=0.8, so near-dups are a floor); per-source held-out bits/byte SEC 0.463 / case law 0.687 / web 1.017. Corrected the decontam share to 10.3%. Vercel deploy waiting on login. |
| 2026-10-03 | 6 | **DONE. Website live at https://slm-125m-legal-sandy.vercel.app** (Vercel production, public). Live phone-viewport test from a cold backend: 80 tokens, 13.9 tok/s, 9.6 s to first token, cross-origin to Modal OK, 0 console errors. All Phase 6 deliverables live. |
| 2026-10-06 | 7 | **Planned** (`docs/08-raft.md`). Teacher: `gemini-3.8-flash` (generation + blind verification; the key already works) with `claude-sonnet-5` as the independent judge for dev/test, evaluation and a 5% audit. 20K examples (55% answerable / 25% refusal / 20% grounded tasks) + ~20% pretraining replay; dev/test only from val documents. Staged caps: $5 bake-off → $25 pilot → $5 pilot GPU → $60 scale + $3 GPU. Nothing executed; no spend. |
| 2026-10-16 | 7 | **Plan revised** (D-16, E-18): `deepseek-flash` (off-peak) writes and checks; `gemini-3.8-flash` is the independent checker; 10K examples; source passages pre-filtered; thinking only on refusal-label checks. ≈ $20 (API ≈ $17, GPU ≈ $3); stage caps $3 → $10 → $4 GPU → $12 + $2 GPU. No Anthropic key needed. Nothing executed. |
| 2026-10-16 | 7 | **Plan reviewed; patch applied** (D-17, E-19), 57 edits.
  Fixed: replay getting 84% of the loss weight; collapsed chat-token rows (pairwise cosine 0.95); test leakage checked against the wrong reference (now the whole pretraining corpus); the entity-name refusal shortcut (paired A/B1 + same-document distractors); an unbuildable C bucket (placeholder + V2 check); DeepSeek's default thinking (explicitly off + `max_tokens` caps); the audit fallback breaking the budget (now per bucket, ≤ $3).
  Pilot compiles once. Estimate unchanged at ≈ $20. Nothing executed. |
| 2026-10-16 | 7 | **Execution started** (user approved; costed steps asked one by one). 7.0 config ✅. 7.1 local held-out baseline ✅ (bpb 0.695 / 0.453 / 1.019). **7.2 ✅, ≈ $0.19** (approved cap $1): 141K val + 754K train passages, 6 BM25 indexes. Found: **86% of SEC dev/test candidates overlapped training filings** (E-20); the dev/test pool was rebuilt from leak-scanned passages only (2,000 passages, 1,332 docs). SEC BM25 top-1 check failed-and-explained (top-10 recall 99.6%). 0 containers left. |
| 2026-10-16 | 7 | **7.3 trial ✅, $0.553** (cap $3): 150 sources, both writers, three checkers, 541 contexts. Keep DeepSeek (64.0% vs 71.4%), thinking off (89.5% vs 90.6%), checker agreement 90.0%, JSON 100%. Fixed an answer-agreement scoring bug after the first scoring (C-11). Measured unit costs are ~4× under the model, so the re-quote is **7.4 ≈ $1.90, Phase 7 ≈ $7** (was $20). |
| 2026-10-16 | 7 | **7.4 pilot data ✅, $1.81** (cap lowered to $4): 2,400 train / 500 dev / 1,000 test (150 pairs) / 100 probe, every quota exact. A review of the trial data first added a vague-reference rule and per-call question types. The budget guard refused one over-cap Gemini batch (worst case $4.62) before spending; fixed with token caps and chunking. Phase 7 spend so far ≈ $2.56. Awaiting the user's spot-check. |
| 2026-10-16 | 7 | **7.5 ✅ ($0):** `sft.py` / `eval_raft.py` / `raft_gpu_app.py`; 53/53 local checks plus 5 sweep checks. Caught 4 bugs before any GPU spend: mask slicing, `min_new_tokens=16` leaking into generation, over-strict resume fingerprint, export after resume. A full review of 7.0–7.4 artifacts first: all boundaries, leakage, format and pairs verified. Next: 7.6 GPU gate. |
| 2026-10-17 | 7 | **7.6 pilot sweep ✅, $1.37** (cap $4; 2 failed launches fixed, ≈ $0.18 wasted): 11 runs compiled once in 880 s. Selected LR 2e-4 / refusal 35%. Curve still climbing (10K justified); AUROC 0.934. **Forgetting guards failed** (+0.031 bits/byte, −4.4 CaseHOLD PMI points). Low grounded accuracy traced to invented quotes; copy-constrained decoding takes the 80-item score from −0.550 to −0.137 at $0 (D-21). 7.6b forgetting follow-up written and smoke-tested; awaiting GPU approval. 0 containers. |
| 2026-10-17 | 7 | **7.6b ✅, $1.18** (7.6 total $2.55 of $4): forgetting peaks at epochs 2–3 and recovers as the LR anneals. LR 5e-5 ends at +0.006 bits/byte; λ moves the end value only +0.029 → +0.026. CaseHOLD is within limit only at λ 1.0 (−1.1). Copy decoding on all of dev: grounded 0.05–0.07 → 0.21–0.25. Pre-registered projection invalid (C-12); no arm passes both guards; decision returned to the user. 0 containers. |
