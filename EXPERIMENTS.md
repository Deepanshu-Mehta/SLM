# EXPERIMENTS — results and decisions, in order

An append-only history. Each experiment records the question, the setup, the **measured** result
and what it changed. Decisions are logged with their reasons so a later session doesn't re-open
settled questions without new evidence. Corrections are logged so wrong numbers never come back.
Full detail lives in `docs/NN-*.md`; current state is in [`PROJECT_STATE.md`](PROJECT_STATE.md).

## Experiments

### E-01 · Phase 0 smoke test (2026-09-21)
Streamed 10 documents per source through `cleaning.py`. Kept: case law **9/10** (1 too short),
SEC **10/10**, web **10/10**. The text fields are correct; case law shows OCR noise. → `docs/00-setup.md`

### E-02 · How much clean text does each source hold? (2026-09-21)
Cleaned 2,000 documents per source and projected to the full dataset.

| Source | Keep rate | Avg clean chars | Rows | Est. clean tokens |
|---|---|---|---|---|
| case law | 76% | 11,455 | 282,390 | **0.81B** |
| SEC | 98% | 95,371 | 48,543 | **1.16B** |
| web | 96% | 4,827 | 9,670,000 | 11.67B |

→ The legal sources cap at ~1.97B tokens, so 70/20/10 at 10B is impossible (**D-01**).

### E-03 · Phase 1 stream + clean (2026-09-21)
20 workers (case law 10 + SEC 5 + web 5 of 14 parquet shards).

- **Documents:** 718,780 streamed → **697,958 kept (97.1%)**, **2.68B proxy tokens** (chars ÷ 4).
- **Drops:** too_short 20,131 · ocr 685 (578 of them in case-law shard-003) · non_english 6.
- **SEC ran out of documents:** 1.18B, short of its 1.30B budget.
- **Infrastructure:** one container hit SIGABRT at interpreter teardown. A log audit showed no data
  loss (20 distinct shards).

→ `docs/01-data.md`

### E-04 · Phase 2 dedup + decontamination (2026-09-21)
- **Attempts:** two failed for infrastructure reasons, not the pipeline. The first hit a local DNS
  `ConnectionError`; the second a client disconnect, which Modal answered with `Runner terminated`.
  Fixed with `--detach` (**D-02**).
- **Result:** **670,124 documents**, 2.40B proxy tokens.
- **Drops:**
  - near-duplicates: 1,606 (case law)
  - exact duplicates: 2,051 (SEC 1,989, web 62)
  - CaseHOLD-contaminated: 24,177 (case law 24,002 = 10.3% of the 232,292 entering; SEC 175 = 0.37%)
- **Decontamination set:** 480,908 thirteen-word windows from LexGLUE `case_hold`.
  `casehold/casehold` failed to resolve, as documented in GUIDE §8.7.

→ `docs/02-dedup.md`

### E-05 · Phase 3 tokenizer (2026-09-21)
`vocab_size=16384`; both probe sentences round-trip. Legal and financial probes encode at
**5.2–5.4 characters per token**, against the 4.0 proxy, so the vocabulary has specialised to the domain. → `docs/03-tokenizer.md`

### E-06 · Phase 4 v1, every 100th window held out (2026-09-22)
- **Result:** 2,039,072,768 train + 20,601,856 val tokens; val exactly 1.000%; 0 alignment errors;
  two preemptions absorbed by the per-shard fan-out.
- **Shortfall:** 7% under the guide's 2.19B, all of it in case law. Measured case law encodes at
  **4.127 characters per token**, against ~3.76 implied by the guide. That's a more efficient
  tokenizer, not lost data (**D-03**).
- **Later superseded by E-08.**

→ `docs/04-tokenize.md`

### E-07 · Phase 5 preflight (2026-10-03)
`verify_model.py` confirmed:
- exact params **125,848,320**, tied embeddings, fp32 master weights;
- token ids 0/1/2/3 as expected;
- **`<|bos|>` occurs 0 times** in 2.06B packed tokens (**D-07**);
- the v1 val split encodes 4.666731 bytes per token.

### E-08 · Phase 4 v2, document-level split (2026-10-03)
Each whole document goes to val when `blake2b(text) % 100 == 0` (**D-06**).

- **Tokens:** **2,040,088,576 train + 19,577,856 val**.
- **Documents:** 6,638 val (0.991%). Train + val = 670,124, exactly the corpus.
- **Mix:** the val source mix tracks the train mix within about a point.
- **Downstream:** the bits-per-byte factor becomes **0.308978** (4.669241 bytes/token); the run is **15,564 steps**.

### E-09 · Local smoke test on the M2 Pro CPU (2026-10-03)
`smoke_local.py` passed **31/31**:
- loss fell 9.896 → 7.867 in 20 steps;
- a run killed at step 13 and resumed was **bit-identical** to an uninterrupted run;
- 2-process DDP losses were identical to single-process;
- the HF export carries bos 0 / eos 1 / pad 2, and its argmax agrees with the fp32 model 100%.

**Bugs fixed:** `ReduceOp.AVG` doesn't exist on gloo; warmup would have been 97,656 steps at
the tiny smoke batch size.

### E-10 · Calibration on 1 × H100, 50 steps (2026-10-03)

| Mode | tokens/s | MFU | Projected 4-epoch cost |
|---|---|---|---|
| eager | 199,615 | 15.2% | **$51.67**, over the cap |
| `torch.compile` | 424,826 | **32.4%** | **$24.28** |

Losses were identical to 4 decimals and resume worked on CUDA. ≤ $0.54 (7 m 46 s). → **D-05**

### E-11 · Pre-launch review (2026-10-03)
- **Fixed:** the 8-GPU function was relying on Modal's tiny default host request. It now asks for
  16 cores and 96 GiB, with 4 compile threads per rank, avoiding an OOM risk during compilation.
- **Tested:** the budget-stop branch, for the first time. It checkpoints, exits with code 3, and the
  spend ledger carries across launches.

### E-12 · Full pretraining, 4 epochs on 8 × H100 (2026-10-03)
- **Throughput:** 3.36M tokens/s, **32.1% median MFU** (~99% scaling from 1 GPU).
- **Time and cost:** **44.9 min**, **≤ $24.78**. No preemptions.

| Step | Epoch | Val loss | Val ppl | Val bits/byte |
|---|---|---|---|---|
| 1,000 | 1 | 2.7763 | 16.06 | 0.8578 |
| 3,000 | 1 | 2.4271 | 11.33 | 0.7499 |
| 7,000 | 2 | 2.2546 | 9.53 | 0.6966 |
| 11,000 | 3 | 2.1697 | 8.76 | 0.6704 |
| **15,564** | 4 | **2.1269** | **8.39** | **0.6572** |

- **No overfitting:** held-out loss improved at all 16 evaluations. Training loss steps down at
  each epoch boundary (partial memorisation) while held-out loss falls smoothly.
- **Diminishing returns:** gains per epoch roughly halve (−0.053, −0.026, −0.013 bits/byte), so the
  corpus is close to exhausted at this size.

Full table in `docs/05-pretrain.md`.

### E-13 · Artifact verification and per-source evaluation (2026-10-03)
- The Hub's `model.safetensors` has the same SHA-256 as the local verified copy.
- **Per source**, 96 held-out windows each, each source scored with its own bytes per token:

| Source | Loss | Perplexity | Bits/byte |
|---|---|---|---|
| SEC | 1.619 | 5.05 | **0.463** |
| case law | 2.134 | 8.45 | **0.687** |
| web | 3.026 | 20.61 | **1.017** |

- **Samples:** fluent legal and SEC prose (word-perfect 10-K boilerplate), invented citations, and
  wrong general facts (photosynthesis).

### E-14 · Serving (2026-10-03)
- **HF Docker Space:** refused with **402**, since it requires HF PRO.
- **Modal CPU endpoint instead** (**D-11**):
  - cold start **11.9 s**; **11.8–17.5 tokens/s**;
  - `/score` gives perplexity 5.856 on a legal sentence vs 488.297 on nonsense;
  - returns 429 after 30 POSTs/min;
  - **scales to 0 containers about 218 s after the last request**.
- **Bugs fixed:** `config` was missing from the image (a crash-loop ending in HTTP 303 after 150 s);
  stream events were text chunks rather than tokens (27 events for 30 tokens), so the true count now
  comes in the final `done` event.

### E-15 · Website (2026-10-03)
- The TypeScript BPE port matches the Python tokenizer on **36/36** golden cases.
- **Browser tests:** desktop and 390 px, light and dark, live generation; **0 console errors**.
- **Live cold start in production:** 9.6 s to the first token, then 13.9 tokens/s.

### E-16 · Phase 2's actual LSH parameters (2026-10-03)
With 32 permutations at threshold 0.8, `datasketch` chose **3 bands × 10 rows**. Probability a pair
is flagged:

| Setting | Jaccard 0.8 | Jaccard 0.9 |
|---|---|---|
| 32 permutations (ours): 3 × 10 | **0.289** | 0.724 |
| 128 permutations: 9 × 13 | 0.399 | 0.929 |

→ The 1,606 near-duplicates are a **floor**, not a full count.

### E-17 · Phase 7 planning measurements (2026-10-06)
- **Gemini key:** the `GEMINI_API_KEY` in `.env.local` lists `gemini-3.8-flash`, `gemini-3.5-flash`,
  `gemini-3.1-pro-preview` and others (a metadata call, $0). There is no Anthropic key yet.
- **Tokenizer on QA-style text:** a numeric SEC passage + question + answer encodes at
  **3.3 characters per token**, against 4.67 bytes/token on held-out prose. That's why RAFT
  passages are capped at 160 tokens, so 4 fit in the 1,024-token context.
- **Teacher prices on 2026-10-06** (Batch API, per million tokens, in / out):
  - `gemini-3.8-flash`: $0.375 / $1.875, the pricing page says "through Dec 31, 2026"
  - `claude-haiku-4-5`: $0.50 / $2.50
  - `claude-sonnet-5`: $1 / $5
  - `gpt-5-mini`: $0.125 / $1
  - `gemini-3.1-pro-preview`: $1 / $6
- **Retired:** Claude 3.5 Haiku and 3.5 Sonnet, which the brief named.

→ `docs/08-raft.md`

### E-18 · Teacher cost comparison: Gemini vs DeepSeek (2026-10-15)
- **Prices (per million tokens):**
  - `deepseek-flash` (DeepSeek-V4.1-Flash), off-peak: $0.15 in / $0.60 out / $0.003 cache hit.
    Peak (01:00–04:00 and 06:00–10:00 UTC, Mon–Fri) is double.
  - `deepseek-v4-pro`, off-peak: $0.66 / $1.98.
  - There is no batch API. The terms explicitly allow training other models on outputs.
- **Modelled cost per 1,000 candidates** (~1,700 in, ~1,000 of it a cached prefix, ~600 out):

  | Setup | Per 1,000 candidates |
  |---|---|
  | `gemini-3.8-flash`, batch | $1.76 |
  | DeepSeek Flash, no thinking | **$0.47** |
  | DeepSeek Flash, +1,000 thinking tokens on every check | $1.07 |
  | DeepSeek Pro, no thinking | $1.67 |

- **Phase 7 totals (modelled):**

  | Setup | Total |
  |---|---|
  | Gemini writer + Sonnet 5 judge, 20K | ≈ $77 |
  | DeepSeek Flash + Gemini checker, 20K | ≈ $46 |
  | Same, with the levers below | ≈ $28 |
  | **Same, at 10K** | **≈ $20** |

- **Levers:** off-peak; 10K examples; pre-filter the source passages (assumed keep rate 65 → 80%);
  thinking only on refusal-label checks.
- **Packing more questions into one call saves ~2%** (the instruction prefix is already a cache hit,
  and the blind check needs one call per candidate). Rejected for its cost to diversity.

These are models, not measurements; the step 7.3 trial measures them. → `docs/08-raft.md` §9

## Decision log

| ID | Date | Decision | Why |
|---|---|---|---|
| D-01 | 09-21 | Legal-first mix: take all legal text, cap web at 0.5B; not 70/20/10 | E-02: the legal sources can't supply 70% of 10B |
| D-02 | 09-21 | Use `modal run --detach` for long runs | E-04: local network drops killed attached apps |
| D-03 | 09-22 | Accept 2.04B tokens; don't pad with web text | Padding would dilute the legal share; the gap is tokenizer efficiency, not missing data |
| D-04 | 10-03 | 4 epochs, $50 cap, 8 × H100 (user's choice) | Fixed 2.04B-token corpus; 4 epochs is the edge of useful repetition |
| D-05 | 10-03 | SDPA + fused AdamW + TF32 + **`torch.compile`**; skip flash-attn 2 | E-10. Attention is only 14% of FLOPs while the MLP is 64%; compile fuses the small ops around it |
| D-06 | 10-03 | Rebuild the validation split **by document** (user approved) | A window split leaks ~18 sibling windows per SEC filing, and 4 epochs would hide overfitting |
| D-07 | 10-03 | Prompt with `<|eos|>`, never `<|bos|>` | E-07: BOS never occurs in the training data |
| D-08 | 10-03 | Weight decay on matrices and embeddings only, not norms | Decaying RMSNorm gains toward 0 hurts |
| D-09 | 10-03 | Full run at `--max-usd 49` | $50 approved minus calibration spend |
| D-10 | 10-03 | Publish under **Deepanshu027**, public | `mdeepanshu` doesn't exist on HF; a public website needs a public model |
| D-11 | 10-03 | Serve on **Modal CPU, scaling to zero** (user's choice) | HF Docker Spaces need PRO. Modal costs $0 idle, capped at 1 container |
| D-12 | 10-03 | Keep the website code **out of git** (user) | It isn't part of model training |
| D-13 | 10-03 | **License deferred** (user) | The owner's decision. The README says all rights are reserved |
| D-14 | 10-05 | Rewrite history to drop `initial.txt`; keep the email; **the user pushes manually** | The user's choice before the first push |
| D-15 | 10-05 | Phase 7 plan (recommendation, not yet executed): **direct RAFT with teacher-written, quote-grounded targets**; refuse when the correct passage is missing; $0 baseline first; ~2K-example pilot | 125M params can't absorb a teacher's knowledge; logit distillation is impossible (different vocabulary); 1,024-token context; chat format never seen |
| D-16 | 10-16 | Phase 7 setup (user's choice): **`deepseek-flash` writes and checks, off-peak; `gemini-3.8-flash` is the independent checker; 10K training examples; pre-filter source passages; thinking only on refusal-label checks**; ≈ $20 | E-18: about 4× cheaper than Gemini + Sonnet at 20K. DeepSeek's terms allow training on outputs. Quality is protected by two-family agreement on dev/test labels, the 5% audit, and incremental top-up if targets are missed |

## Corrections log

| ID | Was | Is | Note |
|---|---|---|---|
| C-01 | 16 Phase 1 workers (the guide's figure) | **20** (10 + 5 + 5) | The guide's arithmetic was off |
| C-02 | 125,847,552 params | **125,848,320** | `approx_params()` omitted the final RMSNorm (768) |
| C-03 | CaseHOLD removed 11.6% of case law | **10.3%** (24,002 / 232,292) | The 11.6% divided by documents kept |
| C-04 | 8,160,354,304 tokens seen | **8,160,018,432** (15,564 × 524,288) | 328 windows after the last full step go unused |
| C-05 | One bits-per-byte constant applied to every source | Each source needs its own bytes per token | Applying the pooled constant mis-states the web score (0.94 vs the correct 1.02) |
| C-06 | "The repo has no git remote" | `origin` already existed | It was never checked before the claim |
| C-07 | — | A blind `git add -A` once committed an unnoticed deletion of the user's `initial.txt` | That's why staging is now explicit; the file was later removed from history (D-14) |
| C-08 | HF model card linked `Deepanshu027/slm-125m-api` | Links the live website and Modal API | That Space was never created (E-14) |
| C-09 | Stream events counted as tokens | The true count comes in the `done` event | E-14 |
