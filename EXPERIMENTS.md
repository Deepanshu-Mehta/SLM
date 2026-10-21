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

### E-19 · Phase 7 plan review: measurements (2026-10-16)
Measured on the released weights and tokenizer (`.smoke/final/base`):

- **Never-seen token rows have collapsed.**
  - Embedding rows 0 and 2–6 (bos, pad, unk, user, assistant, system) point almost the same way:
    pairwise cosine ≈ 0.95, cosine 0.88 with the mean embedding, norm ≈ 1.1 (like trained rows).
  - As outputs they sit at log-prob ≈ −25, against a median of −18 for trained tokens.
  - So `<|user|>` and `<|assistant|>` are nearly the same input vector. "Mean + small noise" would
    keep them so; the init now samples from N(μ, Σ) of the trained rows.
- **The decision token pair.**
  - `Source:` = `S` `ource` `:`; ` none` = `Ġnone` (id 5494), one token.
  - ` Passage` = `ĠPass` (id 12175) + `age`.
  - The answer/refuse decision is therefore `Ġnone` vs `ĠPass`, at one position.
- **The refusal target** is 19 tokens including `<|eos|>`.
- **Encoding is boundary-safe.** Encoding segment by segment at special-token boundaries gives the
  same ids as encoding the whole string. The tokenizer has no normaliser, and curly quotes and `§`
  round-trip exactly.
- **Loss weighting.**
  - Measured target lengths: answerable 62, refusal 19, summary 45 tokens.
  - With 1 replay window per 4 SFT sequences, a pooled per-token mean gives replay **84%** of the
    loss weight (13.1K vs 2.5K target tokens per 64-sequence batch).
  - A per-example mean would raise the effective refusal prior from 25% to **~50%**.
- **Validation shard order:** shards sort as case-law, fineweb-edu, sec. A capped `train.evaluate()`
  therefore reads case law only.

→ `docs/08-raft.md` (review edits applied)

### E-20 · Step 7.2: passage pools and the SEC leakage finding (2026-10-16)
- **Pools:** 141,189 val + 753,575 train passages; train distractor pools of ~250K per source;
  6 BM25 indexes. Cost ≈ $0.19 for the run and its diagnosis.
- **The SEC val split is largely near-duplicate of training text.**
  - 61% of the 56,873 source-eligible SEC val passages share ≥ 50% of their 13-grams with train
    documents, and 27% share ≥ 90%. The per-document median is 0.69.
  - Comparable figures: case law ~19% of candidates over 10%; web ~3%.
  - **Cause:** annual filings of one company restate the previous year's text. Phase 2's exact dedup
    was per shard, and its LSH catches only 29% of pairs at J = 0.8 (E-16).
  - **Consequence:** the base model's SEC held-out 0.463 bits/byte (E-13) is optimistic. Part of it
    is reproduction of near-duplicate filings.
  - **Phase 7 handling:** dev/test passages must have < 10% 13-gram overlap with the whole train
    corpus. 9,916 SEC passages from 369 of 428 documents qualify. The dev/test pool (2,000 passages,
    40/40/20, 1,332 documents) is built only from those.
- **BM25 sanity:** top-1 ≥ 95% held for case law and web but **failed for SEC** (92.4% / 93.6%).
  Top-10 recall is 99.6% / 98.6%. The misses are generic boilerplate sentences shared across
  filings, not index faults.
- **Local MPS limits:** a full CaseHOLD run on the 8.6 GB Mac swapped to 9.3 GB and stalled. It was
  stopped; the official baselines are measured in the GPU container.

### E-21 · CaseHOLD zero-shot sanity on the base model (2026-10-16)
- **Data:** 500 test items on CPU, scored in fp32 from the released bf16 weights.

  | Metric | Value | 95% CI |
  |---|---|---|
  | acc (summed log-prob) | **0.122** | 0.096–0.154 |
  | acc_norm (per byte) | **0.172** | 0.141–0.208 |

- **Below chance is a length artifact.** Correct holdings average 168 characters against 152 for
  the wrong ones. "Pick the shortest" scores 0.114 and "pick the longest" 0.246 on all 3,600 items.
- **Conclusion:** the base model has no zero-shot CaseHOLD skill, which is expected at 125M.
  - PMI scoring was added to `eval_forgetting.py`.
  - If PMI is also at chance, CaseHOLD can't detect forgetting, and held-out bits/byte stays the
    guard (the user decides).

### E-22 · Step 7.3 teacher trial (2026-10-16, $0.553)
150 train passages, both writers, three checkers (`docs/08-raft.md` §13).

| Rule | Result | Outcome |
|---|---|---|
| Writer keep rate | DeepSeek **64.0%** vs Gemini **71.4%** | keep DeepSeek |
| Thinking on refusal labels (agreement with Gemini) | off 89.5% vs on 90.6% | thinking off |
| JSON parse | 100% | ✅ |
| Checker agreement | **90.0%** | ✅ |

- **These outcomes come after fixing `answers_agree`.** Token-F1 counted a verbose but correct
  checker answer (same passage, same quote) as disagreement. It is now symmetric containment with
  light stemming. Before the fix, the writer and checker rules failed (C-11).
- **Keep rates.** Train rule: A 86%, B1 86%, B2 93%, C 100%. Dev/test rule: 81% / 83% / 90% / 100%.
- **Units.** Per call: DeepSeek $0.00013 for generation and $0.00014 for a check; Gemini batch
  $0.00040 for a check. That is ~4× under the plan's model. DeepSeek's prefix cache barely hit (~3%).
- **Main generation loss:** questions copying ≥ 6 passage words (DeepSeek 23% of items, Gemini 12%).
- **Re-quote:** 7.4 ≈ $1.90; 7.8 ≈ $2.51; Phase 7 ≈ $7 (was $20).

### E-23 · Step 7.4 pilot data (2026-10-16, $1.81 of the $4 cap)
- **Datasets:** 2,400 train / 500 dev / 1,000 test (150 A/B1 pairs) / 100 probe. Every quota was
  filled exactly.
- **Fixes applied first (review of the trial data):**
  - a vague-reference rule (9% of trial questions said "the passage");
  - a per-call requested question type;
  - stricter paraphrase in G1, the literal "the passage" in G3, brief V1 answers.
- **Funnel:**
  - G1 60% pass. Copying ≥ 6 words rose to 26.2% (the prompt did not fix it); vague references 9.6%.
  - G3 91% (placeholder misses 4/25 → 0).
- **Keep rates:** train 88–99%; dev/test 82–100%. Dev/test A rose from 81% (trial) to 90%.
- **Budget guard refused a Gemini batch** whose worst case was $4.62 (1,024 output tokens per
  request). Nothing was spent. Fixed with a 384-token cap (measured maximum 202) and 800-request
  chunks.
- **Type mix (train A):** number 29% / fact 27% / party 18% / yes-no 14% / rule-holding 6% /
  definition 6%. The rule-holding type request is largely ignored.

### E-24 · Pilot spot-check and near-miss label noise (2026-10-16)
- **The user's spot-check found 1 real error in 50** (item 31, B2: the passage gives the asked
  figure).
- **Item 29 was not an error.** The review quoted text that exists nowhere in the item or its removed
  oracle; the refusal is correct.
- **Assistant read-through of 15 dev/test B2 items:** 13 correct, 1 borderline (dropped). The
  estimated B2 label noise is ≈ 5–10%.
- **Both bad test items were swapped** for verified spares through `exclusions.json`; only those
  2 rows changed.
- **Lesson:** two-checker agreement still passes near misses whose qualifier ("including HSI") makes
  a stated figure look absent. The near-miss bucket needs extra scrutiny wherever it defines
  reported numbers.

### E-25 · Full read-through of dev/test near misses (2026-10-16, $0)
- **Coverage:** 137 dev/test B2 items + 41 spares, read against all passages.
- **Result:** 24 of 137 (17.5%) dropped under a sharpened rule.
  - **Drop if** stated, computable from figures for the same item, partial/qualitative for the same
    item, or underspecified.
  - **Keep if** the tempting figure belongs to a different year, entity or metric.
  - 7 of 41 spares also failed.
- **Breakdown of drops:** computable 8 / partial 6 / underspecified or case-name 6 / answerable 4.
- **Effect:** dev now has 36 B2 (499 total, 1 short), test 100. Every B2 is reviewed; other rows
  are unchanged.
- **Lesson.** Two independent checkers agreeing is not enough for near misses: both accept items
  whose answer is one subtraction or one qualitative phrase away. Train B2 likely has similar noise.
  For 7.8: G2 prohibitions plus a B2-specific "plausible answer" check.

### E-26 · Step 7.5 local SFT smoke test (2026-10-16, $0)
- **53/53 checks pass, plus 5 sweep checks.**
  - The loss equals a hand computation, and is identical when split into micro-batches. A pooled
    mean would differ (4.01 vs 6.04).
  - 32 examples overfit to 0.0001; generations reproduce them 8/8.
  - A hard kill and resume is bit-identical.
  - The chat rows are distinct (cosine ≤ 0.24).
  - In-place reload keeps the Parameter objects, so the module compiles once.
- **Bugs caught before any GPU spend:**
  - SFT/replay mask slicing (would have crashed).
  - Generation inherited `min_new_tokens=16` from the base model's config, making refusals repeat
    instead of stopping.
  - An over-strict resume fingerprint.
  - Export after a resume could ship the last weights instead of the best.

### E-27 · Step 7.6 pilot sweep on 1 × H100 (2026-10-16/17, ≈ $1.37 of the $4 cap)
- **Launches.** Three. The first two failed before training: a KeyError on the cap ($0), then a CPU/CUDA
  device mismatch in `init_chat_tokens` (≈ $0.18). Both fixed. The third completed in 880 s ($1.19);
  0 containers afterwards.
- **Official baselines** (fp32 init, bf16): held-out pooled **0.7090** bits/byte; CaseHOLD acc 0.121 /
  acc_norm 0.177 / **PMI 0.598**. PMI is well above chance, so CaseHOLD is an informative guard.
- **Grid** (9 runs at 2,000 examples, best epoch of ≤ 8 by dev macro-F1, proxies at τ 0.5):
  - **Selected: LR 2e-4, refusal 35%** (score −0.423, grounded 0.102, hallucination 0.445, macro-F1
    0.864, best epoch 5).
  - Highest macro-F1: LR 5e-5 / 25% (0.887, AUROC 0.937).
- **Learning curve** at the selected setting: score −0.730 / −0.566 / −0.423 at 500 / 1,000 / 2,000
  examples. Still climbing, so 10K is justified.
- **τ** (teacher-forced): AUROC 0.934; macro-F1 0.851 at τ 0.5, best 0.872 at τ 0.7.
- **Forgetting guards FAILED.**
  - Pooled bits/byte +0.0314 (limit +0.02): case law +0.031, SEC +0.028, web +0.036.
  - CaseHOLD PMI −4.4 points (limit −2), CIs disjoint.
  - acc and acc_norm moved < 0.5 points; they only track length.
- **Low grounded accuracy is invented quotes.** On 40 dev A items, 17 quotes weren't from the passage
  and 8 were near-verbatim; only 3 were correct.
  - Copy-constrained decoding (`CopyConstraint`) on 80 dev items: grounded 0.075 → 0.200,
    hallucination 0.450 → 0.163, score −0.550 → −0.137. Decisions identical (80/80).
  - The leftover error is wrong-span selection (17 of 40 A items quote a real but wrong sentence).
- **Bug caught** before the third launch: `proxy_metrics` let a malformed answer on an unanswerable
  item count as neither answer nor refusal. It now uses the constrained decision; unit-tested.
- **Follow-up planned (7.6b, needs approval).** Per-epoch held-out bits/byte for 4 (LR, λ) arms, to
  project forgetting to 7.8's 579 steps. Pre-registered rule in `docs/08-raft.md` §13.

### E-28 · Step 7.6b forgetting follow-up on 1 × H100 (2026-10-17, $1.18)
- 4 arms at the 7.6 selection (n 2,000, refusal 35%, 8 epochs = 312 steps), with held-out bits/byte every
  epoch. Full table in `docs/08-raft.md` §13.
- **Forgetting is transient.** In every arm, bits/byte peaks at epochs 2–3 and recovers as the cosine LR
  anneals:
  - LR 2e-4 λ 0.25: peak +0.056, end +0.029;
  - λ 0.5: +0.046 / +0.027;
  - λ 1.0: +0.043 / +0.026;
  - LR 5e-5 λ 0.5: +0.010 / +0.006.
- **LR is the lever for bits/byte; λ barely matters.** CaseHOLD PMI at the best epoch: −4.9 / −3.7 /
  **−1.1** / −4.8 points. Only λ 1.0 is within the −2 limit; it isn't explained by bits/byte.
- **Copy-constrained decoding on the full 499-item dev set:**
  - score −0.49…−0.52 → −0.03…+0.09;
  - grounded 0.05–0.07 → 0.21–0.25;
  - hallucination 0.41–0.52 → 0.08–0.13.
  - Confirms the E-27 80-item result.
- **The pre-registered projection was invalid (C-12).** Its mechanical pick (LR 2e-4 / λ 1.0) measures
  +0.026 at the end of the run, so it isn't adopted. No arm passes both guards on measured numbers, so the
  decision goes back to the user.
- **Reproducibility:** the λ 0.25 arm's epoch 7 (+0.0315) matches 7.6's guard value (+0.0314). The
  best-epoch pick moved (4 vs 5) because dev macro-F1 differs by < 0.03 between epochs.

### E-29 · Step 7.6c confirmation run (2026-10-18, $0.52)
- LR 5e-5, λ 1.0, n 2,000, 8 epochs, judged on the final epoch:
  - bits/byte **+0.0051** ✅ (peak +0.0075);
  - CaseHOLD PMI **−4.7 points** ❌;
  - copy score −0.104, grounded 0.193, hallucination 0.136.
- **CaseHOLD drops at once and doesn't track bits/byte:** −3.4 at step 39 (bits/byte +0.007), then
  −2.3 to −5.0 across epochs. 7.6b's −1.1 at λ 1.0 / LR 2e-4 looks like the favourable end of that
  scatter. No tested setting meets the CaseHOLD guard; LR 5e-5 meets the bits/byte guard.

### E-30 · Step 7.6d CaseHOLD PMI decomposition (2026-10-18, $0.125, 1 × L40S)
- Base vs `best_pilot.pt` (7.6: LR 2e-4, λ 0.25) on all 3,600 test items. PMI accuracy:
  - base 0.599, tuned 0.555 (−4.4); paired flips 455 vs 297 (z ≈ 5.8);
  - swapping one term at a time: tuned cond + base null 0.573, base cond + tuned null 0.535.
- **Margin changes** (correct minus the mean of the others):
  - cond **−0.32 ± 0.06** nats;
  - null **−0.26 ± 0.09**;
  - PMI −0.06 ± 0.08, not significant.
- **Neither a pure prior shift nor a loss of context use.** Both terms fell about equally, and the
  context-attributable margin is unchanged on average. The accuracy loss comes from noisier per-item
  scores.
- Plain accuracy is unchanged (0.121 → 0.119).
- Caveat: these are LR 2e-4 weights; 7.6c's LR 5e-5 weights weren't saved.

### E-31 · Step 7.7 pilot evaluation with the J1 judge (2026-10-21, ≈ $0.61)
- **Retrain of the chosen setting reproduces 7.6c:** macro-F1 0.842, bits/byte +0.0052. τ = 0.75 by the
  pre-registered rule, which meets both decision targets (over-refusal 0.105, recall 0.815).
- **Copy-decoding format fixes** (forced `\nAnswer:` scaffold, forced close, 240-char cap): format 0.966 → 0.998,
  quote fidelity 0.841 → 0.993.
- **J1 on dev, serving mode (`copy2`):**

  | Metric | Value | Target | |
  |---|---|---|---|
  | Grounded accuracy | **0.215** | 0.60 | ❌ |
  | Hallucination | **0.349** | 0.15 | ❌ |
  | Citation | 0.780 | 0.80 | ❌ (close) |
  | Refusal recall | 0.815 | | ✅ |
  | Over-refusal | 0.109 | | ✅ |
  | Quote fidelity | 0.993 | | ✅ |
  | Format | 0.998 | | ✅ |

- **The bottleneck is reading within the passage.** The model cites the oracle 78% of the time but is
  correct on only 31% of those.
- **Composition of the 34.9% hallucination** (shares of all dev items): unsupported A 17.2%,
  unsupported C 12.6% (63 of 100 grounded tasks), answered unanswerable 4.6%.
- **The proxy matches J1 under copy decoding** (55/8/4/178).
- **Projection:** 10K → grounded accuracy ≈ 0.30–0.45, so the 0.60 target likely missed.

### E-32 · Review before 7.8: the pilot memorises (2026-10-21, $0)
- **7.7a dev target loss** bottoms at epoch 2 (0.482) and rises to 0.592 by epoch 8; decision macro-F1 peaks late.
- **On 40 of its own training A items:** proxy grounded **0.95** and oracle citation 40/40, versus 0.23 and 78% on dev.
  This is a generalisation gap, so more unique data is the lever.
- **J1 spot-check:** 15/15 verdicts agree with a manual reading.

### E-33 · 7.8 sources: SEC has too few eligible filings for one passage each (2026-10-21)
- One source passage per document gave SEC 1,804 of 6,680 (case law and web were full).
- Fix: up to 4 passages per SEC filing, which gives 6,680 from 1,757 filings. The mix stays 40/40/20,
  with 0 overlap with trial/pilot sources.

### E-34 · 7.8 selection: probe documents in the training data (2026-10-21)
- **Cause.** Source exclusion was by passage, so other passages of the probe's train documents entered
  training: as BM25 distractors (no gradient) and in 562 A/C targets (57 pilot).
- **Fix.** Rows whose cited passage comes from a probe document are excluded and replaced. 0 probe documents
  are cited, 0 dev/test documents are used. The 7.7 pilot model also trained on 57 such rows; its probe was
  never evaluated.

### E-35 · Step 7.8: 20K data, learning curve, final test (2026-10-21; API $6.26, GPU $1.82)
- **Data:** 20,340 train rows (audit OK).
- **Judged dev curve** (grounded accuracy): 2K 0.215, 5K 0.273, 10K 0.287, 20K 0.349, about +0.045 per doubling.
- **Final test** (20K, τ 0.7, copy2):

  | Metric | Value | Target | |
  |---|---|---|---|
  | Grounded accuracy | **0.316** | 0.60 | ❌ |
  | Hallucination | **0.325** | 0.15 | ❌ |
  | Refusal recall | **0.780** | 0.80 | ❌ |
  | Over-refusal | 0.048 | | ✅ |
  | Citation | 0.903 | | ✅ |
  | Quote fidelity | 0.998 | | ✅ |
  | Format | 1.000 | | ✅ |
  | AUROC | 0.923 | reported | |

- **Hallucination** = unsupported A 15.3% + unsupported C 10.6% + answered B2 4.4% + answered B1 2.2%.
  B2 refusal recall is only 0.55–0.61.
- **Forgetting:** bits/byte +0.009 ✅; CaseHOLD PMI −4.8 (reported).
- **Teacher ceiling** 1.00 / 0.01 / 0.96 (inflated by construction).
- **Conclusion:** the gap is the 125M student's reading. Reaching 0.60 by data alone would take on the order of 1M examples.

### E-36 · 7.8b answer-copy decoding on dev (2026-10-21; $0.26)
- **Effect:** `copy3` vs `copy2` on the 20K weights: hallucination 0.309 → 0.299, grounded 0.349 → 0.367; 64 of 499 outputs changed.
- **Not adopted.** The pre-registered bar was −0.03 hallucination. The unsupported answers are already spans: wrong spans.

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
| D-17 | 10-16 | Phase 7 review fixes (user-approved patch): SFT and replay losses normalised separately (λ = 0.25, per-token within SFT); rows 4–6 sampled from N(μ, Σ); dev/test leakage scanned against the whole pretraining corpus; paired A/B1 with same-document distractors; C-bucket placeholder and V2 check; DeepSeek thinking explicitly off, with `max_tokens` caps; per-bucket audit fallback (≤ $3); pilot compiled once | E-19. Net cost ≈ $0; estimate stays ≈ $20 |
| D-18 | 10-16 | Keep the 10% leak rule; build the dev/test pool only from leak-scanned passages (SEC per-document cap 3, numeric quota filled first); record the SEC BM25 top-1 check as failed-and-explained (top-10 recall 99.6%) | E-20: relaxing the rule would let near-copies of training filings into the test; enough clean SEC passages exist |
| D-19 | 10-16 | After the 7.3 trial: DeepSeek stays the writer; DeepSeek checks run **without thinking** (`TEACHER.think_on_refusal_checks = False`); answer agreement is symmetric containment | Pre-registered rules (E-22). Thinking's extra rejections were mostly "ambiguous" flags, and Gemini sided with thinking-off on 10 of 16 |
| D-20 | 10-16 | Dev/test near misses must pass a manual read under the sharpened rule (drop if stated/computable/partial/underspecified; keep different-item hard negatives); dev accepts a 1-item B2 shortfall rather than lowering the bar | E-25. The user chose the $0 full read over accepting ~5–10% noise in the numbers that define results |
| D-21 | 10-17 | Pilot selection LR 2e-4 / refusal 35% (pre-registered rule: best selection score). Report every proxy and the final test under **both** plain and copy-constrained decoding; recommend copy-constrained for serving. τ 0.7 is provisional until 7.7 | E-27: copying is the dominant failure (score −0.550 → −0.137 on 80 dev items, decisions unchanged); the constraint costs nothing at training time |
| D-22 | 10-21 | Move Modal work to the new account `deepanshu-m23csai` (user's instruction); copy the Volume volume to volume inside Modal (`migrate_app.py`, one worker per file), skipping `clean/` and `checkpoints/calibration/` | The old account's credits are ending. A Modal-side copy avoids routing 16 GiB through the laptop. 184/184 files verified by size, weights by SHA-256 |
| D-23 | 10-21 | (User's choice, option 2) 7.8 uses LR 5e-5 / λ 1.0 and keeps the final epoch. Bits/byte stays a gate; CaseHOLD PMI is reported, not gated, and its −2 point limit is recorded as missed (−4.7) | E-28–E-30: LR 5e-5 holds bits/byte (+0.005); the CaseHOLD drop appears in epoch 1 whatever the setting, and the mean context-attributable margin is unchanged |
| D-24 | 10-21 | Serving API redeployed on the new account: `https://deepanshu-m23csai--slm-125m-serve-api.us-east.modal.run`; `config.API_URL` and the website updated, Vercel redeployed | The user's instruction (old account's credits ending). All endpoints and CORS verified from the public URL; the live site's bundle references only the new URL |
| D-25 | 10-21 | 7.7 as planned in `docs/08-raft.md` §13. Retrain the chosen pilot setting once and save it. τ = the highest dev macro-F1 with over-refusal(A) ≤ 0.20 and recall(B) ≥ 0.80. J1 (Gemini) on both decodings (new cap `7.7-judge` $1, pending approval). `config.SFT` now carries D-23 (LR 5e-5, λ 1.0, keep final, refusal 35%); the 7.6 entrypoints pin their original recipe | Review before 7.7: the judged grounded accuracy is unknown (proxy 0.193 vs target 0.60), and nothing from the chosen setting was saved |
| D-26 | 10-21 | Serving/eval decoding = `copy2` (copy-constrained with forced scaffold, forced close, 240-char cap), τ = 0.75 for the pilot weights (7.8 re-picks τ by the same rule on its own dev). Recommendation: go to 7.8 | E-31: format 0.998 and quote fidelity 0.993; τ rule met. Data is the only evidenced lever, and it's cheap (≈ $0.25 per 1K) |
| D-27 | 10-21 | (User) 7.8 at **20K** training examples; checkpoint selected by the dev proxy score under `copy2` among evals with bits/byte ≤ +0.02; learning curve 5K/10K/20K judged by J1 | E-32: the pilot memorises (train 0.95 vs dev 0.23), and content overfits from epoch 3 while decision-F1 rises; data ≈ $0.25 per 1K |

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
| C-10 | SEC held-out 0.463 bits/byte read as generalisation | Optimistic: 61% of SEC val passages share ≥ 50% of their 13-grams with training filings | E-20 |
| C-11 | Trial 7.3 first scoring: Gemini ≥ 10 points better as writer; checker agreement 82.4% | DeepSeek 64.0% vs Gemini 71.4% (keep DeepSeek); agreement 90.0% | `answers_agree` used token-F1. 55 of 60 "disagreements" had the same passage and quote with a verbose answer (E-22) |
| C-12 | 7.6b rule: forgetting grows with steps, so extrapolate the per-epoch bits/byte linearly to 7.8's 579 steps | Forgetting peaks at epochs 2–3 and recovers as the cosine LR anneals; the linear fit extrapolated the recovery | The rule was written before seeing any per-epoch data. Its pick isn't adopted; measured end-of-run numbers are used (E-28) |
| C-13 | 7.7 is "$0" (PROJECT_STATE and two messages, 10-18/10-21) | 7.7 includes the J1 judge on the pilot candidate (≈ $0.50 for both decodings) plus a ≈ $0.45 GPU retrain | I misread the plan's step table (§5 says ≈ $0.40); caught in the review before 7.7 |
| C-14 | (a) Keep the best-decision-F1 / final epoch (§7.1, D-23); (b) 10K → grounded 0.30–0.45 (E-31) | (a) Select on the dev proxy score under `copy2` (content + decisions); (b) no judged learning curve exists yet, so no projection | (a) Answer content overfits from epoch 3 while decision-F1 still rises (E-32). (b) The projection used the plain-decoding proxy, which measured invented quotes |
| C-15 | 7.8b regeneration capped at `--max-usd 0.15` | It cost $0.245; the 7.8 GPU stage is $2.07, $0.07 over its $2 cap | The guard checked only before generation, and the answer constraint's fallback is slow. Fixed: the budget is re-checked per 64-row chunk |
