# Phase 7: Grounded SFT / RAFT (plan)

**Status: PLANNED, not executed.**

- **Written 2026-10-06; revised 2026-10-16** to the low-cost setup the user chose: DeepSeek Flash
  as the teacher, Gemini Flash as the independent checker, 10,000 training examples, ≈ $20.
- **Technical review 2026-10-16** (before step 7.0) fixed six things. (1) Replay was getting 84%
  of the loss weight. (2) The chat-token init didn't separate rows 4–6. (3) Leakage was checked
  against the wrong reference set. (4) The C bucket had no way to name its passage and no checks.
  (5) DeepSeek's thinking is on by default. (6) The audit fallback broke the budget. Details are in
  §1, §3.2, §4, §5, §7 and §8.
- Every paid step below waits for explicit approval. Results get recorded in §13 as each step runs.
- Brief: `instruction.txt` (the user's draft). Earlier recommendation: `EXPERIMENTS.md` D-15.
  Cost comparison behind this revision: `EXPERIMENTS.md` E-18.

## 0. Summary

**Goal.** Teach the 125.8M base model to answer **only** from passages it is given. It should cite
which passage and quote it, and it should **refuse** when the passages don't hold the answer.

**Approach.** Direct RAFT on teacher-written, quote-grounded targets; no closed-book distillation
first. Every training example puts 4 passages in the prompt. In answerable examples one passage holds
the answer and 3 are hard distractors. In refusal examples none holds it, and the target is a fixed
refusal string.

| | Plan |
|---|---|
| Teacher | **`deepseek-flash`** (DeepSeek-V4.1-Flash), run **off-peak**. It writes every training target and blindly checks every candidate. Thinking is **off**, except when checking refusal labels |
| Independent checker | **`gemini-3.8-flash`** (Batch API), a different model family. It co-labels dev/test, audits 5% of training labels, and judges the student's answers. It never writes training targets |
| Dataset | **10,000 training examples**: 55% answerable QA, 25% unanswerable → refusal, 20% grounded non-QA tasks. Plus **pretraining replay** (weighted to ~20% of the loss) against forgetting. Dev 500 and test 1,000, built only from held-out documents. If the test targets are missed, generate more data and retrain; that's incremental, not a restart |
| Cost levers adopted | (1) DeepSeek off-peak. (2) 10K examples. (3) Pre-filter the passages questions are written from. (5) Thinking only where labels are hard. **Not adopted:** packing more questions into one call (saves ~2% and hurts diversity) |
| Training | Full fine-tune from the fp32 Phase 5 checkpoint. LR 1e-4 (swept in the pilot), 3 epochs, loss on assistant tokens only, 1 × H100, a few minutes |
| Metrics | Hallucination rate, over-refusal rate, grounded accuracy, citation accuracy, verbatim-quote rate, and forgetting (held-out bits/byte, CaseHOLD) |
| Cost | **≈ $20 in total** (API ≈ $17, GPU ≈ $3). Stage caps sum to $31, and each stage is approved separately |

## 1. Starting point (review findings that shape this phase)

| Fact | Consequence |
|---|---|
| Context is **1,024 tokens**; the tokenizer averages 4.67 bytes/token on prose but **3.3 chars/token** on QA-style text full of numbers (E-17) | k = 4 passages of ≤ 160 tokens each (≈ 650 tokens), question ≤ 48, answer ≤ 128. Anything over 1,024 is dropped, never truncated |
| Chat tokens `<\|user\|>` 4, `<\|assistant\|>` 5, `<\|system\|>` 6 **never occurred in training**. Measured 2026-10-16 on the released weights: rows 0 and 2–6 have **collapsed onto one shared direction** (pairwise cosine ≈ 0.95; cosine 0.88 with the mean embedding; norm ≈ 1.1, like trained rows). As outputs they sit at log-prob ≈ −25, against ≈ −18 for a typical token | As inputs, `<\|user\|>` and `<\|assistant\|>` are nearly the same vector, and "mean + small noise" would keep them that way. Initialise rows 4–6 by **sampling from N(μ, Σ) of the trained rows** (distinct, in-distribution). The smoke test checks pairwise cosine < 0.5 and that their output log-prob stays ≤ the median trained token. Keep suppressing ids 0 and 2–6 at decode time |
| Prompts must start with `<\|eos\|>` (D-07) | Every SFT sequence starts with id 1 |
| The corpus is one document per line in `/data/corpus/<source>/*.txt`; the val split is `blake2b(text, 8 bytes) % 100 == 0` (6,638 documents never trained on) | **Train examples come only from train documents; dev/test only from val documents.** The test then measures grounding on text the model never saw. Reuse `modal_app._is_val_doc`'s exact rule |
| The near-duplicate count is a floor (LSH catches 29% at J = 0.8), and exact dedup ran only within each shard. The model saw every train document 4 times | Near-duplicates of val documents can sit in the **pretraining** train split, so checking dev/test only against SFT-train passages is not enough. Step 7.2 scans the **whole train corpus**: a fan-out (one worker per corpus shard) builds the small dev/test 13-gram set from blake2b hashes and streams its shard. Passages with ≥ 10% of their 13-grams found in train are dropped, and the overlap distribution is reported. ≈ $0.10 |
| `/data/checkpoints/ckpt.pt` holds the **fp32** final weights; `base/` is a bf16 export | SFT initialises from `ckpt.pt`, never from the bf16 export |
| `train.py` already has autocast, the optimizer builder (no decay on norms), `lr_at`, atomic saves, the spend ledger, `export_hf` and the evaluator | `sft.py` imports them rather than duplicating them, with four exceptions. (1) The loss: `train.py` uses HF's per-micro-batch mean ÷ accumulation steps, and SFT needs its own normaliser (§7.1). (2) `evaluate()` takes the *first* N val windows; val shards sort as case-law, fineweb-edu, sec, so a capped evaluation is all case law. It also uses the pooled bits/byte constant. The forgetting guard needs explicit per-source index lists and per-source bytes/token (C-05). (3) `export_hf` writes a sampling `generation_config` (`do_sample`, `top_k` 50, `min_new_tokens` 16). The SFT export must overwrite it with greedy decoding and `min_new_tokens` 0. (4) Load only `state["model"]` from `ckpt.pt`, never its optimizer state |
| `.env.local` has `GEMINI_API_KEY` (verified working) but **no `DEEPSEEK_API_KEY`** | The user adds `DEEPSEEK_API_KEY` before step 7.3. No Anthropic key is needed |
| `requirements.txt` has no `google-genai`, `openai` (DeepSeek's API is OpenAI-compatible), or `bm25s` | Add them, pinned, when implementing |
| No CaseHOLD evaluator exists | Write `eval_casehold.py` (§8.4) |

## 2. Teachers (brief task 1)

The brief's candidates are dated: Claude 3.5 Haiku and 3.5 Sonnet are retired, and GPT-4o-mini and
Llama 3.1 have been superseded. Current prices, per million tokens, checked 2026-10-06 to 2026-10-15:

| Model | In / out | Notes |
|---|---|---|
| **`deepseek-flash`** (V4.1-Flash) | **off-peak** $0.15 / $0.60; cache hits $0.003. **Peak** is double | Thinking switchable per request. No batch API. Peak is 01:00–04:00 and 06:00–10:00 UTC, Mon–Fri. **Terms explicitly allow training other models on outputs** |
| `deepseek-v4-pro` | off-peak $0.66 / $1.98 | As expensive as Gemini Flash with no quality need here; not used |
| **`gemini-3.8-flash`** | batch $0.375 / $1.875 | The price is listed "through Dec 31, 2026". The key already works. Used as the checker only |
| `claude-sonnet-5` | batch $1 / $5 | Was the judge in the first draft. Swapped for Gemini, saving ~$8 and removing the need for an Anthropic key |
| `gpt-5-mini` | batch $0.125 / $1 | No OpenAI key; not needed |
| Llama 3.1 | self-hosted | Rejected: its license requires derived models to carry "Llama" in their name |

**Why this split:**
- **Writing targets is the easy part.** A question about one passage plus a copied quote is well
  within a current flash-tier model. Its mistakes are mostly *mechanical* (a quote that isn't
  verbatim, a copied question, a changed number), and §5.2's deterministic checks catch those. The
  cheapest capable model is the right writer.
- **What decides quality is the label on hard cases:** does any of the 4 passages actually answer the
  question? A wrong "unanswerable" label teaches **over-refusal**; a wrong "answerable" label teaches
  **hallucination**. Four safeguards cover this:
  - Every candidate gets a **blind** check. The checker sees only the 4 passages and the question,
    never the intended answer.
  - DeepSeek **thinks** on refusal-label checks (buckets B1/B2), where reasoning pays.
  - **Dev/test labels**, which define every reported number, are kept only where DeepSeek and Gemini
    **independently agree**.
  - A **5% audit** of training labels by Gemini, **stratified to oversample B1/B2** (200 A, 150 B1,
    100 B2, 50 C). The rule applies **per bucket**: any bucket with > 5% disagreement has its
    remaining labels re-checked by Gemini. B1 + B2 together is ≈ $3, inside the $12 stage cap, and a
    full re-check is never needed.
- **About refusal-constraint adherence** (the brief asks specifically): the student's refusal comes
  from the *labels*, not from the teacher refusing. Teachers are never asked to refuse; they judge
  answerability as structured JSON, and that's what the trial measures.
- **Caveat:** DeepSeek-V4.1-Flash and Gemini 3.8 Flash were released after my training data, so I
  can't vouch for them from experience. **Step 7.3, a ≈ $1.50 trial**, measures them before any
  real spend, with decision rules fixed in advance (§5.3).
- **Terms and data:**
  - DeepSeek's terms explicitly permit "training other models (such as model distillation)" and
    assign outputs to the user. All training targets come from DeepSeek.
  - Gemini's terms bar developing competing models. Here Gemini only labels and grades; the user
    should still read its terms before publishing anything.
  - DeepSeek processes requests in China. Only public corpus text (court opinions, SEC filings,
    educational web pages) is ever sent to it.
  - Use the **paid** Gemini tier: free-tier prompts may be used by Google, and its rate limits are too
    low.

## 3. Dataset architecture and sizing (brief task 2)

### 3.1 Size: 10,000 training examples, topped up only if the test demands it

- A 125M model has to learn a new turn format, span copying across distracting context, and a
  calibrated answer/refuse decision. 10K examples is enough to learn those, and it is the cheapest
  size the pilot can justify.
- **The pilot learning curve (7.6)** trains at 500, 1,000 and 2,000 examples. "Flat" means the gain
  from 1K → 2K is within dev noise (about ±3–4 points on 500 items).
  - **Flat:** 10K is ample.
  - **Still climbing clearly:** 10K is still the first step. After the final test, if targets are
    missed and the curve suggests data will help, generate more (≈ $0.85 per extra 1,000 kept
    examples) and retrain (< $1 GPU), with approval.
- **Dev 500 / test 1,000**, from val documents only. They are built once and frozen before any training.

### 3.2 Composition

| Bucket | Share | Count (of 10K) | What the example contains | Target |
|---|---|---|---|---|
| **A. Answerable QA** | **55%** | 5,500 | oracle + 3 hard distractors (§4.1 step 4: up to 2 non-adjacent chunks of the oracle's own document, the rest from BM25); oracle position uniform over 1–4 | source, verbatim quote, short answer |
| **B1. Unanswerable, oracle removed** | 15% | 1,500 | **the same question as an accepted A example** (a contrastive pair), oracle removed, 4 hard distractors by the same rule. No new generation call | refusal |
| **B2. Unanswerable, near miss** | 7.5% | 750 | the question's topical passage *is present* but lacks the asked detail (e.g. asks the 2021 figure when the passage gives 2020), plus 3 distractors | refusal |
| **B3. No usable context** | 2.5% | 250 | the question with 0 passages, or 4 passages from an unrelated source | refusal |
| **C. Grounded tasks** | **20%** | 2,000 | 4 passages; an instruction about one of them, which names it as `Passage {k}` (filled in after shuffling): one-sentence summary, extract a list (parties, dates, amounts), or yes/no check with a quote (**about half "No"**) | the response, citing its source |
| **D. Replay** (outside the 10K) | 1 window per 4 SFT sequences | — | raw 1,024-token windows from the pretraining **train** split | LM loss, **normalised separately and weighted λ = 0.25** (§7.1) |

- **Source mix:** case law 40 / SEC 40 / web 20, matching the pretraining mix. Questions and passages
  for a given example come from the same source.

**Why these ratios:**
- **25% refusal (B1 + B2 + B3)** sits inside both reference points: SQuAD 2.0 is ~1/3 unanswerable,
  and RAFT found it best to keep the oracle in 60–80% of training examples.
  - More refusals push toward **over-refusal syndrome**; fewer leave **hallucination**.
  - It is treated as a hyperparameter: the pilot trains at **15 / 25 / 35%** and picks the share
    with the best balance on dev (§7.2).
- **B2 matters most.** Without near misses, the model learns a lexical shortcut, "refuse when no
  passage shares words with the question". B2 forces it to read whether the specific detail is there.
- **Paired A/B1 and entity-matched negatives close the second shortcut.** G1 makes every question
  name its entity. If distractors never mention that entity, "the named entity appears nowhere →
  refuse" solves B1 without reading. Three measures counter that:
  - B1 reuses accepted A questions, so the same question appears with and without its answer.
  - A and B1 contexts include same-document chunks that mention the entity but not the answer.
  - Step 7.4 reports the share of B1 examples where the question's entity appears in some passage
    (target ≥ 50%), and §8.1 reports refusal recall split by entity present vs absent.
- **Pretraining memory is part of the lesson.** Train passages were seen 4 times in pretraining, so
  for B1 the model may *know* the answer. The target still refuses, which directly trains the
  behaviour this phase wants: no answering from parametric memory.
- **C is the "general instruction" bucket, deliberately grounded.** Open-domain instruction data
  (chat answered from memory) teaches exactly the behaviour this phase is trying to remove, and a 125M
  model holds too little world knowledge to answer from memory reliably (E-13: it invents facts).
  Grounded tasks still give the format diversity general data normally provides. The **"No" answers
  in C** teach the difference between "the passage says no" and "the passage doesn't say", which is
  the other main cause of over-refusal.
- **D (replay) is the forgetting guard.** Mixing in a slice of the original LM objective is the
  standard fix for catastrophic forgetting. Its effect is measured with held-out bits/byte against
  the 0.6572 baseline.

## 4. Data format (brief task 3, part 2)

Role boundaries use the special tokens; field labels are plain text the base model already
handles. There is no system turn: the behaviour is fixed, and the ~25 tokens are better spent on
passages.

```
<|eos|><|user|>Passage 1: {p1}
Passage 2: {p2}
Passage 3: {p3}
Passage 4: {p4}
Question: {question}<|assistant|>Source: Passage {k}
Quote: "{verbatim span ≤ 40 words}"
Answer: {answer ≤ 30 words}<|eos|>
```

Refusal (B1/B2/B3). The text is a single canonical string, stored as `config.RAFT.refusal`:

```
<|assistant|>Source: none
Answer: The passages do not contain the answer.<|eos|>
```

Grounded task (C) uses `Task: {instruction}` in place of `Question:`, and the target is
`Source: Passage {k}` then `Answer: {response}`. Yes/no tasks also carry `Quote:`.

**Design choices:**
- **One token position carries the decision.** Measured tokenisation:
  - `Source:` = `S` `ource` `:` (3 tokens);
  - ` none` = **`Ġnone`** (id 5494), one token;
  - ` Passage 3` = **`ĠPass`** (id 12175) `age` `Ġ3`.

  So the decision is `Ġnone` vs `ĠPass` at the position after `:`.
  - **Training:** `Source:` is part of the prompt (masked), so the first *target* token is the
    decision.
  - **Inference:** prefill `Source:`, compute s = P(`Ġnone`) / (P(`Ġnone`) + P(`ĠPass`)), and
    refuse iff s ≥ τ. After `ĠPass`, force `age` and restrict the next token to the passage numbers
    present (`Ġ1`–`Ġk`). Then decode greedily. Log the probability mass outside the two decision
    tokens as a format-drift signal, and report unconstrained format validity too.
  - This gives (a) a **calibrated** answer/refuse classifier, with τ chosen on dev; (b) the AUROC
    metric (§8); and (c) citations that are always well-formed.
- Field labels at line starts fragment (`Answer` = `An` `sw` `er`, `Quote` = `Qu` `ote`, `Passage`
  in the prompt = `P` `ass` `age`), costing ~15 tokens per example. That's acceptable; SFT learns the
  format. Segment-wise encoding at special-token boundaries equals whole-string encoding (verified),
  so the loss mask can be built per segment safely.
- **Quote before answer.** The model copies a span and then answers from it. That's the RAFT
  chain-of-thought idea cut down to a copy step, which small models are relatively good at
  (induction heads). Quotes can also be **checked mechanically** at evaluation.

### 4.1 Preprocessing

1. Chunk each document into sentence-packed passages of 60–160 tokens (our tokenizer).
2. **Two passage pools:**
   - **Source pool** (passages questions are written from) is **pre-filtered**. Drop passages with
     alphabetic ratio < 0.6, boilerplate (legal-notice and cover-page patterns), citation lists, and
     OCR noise. Keep **~10% table- or number-heavy SEC passages** on purpose, so numeric questions
     survive. This raises the keep rate (assumed 65% → 80%; the trial measures it).
   - **Distractor pool** is **unfiltered**, so the model still sees tables, citation lists and messy
     text in its context and doesn't become brittle.
3. Exact-dedupe passages; 13-gram filter test against train (§1).
4. **Distractors** (same source as the oracle), identical rules for A, B1 and B2:
   - up to **2 chunks of the oracle's own document**, excluding its neighbours (±1). These mention
     the same parties but usually not the asked detail;
   - the rest from **BM25** (query = the question) over other documents: sample from the top 20, at
     most one passage per document;
   - never a passage with word-5-gram Jaccard > 0.5 against the oracle (restatements);
   - the blind check (§5.4) still decides.
   - **Pool sizes:** chunking the whole train corpus gives ~15M passages, too many for an in-memory
     BM25. The train **distractor pool** is ~250K passages per source, sampled by document. The val
     pool is every val passage (~150K).
   - The **source pool** only needs the ~7K train + ~2K dev/test passages that questions are written
     from.
5. Shuffle passage order so the oracle is uniform over positions 1–4.
6. Tokenise, **drop** anything over 1,024 tokens, and right-pad to exactly 1,024. The fixed shape
   means `torch.compile` compiles once.
7. Labels: `label[t] = input[t]` for the target span, from the first token after `Source:` through
   the final `<|eos|>`, and −100 everywhere else (prompt, `<|assistant|>`, `Source:`, padding). HF
   shifts labels internally, so this trains exactly the target tokens. Right padding plus causal
   attention means training needs **no attention mask** (the SDPA causal fast path).
8. Store as `uint16` input ids plus a loss mask and a per-sequence kind (SFT or replay), with a JSONL
   sidecar holding the raw fields for evaluation.
9. **Dev vs test split by document** within the val documents: `blake2b(text + "|dev") % 3 == 0` →
   dev, else test. No document contributes to both.

## 5. Execution steps

The Volume layout is `/data/raft/{passages,bm25,teacher,datasets/{pilot,full},runs}`. Each step
records its actual results in §13.

| Step | What | Where | Cost | Gate |
|---|---|---|---|---|
| 7.0 | This plan; add `RAFT`/`SFT` constants to `config.py` | local | $0 | the user approves the plan |
| 7.1 | **Baselines, sanity run**: CaseHOLD zero-shot accuracy; held-out bits/byte on a fixed, seeded 600-window subset (200 per source, each source with its own bytes/token). The **official** baselines for the forgetting guards are re-measured on the init weights inside the pilot GPU container, so before and after use the same numerics (bf16 autocast) | local (MPS/CPU) | $0 | — |
| 7.2 | **Passage pools + BM25 + leakage scan**: fan-out, one worker per corpus shard → source and distractor pools; BM25 index per split and source; dev/test 13-gram scan against the whole train corpus (§1) | Modal CPU | ≈ $0.2 | — |
| 7.3 | **Trial**: 150 source passages; both models generate; every candidate checked three ways (DeepSeek thinking on, thinking off, Gemini) | local → APIs | ≈ $1.50, cap **$3** | **spend approval** + `DEEPSEEK_API_KEY` |
| 7.4 | **Pilot data**: 2,400 kept train examples (enough for 15/25/35% refusal mixes of 2,000: target 1,247 A / 453 C / 420 B1 / 210 B2 / 70 B3), dev 500, test 1,000 | local → APIs | ≈ $7.50, cap **$10** | **spend approval** |
| 7.5 | **Local SFT smoke test** (`smoke_sft.py`) | local CPU | $0 | — |
| 7.6 | **Pilot sweep**: 3 LRs × 3 refusal shares at 2,000 examples, plus 500- and 1,000-example curve points = 11 short runs in one container, **compiled once** (§7.1); official baselines measured first | 1 × H100 | ≈ $1.50, cap **$4** | **GPU gate** |
| 7.7 | Evaluate the pilot on dev → pick LR, refusal share and τ; read the learning curve | local / Modal CPU | ≈ $0.40 (Gemini judge) | — |
| 7.8 | **Scale to 10K**: generate the remaining 7,600; 5% Gemini audit; full SFT; final test evaluation | APIs + 1 × H100 | API ≈ $7.90, cap **$12**; GPU ≈ $0.50, cap **$2** | **spend approval + GPU gate** |
| 7.9 | Record results; export `/data/raft/runs/final/hf/`; update MASTER_PLAN / PROJECT_STATE / EXPERIMENTS | local | $0 | — |

Publishing the SFT model to HF, adding a grounded endpoint, and a playground mode on the website
are **not in this phase**. They're a proposed Phase 8, and each is outward-facing, so each gets
confirmed first.

### 5.1 Validation per step (a failed check stops the phase)

- **7.1:**
  - The bits/byte on the subset is reproducible (two runs identical).
  - CaseHOLD accuracy is reported with its 95% CI against 20% chance.
- **7.2:**
  - Every passage's document hash agrees with its split.
  - 0 val passages in either train pool.
  - The token-length histogram is inside 60–160.
  - Filter drop rates are reported per source, and table-heavy SEC passages are ≈ 10% of the source pool.
  - BM25 sanity: querying with a sentence from a passage returns that passage top-1 for ≥ 95% of
    500 probes.
- **7.3 / 7.4 / 7.8:**
  - Mechanical-check pass rates and keep rates are reported per bucket and source.
  - The spend ledger is ≤ the stage cap.
  - Every DeepSeek call is timestamped, to confirm off-peak.
  - The user spot-checks 50 random dev/test items.
  - Entity-present share in B1 ≥ 50% (§3.2), and 0 documents shared between dev and test.
- **7.5:**
  - The loss mask covers exactly the target span after `Source:` plus EOS.
  - A unit test reproduces the hand-computed λ-weighted loss on a 2-example batch.
  - Chat rows 4–6 are re-initialised (pairwise cosine < 0.5) and the other rows are untouched.
  - 32 examples overfit to train loss < 0.05.
  - Kill-and-resume is bit-identical.
  - Generations parse.
  - No sequence exceeds 1,024.
- **7.6 / 7.8:**
  - Loss is finite.
  - Dev decision metrics (one forward pass, no generation) are logged every epoch in the pilot and
    every 50 steps in the full run.
  - The budget guard holds.
  - `modal container list` → 0 afterwards.

### 5.2 Mechanical checks on teacher output (model-free, all buckets)

1. Output is valid JSON matching the schema. DeepSeek is called in JSON mode; anything that fails to
   parse or validate is discarded, never repaired.
2. The `quote` is a **verbatim substring** of the source passage, ≤ 40 words. Matching
   canonicalises both sides (NFKC; curly→straight quotes; dash and ellipsis variants; whitespace).
   The stored quote is then **re-cut from the passage at the matched offsets**, so the student's
   target is an exact copy (the tokenizer has no normaliser; verified).
3. Every token containing a digit in `answer` appears verbatim in the passage. Capitalised words
   other than the first word and Yes/No appear case-insensitively (catches numeric and name
   hallucination without rejecting "The …" or "No.").
4. The question is 5–40 words and shares no run of ≥ 6 consecutive words with the passage, **not
   counting capitalised name spans** (G1 rule 2 requires names, and case captions or company names
   can be 6+ words).
5. The answer is ≤ 30 words, and for factoid types ≥ 50% of its content words appear in the quote.
6. No question is duplicated within or across splits (normalised exact match).
7. The fully formatted example is ≤ 1,024 tokens.
8. A-answers that hedge ("not stated", "does not specify", "unclear") are dropped.
9. C extract_list: every listed item appears verbatim in the passage. C yes_no_check: the quote
   passes check 2.

### 5.3 Trial decision rules (fixed before the trial runs)

| Question | Measured on the trial | Rule |
|---|---|---|
| Is DeepSeek a good enough writer? | keep rate (mechanical + blind check) for DeepSeek vs Gemini as generator | DeepSeek stays unless Gemini's keep rate is **≥ 10 points higher**; otherwise re-quote and ask |
| Does thinking help the refusal labels? | agreement with Gemini on B1/B2 candidates, DeepSeek thinking on vs off | if thinking-off agreement is **within 2 points**, turn thinking off everywhere (saves ≈ $2.60 more on the ~12,500 training candidates) |
| Are the cost assumptions right? | real tokens per call, keep rate, thinking length | re-quote every later stage cap from measured numbers before 7.4 |
| Does thinking stay off when asked? | reasoning tokens reported on no-think calls | any > 0 aborts the run (thinking is DeepSeek's default and would roughly triple the cost) |
| Does JSON mode work with thinking on? | V1 parse rate with thinking | < 98% → plain-text replies with strict parsing for those calls |
| Are the two checkers compatible? | DeepSeek-vs-Gemini agreement over all candidates | if below **85%**, inspect disagreements before going on: the prompt, not the model, is usually the problem |

### 5.4 Acceptance after the blind check

| Bucket | Keep only if the checker says |
|---|---|
| A | `answerable` true, `support == [oracle]` (unique), not `ambiguous`, the checker's `quote` is verbatim in the oracle **and overlaps the generator's quote span**, and its answer agrees with the generator's (identical numbers/names, or token-F1 ≥ 0.5) |
| B1, B2 | `answerable` false and not `ambiguous` (DeepSeek thinking on) |
| B3 | no check needed (no passages, or a different source) |
| C | extract_list: model-free (§5.2 check 9). yes_no_check: V2's blind Yes/No matches. summarize: V2 marks it `supported` and `on_task` |

- **Dev/test labels:** kept only when DeepSeek (thinking on for every dev/test check) **and** Gemini
  independently agree with the intended label. Disagreements are dropped. Expected yield is ~62% of
  candidates.

### 5.5 Running DeepSeek without a batch API

- Concurrent ordinary requests (OpenAI-compatible client), with a bounded worker pool, exponential
  backoff on 429/5xx, and an append-only JSONL of responses, so a crash resumes without re-paying.
- `teacher.py` **refuses to send during peak hours** (01:00–04:00 and 06:00–10:00 UTC, Mon–Fri)
  unless `--allow-peak` is passed. Peak doubles the price.
- The fixed instructions come first in every request and the passage last, so the instruction prefix
  is served from DeepSeek's cache ($0.003 per million tokens).
- The spend guard sums `usage` × price after every response and stops at `--max-usd`.
- **Thinking is DeepSeek's default.** Every no-think request disables it explicitly, and `teacher.py`
  aborts if a no-think response reports any reasoning tokens.
- **Every request has a `max_tokens` cap:** G1–G3 600; V1/V2 without thinking 250; with thinking
  2,500. That bounds the worst case.
- **Gemini batch usage is known only after completion**, so a batch is submitted only if its worst
  case fits the remaining budget (counted input tokens + `max_output_tokens` × price).

## 6. Teacher prompt templates (brief task 3, part 1)

One question type per call; prompts are deliberately **not** merged (§0). `{source_desc}` is one of
"US court opinion", "SEC filing" or "educational web page".

**G1: answerable QA (writer: DeepSeek, thinking off)**

```
You write training data for a small model that must answer ONLY from passages it is given.
You receive one passage from a {source_desc}. Write up to 2 question-answer pairs that this
passage alone answers.

Rules:
1. The answer must be stated explicitly in the passage, in so many words. No outside knowledge
   and no inference (the checker rejects anything implied).
2. Each question must be self-contained and specific: name the party, company, court, statute,
   period or amount it concerns, using names that appear in the passage. Someone reading a
   different passage on a similar topic must not be able to answer it.
3. Paraphrase. Apart from names, never copy more than 5 consecutive words of the passage into
   the question.
4. "quote" is the shortest exact span of the passage (at most 40 words, copied character for
   character) that contains the answer.
5. "answer" is at most 30 words, direct, and consistent with the quote. Copy numbers, dates and
   names exactly as written.
6. Vary "type" across: fact, number_or_date, party_or_entity, rule_or_holding, definition, yes_no.
   A yes_no answer may be "No" when the passage states the opposite.
7. If the passage is boilerplate, a table fragment, a list of citations, garbled OCR, or lacks
   the context for a specific question, return {"items": []}.

Reply with JSON only: {"items": [{"question": ..., "answer": ..., "quote": ..., "type": ...}]}

Source: {source_desc}
Passage:
"""{passage}"""
```

**G2: near-miss unanswerable (writer: DeepSeek, thinking off)**

```
You write test questions for a model that must refuse when its passages lack the answer.
You receive one passage from a {source_desc}. Write ONE question that a reader would expect this
passage to answer (same parties, subject and time period) but that it does NOT answer: it asks
for a specific detail the passage never states. The question must not be answerable by inference
from the passage, and must not be about an obviously unrelated topic. Name the entity it concerns.

Reply with JSON only: {"question": ..., "missing_detail": ...}, or {"question": null} if no
such question is possible.

Source: {source_desc}
Passage:
"""{passage}"""
```

**G3: grounded task (writer: DeepSeek, thinking off)**, where `{task}` ∈ summarize, extract_list, yes_no_check

```
You write instructions for a model that must work ONLY from passages it is given.
Write ONE instruction of type {task} that can be completed using only this passage, and its ideal
response. Refer to the passage only as "the passage" (it will be replaced with its number).
- summarize: the instruction asks for a one-sentence summary; response at most 35 words, with no
  fact that is not in the passage.
- extract_list: the instruction asks for all parties, dates, amounts, or defined terms of one
  kind; response lists them exactly as written, separated by "; ".
- yes_no_check: the instruction asks whether the passage states a specific claim; response is
  "Yes." or "No." then a verbatim quote of at most 30 words that decides it. Make the claim false
  ("No.") about half of the time, by changing one detail of something the passage does state.

Reply with JSON only: {"instruction": ..., "response": ..., "quote": ... or null}

Source: {source_desc}
Passage:
"""{passage}"""
```

**V1: blind check.** DeepSeek runs it on every candidate, with thinking on for B1/B2 and all
dev/test checks. Gemini runs the same prompt on dev/test and the 5% audit.

```
You check training examples for a model that must answer only from the passages it is given.
Using ONLY the passages below (no outside knowledge), decide:
- answerable: true only if at least one passage explicitly states the answer. Partial, implied
  or "probably" answers count as false.
- support: the numbers of every passage that states the answer on its own.
- answer: the answer in at most 30 words if answerable, else null.
- quote: if answerable, the shortest exact span (at most 40 words) of the first supporting
  passage that states the answer, copied character for character; else null.
- ambiguous: true if the question is unclear, or could refer to different things in different
  passages.

Reply with JSON only: {"answerable": bool, "support": [int], "answer": str or null,
"quote": str or null, "ambiguous": bool}

{passages, numbered 1..k, exactly as the student will see them}
Question: {question}
```

**V2: task check** (DeepSeek, thinking off; bucket C only). For yes/no instructions the proposed
response is omitted, so that check is blind. The fixed text says "the passage" because `k` varies,
which keeps the system prompt identical, and so cacheable, across requests.

```
You check a training example for a model that must work only from the passages it is given.
Using ONLY the passage below:
- If the instruction is a yes/no question, answer it yourself: "Yes" or "No".
- Otherwise, decide whether every factual claim in the proposed response is stated in the passage
  ("supported"), and whether the response does what the instruction asks ("on_task").

Reply with JSON only: {"yes_no": "Yes" or "No" or null, "supported": true or false or null, "on_task": true or false or null}

Passage {k}: {passage}
Instruction: {instruction}
Proposed response: {response, or omitted for yes/no}
```

**J1: evaluation judge** (Gemini; used only on QA and summary outputs that answer rather than
refuse. Yes/no is scored by exact match plus quote check, and extract lists by set-F1, so neither
needs a judge)

```
You grade a small model's answer. Use ONLY the passages; ignore outside knowledge.
- correct: the model's answer means the same as the reference answer. Missing the key fact, or
  adding a wrong claim, makes it false.
- supported: every factual claim in the model's answer is stated in the passages.

Reply with JSON only: {"correct": bool, "supported": bool}

{passages}
Question: {question}
Reference answer: {reference}
Model answer: {model_answer}
```

## 7. Training recipe (brief task 3, part 3)

### 7.1 Hyperparameters

| Setting | Value | Why |
|---|---|---|
| Initialisation | fp32 weights from `/data/checkpoints/ckpt.pt` (step 15,564) | the bf16 export would lose master precision |
| Chat tokens 4–6 | rows sampled from N(μ, Σ) of the trained rows, seeded and identical across runs | rows 4–6 are currently near-duplicates of each other (§1); sampling makes them distinct and in-distribution |
| Trained parameters | all (full fine-tune) | 125.8M fits trivially; LoRA would cost quality for no memory benefit |
| Precision | fp32 master weights + bf16 autocast, TF32, `torch.compile` | project invariant |
| Optimiser | fused AdamW, β (0.9, 0.95), ε 1e-8, weight decay 0.1 on matrices and embeddings only | same as pretraining (D-08) |
| Peak LR | **1e-4**; the pilot sweeps 5e-5 / 1e-4 / 2e-4 | pretraining ran 6e-4 → 6e-5; SFT should sit near the end of that range |
| Schedule | linear warmup over 3% of steps, cosine decay to 10% of peak | |
| Batch | 64 sequences (micro 32 × 2 accumulation) ≈ 65K tokens incl. padding | ~195 steps/epoch at 10K + replay |
| Epochs | full run: **3**, keeping the checkpoint with the best dev decision-F1. Pilot: up to 8, evaluated every epoch, best dev kept | the pilot's best epoch count sets the full run's; small curve points aren't starved of steps |
| Sequence | exactly 1,024 tokens, right-padded | one compiled shape |
| Loss | **L = L_sft + λ·L_replay, λ = 0.25.** Each term is a per-token mean of fp32 cross-entropy over *its own* target tokens in the **whole global batch** (summed over micro-batches, then divided by the global count; not HF's per-micro-batch mean ÷ accumulation steps) | a single pooled mean would give replay **84%** of the loss weight (13.1K replay vs 2.5K SFT target tokens per batch; measured target lengths: A 62, refusal 19, summary 45), diluting SFT ~6× and confounding the LR sweep. Within SFT, keep the per-token mean: every example then has exactly one decision token of equal weight, so the refusal share *is* the decision prior. A per-example mean would silently turn 25% into ~50% |
| Replay | 6 raw train windows per 26 SFT rows in every 32-row micro-batch (≈ 1 per 4.3; fixed shape), weighted λ = 0.25. Each term is normalised by its own token count, so replay is ≈ 20% of the loss whatever the row ratio | forgetting guard. If the bits/byte guard fails, raise λ to 0.5 |
| Grad clip / dropout | 1.0 / 0 | as pretraining |
| Eval decoding | greedy, ≤ 128 new tokens, stop at `<\|eos\|>`, suppress ids 0, 2–6 | deterministic metrics |
| Seed | 1337 | |
| Logged separately | decision-token CE and accuracy, L_sft, L_replay | the decision is 1 of ~48 target tokens, so watch it directly. If dev AUROC lags, a uniform decision-token weight (equal for both classes, so the prior is unchanged) is the first knob |

**Compute.**
- **Full run:** 10K examples × 1.25 (replay) × 1,024 × 3 epochs ≈ 38M tokens. At the measured 425K
  tokens/s (E-10), that's ~90 s of training plus ~4 min of startup, compilation and evaluation on
  1 × H100, ≈ **$0.50**.
- **Pilot:** the 11 runs share one container and **one compilation**.
  - The fp32 init weights are loaded once into CPU memory. Each run copies them into the *same*
    compiled module (`load_state_dict` copies in place, so Dynamo's guards still hold), re-applies
    the seeded row 4–6 init, and builds a fresh optimizer.
  - Evaluation runs on the uncompiled module, so generation's varying shapes never trigger
    recompiles.
  - There are no on-disk checkpoints inside the pilot: the best-dev weights stay in memory, and a
    crash resumes at the next unfinished run (results JSONL).
  - Each run trains ~18 s, so the container takes ≈ 15 min, ≈ **$1.50**.
- **Guards:** each launch uses `--max-usd` (4 for the pilot, 2 for the full run), with checkpoints
  every 100 steps.

### 7.2 Pilot sweep → choice

- **Runs:** LR {5e-5, 1e-4, 2e-4} × refusal share {15, 25, 35%} at 2,000 examples, plus learning-curve
  points at 500 and 1,000 examples using the best setting.
- **Selection score on dev** (deterministic proxies, no judge): maximise *proxy grounded accuracy*
  − *proxy hallucination rate* − *over-refusal rate (A)*.
  - Proxy grounded accuracy: cites the oracle, quote verbatim, answer token-F1 ≥ 0.5.
  - Proxy hallucination rate: answers an unanswerable item, or answers with a non-verbatim quote.
  - J1 is used only on the final candidate.
- **Ties:** prefer the lower hallucination rate.

## 8. Evaluation (brief task 3, part 4)

All metrics are computed on the **frozen test set (1,000 items)**: 500 answerable QA, 300 unanswerable
(150 B1, each paired with an A item / 100 B2 / 50 B3) and 200 grounded tasks, all from documents
the model never trained on.
Dev is used only for selection and τ.

### 8.1 Headline metrics

| Metric | Definition | Target |
|---|---|---|
| **Hallucination rate** | share of **all 1,000** test items where the model **asserts an answer the passages don't support**: it answers an unanswerable item, or J1 marks its answer `supported = false`, or its quote isn't verbatim | ≤ **15%** |
| **Refusal recall** (unanswerable) | refused / unanswerable, also reported per B1 / B2 / B3 | ≥ **0.80** |
| **Over-refusal rate** | refused / answerable QA (A). C is reported separately: its instructions name the passage | ≤ **0.20** |
| **Grounded accuracy** (A) | correct (J1) **and** cited the oracle passage, over all answerable QA | ≥ **0.60** |
| Citation accuracy | cited passage = oracle, among answered A items | ≥ 0.80 |
| Quote fidelity | quote is a verbatim substring of the cited passage | ≥ 0.90 |
| Format validity | output parses into `Source:` / `Quote:` / `Answer:` | ≥ 0.99 |
| Refusal precision / F1, **AUROC of s** | answer/refuse as a classifier at the decision position, A vs B1 + B2 (B3 and C excluded) | reported |
| **Paired consistency** | of the A/B1 test pairs (same question), share where the model answers correctly *with* the oracle and refuses *without* it | reported (the cleanest single grounding number) |

- **Breakdowns:** by source (case law / SEC / web), oracle position (1–4), bucket, table-heavy vs
  prose source passages (checks the pre-filter didn't cost numeric skill), and refusal recall by
  **entity present vs absent** in the passages (the shortcut detector).
- **Temptation probe** (diagnostic, not a target): 100 B1-style items built from *train* documents
  that were pretrained on but not used in SFT. They measure refusal when the model may know the
  answer from memory. ≈ $0.10.
- **Position bias:** accuracy across positions should vary by < 10 points.

### 8.2 Forgetting guards

- **Held-out bits/byte** on the fixed 600-window subset (200 per source, each with its own
  bytes/token), measured before and after SFT on the **same GPU numerics**: increase ≤ **0.02**,
  pooled and per source.
- **CaseHOLD** zero-shot accuracy no more than 2 points below baseline.

### 8.3 Reference points (for reading the numbers)

- The **base model** with a 1-shot plain-text prompt.
- **Always refuse:** hallucination 0, over-refusal 100%.
- **Teacher ceiling:** `deepseek-flash` on 200 test items, judged by Gemini. This is *inflated by
  construction*: test items were kept only where DeepSeek's check agreed. Read it as an upper bound.
- **Known bias:** two-model agreement keeps the *easier* items, so the test leans easy. Drop rates
  per bucket are reported with the results.

### 8.4 CaseHOLD baseline (7.1)

- **Data:** LexGLUE `case_hold`, test split, 3,600 items. Phase 2 already decontaminated the corpus
  against it.
- **Method:** zero-shot. Score each of the 5 candidate holdings by its log-likelihood after the
  citing context, with an `<|eos|>` prefix, left-truncating to 1,024. Report **byte-length-
  normalised** accuracy (acc_norm, as in lm-eval-harness; token counts are tokenizer-specific) and
  unnormalised accuracy, against 20% chance, with 95% CIs.
  **Added 2026-10-16 (E-21):** PMI accuracy, log P(holding | context) − log P(holding | "("),
  because summed log-prob only tracks holding length (correct holdings are ~11% longer). If PMI is
  also at chance, CaseHOLD cannot show forgetting (floor effect): it is then reported, and the
  held-out bits/byte stays the forgetting guard. That decision goes to the user.
- **Purpose:** a general legal-reasoning measure that moves independently of the RAFT format.

The targets were set **before** seeing any results. If one is missed, the miss is recorded and
diagnosed, and the target is never quietly lowered. At 10K examples, a miss first triggers the
learning-curve check in §3.1 (more data or not).

## 9. Budget and gates

| Stage | Cap | Estimate | Basis |
|---|---|---|---|
| 7.3 trial | $3 | ≈ $1.50 | 150 passages × 2 writers; ~300 candidates checked 3 ways |
| 7.4 pilot data | $10 | ≈ $7.50 | ~3,000 train + ~2,400 dev/test candidates through DeepSeek (≈ $4.60); Gemini co-labels dev/test and judges dev (≈ $2.90) |
| 7.6 pilot GPU | $4 | ≈ $1.50 | 1 × H100, ~15 min, compiled once |
| 7.8 scale data | $12 | ≈ $7.90 | ~9,500 candidates through DeepSeek (≈ $6.40); 5% Gemini audit + final J1 + ceiling (≈ $1.50) |
| 7.8 full SFT | $2 | ≈ $0.50 | 1 × H100, ~6 min |
| **Phase 7 total** | **$31** | **≈ $20** | API ≈ $17, GPU ≈ $3 |

- **Assumptions behind the estimate:**
  - Off-peak DeepSeek.
  - Per candidate: ~1,700 input tokens, ~1,000 of them a cached instruction prefix; ~600 output
    tokens, plus ~1,000 thinking tokens on refusal-label and dev/test checks.
  - Thinking on for ~35% of training checks.
  - Keep rate 80% (train) and 62% (dev/test, which needs two models to agree).
  - The trial measures all of these, and every cap is re-quoted before 7.4.
- **Not included:**
  - Nothing for the audit: its per-bucket fallback (≤ $3) fits inside the 7.8 cap.
  - Any top-up beyond 10K (≈ $0.85 per extra 1,000 kept examples).
- **Net effect of the review edits ≈ $0.** Additions: the V1 quote field (+$0.20), the leakage scan
  and temptation probe (+$0.20). Savings: paired B1 needs no generation calls (−$0.40), and the pilot
  compiles once (−$0.50).
- **GPU:** every entrypoint requires `--confirm`, and nothing may be left running afterwards
  (`modal container list` → 0).

## 10. Files to create or modify

| File | Purpose |
|---|---|
| `config.py` | add `RAFT_DIR`, a `RaftConfig` (k = 4, passage 60–160 tokens, mix ratios, sizes, refusal string, model ids, prices, peak hours) and an `SFTConfig` (§7.1) |
| `raft_data.py` | pure, unit-tested logic: chunking, passage filters, the split rule, prompt/target formatting, mechanical checks, output parsing, metrics |
| `raft_app.py` | Modal: passage-pool fan-out (one worker per corpus shard, keeping the fan-out design), BM25 build, distractor mining, dataset assembly; SFT GPU entrypoints gated by `--confirm` |
| `teacher.py` | local: concurrent DeepSeek calls (off-peak guard, retries, resumable JSONL) and Gemini batch jobs; spend ledger; keys read from `.env.local` and never logged |
| `raft_prompts.py` | the §6 prompts, split into a fixed (cacheable) system part and a variable user part |
| `raft_pipeline.py` | the local stage CLI: sources → generate → candidates → assemble → check → report; paid steps need `--confirm` |
| `sft.py` | SFT trainer: the §7.1 loss, fixed-shape micro-batches, chat-token init, exact resume, best-dev tracking, greedy export, and `run_sweep` (the 7.6 loop). Imports helpers from `train.py` |
| `raft_gpu_app.py` | Modal GPU launcher for 7.6 (and later 7.8); `--confirm` required |
| `eval_forgetting.py` | the forgetting guards: CaseHOLD and the per-source held-out bits/byte subset (§8.2, §8.4); importable by the GPU container |
| `eval_raft.py` | the RAFT metrics in §8.1 |
| `smoke_raft.py` | the local checks: data helpers and the 7.2 core now, the SFT checks of 7.5 later |
| `requirements.txt` | add `bm25s==0.3.13` now (local tests); `openai` (DeepSeek client) and `google-genai` at 7.3 (pinned) |

`cleaning.py`, `dedup.py` and `modal_app.py` are not touched.

## 11. Risks

| Risk | Mitigation |
|---|---|
| A 125M model may plateau at modest accuracy | Mostly extractive targets, a quote-first format, pilot learning curves. A plateau is a result to record, not something to hide |
| 10K examples may be too few | Learning curve in the pilot; incremental top-up (≈ $0.85 per 1,000) if test targets are missed |
| Pre-filter narrows the training distribution (fewer numeric/table questions) | Filter only the source pool; distractors unfiltered; ~10% table-heavy SEC sources kept; table-vs-prose test breakdown |
| Label noise in refusals (a distractor actually answers) → over-refusal | Blind check with thinking for B1/B2, excluding adjacent chunks, the 5% cross-family audit with a fallback rule |
| Flash-tier models on both sides may share blind spots | Different families; dev/test needs both to agree; the user spot-checks 50 items |
| DeepSeek has no batch API; peak pricing doubles cost | Resumable concurrent client; off-peak guard in `teacher.py` |
| DeepSeek JSON mode is not schema-enforced | Strict validation; failures are dropped and counted, never repaired |
| Ambiguous questions (a passage that never names its company) | G1 rule 2, the checker's `ambiguous` flag, the context filter |
| Lexical shortcut for refusal | BM25 hard negatives retrieved with the question, B2 near misses, paraphrase enforcement |
| Entity-name shortcut ("entity absent → refuse") | Paired A/B1, same-document distractors, entity-present share ≥ 50%, entity-split recall at eval |
| Replay drowning the SFT signal | Separately normalised losses with λ = 0.25 (§7.1) |
| Position bias | Uniform oracle position; per-position metrics |
| Forgetting | Replay; bits/byte and CaseHOLD guards |
| Train/test leakage through near-duplicates | Split by document plus the 13-gram filter on test |
| Price and terms drift | Prices dated 2026-10-15 (Gemini "through Dec 31, 2026"); re-quoted at each gate; the user reviews the terms |
| Data sent to DeepSeek is processed in China | Only public corpus text is ever sent |
| 1,024-token context limits multi-passage reasoning | Out of scope; extending context to 2,048 is a separate option (a few dollars of continued pretraining) |

## 12. Decisions needed from the user

1. **Approve this plan** (DeepSeek Flash writer + Gemini Flash checker, 10K, ≈ $20).
2. **Add `DEEPSEEK_API_KEY`** to `.env.local` before step 7.3.
3. **Gemini billing:** confirm the key's project is on the paid tier.
4. **Terms:** acknowledge §2's note on Gemini's terms and on sending public text to DeepSeek.
5. **Stage approvals** as each gate comes up: 7.3 ($3) → 7.4 ($10) → 7.6 GPU ($4) → 7.8 ($12 + $2 GPU).

## 13. Results

### 7.0 Configuration: DONE (2026-10-16, $0)
`config.py` now holds `RAFT_DIR`, `RAFT_INIT_CKPT`, `RaftConfig`, `TeacherConfig`,
`RAFT_STAGE_CAPS_USD` and `SFTConfig`; `"sft"` is added to `STAGES`.
- **Checks:** the mixes sum to 1, the caps sum to $31, the batch divides into micro-batches,
  `approx_params()` is still 125,848,320, and pyflakes is clean.

### 7.1 Baselines, local sanity run (2026-10-16, $0)
`eval_forgetting.py`, run on MPS in fp32, from the released bf16 weights (`.smoke/final/base`).

**Held-out subset:** 200 seeded windows from each source's `-000` val shard. The shard sha256s are
recorded in `.smoke/reports/raft_baseline_local.json` (written 2026-10-16 from the run's log, after the
full run was stopped for swapping). The Volume's shards were verified identical before 7.6.

| Source | Loss | Bytes/token | Bits/byte | E-13 (96 windows) |
|---|---|---|---|---|
| case law | 2.1745 | 4.512 | **0.6953** | 0.687 |
| SEC | 1.5791 | 5.027 | **0.4532** | 0.463 |
| web | 3.0685 | 4.344 | **1.0191** | 1.017 |
| pooled (equal-weight subset) | | | **0.7089** | — |

- The pooled figure is above the full-validation 0.6572 because this subset weights the sources
  equally, and the full split is SEC-heavy.
- **CaseHOLD:** test split downloaded (3,600 items, labels balanced 713–727 per class) and cached at
  `.smoke/casehold_test.jsonl`. The first download hit HTTP 429, so the fetch now paces requests and
  honours `Retry-After`. The score is recorded below when the run finishes.
- These are sanity numbers. The **official** before/after pair is measured in the GPU container
  (§5, step 7.1).

### 7.2 Code written and tested locally
- **New files:** `raft_data.py` (pure helpers) and `raft_app.py` (Modal: a remote driver, pass 1
  over 20 corpus shards, candidate selection, a leakage scan over 20 shards, 6 BM25 indexes).
- **`smoke_raft.py` passes 25/25 on 99 real documents.** Three problems were found and fixed:
  - The passage OCR tag misfired on SEC: 53% of SEC passages were tagged OCR because the macOS word
    list lacks plurals ("products", "engines", even "has"). Like Phase 1, the OCR check now applies
    only to case law. SEC clean share went from 40% to 93%.
  - The per-document cap in candidate selection was per stratum; it is now global.
  - BM25 probe misses were SEC filings repeating a sentence in sibling passages, so a hit now counts
    when the top-1 passage contains the query sentence verbatim. Locally 96.7% (bar 95%).
- **Chunk lengths:** encoding a passage standalone differs from the offset-based count by −1…+4
  tokens (6,345 passages; max 163). The 1,024 limit is enforced per formatted example.
- **Leakage scan:** passages copied from train documents score 1.00; novel text scores 0.00.
- **Throughput:** 3.4M n-grams/s locally, so the whole-corpus scan is ≈ 12 CPU-minutes.

### 7.2 Passage pools, leakage scan, BM25: DONE (2026-10-16, ≈ $0.19 of the $1 cap)
**First run.** `modal run --detach raft_app.py::prepare --confirm`: 218 s, ≈ $0.09; 0 containers
left afterwards.

| Output | Result |
|---|---|
| Passages | **141,189 val** (all val documents) + **753,575 train** (pool sample: 4.9% of case-law, 4.0% of SEC, 7.5% of web documents) |
| Train distractor pools | 251,710 case law / 252,941 SEC / 248,924 web |
| Length | every passage 60–160 tokens by offset count (histogram in `reports/7.2.json`) |
| Source-eligible | case law 94% / SEC 95% / web 99%. On Modal (Debian word list) the case-law OCR tag fires on 0.3% of passages, against 22% with the macOS list, which confirms the OCR fix |
| BM25 top-1 probe (bar 95%) | case law 98.6 / 99.2%, web 100 / 99.6% ✅. **SEC 92.4% (train) / 93.6% (val) ❌** |
| Dev/test candidates | 2,656 of 3,000 (SEC short: 2 per document × 428 SEC val documents) |
| Leakage scan (bar < 10% of 13-grams in train) | **981 of 2,656 failed (37%): SEC 736 of 856 (86%)**, case law 227 of 1,200 (19%), web 18 of 600 (3%) |

**Diagnosis.** `raft_app.py::diagnose`, 216 s, ≈ $0.11; two container preemptions were absorbed by
restarts.

- **Leakage, measured on all 56,873 source-eligible SEC val passages:**

  | Overlap with train | Passages |
  |---|---|
  | 0 | 6,077 |
  | < 10% | 9,916 in total (17.4%) |
  | 50–90% | 19,098 |
  | ≥ 90% | 15,395 |

  - The per-document median overlap is 0.69.
  - **The SEC val split is mostly near-copies of training filings**: same company, another year.
    Phase 2's exact dedup was per shard, and its LSH catches only 29% of pairs at J = 0.8.
  - The clean 9,916 still come from 369 of the 428 SEC val documents, so **the 10% rule is kept**,
    not relaxed.
- **BM25 SEC misses are not index faults.**
  - The target was in the top 10 for 30 of 32 val misses (**99.6% top-10 recall**; train 98.6%).
  - Only 1 miss had a near-duplicate top-1. The misses are generic boilerplate queries ("SFAS No.
    109…", "not engaged in any material litigation…") that many filings share, so the "longest
    sentence identifies its passage" probe can't work for them.
  - **The top-1 bar fails as defined for SEC.** The index is verified correct by top-10 recall; the
    check is recorded as failed-and-explained, not silently passed.

**Fix: the dev/test source pool is rebuilt from scanned passages only.**
- Every candidate has a measured leak fraction < 10%: max 0.099, and 72% have exactly 0.
- **2,000 passages:** 40/40/20 (case law 800 / SEC 800 / web 400), dev:test 1:2 by document, no
  document in both.
- **1,332 distinct documents** (650 / 314 / 368). The per-document cap is 2, or 3 for SEC, which has
  fewer documents.
- **SEC numeric passages are exactly 10%;** selection now fills the numeric quota first.
- Written to `raft/devtest/source_pool.jsonl`; the v1 pool is kept locally.

**Consequence beyond Phase 7 (E-20).** The base model's SEC held-out score (0.463 bits/byte) is
optimistic: most SEC "held-out" text shares substantial n-grams with training filings.

### 7.1 addendum: CaseHOLD sanity, 500 items on CPU (2026-10-16, $0)
The full 3,600-item MPS run swapped the 8.6 GB Mac to 9.3 GB and stalled, so it was stopped and
replaced by 500 items on CPU (26 min).

| Metric | Value | 95% CI |
|---|---|---|
| acc (summed log-prob) | **0.122** | 0.096–0.154 (below chance 0.20) |
| acc_norm (per byte) | **0.172** | 0.141–0.208 (chance) |

- **Diagnosis:** correct holdings are longer (168 vs 152 characters). "Pick the shortest" scores
  0.114 and "pick the longest" 0.246, so summed log-prob reduces to a length preference.
- **The base model shows no zero-shot CaseHOLD skill.** PMI scoring was added. The official
  numbers (all three, 3,600 items) are measured on the init weights in the GPU container.

### 7.3 Code written and tested locally (no API calls yet)
- **New files:** `raft_prompts.py`, `teacher.py`, `raft_pipeline.py`. `raft_app.py` gained
  `select_train_sources`, `ContextBuilder`, `train_sources` and `assemble_source`.
- **`smoke_raft.py`: 42/42.** Coverage:
  - quote re-cut (curly vs straight quotes) and elided quotes;
  - name / Yes-No / leading-"The" exemptions, copied questions, hedges;
  - list and yes/no tasks; answer agreement;
  - the first target token is `Ġnone` / `ĠPass`, with segment encoding equal to whole-string
    encoding;
  - the context rules (oracle at k, no adjacent chunk, ≤ 2 same-document passages, one per other
    document, no near-duplicates, task names its passage, ≤ 1,024 tokens, deterministic);
  - a mocked end-to-end dry run of candidates → assemble → report (the planted duplicate is dropped,
    and the thinking agreement is computed correctly).
- **Guards tested:** the peak-hour guard (Fri 02:30 and 07:30 UTC peak; 05:30, 12:30 and Sat
  off-peak). The ledger refuses an over-cap reservation and persists across runs.

### 7.3 Teacher trial: DONE (2026-10-16, **$0.553** of the $3 cap)
**Inputs.** 150 train source passages (case law 60 / SEC 60 / web 30; 100 G1, 25 G2, 25 G3). Both
models wrote from every passage, and every candidate was checked three ways. Spend was $0.549 on
APIs plus $0.004 on Modal; 0 containers left afterwards.

| Step | Result |
|---|---|
| DeepSeek generation | 150/150 OK in 13 s, $0.020. **0 reasoning tokens** on every no-think call |
| Gemini generation (batch) | 150/150 OK, 166 s turnaround, $0.063 |
| Mechanical checks | DeepSeek keeps 179 of 239 items (75%), Gemini 213 of 245 (87%). Top drop for both: questions copying ≥ 6 passage words (DeepSeek 43, Gemini 23). Also 4 DeepSeek summaries lacked "the passage" |
| Assembly | 541 of 545 (4: too few valid distractors). Median 640 tokens (max 795). Oracle position 84/89/96/81. Same-document distractors 1–2 in every context. **Entity present in 74% of B1** (target ≥ 50%) |
| Checks | DeepSeek off 525/525 ($0.074), DeepSeek on 496/496 ($0.185, including 15 runaway-thinking retries), Gemini 525/525 ($0.207). **JSON parse rate 100% for all three** |

**Decision rules.**

| Rule | First scoring | After the agreement fix | Outcome |
|---|---|---|---|
| Writer (DeepSeek stays unless Gemini is ≥ 10 points better) | 52.7% vs 66.9% ❌ | **64.0% vs 71.4%** | **keep DeepSeek** |
| Thinking on refusal labels (off if within 2 points) | 90.6% vs 89.5% | 90.6% vs 89.5% | **thinking OFF everywhere** |
| JSON with thinking ≥ 98% | 100% | 100% | ✅ |
| Checker agreement ≥ 85% | 82.4% ❌ | **90.0%** | ✅ |

**The fix applied after the first scoring, and why it is legitimate.** The first scoring failed two
rules because of a bug in `answers_agree`, not in the labels.
- In 55 of the 60 answerable items that DeepSeek accepted and Gemini rejected, **Gemini chose the
  same passage and the same verbatim quote**, and answered in a full sentence where the writer gave a
  few words. Token-F1 penalised that verbosity.
- The rule is now **symmetric containment** with light suffix stripping: the shorter answer must be
  inside the longer one. It applies only after the passage and quote already match.
- It has its own tests, including a different-fact negative. The decision rules themselves were not
  changed.
- The remaining 5 were genuine Gemini misses: "unanswerable" where the passage states the answer.

**Thinking, looked at directly.**
- On the 16 refusal items where thinking-on rejects and thinking-off accepts, Gemini sides with
  thinking-off 10 times.
- Thinking mostly adds "ambiguous" flags; it does not find answers hidden in distractors. Thinking
  stays off.

**Keep rates.**

| Bucket | Train rule (DeepSeek, no thinking) | Dev/test rule (DeepSeek and Gemini) |
|---|---|---|
| A | 263/305 (86%) | 246/305 (81%) |
| B1 | 129/150 (86%) | 124/150 (83%) |
| B2 | 38/41 (93%) | 37/41 (90%) |
| C | 45/45 | 45/45 |

**Measured unit costs** (per call) against the plan's model:
- DeepSeek generation $0.00013 and DeepSeek check $0.00014; the plan assumed ~$0.0007 and
  ~$0.0005, because outputs are 62–200 tokens rather than 600.
- Gemini batch check $0.00040; the plan assumed ~$0.0010.
- DeepSeek's prompt cache barely hit (~3% of prompt tokens). Costs came in under the model anyway.

**Re-quote.**
- 7.4 pilot ≈ **$1.90** (estimate was $7.50). Train data ≈ **$0.25 per 1,000 kept examples**.
- 7.8 scale ≈ **$2.51**.
- Phase 7 total ≈ **$7** (estimate was $20).

**Prompt fixes proposed for 7.4** (their effect is measured in the 7.4 funnel):
1. G1 rule 3: paraphrase more strictly.
2. G3: the instruction must literally say "the passage".
3. V1: "answer as briefly as possible: a few words, not a sentence".

### 7.4 Pilot data: DONE except the user's spot-check (2026-10-16, **$1.81** of the $4 cap)
**Review of 7.0–7.3 before running.** The aggregate numbers hid two data problems in the trial
output:
- **Vague references.** 9% of accepted questions said "the passage" or "the text", which is
  ambiguous when the student sees 4 passages; 20% referred to an unnamed party.
  → Mechanical rule `question_vague_reference`, plus a G1 instruction to use the names the passage
  gives.
- **Skewed question mix.** 45% of questions were `number_or_date` and only 10% `rule_or_holding`.
  → A per-call requested type, rotated by source, placed in the user message so the system prompt
  stays cacheable.

The three trial fixes were also applied: stricter paraphrase in G1, the literal "the passage" in G3,
and brief V1 answers. The cap was lowered from $10 to $4 after the re-quote.

**Run** (Fri 12:43–13:40 UTC, off-peak; enforced by the guard, but per-call timestamps were not
recorded, so `ts`/`peak` fields are added from now on).

| Step | Result | Cost |
|---|---|---|
| Sources | 3,964: train 2,310 + probe 100 (train documents, trial excluded) + dev 519 + test 1,035 (from the leak-checked pool). 0 overlaps with the trial; 0 dev/test documents shared | ≈ $0.002 |
| Generation (DeepSeek) | 3,963 of 3,964 OK | $0.448 |
| Mechanical | G1 2,622 of 4,383 items (60%); G2 519 of 673 (77%); G3 942 of 1,030 (91%). G1 drops: **copied ≥ 6 words 26.2%** (trial 22.8%: the stricter prompt did not help; samples confirm genuine copying) and vague references 9.6% (new rule). G3 placeholder misses 0 (trial 4/25) | $0 |
| Assembly | 4,953 of 4,963 | ≈ $0.005 |
| DeepSeek checks | 4,635 of 4,636 | ≈ $0.62 |
| Gemini checks (dev/test/probe) | 1,979 of 1,979, in 3 batches of ≤ 800 | $0.737 |

- **The guard refused the first Gemini submission.** Its worst case was $4.62 with 1,024
  output tokens per request, against $2.93 remaining. No money was spent. Fixed with
  `max_output_tokens` 384 (trial maximum 202) and 800-request chunks.
- **Dev/test refusal pairs.** These need both twins accepted by both checkers (~0.67 per pair), so
  B1 pairing was raised to 120 (dev) and 240 (test) candidates before checking.

**Keep rates** (accepted / assembled):

| Group | A | B1 | B2 | C |
|---|---|---|---|---|
| Train | 88% | 88% | 88% | 99% |
| Dev | 90% | 90% | 91% | 100% |
| Test | 90% | 93% | 82% | 95% |
| Probe | | 88% | | |

- By source, every group is 89–93%.
- Dev/test A rose from the trial's 81% to 90%, which is the brief-answer fix.

**Datasets** (`.smoke/raft/pilot/datasets/`, uploaded to `raft/datasets/pilot/`). **Every quota is
filled exactly; no shortfall.**

| Set | Contents |
|---|---|
| Train 2,400 | A 1,247 · B1 420 · B2 210 · B3 70 · C 453 (extract 157 / yes-no 137 / summary 159) |
| Dev 500 | A 275 · B1 75 · B2 37 · B3 13 · C 100 |
| Test 1,000 | A 500 · B1 150 · B2 100 · B3 50 · C 200. **150 B1 items each paired with its A twin** |
| Probe | 100 B1 items from pretrained-on documents |

- **Balance:**
  - Source mix ≈ 40/40/20.
  - Oracle positions are balanced (train 419/423/422/436).
  - Entity present in B1: 70–76% (target ≥ 50%).
  - Max length 773 tokens.
  - Train A yes/no answers: 101 Yes / 75 No.
- **Train A types:**

  | Type | Share |
  |---|---|
  | number_or_date | 29% |
  | fact | 27% |
  | party_or_entity | 18% |
  | yes_no | 14% |
  | rule_or_holding | 6% |
  | definition | 6% |

  The type request raised yes/no from 4% to 14%, but `rule_or_holding` stayed low (6%):
  DeepSeek often ignores the request.
- **Pending: the user's spot-check** of 50 dev/test items in `.smoke/raft/pilot/spot_check.md`.

### 7.4 Spot-check outcome (2026-10-16)
**The user's review of `spot_check.md` raised two items. Verified against the stored passages:**
- **Item 31** (B2, SEC, "US segment revenue in 1999, including HSI"): **a real label error.**
  - Passage 3 states US revenue of $256.4M for 1999, and "Excluding HSI, revenue increased by
    37.0%" implies the reported totals include HSI.
  - The review's own argument relied on outside facts ("the underlying 10-K… HSI attributed to the
    Europe segment"), which our labels may not use. The in-passage reading reaches the same verdict.
  - **Dropped, not converted to an A item:** "including HSI" needs an inference, so the item fails
    A's explicit-statement rule as well.
- **Item 29** (B1, Delaware §228(d)): **the refusal label is correct.** The review quoted Passage 2 as
  saying "prompt notice thereof was provided to all stockholders". In fact Passage 2 is an executive
  biography, and that sentence appears in none of the passages, nor in the removed oracle (which says
  "The remaining stockholders of the Company were notified…").

**Near-miss (B2) read-through by the assistant**, 15 random dev/test items:
- 13 are correct, including several excellent ones (e.g. a passage that says the representatives
  are *not identified*).
- 1 is borderline ("the director's handbook requirement for operable windshield washers": the
  passage says the requirement was "an operable windshield washer"), so it was **dropped**.
- With item 31, the estimated B2 label noise is ≈ 5–10%. B2 is the bucket closest to the decision
  boundary by design.

**Fix.**
- `exclusions.json` records each dropped id with its reason. `select` swaps each excluded row for the
  next unused accepted row of the same bucket.
- Verified: exactly 2 test rows changed; train, dev and probe are identical.
- The two replacements were read and are valid near misses (Personnel Officer *qualifications*; a
  Y2K *budget*).
- Re-uploaded to `raft/datasets/pilot/`. The reviewed `spot_check.md` is kept unchanged.

### 7.4 Near-miss (B2) read-through: all dev/test items (2026-10-16, $0)
**Scope.** The assistant read all 137 dev/test B2 items and the 41 unused accepted spares: each
question against its full topical passage, plus a name/term screen of the other three passages
(any hit read in full). Every verdict and reason is in `b2_review.json` / `b2_spare_review.json`.

**Rule** (sharpened in batch 6 and applied consistently, including retroactively).
- **Drop** when:
  - (a) a passage states the fact, or it follows by simple arithmetic from figures that clearly
    belong to the asked item;
  - (b) the question is underspecified (an unnamed party, or two parts) and some passage answers a
    reasonable reading;
  - (c) a passage gives a partial or qualitative answer for the same item.
- **Keep** when the tempting statement clearly concerns a *different* item (a different year,
  entity, facility or metric). Those are the valuable hard negatives. Under the sharpened rule, 3
  earlier drops (#59, #70, #86) were restored.

**Result.**
- **24 of 137 dropped (17.5%)** in the reviewed sets.

  | Reason | Count |
  |---|---|
  | Computable from stated figures | 8 |
  | Partial / qualitative answer | 6 |
  | Underspecified party or case-name question | 6 |
  | Answerable or near-answerable | 4 |

- **7 of 41 spares also failed.**
- **Recurring generator patterns:**
  - "name of the defendant in *Payne*";
  - figures that subtract or divide to the answer;
  - "a modest/growing portion" for "what percentage";
  - unlabelled table columns paired with hallucinated years (2020, 2023);
  - yes/no questions about something being absent;
  - two-part questions.
- **Excellent hard negatives kept** (the tempting figure belongs to another item):
  - interest expense for 1998 when 1999 is asked;
  - the street value for the bathroom, not the bedroom;
  - another company's revenue;
  - an offset well's production rate;
  - "unspecified monetary damages".

**Applied** through `exclusions.json` (now 33 entries, each with its reason). `select` now records
an honest shortfall when no reviewed spare is left.

| Set | Near misses changed | Now |
|---|---|---|
| Dev | 13 out / 12 in | **36 B2, 1 short: dev = 499** |
| Test | 13 out / 13 in | 100 B2 |

- Every dev/test B2 now carries a reviewed "keep" verdict. No non-B2 row changed. Train and probe
  are identical, and the 150 test pairs are intact. Re-uploaded.

**Consequences.**
- **Train B2 (210 items) is unreviewed** and likely carries similar noise (≈ 5% clearly
  answerable, ≈ 17% under the strict rule). That is acceptable for the pilot sweep, and recorded as
  a caveat.
- **For 7.8:**
  - add the patterns above to G2 as prohibitions;
  - add a B2-specific check ("could a careful reader give a plausible answer, including by
    arithmetic or a partial statement?") before scaling.

### 7.0–7.4 review before 7.5 (2026-10-16, $0)
**A full re-check against the artifacts passed:**
- **No document is shared** between train and dev/test, counting *every* passage in every context
  (distractors included).
- **All 1,211 dev/test oracles** come from the leak-scanned pool.
- **The probe's 68 answer passages** never appear in any SFT train context.
- **No trial source was reused.**
- **All 3,999 examples meet the format contract:** ≤ 773 tokens, `<|eos|>` framing, the decision
  token first, the cited k = the oracle, every quote verbatim.
- **The 150 test pairs are intact.**
- **Spend reconciles to $2.56.** 0 Modal containers.

### 7.5 Local SFT smoke test: DONE (2026-10-16, $0)
**Code:**
- `eval_raft.py`: constrained decoding as a `LogitsProcessor` over HF batched `generate`, output
  parsing, decision metrics/AUROC, and the §7.2 proxies.
- `sft.py`: the trainer.
- `raft_gpu_app.py`: the 7.6 launcher.

**`smoke_raft.py --sft`: 53/53 checks**, plus 5 sweep checks with a tiny model.

| Check | Result |
|---|---|
| Decision ids from the tokenizer | `Ġnone` 5494, `ĠPass` 12175, `age` 479, digits 307/393/421/559 |
| Mask on 200 real examples | exactly decision token … final `<|eos|>` |
| Loss vs a hand computation | 6.037719 = 6.037719 |
| Same loss split into micro-batches | 6.037719 |
| A pooled per-token mean would have given | 4.0106: the §7.1 flaw, avoided |
| Chat-token init | pairwise cosine max 0.24; every other row untouched; output log-probs −21.5/−14.1/−19.3, below the trained p95 (−13.2) |
| Overfit 32 examples | SFT loss **0.0001** after 180 steps |
| Constrained generations | **8/8** reproduce the overfit targets exactly |
| Hard kill at step 6, resume from step 4 | **bit-identical** to an uninterrupted run |
| Sweep loop | in-place reload keeps the same Parameter objects (one compile for all runs); restart skips finished runs; budget guard stops |

**Bugs the smoke test caught (all fixed):**
- **`loss_sums` indexed SFT rows with the whole batch's mask.** It would have crashed on the GPU.
- **HF `generate` inherited the base model's saved `min_new_tokens=16`.** That blocked `<|eos|>`
  after the 15-token refusal, so the model repeated it. This is the export hazard flagged in the
  review. Generation now uses an explicit greedy `GenerationConfig`.
- **The resume fingerprint included cadence settings,** which would have blocked a legitimate
  resume.
- **After a resume, the export could ship the last weights instead of the best;** it now falls back
  to `best.pt`.

**Adjusted check (documented).** The chat-token criterion was "output log-prob ≤ the median trained
token". A correct in-distribution sample lands near the median about half the time, so that test
would fail at random. The check is now "does not dominate: below the trained p95".

**Local limits.** The 8.6 GB Mac swaps heavily during CPU training (≈ 25 min for the suite). That's
fine for the smoke test; all real training is on the GPU.

### 7.6 pre-launch review (2026-10-16, $0)
- **The Volume matches local:**
  - `raft/datasets/pilot/{train,dev,test,probe}.jsonl` are byte-identical (sha256) to the local sets;
  - the three `-000` val shards are identical to those behind the local baseline;
  - `checkpoints/ckpt.pt` is present (1.4 GiB).
- **All 9 sweep mixes and the curve points are buildable** with exact refusal shares (0.150 / 0.250 /
  0.350). At 15% and 35% the pool is exactly exhausted for A, or for B1/B2/B3, by design.
- **Per run:** 26 SFT + 6 replay rows per micro-batch, accumulation 2. At 2,000 examples that is
  39 steps/epoch, 312 steps maximum and 20.4M tokens; across the whole sweep ≈ 220M tokens.
- **Changes:**
  - the container guard's per-run estimate is raised from 90 s to 120 s (the revised estimate is
    ~95 s per run);
  - the replay-ratio wording is corrected (≈ 1 per 4.3 rows; the loss share is unchanged at ≈ 20%).

### 7.6 Pilot sweep: DONE, with two failed guards (2026-10-16/17, ≈ $1.37 of the $4 cap)
**Three launches.**
1. **KeyError on the 7.6 cap**, before any GPU was requested. A trailing comment from the 7.4 cap
   edit had swallowed the key; fixed, and a config check was added.
2. **Device mismatch in `init_chat_tokens`** (CPU generator vs CUDA tensors), after the official
   baselines; ≈ $0.18. Fixed (CPU float64). The sweep path now also passes on MPS.
3. **Completed** in 880 s, $1.19 by the container's meter; 0 containers afterwards.

Also fixed before the third launch: `proxy_metrics` now takes answer/refuse from the constrained
decision, so a malformed answer on an unanswerable item counts as hallucination.

**Official baselines** (init weights, bf16, 1 × H100):
- held-out pooled **0.7090** bits/byte (local sanity 0.7089);
- CaseHOLD acc 0.121 / acc_norm 0.177 / **PMI 0.598** (CI 0.582–0.614).
- **PMI shows the base model does use the citing context.** CaseHOLD is therefore an informative
  forgetting guard; it is not at the floor.

**Sweep** (dev, 499 items; constrained decoding at τ = 0.5; best epoch of ≤ 8 by dev macro-F1):

| n | LR | refusal | best epoch | macro-F1 | AUROC | grounded | halluc. | over-ref. | ref. recall | format | score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 500 | 0.0002 | 35% | 8 | 0.707 | 0.744 | 0.015 | 0.461 | 0.284 | 0.661 | 0.972 | -0.730 |
| 1,000 | 0.0002 | 35% | 7 | 0.838 | 0.901 | 0.044 | 0.493 | 0.116 | 0.774 | 0.996 | -0.566 |
| 2,000 | 5e-05 | 15% | 8 | 0.853 | 0.910 | 0.091 | 0.521 | 0.022 | 0.653 | 0.984 | -0.452 |
| 2,000 | 5e-05 | 25% | 7 | 0.887 | 0.937 | 0.069 | 0.475 | 0.076 | 0.823 | 0.986 | -0.482 |
| 2,000 | 5e-05 | 35% | 4 | 0.863 | 0.915 | 0.062 | 0.463 | 0.105 | 0.815 | 0.980 | -0.507 |
| 2,000 | 0.0001 | 15% | 5 | 0.778 | 0.830 | 0.069 | 0.527 | 0.069 | 0.556 | 0.970 | -0.527 |
| 2,000 | 0.0001 | 25% | 8 | 0.869 | 0.921 | 0.080 | 0.473 | 0.076 | 0.774 | 0.988 | -0.469 |
| 2,000 | 0.0001 | 35% | 4 | 0.850 | 0.902 | 0.073 | 0.503 | 0.051 | 0.694 | 0.978 | -0.481 |
| 2,000 | 0.0002 | 15% | 5 | 0.848 | 0.898 | 0.062 | 0.493 | 0.069 | 0.718 | 0.990 | -0.500 |
| 2,000 | 0.0002 | 25% | 7 | 0.854 | 0.919 | 0.047 | 0.471 | 0.098 | 0.766 | 0.988 | -0.522 |
| 2,000 | 0.0002 | 35% | 5 | 0.864 | 0.927 | 0.102 | 0.445 | 0.080 | 0.782 | 0.986 | -0.423 |

**Selected:** LR 2e-4, refusal 35% (score −0.423).
- **Learning curve** at that setting: 500 → 1,000 → 2,000 examples gives grounded 0.015 → 0.044 →
  0.102 and score −0.730 → −0.566 → −0.423. **Still climbing clearly**, so 10K is justified (§3.1).

**τ table** (best weights, teacher-forced decisions on dev):
- AUROC **0.934**.
- Macro-F1 0.851 at τ 0.5, best 0.872 at τ 0.7.
- Refusal precision/recall: 0.721/0.855 at τ 0.5 and 0.782/0.839 at τ 0.7.

**Forgetting guards: FAILED** (pre-registered limits: bits/byte +0.02; CaseHOLD −2 points).

| Guard | Base | After | Change |
|---|---|---|---|
| Pooled bits/byte | 0.7090 | 0.7404 | **+0.0314** |
| Case law bits/byte | 0.6954 | 0.7261 | +0.031 |
| SEC bits/byte | 0.4532 | 0.4813 | +0.028 |
| Web bits/byte | 1.0191 | 1.0549 | +0.036 |
| CaseHOLD PMI | 0.598 | 0.555 (CI 0.538–0.571) | **−4.4 points**, CIs disjoint |

acc and acc_norm are unchanged (−0.3/−0.4); they only track length.

**Diagnosis of low grounded accuracy** (best pilot weights, 40 random dev A items, local CPU):

| Outcome | Count |
|---|---|
| Quote not from the passage | 17 |
| Near-verbatim quote | 8 |
| Refused | 7 |
| Wrong passage | 4 |
| Correct | 3 |
| Answer mismatch | 1 |

- The model usually picks the right passage and the right answer/refuse decision. It **does not
  copy**: it writes a fluent paraphrase in the `Quote:` field (e.g. "We hold that Tinsley was
  immune…" where the passage says "…argue that they are immune… We agree."). The answers are often
  semantically right, but the quotes are invented.


**Fix for the invented quotes: copy-constrained decoding** (free, inference-time; `eval_raft.CopyConstraint`).
- Inside `Quote: "`, a token is allowed only if the quote stays a substring of the cited passage.
- The quote may close only with `"` or `"\n` (a `")` token would break the format). A 300-character
  cap forces it to close.
- Unit-tested (clean close, cap) before use.

Same 80 dev items (40 A, 40 B1/B2/B3), best pilot weights, local CPU:

| Decoding | grounded | hallucination | format | score | A outcomes |
|---|---|---|---|---|---|
| Plain | 0.075 | 0.450 | 1.000 | −0.550 | 3 correct; 17 quote not from passage; 8 near-verbatim; 7 refused; 4 wrong passage; 1 answer mismatch |
| **Copy-constrained** | **0.200** | **0.163** | 1.000 | **−0.137** | 8 correct; **17 answer mismatch**; 4 quote not verbatim; 7 refused; 4 wrong passage |

- The answer/refuse decisions are identical in both modes (80/80): the constraint changes only the
  quote. B items are unchanged (33 refused, 7 answered).
- **What remains is wrong-span selection.** The quote is now verbatim, but it is often the wrong
  sentence. Example: asked what the appeals court did after finding error, it quotes "the court
  found that the Commission erred" instead of the remand. More and better data (7.8) is the lever
  for this; decoding can't fix it.
- **Decision (D-21):** report every proxy and the final test under **both** decodings. Copy-constrained
  decoding is the recommended serving mode.

### 7.6b Forgetting follow-up: PLANNED, awaiting GPU approval (written 2026-10-17, before any run)
**Why.** Both guards failed at the pilot's best epoch (≈ 195 steps). The pre-registered remedy is
λ = 0.5. But the full 7.8 run is **579 steps** (10K × 3 epochs; pilot 2K × 8 epochs = 312), so the
end-point numbers from 7.6 can't tell us whether any λ holds the guard at 7.8's length. We need
the **slope**.

**What.** One 1 × H100 container (`raft_gpu_app.py::lambda_check`):
- 4 arms from `config.SFT.forgetting_arms`, each at the 7.6 selection (n 2,000, refusal 35%,
  ≤ 8 epochs):
  - LR 2e-4 at λ 0.25 / **0.5** / 1.0;
  - LR 5e-5 (the 7.6 runner-up, best macro-F1) at λ 0.5, to test whether a lower LR forgets less.
- Per arm:
  - held-out pooled bits/byte after **every** epoch (`train_one(extra_eval=…)`, same 600-window
    subset and batch as the official baseline);
  - CaseHOLD PMI at the best epoch;
  - dev proxies with plain and copy-constrained decoding.
- The λ 0.25 arm repeats the 7.6 run, so it doubles as a reproducibility check against +0.0314.

**Pre-registered rule.**
- **Projection.** For each arm, fit bits/byte delta against step on the last 4 per-epoch points,
  and extrapolate to step 579.
- **Eligibility.** An arm passes if its projected delta is ≤ +0.02 **and** its CaseHOLD PMI drop at
  the best epoch is ≤ 2 points.
- **Choice.** Among passing arms, take the highest copy-decoding selection score.
- **If no arm passes,** stop and bring the options to the user: λ > 1, a lower replay ratio
  (`sft_per_replay` 4 → 2), fewer 7.8 epochs, or accept a documented guard miss. Don't pick one
  silently.

**Cost.**
- About 170 s per arm (7.6 runs took 73 s; adding 8 held-out evals, CaseHOLD and a second
  generation pass) + staging and compile. ≈ 12 min, **≈ $1.00** at the measured $0.00127/s.
- `--max-usd 1.6`. The entrypoint refuses anything above what is left of the 7.6 cap
  ($4 − $1.37 = $2.63). The container stops before any arm that would cross `--max-usd`.

**Resume.** Results are appended per arm to `/data/raft/runs/pilot_lambda/results.jsonl`, and a
restarted container skips finished arms. `--detach`; afterwards, verify 0 containers.

**Validation.**
- The λ 0.25 arm's bits/byte delta at its best epoch is within ±0.005 of 7.6's +0.0314.
- Every arm logs one history point per epoch (smoke-tested: 2 calls for 2 epochs). The full local
  suite `smoke_raft.py --sft` passes **62/62** (2026-10-17), including the bit-identical
  crash-and-resume test.

### 7.6b Forgetting follow-up: DONE (2026-10-17, $1.18; 7.6 total $2.55 of the $4 cap)
One container, 877 s; 0 containers afterwards. Staging took 205 s (a cold Volume read; 7.6 took 3 s).

Held-out pooled bits/byte change after each epoch (39 steps per epoch, 312 steps total):

| Arm | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 (end) | best step | CaseHOLD PMI at best | macro-F1 | copy score | plain score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LR 2e-4, λ 0.25 | +.052 | **+.056** | +.055 | +.048 | +.041 | +.036 | +.032 | +.029 | 156 | −4.9 | 0.868 | **+0.086** | −0.490 |
| LR 2e-4, λ 0.5 | +.042 | +.046 | +.046 | +.041 | +.036 | +.032 | +.029 | +.027 | 312 | −3.7 | 0.843 | −0.001 | −0.505 |
| LR 2e-4, λ 1.0 | +.036 | +.042 | +.043 | +.039 | +.035 | +.031 | +.028 | +.026 | 273 | **−1.1** | 0.840 | −0.029 | −0.510 |
| LR 5e-5, λ 0.5 | +.010 | +.009 | +.009 | +.009 | +.007 | +.006 | +.006 | **+.006** | 234 | −4.8 | 0.861 | −0.028 | −0.521 |

Copy-constrained grounded accuracy is 0.21–0.25 in every arm, versus 0.05–0.07 plain. Hallucination is
0.08–0.13 copy, versus 0.41–0.52 plain.

**Findings.**
1. **Forgetting is a transient, not a slope.** It peaks at epochs 2–3, while the LR is high, then recovers
   as the cosine schedule anneals. The rule's premise (forgetting grows with steps, so extrapolate to
   step 579) is **false** (C-12). Its linear fit on epochs 5–8 extrapolates the *recovery* and projects
   +0.002 to +0.006 for every arm. That is an artefact: 7.8's schedule stretches the same peak-then-anneal
   shape over 579 steps. Read mechanically, the rule picks LR 2e-4 / λ 1.0. **I am not adopting that
   pick**, because its measured end-of-run change is +0.026, above the +0.02 limit.
2. **LR is the bits/byte lever; λ is weak.** At the end of the schedule, λ 0.25 → 1.0 moves bits/byte only
   from +0.029 to +0.026. LR 2e-4 → 5e-5 moves it to +0.006 and flattens the peak.
3. **CaseHOLD PMI doesn't track bits/byte.** LR 5e-5 barely moves bits/byte but loses 4.8 points. λ 1.0 is
   the only arm within 2 points (−1.1). CaseHOLD was measured at a different epoch in each arm (the
   best-by-macro-F1 epoch), so the comparison is confounded by epoch.
4. **Reproducibility check: failed as specified, consistent on inspection.** The λ 0.25 arm's best epoch is
   4 here (+0.048); 7.6's sweep picked epoch 5, and here epoch 5 is +0.041, not +0.031. But **epoch 7 here
   is +0.0315 vs 7.6's +0.0314**, which suggests 7.6's guard retrain landed on epoch 7. Unverified:
   7.6 didn't record the retrain's best epoch. Choosing the best epoch by dev macro-F1 is itself noisy
   (arms differ by ≤ 0.03), so it moves between otherwise identical runs.
5. **No arm passes both guards on measured end-of-run numbers.** By the rule's own fallback: stop and ask.
