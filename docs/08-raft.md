# Phase 7: Grounded SFT / RAFT (plan)

**Status: PLANNED, not executed.** Written 2026-10-06. Every paid step below waits for explicit
approval. Results get recorded in §13 as each step runs.

Brief: `instruction.txt` (the user's draft). Earlier recommendation: `EXPERIMENTS.md` D-15.

## 0. Summary

**Goal.** Teach the 125.8M base model to answer **only** from passages it is given. It should cite
which passage and quote it, and it should **refuse** when the passages don't hold the answer.

**Approach.** Direct RAFT on teacher-written, quote-grounded targets; no closed-book distillation
first. Every training example puts 4 passages in the prompt. In answerable examples one passage holds
the answer and 3 are hard distractors. In refusal examples none holds it, and the target is a fixed
refusal string.

| | Recommendation |
|---|---|
| Teacher | **`gemini-3.8-flash` (Batch API)** generates the data and blindly verifies the training examples. **`claude-sonnet-5` (Batch API)** independently adjudicates the dev/test labels, judges the student's answers, and audits a 5% sample of training labels |
| Dataset | **20,000 training examples**, sized after a 2,000-example pilot: 55% answerable QA, 25% unanswerable → refusal, 20% grounded non-QA tasks. Plus **pretraining replay** (~20% of tokens) against forgetting. Dev 500 and test 1,000, built only from held-out documents |
| Training | Full fine-tune from the fp32 Phase 5 checkpoint. LR 1e-4 (swept in the pilot), 3 epochs, loss on assistant tokens only, 1 × H100, a few minutes |
| Metrics | Hallucination rate, over-refusal rate, grounded accuracy, citation accuracy, verbatim-quote rate, and forgetting (held-out bits/byte, CaseHOLD) |
| Cost | Teacher API ≈ **$80** in total, staged ($5 bake-off → $25 pilot → $60 scale-up), each stage approved separately. GPU ≈ **$3**, capped at $5 |

## 1. Starting point (review findings that shape this phase)

| Fact | Consequence |
|---|---|
| Context is **1,024 tokens**; the tokenizer averages 4.67 bytes/token on prose but **3.3 chars/token** on QA-style text full of numbers (measured 2026-10-06) | k = 4 passages of ≤ 160 tokens each (≈ 650 tokens), question ≤ 48, answer ≤ 128. Anything over 1,024 is dropped, never truncated |
| Chat tokens `<|user|>` 4, `<|assistant|>` 5, `<|system|>` 6 exist in the tokenizer but **never occurred in training**. With tied embeddings, those rows only learned "never predict me" | Before SFT, initialise rows 4–6 to the mean trained embedding plus small noise. Keep suppressing them at decode time |
| Prompts must start with `<|eos|>` (D-07) | Every SFT sequence starts with id 1 |
| The corpus is one document per line in `/data/corpus/<source>/*.txt`; the val split is `blake2b(text, 8 bytes) % 100 == 0` (6,638 documents never trained on) | **Train examples come only from train documents; dev/test only from val documents.** The test then measures grounding on text the model never saw. Reuse `modal_app._is_val_doc`'s exact rule |
| The near-duplicate count is a floor (LSH catches 29% at J = 0.8) | Near-duplicates can cross the split, so test passages that share 13-grams with any SFT training passage are filtered out |
| `/data/checkpoints/ckpt.pt` holds the **fp32** final weights; `base/` is a bf16 export | SFT initialises from `ckpt.pt`, never from the bf16 export |
| `train.py` already has autocast, the optimizer builder (no decay on norms), atomic saves, the spend ledger, `export_hf` and the evaluator | `sft.py` imports them rather than duplicating them. The pretraining loop itself (`WindowStore` order) doesn't fit SFT, so it needs a new loop |
| `.env.local` holds `GEMINI_API_KEY` (verified: the key lists `gemini-3.8-flash`, `gemini-3.5-flash`, `gemini-3.1-pro-preview`, …). It has **no Anthropic key** | The Sonnet 5 role needs `ANTHROPIC_API_KEY` added to `.env.local` by the user |
| `requirements.txt` has no `google-genai`, `anthropic`, or `bm25s` | Add them, pinned, when implementing |
| No CaseHOLD evaluator exists | Write `eval_casehold.py` (§8.4) |

## 2. Teacher model (brief task 1)

The brief's candidate list is dated. **Claude 3.5 Haiku and 3.5 Sonnet were retired in 2025–26**, and
GPT-4o-mini and Llama 3.1 have been superseded. These are the current equivalents. Prices are per
million tokens, **Batch API** (50% off standard), taken from the vendors' pricing pages on 2026-10-06:

| Model | Batch in / out | Role it could play | Notes |
|---|---|---|---|
| **`gemini-3.8-flash`** | $0.375 / $1.875 | generator + blind verifier | Newest Flash on the user's key. Pricing page shows this price "through Dec 31, 2026". Thinking level is configurable |
| `gemini-3.5-flash-lite` | $0.15 / $1.25 | cheap generator | Weaker; only worth it if the bake-off shows parity |
| `gemini-3.1-pro-preview` | $1.00 / $6.00 | judge, if Gemini-only | Preview status |
| `claude-haiku-4-5` | $0.50 / $2.50 | generator (bake-off rival) | Successor to the brief's "Claude 3.5 Haiku" |
| **`claude-sonnet-5`** | $1.00 / $5.00 | **independent judge / adjudicator** | Successor to "Claude 3.5 Sonnet"; a different model family from the generator |
| `gpt-5-mini` | $0.125 / $1.00 | generator | Cheapest strong option; needs an OpenAI key the user doesn't have |
| Llama 3.1 / Qwen3 (self-hosted) | GPU time | generator | **Reject Llama 3.1**: its license requires models trained on its outputs to carry "Llama" in their name, and it is weak by 2026 standards. Qwen3 is Apache-2.0 but means a vLLM deployment and a GPU gate, for lower verification quality |

**Recommendation: generate and verify the training data with `gemini-3.8-flash`, and use
`claude-sonnet-5` for everything that has to be independent.**

- **Generation is the easy part.** Writing a question about one passage and copying out a quote is
  well within any current model. Mistakes are mostly *mechanical* (a quote that isn't verbatim, a
  copied question, a number that differs from the passage), and the deterministic checks in §5.4
  catch those whichever model writes the data. That makes the cheapest strong model the right choice,
  and the user's Gemini key is already set up.
- **What decides quality is the label on hard cases:** does any of the 4 passages actually answer the
  question? A wrong "unanswerable" label teaches **over-refusal**; a wrong "answerable" label teaches
  **hallucination**. Two safeguards cover this:
  - Every candidate gets a **blind** verification pass. The verifier sees only the 4 passages and the
    question, never the intended answer.
  - Dev/test labels, which define every number reported, are adjudicated by a **different model
    family** (Sonnet 5). That keeps Gemini's blind spots out of the measurement.
- **The 5% audit** has Sonnet 5 re-label a random 5% of the training set. If Sonnet and Gemini
  disagree on more than **5%** of it, switch the remaining training verification to Sonnet 5 (about
  +$45 at 20K). This decision is made on data, not taste.
- **About refusal-constraint adherence** (the brief asks specifically): the student's refusal comes
  from the *labels*, not from the teacher refusing. Teachers are never asked to refuse; they are asked
  to judge answerability as structured JSON. That measures exactly the property the brief cares about,
  and the bake-off measures it per model.
- **Caveat:** these model versions were released after my training data, so I can't vouch for them
  from experience. **Step 7.3, a ~$2 bake-off**, compares `gemini-3.8-flash` with `claude-haiku-4-5`
  on the same 100 passages before any real spend. The rule is set in advance: Gemini stays unless
  Haiku's yield of verified examples is ≥ 10 points higher.

**Terms of use (the user's call, not legal advice).** Both vendors restrict using their services to
build competing models:
- Gemini API terms: "you may not use the Services to develop models that compete with" them.
- Anthropic's commercial terms prohibit using the services "to train competing AI models"; customers
  own their Outputs.

A 125M legal-domain research model is a long way from a competing general assistant, but read the
current terms before publishing a model or dataset made from teacher outputs. Use the **paid** Gemini
tier: on the free tier, Google may use prompts and outputs to improve its products, and the rate limits
are far too low for 15K+ requests.

## 3. Dataset architecture and sizing (brief task 2)

### 3.1 Size: 20,000 training examples (set by the pilot)

- A 125M model has to learn three things at once: a new turn format, span copying across distracting
  context, and a calibrated answer/refuse decision. Comparable extractive-QA fine-tunes of ~100M
  models use tens of thousands of examples (SQuAD 2.0 has 130K), and a pilot below ~2K would be too
  noisy to read.
- The pilot trains on 1K and 2K examples. If dev metrics still rise clearly from 1K to 2K, scale to
  20K; if they're flat, stop at about 10K and save ~$25.
- **Dev 500 / test 1,000**, from val documents only, adjudicated by Sonnet 5. They are built once and
  frozen before any training.

### 3.2 Composition

| Bucket | Share (of 20K) | Count | What the example contains | Target |
|---|---|---|---|---|
| **A. Answerable QA** | **55%** | 11,000 | oracle + 3 hard distractors; oracle position uniform over 1–4 | source, verbatim quote, short answer |
| **B1. Unanswerable, oracle removed** | 15% | 3,000 | an A-style question with 4 hard distractors; adjacent chunks of the oracle's document excluded | refusal |
| **B2. Unanswerable, near miss** | 7.5% | 1,500 | the question's topical passage *is present* but lacks the asked detail (e.g. asks the 2021 figure when the passage gives 2020), plus 3 distractors | refusal |
| **B3. No usable context** | 2.5% | 500 | the question with 0 passages, or 4 passages from an unrelated source | refusal |
| **C. Grounded tasks** | **20%** | 4,000 | 4 passages; an instruction about one of them: one-sentence summary, extract a list (parties, dates, amounts), or yes/no check with a quote (**about half "No"**) | the response, citing its source |
| **D. Replay** (outside the 20K) | ~20% of training tokens | — | raw 1,024-token windows from the pretraining **train** split | plain LM loss |

- **Source mix:** case law 40 / SEC 40 / web 20, matching the pretraining mix. Questions and passages
  for a given example come from the same source.

**Why these ratios:**
- **25% refusal (B1 + B2 + B3)** sits inside both reference points: SQuAD 2.0 is ~1/3 unanswerable, and
  RAFT found it best to keep the oracle in 60–80% of training examples.
  - More refusals push toward **over-refusal syndrome**; fewer leave **hallucination**.
  - It is treated as a hyperparameter: the pilot trains at **15 / 25 / 35%** and picks the share
    with the best balance on dev (§7.2).
- **B2 matters most.** Without near misses, the model learns a lexical shortcut, "refuse when no
  passage shares words with the question". B2 forces it to read whether the specific detail is there.
- **C is the "general instruction" bucket, deliberately grounded.** Open-domain instruction data
  (Alpaca-style chat answered from memory) teaches exactly the behaviour this phase is trying to
  remove, and a 125M model holds too little world knowledge to answer from memory reliably (E-13: it
  invents facts). Grounded tasks still provide the format diversity that general data normally gives.
  The **"No" answers in C** teach the difference between "the passage says no" and "the passage
  doesn't say", which is the other main cause of over-refusal.
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
- **The first generated word is the decision.** After `Source:`, the token " none" means refuse and
  " Passage" means answer. That gives:
  - (a) a single-position answer/refuse classifier whose probability can be **calibrated**: on dev,
    pick a threshold τ on P(none) to trade hallucination against over-refusal;
  - (b) the AUROC metric (§8).
- **Quote before answer.** The model copies a span and then answers from it. That's the RAFT
  chain-of-thought idea cut down to a copy step, which small models are relatively good at
  (induction heads). Quotes can also be **checked mechanically** at evaluation.
- **Preprocessing:**
  - (1) Chunk each document into sentence-packed passages of 60–160 tokens (our tokenizer).
  - (2) Drop low-quality passages: alphabetic ratio < 0.6, table/number fragments, citation lists.
  - (3) Deduplicate passages exactly, and 13-gram filter test against train (§1).
  - (4) Mine distractors with BM25 using the *question* as the query, from other documents of the same
    source, so they're lexically close but don't answer.
  - (5) Shuffle passage order so the oracle is uniform over positions 1–4.
  - (6) Tokenise, **drop** anything over 1,024 tokens, and right-pad to exactly 1,024. The fixed shape
    means `torch.compile` compiles once.
  - (7) Labels are −100 everywhere except the assistant span and its final `<|eos|>`.
  - (8) Store as `uint16` input ids plus a loss mask (npz or memmap), with a JSONL sidecar holding the
    raw fields for evaluation.

## 5. Execution steps

The Volume layout is `/data/raft/{passages,bm25,teacher,datasets/{pilot,full},runs}`. Each step
records its actual results in §13.

| Step | What | Where | Cost | Gate |
|---|---|---|---|---|
| 7.0 | This plan; add `RAFT`/`SFT` constants to `config.py` | local | $0 | the user approves the plan |
| 7.1 | **Baselines**: CaseHOLD zero-shot accuracy; held-out bits/byte on a fixed 600-window subset (200 per source) | local CPU | $0 | — |
| 7.2 | **Passage pool + BM25**: fan-out, one worker per corpus shard → passages; BM25 index per split and source | Modal CPU | ≈ $0.1 | — |
| 7.3 | **Teacher bake-off**: 100 passages; generators `gemini-3.8-flash` vs `claude-haiku-4-5`; labels verified by Sonnet 5 | local → APIs | ≤ **$5** | **spend approval** + Anthropic key |
| 7.4 | **Pilot data**: 2,400 kept train examples (enough to subsample 15/25/35% refusal mixes of 2,000), dev 500, test 1,000 | local → APIs | ≤ **$25** | **spend approval** |
| 7.5 | **Local SFT smoke test** (`smoke_sft.py`) | local CPU | $0 | — |
| 7.6 | **Pilot sweep**: 3 LRs × 3 refusal shares + a 1K-example learning-curve point = 10 short runs in one container | 1 × H100 | ≤ **$3** | **GPU gate** |
| 7.7 | Evaluate the pilot on dev → pick LR, refusal share, τ; decide 10K vs 20K | local / Modal CPU | ≈ $1 (judge) | — |
| 7.8 | **Scale**: generate the rest of the training data; 5% Sonnet audit; full SFT; final test evaluation | APIs + 1 × H100 | ≤ **$60** API + ≤ **$2** GPU | **spend approval + GPU gate** |
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
  - 0 val passages in the train pool.
  - A token-length histogram is inside 60–160.
  - BM25 sanity: querying with a sentence from a passage returns that passage top-1 for ≥ 95% of
    500 probes.
- **7.3 / 7.4 / 7.8:**
  - Mechanical-check pass rates and verification yield are reported per bucket and source.
  - The spend ledger is ≤ the stage cap.
  - The user spot-checks 50 random dev/test items.
- **7.5:**
  - The loss mask covers exactly the assistant span plus EOS.
  - Chat rows 4–6 are initialised and the other rows are untouched.
  - 32 examples overfit to train loss < 0.05.
  - Kill-and-resume is bit-identical.
  - Generations parse.
  - No sequence exceeds 1,024.
- **7.6 / 7.8:**
  - Loss is finite.
  - Dev metrics are logged every 50 steps.
  - The budget guard holds.
  - `modal container list` → 0 afterwards.

### 5.2 Mechanical checks on teacher output (model-free, all buckets)

1. Output is valid JSON matching the schema; teachers are called with JSON-schema-constrained output.
2. The `quote` is a **verbatim substring** of the oracle passage after whitespace normalisation, and
   ≤ 40 words.
3. Every number, date and capitalised name in `answer` appears verbatim in the passage (catches
   numeric hallucination).
4. The question is 5–40 words and shares no run of ≥ 6 consecutive words with the passage, which
   forces paraphrase and blocks the lexical shortcut.
5. The answer is ≤ 30 words, and for factoid types ≥ 50% of its content words appear in the quote.
6. No question is duplicated within or across splits (normalised exact match).
7. The fully formatted example is ≤ 1,024 tokens.

### 5.3 Acceptance after blind verification

| Bucket | Keep only if the verifier says |
|---|---|
| A | `answerable` true, `support == [oracle]` (unique), not `ambiguous`, and its answer agrees with the generator's (identical numbers/names, or token-F1 ≥ 0.5) |
| B1, B2 | `answerable` false and not `ambiguous` |
| B3 | no verification needed (no passages, or a different source) |
| C | the verifier completes the task with the same Yes/No (yes/no), or confirms every item appears in the passage (extract) |

- **Dev/test labels:** kept only when Gemini's verification **and** Sonnet 5's both agree with the
  intended label; disagreements are dropped. Expected yield is ~65% of candidates; the bake-off
  measures it.

## 6. Teacher prompt templates (brief task 3, part 1)

All calls use JSON-schema output. `{source_desc}` is one of "US court opinion", "SEC filing" or
"educational web page". Thinking is set to the lowest level that passes the bake-off.

**G1: answerable QA (generator)**

```
You write training data for a small model that must answer ONLY from passages it is given.
You receive one passage from a {source_desc}. Write up to 2 question-answer pairs that this
passage alone answers.

Rules:
1. The answer must be stated explicitly in the passage. No outside knowledge; at most one
   obvious step of inference.
2. Each question must be self-contained and specific: name the party, company, court, statute,
   period or amount it concerns, using names that appear in the passage. Someone reading a
   different passage on a similar topic must not be able to answer it.
3. Paraphrase. Never copy more than 5 consecutive words of the passage into the question.
4. "quote" is the shortest exact span of the passage (at most 40 words, copied character for
   character) that contains the answer.
5. "answer" is at most 30 words, direct, and consistent with the quote. Copy numbers, dates and
   names exactly as written.
6. Vary "type" across: fact, number_or_date, party_or_entity, rule_or_holding, definition, yes_no.
   A yes_no answer may be "No" when the passage states the opposite.
7. If the passage is boilerplate, a table fragment, a list of citations, garbled OCR, or lacks
   the context for a specific question, return {"items": []}.

Source: {source_desc}
Passage:
"""{passage}"""
```
Schema: `{"items": [{"question", "answer", "quote", "type"}]}`

**G2: near-miss unanswerable (generator)**

```
You write test questions for a model that must refuse when its passages lack the answer.
You receive one passage from a {source_desc}. Write ONE question that a reader would expect this
passage to answer (same parties, subject and time period) but that it does NOT answer: it asks
for a specific detail the passage never states. The question must not be answerable by inference
from the passage, and must not be about an obviously unrelated topic. Name the entity it concerns.
If no such question is possible, return {"question": null}.

Source: {source_desc}
Passage:
"""{passage}"""
```
Schema: `{"question": str|null, "missing_detail": str}`

**G3: grounded task (generator)**, where `{task}` ∈ summarize, extract_list, yes_no_check

```
You write instructions for a model that must work ONLY from passages it is given.
Write ONE instruction of type {task} that can be completed using only this passage, and its ideal
response.
- summarize: the instruction asks for a one-sentence summary; response at most 35 words, with no
  fact that is not in the passage.
- extract_list: the instruction asks for all parties, dates, amounts, or defined terms of one
  kind; response lists them exactly as written, separated by "; ".
- yes_no_check: the instruction asks whether the passage states a specific claim; response is
  "Yes." or "No." then a verbatim quote of at most 30 words that decides it. Make the claim false
  ("No.") about half of the time, by changing one detail of something the passage does state.

Source: {source_desc}
Passage:
"""{passage}"""
```
Schema: `{"instruction", "response", "quote"|null}`

**V1: blind verifier** (Gemini for training data; Sonnet 5 for dev/test and the 5% audit)

```
You check training examples for a model that must answer only from the passages it is given.
Using ONLY the passages below (no outside knowledge), decide:
- answerable: true only if at least one passage explicitly states the answer. Partial, implied
  or "probably" answers count as false.
- support: the numbers of every passage that states the answer on its own.
- answer: the answer in at most 30 words if answerable, else null.
- ambiguous: true if the question is unclear, or could refer to different things in different
  passages.

{passages, numbered 1..k, exactly as the student will see them}
Question: {question}
```
Schema: `{"answerable": bool, "support": [int], "answer": str|null, "ambiguous": bool}`

**J1: evaluation judge** (Sonnet 5; used only on student outputs that answer rather than refuse)

```
You grade a small model's answer. Use ONLY the passages; ignore outside knowledge.
- correct: the model's answer means the same as the reference answer. Missing the key fact, or
  adding a wrong claim, makes it false.
- supported: every factual claim in the model's answer is stated in the passages.

{passages}
Question: {question}
Reference answer: {reference}
Model answer: {model_answer}
```
Schema: `{"correct": bool, "supported": bool}`

## 7. Training recipe (brief task 3, part 3)

### 7.1 Hyperparameters

| Setting | Value | Why |
|---|---|---|
| Initialisation | fp32 weights from `/data/checkpoints/ckpt.pt` (step 15,564) | the bf16 export would lose master precision |
| Chat tokens 4–6 | embedding row ← mean of trained rows + N(0, 0.02 · row std) | untrained rows would inject noise at every turn boundary |
| Trained parameters | all (full fine-tune) | 125.8M fits trivially; LoRA would cost quality for no memory benefit |
| Precision | fp32 master weights + bf16 autocast, TF32, `torch.compile` | project invariant |
| Optimiser | fused AdamW, β (0.9, 0.95), ε 1e-8, weight decay 0.1 on matrices and embeddings only | same as pretraining (D-08) |
| Peak LR | **1e-4**; the pilot sweeps 5e-5 / 1e-4 / 2e-4 | pretraining ran 6e-4 → 6e-5; SFT should sit near the end of that range |
| Schedule | linear warmup over 3% of steps, cosine decay to 10% of peak | |
| Batch | 64 sequences (micro 32 × 2 accumulation) ≈ 65K tokens incl. padding | ~390 steps/epoch at 20K + replay |
| Epochs | **3**; keep the checkpoint with the best dev decision-F1 | small data, so watch for overfitting |
| Sequence | exactly 1,024 tokens, right-padded | one compiled shape |
| Loss | cross-entropy on assistant tokens + final `<|eos|>`, averaged per token over the batch | the prompt isn't a target |
| Replay | 1 raw train window per 4 SFT sequences (~20% of tokens), full LM loss | forgetting guard |
| Grad clip / dropout | 1.0 / 0 | as pretraining |
| Eval decoding | greedy, ≤ 128 new tokens, stop at `<|eos|>`, suppress ids 0, 2–6 | deterministic metrics |
| Seed | 1337 | |

**Compute.** 20K examples × 1.25 (replay) × 1,024 × 3 epochs ≈ 77M tokens. At the measured 425K
tokens/s (E-10), that's ~3 minutes of training plus ~3 minutes of startup and compilation on 1 × H100,
≈ **$0.75**. The pilot's 10 runs share one container: ≈ 25 min, ≈ **$2**. Each launch uses
`--max-usd` (5 for the pilot, 3 for the full run), with checkpoints every 100 steps.

### 7.2 Pilot sweep → choice

- **Runs:** LR {5e-5, 1e-4, 2e-4} × refusal share {15, 25, 35%} at 2,000 examples, plus a learning-curve
  point at 1,000 examples using the best setting.
- **Selection score on dev:** maximise grounded accuracy − hallucination rate − over-refusal rate, on
  deterministic metrics only (no judge cost).
- **Ties:** prefer the lower hallucination rate.

## 8. Evaluation (brief task 3, part 4)

All metrics are computed on the **frozen test set (1,000 items)**: 500 answerable QA, 300 unanswerable
(150 B1 / 100 B2 / 50 B3) and 200 grounded tasks, all from documents the model never trained on.
Dev is used only for selection and τ.

### 8.1 Headline metrics

| Metric | Definition | Target |
|---|---|---|
| **Hallucination rate** | share of items where the model **asserts an answer that the passages don't support**: it answers an unanswerable item, or J1 marks its answer `supported = false`, or its quote isn't verbatim | ≤ **15%** |
| **Refusal recall** (unanswerable) | refused / unanswerable, also reported per B1 / B2 / B3 | ≥ **0.80** |
| **Over-refusal rate** | refused / answerable (A + C) | ≤ **0.20** |
| **Grounded accuracy** (A) | correct (J1) **and** cited the oracle passage, over all answerable QA | ≥ **0.60** |
| Citation accuracy | cited passage = oracle, among answered A items | ≥ 0.80 |
| Quote fidelity | quote is a verbatim substring of the cited passage | ≥ 0.90 |
| Format validity | output parses into `Source:` / `Quote:` / `Answer:` | ≥ 0.99 |
| Refusal precision / F1, **AUROC of P(none)** | answer/refuse as a classifier, scored at the decision token | reported |

- **Breakdowns:** by source (case law / SEC / web), oracle position (1–4) and bucket.
- **Position bias:** accuracy across positions should vary by < 10 points.

### 8.2 Forgetting guards

- **Held-out bits/byte** on the fixed 600-window subset: increase ≤ **0.02** over the 7.1 baseline.
- **CaseHOLD** zero-shot accuracy no more than 2 points below baseline.

### 8.3 Reference points (for reading the numbers)

- The **base model** with a 1-shot plain-text prompt.
- **Always refuse:** hallucination 0, over-refusal 100%.
- **Teacher ceiling:** `gemini-3.8-flash` on 200 test items.

### 8.4 CaseHOLD baseline (7.1)

- **Data:** LexGLUE `case_hold`, test split, 3,600 items. Phase 2 already decontaminated the corpus
  against it.
- **Method:** zero-shot, scoring each of the 5 candidate holdings by length-normalised
  log-likelihood after the citing context, with an `<|eos|>` prefix. Report accuracy against 20%
  chance, with a 95% CI.
- **Purpose:** a general legal-reasoning measure that moves independently of the RAFT format.

The targets were set **before** seeing any results. If one is missed, the miss is recorded and
diagnosed, and the target is never quietly lowered.

## 9. Budget and gates

| Stage | Cap | Estimate | Basis |
|---|---|---|---|
| 7.3 bake-off | $5 | ≈ $2 | 100 passages × 2 generators + Sonnet verification of ~400 candidates |
| 7.4 pilot data | $25 | ≈ $20 | ~6,400 candidates generated and verified by Gemini (≈ $12) + Sonnet adjudication of the ~2,700 dev/test candidates (≈ $8) |
| 7.6 pilot GPU | $5 | ≈ $2 | 1 × H100, ~25 min |
| 7.8 scale data | $60 | ≈ $56 | ~27K candidates via Gemini ($0.0018 each ≈ $49) + 5% Sonnet audit (≈ $4) + final J1 (≈ $3) |
| 7.8 full SFT | $3 | ≈ $0.75 | 1 × H100, ~6 min |
| **Phase 7 total** | **$98** | **≈ $80** | |

- **Token assumptions:** generation is ~900 in / ~600 out per passage, giving 2 candidates.
  Verification is ~1,250 in / ~300 out. Kept/candidate yield is 65%. The bake-off measures all three,
  and the caps are re-quoted before 7.4 if they're off.
- **Spend guard:** `teacher.py` sums `usage` × price after every batch and refuses to submit a batch
  that would cross `--max-usd`.
- **GPU:** every entrypoint requires `--confirm`, and nothing may be left running afterwards
  (`modal container list` → 0).

## 10. Files to create or modify

| File | Purpose |
|---|---|
| `config.py` | add `RAFT_DIR`, a `RaftConfig` (k = 4, passage 60–160 tokens, mix ratios, sizes, refusal string, teacher model ids, prices) and an `SFTConfig` (§7.1) |
| `raft_data.py` | pure, unit-tested logic: chunking, passage filters, the split rule, prompt/target formatting, mechanical checks, output parsing, metrics |
| `raft_app.py` | Modal: passage-pool fan-out (one worker per corpus shard, keeping the fan-out design), BM25 build, distractor mining, dataset assembly; SFT GPU entrypoints gated by `--confirm` |
| `teacher.py` | local: build, submit and collect Gemini and Anthropic **batch** jobs; spend ledger; keys read from `.env.local` and never logged |
| `sft.py` | SFT trainer; imports helpers from `train.py` |
| `eval_raft.py`, `eval_casehold.py` | the metrics in §8 |
| `smoke_sft.py` | the local checks for 7.5 |
| `requirements.txt` | add `google-genai`, `anthropic`, `bm25s` (pinned) |

`cleaning.py`, `dedup.py` and `modal_app.py` are not touched.

## 11. Risks

| Risk | Mitigation |
|---|---|
| A 125M model may plateau at modest accuracy | Mostly extractive targets, a quote-first format, and pilot learning curves before scaling. A plateau is a result to record, not something to hide |
| Label noise in refusals (a distractor actually answers) → over-refusal | Blind verification, excluding adjacent chunks, the 5% cross-family audit with a switch rule |
| Ambiguous questions (a passage that never names its company) | G1 rule 2, the verifier's `ambiguous` flag, filtering out passages that lack context |
| Lexical shortcut for refusal | BM25 hard negatives retrieved with the question, B2 near misses, paraphrase enforcement |
| Position bias | Uniform oracle position; per-position metrics |
| Forgetting | Replay; bits/byte and CaseHOLD guards |
| Train/test leakage through near-duplicates | Split by document plus the 13-gram filter on test |
| Price and terms drift | Prices dated 2026-10-06 (Gemini "through Dec 31, 2026"); re-quoted at each gate; the user reviews the terms |
| 1,024-token context limits multi-passage reasoning | Out of scope; extending context to 2,048 is a separate option (a few dollars of continued pretraining) |

## 12. Decisions needed from the user

1. **Teacher setup:** `gemini-3.8-flash` (generation + training verification) plus `claude-sonnet-5`
   (dev/test adjudication, judge, audit). That requires adding `ANTHROPIC_API_KEY` to `.env.local`.
   Alternative: Gemini only, with `gemini-3.1-pro-preview` as the independent judge (same family, less
   independent).
2. **Phase 7 budget:** ≈ $80 expected, $98 in total caps, approved stage by stage (7.3 → 7.4 → 7.6
   → 7.8).
3. **Gemini billing:** confirm the key's project is on the paid tier.
4. **Terms:** acknowledge §2's note on using teacher outputs.
5. **Scale:** confirm 20K as the default; it's revisited with pilot data in 7.7.

## 13. Results

_Not started. Each step's measured results get recorded here._
