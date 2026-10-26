# Phase 9: Gemma 2 2B on the same grounded data (comparison with SLM-125M-RAFT)

**Status: APPROVED, LoRA only (D-30, 2026-10-26): "start with gemma2B raft-using lora".** The full
fine-tuning control, Gemma 3 270M, the quantization check and best-of-8 are deferred, not dropped.
9.1 (code, $0) in progress; every GPU step still stops at its own gate.

Phase 7 left three gated targets missed (test, `copy2`, τ 0.7): grounded accuracy 0.316 (target 0.60),
hallucination 0.325 (0.15) and refusal recall 0.780 (0.80). The model cites the right passage 90% of
the time but answers correctly a third of the time, and data scaling adds only ≈ +0.045 per doubling
(E-35). This phase asks one question before any preference optimization:

> **Is the gap the data and recipe, or the size of the model?**

Gemma 2 2B (2,614,341,888 parameters, about 2T pretraining tokens) is trained on **the same 20,340
rows**. It is evaluated on **the same frozen dev (499) and test (1,000) sets**, with the same
decoding, the same τ rule, the same checkpoint rule and the same J1 judge. Everything that differs
between the two models is listed in §3.

## 1. Pre-registered reading of the result (fixed now, before any run)

Using test grounded accuracy (J1, `copy2`) of the 20K Gemma model:

| Gemma 2 2B grounded | Reading | What Phase 10 does |
|---|---|---|
| **≥ 0.60** | The data, labels and scoring are sound. SLM-125M is limited by its size | Preference optimization on Gemma, and/or distil Gemma into the 125M model. Preference optimization on the 125M model only if its best-of-8 headroom (9.8) is ≥ +0.15 |
| **0.45 – 0.60** | Both: the student's size matters and the data has a ceiling | Read 100 of Gemma's dev failures by category before choosing |
| **< 0.45** | The data or labels are the bottleneck, not the model | Audit the training labels before any preference work: preference training on flawed labels amplifies them |

The same reading applies to hallucination (≤ 0.15) and near-miss refusal (B2). The closed-book probe
(9.3) qualifies all of it: if Gemma can answer test questions **without** the passages, its grounded
accuracy is reported both raw and on the items it could not answer closed-book.

## 2. Technique: LoRA in bf16; no QLoRA; quantization is measured at inference only

| Option | Memory for 2.6B on 1 × H100 80 GB | Verdict |
|---|---|---|
| **Full fine-tuning** (fp32 master weights + AdamW + bf16 autocast, our recipe) | ≈ 42 GB for weights, gradients and Adam states, plus activations; fits | **Control arm at 5K only (9.4).** It is what SLM-125M got, so it answers "did LoRA hold Gemma back?" |
| **LoRA, bf16 base** | base 5.2 GB, adapters ≈ 42M params (r 32); trivial | **Main method.** For SFT sets this small, LoRA on all linear layers matches full fine-tuning when its LR is ~10× higher ("LoRA Without Regret", Thinking Machines, 2025), and it forgets less (Biderman et al., 2024). That matters because our bits/byte gate is +0.02 |
| **QLoRA** (4-bit NF4 base) | base ≈ 1.6 GB | **Not used.** It saves memory we don't need (5.2 GB of 80 GB). It is slower (every matmul dequantizes the base) and trains the adapters against a 4-bit-rounded base. That rounding is a confound in a "best Gemma can do" comparison. QLoRA is for a 7B+ base on a small GPU |
| DoRA and other LoRA variants | — | Not used: they add a variable without addressing our question |

**LoRA settings** (fixed now; only the LR is swept):
- rank 32, α 32, no dropout;
- targets: all linear layers in attention and MLP (`q,k,v,o,gate,up,down`);
- embeddings and LM head frozen. They are tied, 256K × 2304 = 590M parameters, and the format uses
  only ordinary text tokens (§3);
- adapters in fp32, frozen base in bf16, bf16 autocast;
- AdamW, weight decay 0 on the adapters, gradient clip 1.0;
- the same schedule as Phase 7: 3% warmup, cosine to 10%;
- **52 SFT rows per optimizer step** (Phase 7's 64-sequence batch held 52 SFT rows and 12 replay
  windows; with no replay, 52 keeps the SFT batch identical), as micro-batches of 13 × 4, 3 epochs.
  LoRA degrades at large batch sizes, and 52 is moderate;
- LR swept over {1e-4, 3e-4} in the pilot (9.4).

**Full fine-tuning control (deferred, D-30):** LR 1e-5, the same schedule and batch. The tied embeddings are trained
(as SLM-125M's were).

**Quantization is for inference only, measured rather than assumed (9.7, optional):**
- the merged model is evaluated in bf16 (the headline), int8 (LLM.int8) and NF4 on dev;
- this tells us what quantization would cost if Gemma were ever served;
- Gemma is **not** served in this phase. The API runs on 2 CPU cores / 4 GiB, which can't hold
  2.6B parameters in bf16 (5.2 GB), and an int8 model there would likely take tens of seconds per answer (not measured).

## 3. What is identical, and what differs

**Identical:**
- train rows (20,340), dev (499), test (1,000) and the probe;
- the target text: `Source: Passage k\nQuote: "…"\nAnswer: …` or the refusal;
- loss on the target tokens only;
- the decision rule s = P(none) / (P(none) + P(Passage)) and the τ rule (best macro-F1 among τ with
  over-refusal ≤ 0.20 and recall ≥ 0.80);
- `copy2` decoding and checkpoint selection `keep="select"` (the best dev `copy2` proxy among evals
  passing the bits/byte gate);
- J1 (`gemini-3.8-flash` batch) and the §8.1 scorer;
- the test set is used once, after the model and τ are fixed.

**Different (each is forced by the model, or chosen and stated):**

| | SLM-125M-RAFT | Gemma 2 2B |
|---|---|---|
| Prompt start | `<|eos|>` (BOS never seen in pretraining) | `<bos>` (Gemma requires it) |
| Role markers | `<|user|>`, `<|assistant|>` | Plain text: `Passage k: …\nQuestion: …\nSource:`. Gemma's `<start_of_turn>` tokens may be untrained in the base model, and LoRA leaves embeddings frozen (9.1 checks their embedding norms) |
| Special ids | bos 0 / eos 1 / pad 2 / unk 3 | **pad 0 / eos 1 / bos 2 / unk 3**: bos and pad are swapped relative to ours |
| Decision tokens | `Ġnone` 5494 vs `ĠPass` 12175 (`ĠPassage` = 2 tokens) | `▁none` 8031 vs `▁Passage` 81975 (1 token); **digits split**: ` 1` → `▁`, `1` |
| Rows, in tokens (measured) | prompt mean 594 / max 709; target mean 32 | prompt mean 607 / p99 804 / max 951; target mean 34 / max 179; 13.0M tokens per epoch |
| Replay | λ 1.0 on our pretraining windows | **None.** Our corpus is not Gemma's pretraining data, so replay would be domain-adaptive pretraining (a confound). LoRA plus the bits/byte gate guards forgetting instead |
| Attention | SDPA, `torch.compile` | `eager`: Gemma 2's attention-logit soft-capping isn't applied under SDPA. Checked in 9.2 |
| Context | 1,024 | 8,192 (the inputs are identical, so it isn't used) |

**Forgetting guards for Gemma** (the same limits, against Gemma's own baseline):
- held-out bits/byte on the same 600 windows: decoded to text, re-tokenized, and scored per UTF-8
  byte, so it doesn't depend on the tokenizer. The gate is ≤ +0.02 over Gemma's base;
- CaseHOLD PMI (reported).

Gemma's **base** bits/byte on our held-out set is itself a reported result, next to our 0.6572.
It carries a caveat: case law and EDGAR filings are public and may be in Gemma's pretraining data.

## 4. Steps

### 9.0 Prerequisites (the user, $0)
1. **Accept the Gemma license on Hugging Face** for the account `Deepanshu027`:
   - `google/gemma-2-2b` and `google/gemma-2-2b-it`;
   - `google/gemma-3-270m` too, if the 9.6 arm is approved.

   All three are gated ("manual"). The token currently gets 403, checked 2026-10-26.
2. **Create the Modal secret** `huggingface-token` on `deepanshu-m23csai`. Containers need it to
   download the gated weights.
   ```bash
   ! bash -c 'set -a; source .env.local; set +a; .venv/bin/modal secret create huggingface-token HF_TOKEN="$HUGGINGFACE_TOKEN" HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN"'
   ```
3. **Terms:** Gemma's Terms of Use allow research fine-tunes. Publishing a Gemma derivative (not
   part of this phase) would carry those terms. That is a separate decision.

### 9.1 Port the code (local, $0)
- **`raft_decode.py` gets a per-family format.** Today it hard-codes:
  - our special ids, the suppress list (ids 4–6) and `<|eos|>`;
  - `Ġ` + `Pass` + `age` for the decision tokens;
  - a single-token ` k` digit.

  A small `Format` (prompt builder, special ids, suppress list, the forced tokens after the decision,
  digit tokens) is derived from the tokenizer, with loud failures like `decision_ids` today.
  **SLM-125M behaviour stays byte-identical:** the full local suite (77/77) must still pass, and
  released-model outputs on a sample of saved test generations must be unchanged.
- **`CopyConstraint` with a 256K vocabulary:**
  - The vocabulary strings are built from single-token decodes. Measured with the Gemma
    `tokenizer.json`, the leading space is kept (`▁none` decodes to `" none"`). This is re-checked
    with the official tokenizer class.
  - Full-vocabulary scans (forced scaffold, empty-quote start) become 16× larger, so their timing is
    measured.
- **New `gemma_app.py` with its own image.** The released 125M image stays untouched:
  - pinned `transformers` (≥ 4.50, so Gemma 3 also loads) and `peft`;
  - the GPU entrypoints require `--confirm` and have a `--max-usd` guard;
  - results are saved per run; each step resumes at its next unfinished run.
- **Trainer** (reusing `sft.py`'s plan, schedule, eval hooks and `keep="select"`):
  - dynamic padding, to the micro-batch maximum rather than 1,024;
  - **logits computed only at target positions.** Targets are about 5% of tokens, and full logits at
    256K vocab would take about 8 GB per micro-batch;
  - **Gemma 2's final logit soft-cap (30) applied** to those logits;
  - micro-batch sized from a measured memory probe.
- **Tests:** a tiny random Gemma 2 configuration runs on CPU (no download): format round-trip,
  decision ids, the copy constraint on Gemma tokens, a target-only loss equal to the full-logit loss,
  LoRA merge equivalence, and the resume path.

### 9.2 Container probe (H100, ≈ $0.3, `--max-usd 0.5`)
- Load `gemma-2-2b`.
- Compare `eager` and SDPA logits and speed on 32 dev prompts.
- Measure tokens/s and peak memory for LoRA and full fine-tuning at the chosen micro-batch.
- Time one dev `copy2` generation pass.
- **The cost estimates for 9.4 and 9.5 are re-derived from these numbers.** If either estimate rises
  more than 30% above this plan, stop and bring it back.

### 9.3 Baselines without fine-tuning (L40S, ≈ $0.6, `--max-usd 1.0`)
- **Gemma 2 2B base:** held-out bits/byte (pooled and per source) and CaseHOLD PMI. These are the
  guard baselines.
- **Gemma 2 2B instruct, few-shot:**
  - two worked examples in the user turn, the model turn prefilled with `Source:`;
  - τ is chosen on dev by the same rule, then test runs once (plain and `copy2`);
  - this is the "just prompt a bigger model" reference.
- **Closed-book probe:** Gemma 2 2B instruct answers the 500 test A questions **without passages**.
  The J1 judge compares them with the references. This measures how much Gemma already knows about
  the test documents.

### 9.4 Pilot at 5K: two LoRA learning rates (H100, ≈ $2.0, `--max-usd 3.0`)
- **Data:** `select_mix(5,000, 0.35)` from the 20K pool. It is the same draw as Phase 7's 5K curve
  point (0.273 judged grounded for SLM-125M).
- **Arms:** LoRA LR 1e-4 and LoRA LR 3e-4 (the full fine-tuning arm is deferred, D-30).
- Each arm has 3 epochs and 2 evals (mid-run and final), with `keep="select"`.
- **J1 on the selected dev outputs of both arms.**
- **Pre-registered choice for 9.5:**
  - the LoRA LR with the higher dev J1 grounded accuracy among gate-passing arms;
  - If no arm passes the gate, stop and bring it back.

### 9.5 Scale to 20K, final test (H100, ≈ $4.3, `--max-usd 6.0`)
- The chosen recipe on all 20,340 rows: 3 epochs, 4 evals, `keep="select"`.
- Then the τ table and dev generations (plain and `copy2`).
- **Test once** (plain and `copy2`).
- CaseHOLD and per-source bits/byte.
- **Saved to `/data/gemma/runs/full/`:**
  - the adapter (or the full weights);
  - the merged bf16 model;
  - `raft_format.json`, `final.json`, `results.jsonl` and `log.txt`.
- Local copies of all generations go in `.smoke/gemma/`.

### 9.6 Optional: Gemma 3 270M, the size-matched control (H100, ≈ $1.0, `--max-usd 1.5`)
- Its transformer has about 100M parameters, against SLM-125M's 113M non-embedding parameters. It saw
  about 6T pretraining tokens, against our 8.2B.
- This separates "more parameters" from "more and better pretraining".
- The same pipeline, at the 9.4 LoRA recipe, re-checked with one LR pair at 5K. Then 20K and test.

### 9.7 Optional: inference quantization (L40S, ≈ $0.3, `--max-usd 0.5`)
- The merged 9.5 model in int8 and NF4: dev `copy2` at the 9.5 τ.
- J1 only on outputs that changed versus bf16.
- Reports the change in grounded accuracy and hallucination. Weights only; the base is not
  re-trained.

### 9.8 Optional: best-of-8 headroom for both models (L40S, ≈ $0.6, `--max-usd 1.0`)
- On dev A and C items:
  - the decision is fixed at τ;
  - the quote and answer are sampled 8 times (temperature 0.7, `copy2` kept on);
  - J1 judges only the distinct outputs.
- **Reported:** best-of-8 vs greedy grounded accuracy for SLM-125M-RAFT and Gemma. This is the input
  to Phase 10's choice (§1).

### 9.9 Write-up ($0)
- §13 results; the comparison table (zero-shot, the 125M model, Gemma LoRA, Gemma full fine-tuning
  at 5K, and Gemma 3 270M if run).
- The pre-registered reading from §1, stated plainly whichever way it falls.
- `EXPERIMENTS.md` entries, `MASTER_PLAN.md`, and 0 containers verified.

## 5. Budget

Estimates from FLOPs, before the 9.2 probe. Assumptions:
- LoRA ≈ 4 × 2.0B non-embedding parameters × tokens; about 39M training tokens for the 20K run;
  25% MFU with eager attention;
- generation timed from Phase 7's evals, scaled by model size.

| Step | GPU | Estimate | `--max-usd` |
|---|---|---|---|
| 9.2 probe | H100 | ≈ $0.3 | 0.5 |
| 9.3 baselines | L40S | ≈ $0.6 | 1.0 |
| 9.4 pilot (2 LoRA arms × 5K) | H100 | ≈ $2.0 | 3.0 |
| 9.5 20K + test | H100 | ≈ $4.3 | 6.0 |
| J1: pilot dev ×2, final dev, test ×2, zero-shot, closed-book (≈ 6,000 items) | — | ≈ $1.1 | stage cap 2.0 |
| **Core total (LoRA only, D-30)** | | **≈ $8.4** | **cap $12.5** |
| 9.6 Gemma 3 270M (optional) | H100 | ≈ $1.0 | 1.5 |
| 9.7 quantization (optional) | L40S | ≈ $0.3 | 0.5 |
| 9.8 best-of-8 (optional) | L40S | ≈ $0.6 (+ J1 ≈ $0.2) | 1.0 |

**Chosen (D-30): LoRA only.** Without the full fine-tuning arm, the objection "LoRA held Gemma back"
stays open; the 5K pilot shows how far LoRA gets, and the arm can still be run later (≈ $1.2).

Every GPU step stops at its own gate with the measured estimate, `--max-usd`, duration, GPU count,
steps and resume strategy. Afterwards, `modal container list` must show 0.

## 6. Validation (the phase counts as done only if all hold)
- The 125M decoding is byte-identical after the port: 77/77 passes and the saved test generations
  are unchanged.
- The CPU tests for the Gemma format, the loss and the LoRA merge pass.
- The Gemma base model's guard baselines are recorded before any training.
- The 9.4 choice follows the pre-registered rule. The 20K model passes the bits/byte gate, or the
  failure is reported, never silently accepted.
- Test is used exactly once per model, after τ is fixed. Every §8.1 metric is reported for Gemma
  and side by side with SLM-125M-RAFT.
- The closed-book probe is reported, with grounded accuracy on items not answered closed-book.
- Nothing is published or deployed in this phase. Spend stays within the caps, and 0 containers
  remain afterwards.

## 7. Risks
- **Contamination:** Gemma may have seen the case law and SEC documents. The closed-book probe
  measures it, and the 13-gram scan can't (Gemma's data isn't public).
- **Format mismatch on the instruct baseline:** a few-shot instruct model may break the format.
  Format validity is reported, and `copy2` plus the forced decision keep it parseable.
- **Copy-constraint speed at 256K vocabulary:** measured in 9.2. If it's too slow, the full-vocab
  scans get a precomputed index (same behaviour, checked by the 125M equivalence test).
- **The pilot LR grid is two points:** if the better LR is at the edge (3e-4 best and still
  improving), one extra 5K LoRA arm at 6e-4 (≈ $0.8) is proposed, not run automatically.

## 8. Results

### 9.1 Port the code: DONE (2026-10-26, $0)
**Code:**
- **`raft_decode.py`:**
  - a per-family `Format` (SLM, GEMMA) picked from the tokenizer by `format_for`;
  - `decision_ids` derives every id from the tokenizer (decision tokens, the tokens forced between
    " Passage" and the digit, suppressed and stop ids). Gemma: `▁none` 8031 / `▁Passage` 81975, forced
    `▁`, digits `1`–`4`, suppress `<pad> <bos> <unk> <mask> <start_of_turn>`, stop `<eos> <end_of_turn>`;
  - the forced-scaffold vocabulary scan is memoised per remaining suffix (same result, once per batch);
  - the single-token vocabulary cache is now keyed per tokenizer.
- **`eval_forgetting.score_casehold`:** optional `prefix_id` and `softcap`. The defaults are the SLM
  behaviour.
- **`gemma_sft.py`:**
  - LoRA via PEFT 0.13.2, adapters fp32 over the bf16 base;
  - target-only soft-capped logits and a chunked CE;
  - the training loop with the sft.py loss contract and keep="select";
  - teacher-forced and prompt-end decision scores;
  - tokenizer-independent held-out bits/byte;
  - the instruct few-shot and closed-book prompts.
- **`gemma_app.py`:**
  - its own image (torch 2.5.1, transformers 4.46.3, peft 0.13.2, accelerate 1.1.1);
  - `fetch` (CPU) and `probe` / `baselines` / `pilot` / `full` (GPU, `--confirm`, per-stage caps);
  - 9.4 and 9.5 refuse to start until the 9.2 probe timings are recorded in `config.GEMMA`.
- **`config.py`:** `GemmaConfig` with pinned Hub revisions (base `c5ebcd4…`, instruct `299a856…`) and
  `GEMMA_STAGE_CAPS_USD`.

**Two plan details changed while coding:**
- **52 SFT rows per step**, not 64 sequences. That is Phase 7's SFT share of its batch, so the SFT
  batch stays identical without replay (§2).
- **The probe cap went from $0.3 to $0.5.** Counting container start and two 5 GB model loads, the
  probe is ≈ 4 H100-minutes, not 2, so it measures subsets and extrapolates.

**Validation:**
- **Phase 7 suite: 77/77** with the ported `raft_decode.py`.
- **`smoke_gemma.py`: 38/38.**
  - **Format, all 21,839 stored rows:**
    - one `<bos>` first and `<eos>` last;
    - every target starts with exactly the decision tokens the constraint forces;
    - longest row 985 tokens (≤ 1,024);
    - no Gemma control strings in any row;
    - the instruct prompt is 1,605 tokens.
  - **SLM unchanged:**
    - the committed vs the working `raft_decode.py` on the released weights (CPU): 24 test rows, plain
      and `copy2`, identical text, choice and s;
    - `answer()` identical;
    - CaseHOLD scorer identical on all 60 log-probs.
    - Reported, not gated: CPU fp32 matches the saved H100 bf16 generations on 23/24.
  - **Copy constraint on Gemma's 256K vocabulary:** 12 random walks on real dev passages all end in a
    verbatim quote of ≤ 240 characters, then the forced `\nAnswer:`.
  - **Tiny random Gemma 2** (real tokenizer, CPU):
    - target-only CE equals the HF full-logit (soft-capped) CE;
    - teacher-forced s = prompt-end s = HF forward;
    - held-out bits/byte equals a hand computation;
    - keep="select" rule;
    - zero-init LoRA leaves outputs unchanged;
    - only fp32 adapters train;
    - the loss falls;
    - the adapter state round-trips; runs are deterministic from the same init;
    - merge matches (max 6e-7);
    - generation forces the decision and a valid digit.
  - **Gates:** refuse without `--confirm` and over the cap.
- **Environment:** the first combined run was killed for memory on the 8 GB laptop (full 256K-vocab
  logits). The LoRA comparisons now use a 2 × 64 slice.
