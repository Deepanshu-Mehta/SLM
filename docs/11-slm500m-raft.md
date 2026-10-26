# Phase 10: RAFT on a 500M model pretrained on the same corpus (does 4× parameters help?)

**Status: PLAN (written 2026-10-27, before any spend). Step 10.0 (investigation, $0) is done; nothing
else has run.** Every GPU step stops at its own gate.

Phase 9 showed SLM-125M's grounded-QA gap is model size, not data: Gemma 2 2B + LoRA met all seven targets
(test grounded 0.912) on rows where SLM-125M-RAFT reached 0.316. But Gemma differs in two ways at once,
**parameters (2.6B) and pretraining (≈ 2T diverse tokens)**. `thesreedath/slm-500m-base` holds pretraining
nearly fixed (the same three sources, same recipe family) and changes only size, so it can separate the two:

> **On the same RAFT data and evaluation, how much does 4× parameters (125M → 518M) improve grounded QA?**

## 1. What the model is (verified 2026-10-27; 10.0)

| | `thesreedath/slm-500m-base` | SLM-125M (ours) |
|---|---|---|
| Hub revision | `4090f98f50ceaa06246649ab5ff38471415d9b9a`, **Apache-2.0** | ours |
| Parameters / geometry | 517,795,840; 24 layers, hidden 1280, 20 heads (MHA, head dim 64), SwiGLU 3456, RoPE 10,000, tied | 125,848,320; 12 × 768 |
| Context | 1,024 | 1,024 |
| Tokenizer | own byte-level BPE, **32,768** | own, 16,384 |
| Special tokens | `<|bos|>` 0, `<|eos|>` 1, `<|pad|>` 2, `<|unk|>` 3, `<|user|>` 4, `<|assistant|>` 5, `<|system|>` 6: **the same layout** | same |
| Documents | each followed by `<|eos|>` (their `modal_app.py` packing) | same |
| Corpus | HFforLegal/case-law `us` (all), PleIAs/SEC (all), fineweb-edu sample-10BT (first ≈ 0.5B tokens): **the same sources and budgets as ours** (their `config.DATA_MIX` = ours); ≈ 2.19B tokens | 2.04B train tokens |
| Validation split | **by window: every 100th 1,024-token window** (`VAL_EVERY_N_WINDOWS`) | **by document** (`blake2b(text) % 100`) |
| Pretraining | ≈ 5 epochs (their `PRETRAIN_EPOCHS = 5`; the card says "n/a"), 8 × B200 | 4 epochs |
| RAFT decision tokens | ` none` 5416 vs ` Pass` 12582 + `age` 483 + digit (308, 389, 423, 547) | 5494 vs 12175 + 479 + digit |

**Checks in our code:**
- `raft_decode.format_for` detects it as the SLM format, so `decision_ids`, `RaftConstraint`, the copy
  constraint and the SLM generation config work unchanged. Its special-token ids equal ours, so the
  hard-coded `SUPPRESS`/`GREEDY` ids are also right.
- **Its `config.json` is inconsistent:** `bos_token_id 1`, `eos_token_id 2`, but the tokenizer has
  `<|eos|>` = 1 and 2 = `<|pad|>`. Our decoding never reads those fields. The exported RAFT model will
  carry corrected ids (bos 0 / eos 1 / pad 2).
- **The same author published a 125M base on the same corpus and split** (`thesreedath/slm-125m-base`,
  16K vocab, 10 epochs, licence "other"). They also published a QA → instruction → DPO → RAFT chain at
  500M (`slm-500m-qa`, `-instr`, `-dpo2`, `-raft-dpo`; licence "other" or none). Those are used only for
  internal comparison (§5), never republished.

## 2. The training/test overlap problem

**By construction, our dev/test passages are probably in its training data.**
- Our dev/test passages come from documents in **our** validation split.
- The 500M's corpus has the same documents, but its window-level split puts ≈ 99% of every document's
  windows into training.
- Unless its cleaning or dedup dropped them (its `cleaning.py` and `dedup.py` differ from ours), it was
  trained on them, about 5 times.

**But the measured effect is small (10.0, local, $0, E-40).**

| Bits/byte (120 passages per set, 40 per source) | our train passages | our dev/test passages |
|---|---|---|
| SLM-125M (never saw dev/test) | 0.819 | 0.867 |
| slm-500m-base | 0.770 | 0.809 |
| **500M / 125M, per passage** | **0.925** | **0.930** |

If the 500M had memorised dev/test the way models memorise training text, its advantage over SLM-125M would
be **larger** on dev/test than on train. Calibration: SLM-125M's own end-of-training train/val gap was
0.074 nats (3.5%), which would put the dev/test ratio near 0.89. It is the same, 0.930 vs 0.925.

What this does and does not show:
- **Shown:** familiarity with the dev/test passages is not measurably higher for the 500M.
- **Not shown that it didn't see them:** passage-level memorisation at this scale is small. SLM-125M itself
  is barely better on its own training passages for case law (+0.011) and web text (−0.045).
- **The SEC gap (+0.18 for both models) is a selection effect, not memorisation.** Dev/test SEC passages
  were chosen with < 10% 13-gram overlap with training filings (E-20), so they are atypical for both
  models.

**How the plan handles it (layered; the headline is chosen in advance, §4):**
1. **Stronger membership test (10.1, $0).**
   - Add a third set that **provably neither model saw**: fineweb-edu sample-10BT text from beyond both
     corpora's ≈ 0.5B-token prefix.
   - Use 600 passages instead of 120, and add the author's 125M (same corpus and split as the 500M).
   - Unseen text gives each model its true "unseen" baseline. Any extra advantage of the 500M on dev/test
     over that baseline estimates memorisation.
2. **Closed-book probe after RAFT (10.6).** Each RAFT model gets the test questions **with no passages**.
   A correct answer without context (J1) is knowledge that didn't come from reading. Grounded accuracy is
   also reported excluding those items (as in Phase 9).
3. **A control with the same overlap (10.7, optional).** RAFT on the author's 125M, which has the same
   corpus and window split as the 500M. 500M vs their 125M is a size comparison with the overlap held
   equal; their 125M vs ours estimates what the overlap is worth at fixed size.
4. **A fresh test set (10.8, optional, only if 10.1 or 10.6 flag overlap).** About 600 items from
   documents published after both corpora (2026 EDGAR filings, recent court opinions, late fineweb-edu
   shards), built with the Phase 7 pipeline. It is free of overlap for every model, including Gemma.

## 3. How we use the model: RAFT recipe (identical to SLM-125M's, D-23/D-27)

**Same:**
- the 20,340 train rows, dev 499, test 1,000 (frozen);
- full fine-tuning, fp32 master weights + bf16 autocast, `torch.compile`;
- LR from a 2-point pilot (below), cosine (3% warmup, to 10%);
- λ 1.0 replay of pretraining windows, 35% refusal, batch 64 (52 SFT + 12 replay), 3 epochs;
- `keep="select"`: the best dev `copy2` proxy among evals with held-out bits/byte ≤ +0.02;
- the pre-registered τ rule, `copy2` decoding, J1, the §8.1 scorer.

**Adapted, because the model uses a different tokenizer:**
- **Rows are re-tokenized** with the 500M tokenizer. The prompt text is unchanged: `<|eos|><|user|>…<|assistant|>Source:`.
- **Replay windows:** ≈ 30M tokens of **our train corpus** (the same documents as its corpus) re-tokenized
  and packed into 1,024-token windows. That is in-distribution replay, unlike Gemma (§3 of
  `docs/10-gemma.md`). 7.8 used 14M replay tokens per 20K run.
- **Chat tokens 4–6** get the same initialisation as SLM-125M (`sft.init_chat_tokens`). Their embedding
  norms don't show whether they were trained, and the documents never contain them.
- **The forgetting gate:**
  - measured on the same 600 held-out windows, decoded and scored per UTF-8 byte (tokenizer-free, as for
    Gemma), relative to the 500M's own base;
  - for the 500M these windows are probably training text, so the gate measures drift, not held-out
    quality;
  - bits/byte on the unseen fineweb set (10.1) is reported alongside;
  - CaseHOLD (decontaminated in both corpora) is reported, not gated (D-23).
- **The LR pilot at 5K: 5e-5 (SLM-125M's choice) and 2e-5.** Larger models usually want a lower LR, and
  Phase 7/9 found LR is the forgetting lever.
  - **Pre-registered choice:** the higher dev J1 grounded accuracy among gate-passing arms.
  - If neither passes, stop and bring it back.
- **Learning curve:** 5K (pilot) and 20K. That matches SLM-125M's 5K and 20K points, so data efficiency
  can be compared too.

**Not used:** LoRA or QLoRA. At 518M, full fine-tuning (≈ 8 GB of weights and optimizer states) is cheap
and is what SLM-125M got, which keeps the size comparison clean.

## 4. Pre-registered reading (fixed now)

**Headline:** test grounded accuracy (`copy2`, J1) of the 20K 500M model against SLM-125M-RAFT's 0.316. It
is taken on the original test set **unless** 10.1 shows memorisation (500M dev/test advantage beyond its
unseen baseline ≥ 2× the run-to-run noise) **or** 10.6 shows the 500M answering ≥ 5 points more test
questions closed-book than SLM-125M. In either case the fresh set (10.8) becomes the headline.

| Δ grounded (500M − 125M) | Reading |
|---|---|
| **≥ +0.15** | Parameters matter a lot at this data scale: a ~1B model on the same corpus is worth trying before larger pretraining |
| **+0.05 – +0.15** | Parameters help but explain a small part of Gemma's gain (+0.60); most of it is pretraining scale and diversity |
| **< +0.05** | 4× parameters on the same 2B-token corpus buys almost nothing; Gemma's gain came from pretraining, not size |

**Prediction, written down to keep us honest:** Δ ≈ +0.05 to +0.15. The 500M is only ≈ 7% better in
bits/byte on this text (Gemma: ≈ 12%), and at 125M the learning curve gave +0.045 per data doubling.

**Also reported:**
- hallucination, refusal recall (B2 especially), forgetting;
- the 5K vs 20K curve for both sizes;
- with 10.7, the size effect with the overlap held equal.

## 5. Optional: does QA → instruction → DPO before RAFT help? (your earlier pipeline question)

The author's chain gives this nearly for free.
- **10.9a: eval only (≈ $0.3 GPU + J1).** Score `slm-500m-raft-dpo` (their base → QA → instruction → DPO →
  RAFT) on our test.
  - Their prompt format differs (`<|bos|><|system|>…<context>`), so it is evaluated in its own format, with
    the answer/refuse decision parsed from text.
  - Their data and refusal mix also differ, so this is indicative, not controlled.
- **10.9b: controlled (≈ $2.5).** Run our exact RAFT recipe starting from `slm-500m-instr` (their base → QA →
  instruction) instead of the base. The only difference from 10.6 is the QA + instruction stages.
  - **Caveat:** we can't inspect their QA and instruction data, which may include our test documents.
    10.9b inherits that risk and is reported with it.

## 6. Steps and cost

| Step | What | Where | Estimate | Cap |
|---|---|---|---|---|
| 10.0 | Investigation: model, corpus, split, tokenizer, licence, first membership test | local | **$0, done** | — |
| 10.1 | Membership test v2: 600 passages × {train, dev/test, unseen fineweb} × {our 125M, their 125M, 500M} | local CPU (or GPU cents) | $0 | — |
| 10.2 | Port: load an HF base, foreign-tokenizer row encoding, tokenizer-free gate, corrected export ids; local tests (125M path byte-identical) | local | $0 | — |
| 10.3 | Replay store: ≈ 30M tokens of our train corpus re-tokenized for the 500M (fan-out per shard) | Modal CPU | cents | — |
| 10.4 | GPU probe: step time and memory with `torch.compile`; base guards | 1 × H100 | ≈ $0.3 | 0.5 |
| 10.5 | Pilot: 5K × LR {5e-5, 2e-5}, 4 evals each; J1 on dev | 1 × H100 | ≈ $2.3 | 3.0 |
| 10.6 | 20K at the chosen LR, 8 evals; τ; test once; closed-book probe; guards; export | 1 × H100 | ≈ $3.4 | 4.5 |
| J1 | pilot dev ×2, 20K dev, test ×2, closed-book | Gemini batch | ≈ $0.7 | 1.5 |
| **Core** | | | **≈ $6.7** | caps $9.5 |
| 10.7 | Control with the same overlap: author's 125M, 20K, test (internal only) | 1 × H100 | ≈ $1.1 | 1.5 |
| 10.8 | Fresh test set (only if flagged) + scoring all models on it | APIs + GPU | ≈ $2.5 | 3.5 |
| 10.9a / b | Their RAFT-DPO chain on our test / our RAFT from their instruction checkpoint | 1 × H100 | ≈ $0.3 / $2.5 | 0.5 / 3.0 |

**Estimates:** compute scales about 4.1× from 7.8's measured 125M run (≈ 1.25 s per step with
`torch.compile`). The 10.4 probe re-derives them. If 10.5 or 10.6 comes out more than 30% higher, it comes
back to you.

**Operations:**
- Launches run under `caffeinate -i` and `nohup` (C-17).
- Per-run results are saved to the Volume, so a restart resumes at the next run.
- 0 containers are verified after each step.

## 7. Validation
- 10.2: the SLM-125M decoding and training path is byte-identical (Phase 7 suite 77/77); all 21,839 rows
  encode with the 500M tokenizer, with targets starting with the forced decision tokens; the replay store
  decodes back to source text.
- 10.5 and 10.6 follow the pre-registered rules. The test is used once per model, after τ is fixed.
- The headline set is chosen by the §4 rule before the test numbers are read.
- Nothing is published (the 500M RAFT model could be, under Apache-2.0, after a separate approval).
