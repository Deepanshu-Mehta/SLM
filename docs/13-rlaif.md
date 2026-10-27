# Phase 12: preference optimization with AI feedback (RLAIF)

**Status: PLAN (written 2026-10-27, before any code or spend). Not approved yet.** References: the
course's RLHF walk-through (https://slm-rlhf.vercel.app: reward model, PPO with a KL leash, reward
hacking) and its preference-data page (https://slm-rlhf-data.vercel.app). The data page's method:
- K on-policy samples per prompt, ranked by an AI judge that follows a written constitution;
- C(K,2) pairs from each ranking;
- the warning that length bias is "the quiet killer".

## 1. What preference optimization should fix (measured, not assumed)
Every RAFT model already gets the format right. What is left is **choosing**:
- whether to answer (near-miss refusal);
- which span to answer from (hallucination on answerable questions).

| Frozen test (J1, `copy2`) | Grounded | Hallucination (target ≤ 0.15) | Near-miss (B2) refusal | Over-refusal | Bits/byte headroom to the +0.02 gate |
|---|---|---|---|---|---|
| **SLM-125M-RAFT** | 0.316 | **0.325** | **0.56** | 0.048 | **+0.011** (now +0.009) |
| 500M RAFT | 0.346 | 0.307 | 0.47 | 0.046 | +0.0025 (now +0.0175) |
| **Gemma 3 270M RAFT** | 0.644 | **0.201** (its only missed target) | 0.79 | 0.034 | gate already missed (+0.308, D-35) |
| Gemma 2 2B LoRA | 0.912 | 0.086 | 0.76 | 0.006 | +0.0002 (now +0.0198) |

**Hallucination breakdown for SLM-125M** (shares of all test items):
- unsupported answers to answerable questions 15.3%;
- unsupported task outputs 10.6%;
- answered near misses 4.4%;
- answered with the passage removed 2.2%.

These are exactly the "wrong choice between options the model can already produce" errors that preference
optimization targets. **But only if the model sometimes produces the right option.** That is measured
first (12.1).

## 2. Recommendation: which models

### SLM: the 125M, not the 500M
- **Forgetting headroom.** DPO moves the model away from its SFT reference. The 125M has 0.011 bits/byte of
  slack to the gate; the 500M has 0.0025, so almost any update would fail it.
- **The gap and the payoff.** The two models are statistically tied on test (0.316 vs 0.346, ≈ 1 standard
  error), and the 500M is worse on near misses (0.47 vs 0.56). Only the 125M is ours, pretrained from
  scratch and live behind `POST /answer`: a gain there reaches the demo.
- **Cost:** the 125M is ≈ 4× cheaper per step.

### Gemma: Gemma 3 270M, not Gemma 2 2B
- **A clear target.** It misses exactly one target, hallucination (0.201 vs ≤ 0.15). It already refuses near
  misses well (0.79), so there is something concrete to fix and a pre-set bar to clear.
- **Gemma 2 2B already meets all seven targets** (hallucination 0.086): little headroom, a gain would be
  hard to show, and it has no forgetting slack (+0.0198 of +0.02).
- **Practical:** 268M parameters train fast, and it is the only Gemma that fits our 2-core / 4 GiB `/answer`
  box, so an improved model could actually be served (a separate decision).
- **Caveat:** its SFT already missed the forgetting gate (+0.308, D-35). For it, the gate in this phase is
  "**no further drift**": bits/byte within +0.02 **of its SFT checkpoint**. The original miss stays
  recorded.

**So two tracks run the same pipeline:**
- **SLM-125M** (primary: our model, our demo);
- **Gemma 3 270M** (secondary: can preference optimization take the best small model past the
  hallucination target?).

The 500M and Gemma 2 2B are not used.

## 3. Method: RLAIF with Gemini Flash labels, optimized with DPO
- **Why DPO, not reward model + PPO (the reference site's pipeline).**
  - DPO optimizes the same KL-regularized objective without a separate reward model or an RL loop. The
    implicit reward is β·log(π/π_ref), and the leash is β.
  - It needs only the policy and a frozen reference (the SFT checkpoint).
  - PPO would need four models in memory, a value head trained on a few thousand AI comparisons (weak at
    125M), and KL tuning. That is the reward-hacking zone the reference site warns about.
  - **Iterated DPO** (re-sample from the improved policy, re-label, train again) gives the on-policy
    benefit of RL at offline cost.
  - **Not planned:** RM + PPO as a later teaching extension, and GRPO.
- **"AI feedback" here (the constitution):** Gemini Flash (`config.TEACHER.checker`, batch) grades every
  sampled answer against a written rubric that encodes the same preferences as the human guideline on the
  reference site:
  1. **Decision:** refusing is right only when the passages do not contain the answer. Answering a question
     the passages can't answer is the worst error.
  2. **Support:** every claim in the answer is stated in the cited passage. The quote is from the passage
     and backs the answer.
  3. **Correctness:** it matches the reference answer when one exists (reference-guided judging is more
     reliable and cheaper than judging blind).
  4. **Concise:** no added content. *Not* longer-is-better.

  The judge returns a score 1–5 with the failed criterion. One call per candidate (pointwise), not
  pairwise, so there is no position bias and K calls per prompt instead of C(K,2).
- **Pairs:** within a prompt, every pair of candidates whose scores differ by ≥ 2 (a C(K,2) expansion,
  filtered). Ties and near-ties are dropped.
  - **Length-bias guard (from the reference site):** log the mean token length of chosen vs rejected. If
    chosen is > 15% longer, rebalance by subsampling.
  - A 5% swap-order pairwise audit checks that the judge's pointwise ranking is consistent.
- **Known labels as a check on the judge:** our prompts carry construction labels (answerable or not, the
  oracle passage). Judge–label agreement on the decision is reported per kind; < 90% on B1/B3 stops the run.
  The judge, not the label, drives the pairs (that is what makes it RLAIF).
- **Loss:** DPO, β 0.1 (β 0.3 as a pilot arm), reference = the SFT checkpoint, loss on target tokens only.
  - Plus an auxiliary SFT term on the chosen answer (weight 0.2; it stops DPO from pushing down the
    likelihood of the chosen answer too).
  - Plus, for the 125M, **replay λ 1.0** as in Phase 7 (its forgetting lever).

## 4. Data
- **Prompts (≈ 3,000 per track):**
  - ≈ 1,500 **spare verified rows** from Phase 7's scale stage: accepted by all checks but not selected
    (≈ 1,500 A, 380 B2, 246 C, minus the probe-cited exclusions);
  - B1 variants derived from A prompts (oracle passage removed) and B3, at no cost;
  - ≈ 1,000 **new rows** from the existing teacher pipeline (DeepSeek G1/G2 off-peak + V1/V2/V3 checks,
    unused train-split documents), weighted toward the known failure modes: **near misses (30%)** and
    answerable SEC/case-law questions with numbers.
  - None from dev/test documents. Probe documents are excluded as in 7.8.
- **Samples:** K = 4 per prompt from the SFT policy.
  - The decision is **sampled** from P(none)/P(Pass) (not thresholded by τ), so pairs contrast refusing
    with answering.
  - The body is sampled at temperature 0.8 under `copy2`, so quotes stay verbatim and preferences are about
    *which* span and *whether* to answer.
  - Prompts where all K samples are identical are dropped before judging.
- **Expected:** ≈ 12K samples per track, ≈ 8K judged after dropping identical ones, ≈ 5–8K pairs after the
  score-gap filter.

## 5. Steps and cost (each GPU and API step stops at its own gate)
| Step | What | Estimate |
|---|---|---|
| 12.0 | Code ($0): `rlaif_data.py` (sampler, constitution prompt, pair builder, length log, judge–label agreement), `dpo.py` (DPO + auxiliary SFT + replay on top of `sft.py`/`gemma_sft.py` target-only log-probs), `rlaif_app.py` (Modal jobs), an independent judge path (§6); local tests incl. a hand-computed DPO loss | $0 |
| 12.1 | **Headroom check (go/no-go):** best-of-8 on dev for each track (sampled as above, J1). **Go only if** best-of-8 grounded ≥ greedy + 0.10 **or** the share of prompts with ≥ 1 correct decision on B2 ≥ greedy + 0.15 | GPU ≈ $0.4, J1 ≈ $1.0 |
| 12.2 | Prompts: spare rows + ≈ 1,000 new (DeepSeek + checks) | ≈ $0.8 |
| 12.3 | Sampling (K = 4) | GPU ≈ $0.5 (both tracks) |
| 12.4 | RLAIF labels (Gemini Flash batch, ≈ 16K calls) + 5% pairwise audit | ≈ $5.0 |
| 12.5 | DPO pilot: β {0.1, 0.3} × 1 LR, dev J1; pre-registered pick | GPU ≈ $0.8, J1 ≈ $0.6 |
| 12.6 | **Round 2 (conditional on round 1 improving dev):** re-sample from the round-1 policy on fresh prompts, re-label, DPO again | ≈ $3.5 |
| 12.7 | Final: test once per track (J1 + independent judge + proxy), guards, export | GPU ≈ $0.3, judges ≈ $1.5 |
| **Total** | **≈ $10 with round 2, ≈ $6.5 without** | caps set per step at approval |

**Gemini prepaid credits:** this phase needs ≈ $7–8 of Gemini batch. Please top up before 12.4.

## 6. The judge-overlap problem, and the pre-registered reading
**Problem:** the labeller (Gemini Flash) and the evaluation judge J1 (also Gemini Flash) are the same model
family. A policy trained toward Gemini's preferences can score better under J1 without being better. That
is reward hacking that the metric would hide.

**Defences:**
1. The constitution prompt is **different** from J1's prompt.
2. **Judge-free metrics are co-primary:**
   - the deterministic proxy (verbatim quote + right passage + token-F1);
   - decision metrics on construction labels (refusal recall by kind, over-refusal, AUROC).
3. **An independent judge (J2):** DeepSeek (no thinking), with J1's rubric, on the full test for the
   final models.
4. **Human spot-check:** 50 changed answers read and logged.
5. **Hacking signs, reported:** length drift; refusal-rate drift; hallucination falling only because
   answers get shorter or vaguer.

**Success, per track (test, `copy2`; fixed now):**
- **Hallucination falls by ≥ 0.05** (J1), **and** the direction is confirmed by J2 and by the proxy;
- **without** grounded accuracy falling more than 0.02, and with over-refusal ≤ 0.20 and format ≥ 0.99;
- **and** bits/byte ≤ +0.02 relative to the SFT model (for the 125M also ≤ +0.02 relative to the
  pretrained base, the original gate).
- **Gemma 3 270M:** the headline is whether it reaches the hallucination target (≤ 0.15) and so meets all
  seven.
- **If 12.1 shows no headroom for a track,** that track stops there. The result ("preference optimization
  can't fix what the policy never samples") is recorded.

## 7. Out of scope (separate decisions)
- Serving an improved model from `/answer`, Hugging Face uploads, and website changes.
- RM + PPO (possible as a later teaching extension), and Gemma 2 2B and the 500M.

## 8. Results

### 12.0 Code: DONE (2026-10-27, $0)
**Code:**
- **`raft_decode`:**
  - `RaftConstraint(sample_decision=True)` draws the decision from s;
  - **`FormatConstraint`** (opt-in): forces the exact training refusal, restricts the openers after
    ` Passage k`, keeps answers on one line with no special or `<unusedN>` tokens, and caps them at 400 characters;
  - `TemperatureLast` and `sample_batch`.
  - Greedy decoding is unchanged.
- **`rlaif_data`:** spare-row prompts, the constitution, judge jobs, score-gap pairs with the length check
  on answer-vs-answer pairs, and judge–label agreement.
- **`dpo`:** target-only log-probs, precomputed reference, DPO + 0.2 SFT + replay.
- **`rlaif_app`:** Modal sampling and DPO jobs, two images.
- **`rlaif_pipeline`:** judging, headroom, labels, pairs.

**Validation:**
- `smoke_rlaif.py` 15/15:
  - sampled decision frequency matches s;
  - forced refusal text; opener restriction;
  - DPO term equals a hand computation; target log-probs equal the HF forward;
  - the reference equals the policy at step 0; training raises the margin.
- **The user's temperature concern, measured:** the real SLM-125M-RAFT at T 0.8 keeps the format on 96/96
  samples even without `FormatConstraint`. It stays on as a safety net, and pairs require both samples to
  be valid.
- **Degenerate quotes:** sampling can close a quote after a fragment (e.g. `"at"`). The constitution scores
  such quotes as weak support.

**Prompts (12.2, $0):** 2,301 spare verified rows: A 1,524 / B2 380 / C 246 / B3 151 (case law 911, SEC 975,
web 415).
- None from train/dev/test rows or dev/test/probe-cited documents.
- Near misses sampled K = 8, the rest K = 4: 10,724 samples per track.
- **Deviation:** no new DeepSeek rows (≈ $0.8 saved); near-miss weighting is by K instead.

### 12.1 Headroom (go/no-go): DONE, GO for both tracks (2026-10-27; GPU ≈ $0.52 of $0.60; J1 $0.70 of $1.20)
**Run notes:**
- **The first Gemma 3 launch crashed:** a CUDA index assert. The tokenizer has 262,145 entries but the
  model has 262,144 logits, and `FormatConstraint` blocked the image token at id 262,144. Fixed by limiting
  every list to the logits width, and retested.
- **Gemma 3 hit its budget stop at 320/399 rows** and was resumed to 399.

**Results:** K = 8 samples per dev A/B row (399 rows each), T 0.8, `copy2`, sampled decision. J1 on 1,486
(SLM) + 868 (Gemma 3) distinct answered A samples, 100% coverage.

| Dev | Format validity | Best-of-8 grounded | Greedy | Gain | B2: ≥ 1 refusal in 8 | Greedy B2 recall | GO |
|---|---|---|---|---|---|---|---|
| SLM-125M | 1.000 | **0.564** | 0.349 | **+0.215** | 0.944 | 0.667 (+0.278) | ✅ (both criteria) |
| Gemma 3 270M | 0.997 | **0.804** | 0.658 | **+0.145** | 0.917 | 0.806 (+0.111) | ✅ (grounded criterion) |

**Reading:** both policies often sample the right span or the right refusal but don't rank it first. That is
the headroom preference optimization needs. Best-of-8 is an upper bound (oracle selection), not a forecast.

**Re-costed 12.3–12.4** at the measured rates: sampling ≈ $1.6 (plan ≈ $0.5) and labels ≈ $5.8. The phase
would cost ≈ $10 without round 2 (plan ≈ $6.5, **+54%**), so it **comes back to the user** (§5 rule).

### 12.2–12.3 Smaller run (user's choice after the re-cost) (2026-10-27): IN PROGRESS
**Prompts:** 1,600, K = 4 for every row (6,400 samples per track): all 380 B2, 151 B3 and 246 C, plus 823 A
(seeded subset). The full 2,301-row set is kept as `.smoke/rlaif/prompts_full.jsonl`.

**Sampling:**
- **The first launch didn't start:** a zsh loop passed malformed arguments (zsh doesn't word-split `$VAR`;
  AGENT_BRIEF gotcha). $0.
- **The second launch was cancelled after ≈ 8 min** (≈ $0.60): the Mac slept on battery while `caffeinate -i`
  was active, so a closed lid or forced sleep, which `caffeinate` can't block.
- **Saved progress** (written per 64-row chunk): SLM-125M 1,088 / 1,600 rows, Gemma 3 832 / 1,600 rows.
- Remaining ≈ $0.42, so 12.3 ≈ $1.02 against its $1.00 cap.

**Noted:** with transformers 4.57, Gemma 3's model generation config silently sets **top-p 0.95** (our explicit
1.0 equals the library default, so it is overridden). Gemma 3's samples (12.1 and 12.3) come from a
slightly narrower nucleus than SLM-125M's. Format validity is unaffected (0.997).

### 12.3–12.5 status (2026-10-27 ≈ 12:45 IST)
**12.3 sampling, finished on the deployed app:**
- SLM-125M 6,400 samples (format 0.9989), $0.13; Gemma 3 6,400 (0.9986), $0.29.
- 12.3 total ≈ $1.03 of $1.10.
- **Deploy mode** (user-approved): `modal deploy rlaif_app.py` + `rlaif_launch.py spawn`, so jobs run on Modal
  whatever the laptop does.

**12.4 labels** ($3.32 of $6.00 so far):
- SLM-125M 4,676 / 4,676 (99.91% parsed).
- Gemma 3 2,400 / 3,369. Its batch stalled, then **Gemini returned 402 (prepaid credits depleted)**.

**SLM-125M pairs** (`pairs_r1_stats.json`): **1,316 pairs from 565 prompts**.
- Kinds: answer>answer 899, refuse>answer 324, answer>refuse 93.
- Length ratio on answer pairs 1.086 (< 1.15: no flag).
- Judge–label agreement: A 1.00, B3 1.00, B2 0.931, C 1.00.
- Scores: 2 → 2,902; 5 → 987; 1 → 490; 3 → 153; 4 → 140. Most samples score 2, and equal scores make no
  pair, hence half the expected pairs.
- **Deviation:** 2 DPO epochs instead of 1 (83 steps).
- Label coverage rule relaxed from 100% to ≥ 99% for pairing: a missing label only drops that sample.

**12.5 DPO pilot, SLM-125M** (deployed, $0.43 of the 12.5 GPU cap):
- **Forgetting:** bits/byte 0.7181 vs SFT 0.7179 (+0.0002) and +0.0091 vs the pretrained base (gate ✅).
- **Decision drift:** at τ 0.7, over-refusal rose to 0.57–0.64 (β 0.1) and 0.20 (β 0.3), because
  refuse>answer pairs dominate. The pre-registered τ rule re-tunes to τ 0.95 for both.

**Dev proxies at the re-tuned τ (`copy2`):**

| | τ | Proxy grounded | Proxy hallucination | Over-refusal | Refusal recall |
|---|---|---|---|---|---|
| SFT (7.8) | 0.70 | 0.389 | 0.032 | 0.018 | 0.871 |
| DPO β 0.1 | 0.95 | 0.276 | 0.014 | 0.236 | 0.952 |
| DPO β 0.3 | 0.95 | 0.371 | 0.040 | 0.029 | 0.847 |

The proxy can't see unsupported answers, the main target. **J1 on these dev outputs is blocked by the
depleted Gemini credits** (402 at submission).

### 12.4–12.5 completed after the credit top-up (2026-10-27 ≈ 13:10 IST)
**Labels:**
- Gemma 3 3,369 / 3,369: the stalled batch resumed once the credits returned.
- **12.4 total $3.77 of $6.00.** The Kaggle key the user offered is not a Gemini API key (`API_KEY_INVALID`).
  Its model proxy is scoped to Kaggle Benchmarks and lists other Gemini versions, so it was not used.

**Gemma 3 pairs:** **872 from 462 prompts**.
- Kinds: answer>answer 708, refuse>answer 98, answer>refuse 66.
- Length ratio 1.057 (no flag).
- Agreement: A 1.00, B3 1.00, B2 0.946, C 0.997.

**Gemma 3 DPO:**
- Deployed, $0.58. The 12.5 GPU total is **$1.01 of $1.50**.
- Bits/byte vs its SFT checkpoint: +0.016 (β 0.1) / +0.014 (β 0.3), within the +0.02 "no further drift" gate.

**Dev, J1, `copy2`** (τ re-chosen by the pre-registered rule; 12.5 J1 $0.42 of $1.00):

| | τ | Grounded | Hallucination | Refusal recall (B2) | Over-refusal | Citation | Quote fidelity |
|---|---|---|---|---|---|---|---|
| **SLM-125M SFT** (7.8) | 0.70 | 0.349 | 0.309 | 0.871 (0.667) | 0.018 | | |
| SLM-125M DPO β 0.1 | 0.95 | 0.291 | **0.226** | 0.952 (**0.917**) | **0.236 ❌** | 0.900 | 0.996 |
| **SLM-125M DPO β 0.3 (pick)** | 0.95 | **0.364** | 0.293 | 0.847 (0.694) | 0.029 | 0.918 | 0.997 |
| **Gemma 3 SFT** (11.4) | 0.75 | 0.658 | 0.192 | 0.927 (0.806) | 0.025 | | |
| **Gemma 3 DPO β 0.1 (pick)** | 0.80 | 0.655 | **0.176** | 0.952 (**0.889**) | 0.062 | 0.969 | 0.986 |
| Gemma 3 DPO β 0.3 | 0.90 | 0.669 | 0.184 | 0.935 (0.833) | 0.040 | 0.970 | 0.976 |

**Pre-registered pick:** the lowest dev hallucination among arms with grounded ≥ SFT − 0.02, over-refusal ≤ 0.20
and the gate passed.
- **SLM-125M: β 0.3.** β 0.1 fails over-refusal, 0.236, and loses 0.058 grounded.
- **Gemma 3: β 0.1.**

**Reading (dev):**
- **Round 1 does not meet the success bar** (hallucination −0.05) for either model:
  - SLM-125M β 0.3: −0.016 hallucination, +0.015 grounded;
  - Gemma 3 β 0.1: −0.016 hallucination, −0.003 grounded.
- **The clearest effect is on the decision.** Near-miss refusal rises by +0.03 (SLM-125M β 0.3) to +0.08
  (Gemma 3 β 0.1). The pairs most often prefer refusing on unanswerable items (refuse>answer 324 / 98).
  At β 0.1 the 125M over-learns it (over-refusal 0.236).
- **Unsupported answers barely move.** Most samples score 2 (wrong or unsupported) and equal scores make
  no pair, so wrong-span answers are rarely contrasted with right ones. Pair counts were 1,316 and 872,
  about half the ≈ 2–3K expected.
- **Not run (not approved):** the test pass (12.7) and round 2 (12.6).
  - Likely levers for a round 2: K = 8 everywhere and the full 2,301 prompts, for more right-vs-wrong
    answer pairs; a higher LR (the margins grew but the policy moved little at 1–2e-6); re-sampling from the
    round-1 policy.

**Phase 12 spend so far ≈ $7.45:**
- 12.1: GPU $0.52 + J1 $0.70;
- 12.3: GPU $1.03;
- 12.4: labels $3.77;
- 12.5: GPU $1.01 + J1 $0.42.

That is against the ≈ $7.1 approved for the smaller run (+5%). The deployed `slm-125m-rlaif` app is idle
($0); 0 containers.
