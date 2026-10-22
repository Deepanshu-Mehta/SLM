# Phase 8: release SLM-125M-RAFT (model, API, website)

Approved by the user on 2026-10-21 (D-28): "publish model and add [it] in the vercel … explain
visually the architecture and things that we use like previously". All three steps are
outward-facing. Each is validated against the public artifact before it counts as done.

## 8.1 Shared decoding code ($0, done before anything is published)
- `raft_decode.py` is standalone (torch + transformers only). It holds:
  - the prompt format;
  - the answer/refuse decision (s = P(none)/(P(none)+P(Pass)), refuse iff s ≥ τ = 0.7);
  - `RaftConstraint`;
  - `CopyConstraint` (the 7.7 format fixes; answer copy stays off by default, E-36);
  - `parse_output`;
  - `answer()`.
- `eval_raft.py` imports from it, so evaluation, the API and the HF repo run the same code.
  Import-time asserts tie its constants to `config.py` and `model.py`.
- **Validation:** all decoding, sweep and eval checks pass. The released weights on CPU (fp32)
  reproduce the H100 test generations on a sample of test items.

## 8.2 Hugging Face: `Deepanshu027/slm-125m-raft` (public, like the base, D-10)
- **Files:** the HF export from `/data/raft/runs/full/hf` (safetensors, config with `architectures`,
  tokenizer, greedy generation config, `raft_format.json`), plus `raft_decode.py` and a model card.
- **Model card:**
  - what it is and its intended use (answer only from given passages, cite, quote, or refuse);
  - the prompt format;
  - a usage snippet with `raft_decode.answer`;
  - training (data, buckets, teachers, recipe, cost);
  - **the final test table with the three misses stated plainly**;
  - forgetting;
  - limitations.
  - License: the same as the base ("all rights reserved"; license still undecided, D-13).
- **Validation:** download from the Hub into a clean directory, load, and answer two smoke questions
  (one answerable, one refusal). Record the SHA-256 of `model.safetensors`.

## 8.3 API: `POST /answer` on the existing Modal app (CPU, scale-to-zero)
- **Request:** `{passages: [≤4 strings], question, copy: bool = true, tau: float = 0.7}`.
- **Response:** `{decision, s, passage, quote, answer, valid, raw, elapsed_ms}`.
- **Weights:** the RAFT weights are baked into the image from the HF repo, next to the base.
- Container memory 3 → 4 GiB (two 125M fp32 models).
- Same rate limit, CORS and caps (0–1 containers, 180 s scale-down).
- **Validation from the public URL:**
  - `/health`;
  - `/answer` on an answerable item (cites the right passage with a verbatim quote) and an
    unanswerable item (refuses);
  - CORS for the site;
  - 0 containers after scale-down.

## 8.4 Website: a new page `/grounded` (plus nav link), in the style of `/build`
Inspired by slm-finetuning-types.vercel.app (numbered sections; "answer in the context / not in the
context" side by side; a loss-mask toggle; a summary table) and slm-gemma-legal.vercel.app (a spec
strip of metric cards; a passages + question playground; "what this is").

1. **Hero:** what RAFT is, the spec strip (125.8M params, 20,340 examples, 4 passages, 3 epochs, ≈ $14.70),
   and the **live grounded playground** (passages + question → decision gauge, cited passage, verbatim
   quote highlighted in the passage, answer; preset examples, including a refusal).
2. **Chapters** (same Chapter / ChapterNav components, generalised to take a chapter list):
   1. *The problem:* a base model invents citations, so the goal is to answer only from passages, or refuse.
   2. *The format:* prompt and target anatomy, with an interactive loss-mask toggle (loss only on the
      target; the first token is the decision).
   3. *Five kinds of example:* A / B1 / B2 / B3 / C, side by side, with real counts.
   4. *Teachers and checks:* the funnel from 16.7K calls to 20,340 kept rows (mechanical checks, V1/V2,
      V3 near-miss check, Gemini audit, leakage rules).
   5. *Training without forgetting:* the λ-weighted replay loss; 7.6b's forgetting curves (peak then
      recovery; LR is the lever).
   6. *Decision and decoding:* s and τ (an interactive threshold over the dev scores); the copy
      constraint before and after (quote fidelity 0.15 → 1.00).
   7. *More data:* the judged learning curve 2K → 20K, and memorisation (train 0.95 vs dev 0.23).
   8. *Results, limits and cost:* the test table against the targets (misses shown in red), the
      hallucination breakdown, the teacher ceiling, and spend per stage.
- **Data:** every number comes from run artifacts through `site_data.py` (a new `raft` block in
  `site.json`). Nothing is typed in by hand.
- **Validation:**
  - `next build` passes;
  - the page renders in a real browser at desktop and phone width, in light and dark;
  - the playground answers through the public API;
  - after the Vercel prod deploy, the live URL serves the page and its JS points at the API URL.

## Cost
≈ $0. Modal CPU only while someone uses the playground (scale-to-zero); HF and Vercel are free.

## Results (2026-10-22): DONE, $0 incremental spend

| Step | Result | Validation |
|---|---|---|
| 8.1 `raft_decode.py` | Decoding moved unchanged out of `eval_raft.py`, which re-imports it. Import-time asserts tie the constants to `config.py` / `model.py` | All decoding/sweep/eval checks pass. The released weights (bf16) on CPU match the H100 test generations on **24/24 decisions** and **19/24 texts verbatim** (the rest differ by bf16-vs-fp32 numerics) |
| 8.2 HF `Deepanshu027/slm-125m-raft` | Public. HF export + `raft_decode.py` + `raft_stats.json` + model card built from artifacts by `publish_raft.py`. Weights SHA-256 `9f832f89…` | `publish_raft.py verify`: fresh Hub download, load, and the card's two asserted examples (cites passage 2 with a verbatim quote; refuses the near miss) |
| 8.3 `POST /answer` | Same Modal app (`deepanshu-m23csai`), RAFT weights baked in from HF, lazy-loaded on the first call; memory 3 → 4 GiB | From the public URL: answer (passage 2, verbatim quote, correct) and refusal; CORS `*`; 422 for over-long input; first call 8.2 s (load), then ~2–5 s |
| 8.4 Website `/grounded` | Hero with spec strip and live playground; 8 chapters (problem, format with a loss-mask toggle, five kinds, data funnel, forgetting curves, τ slider over 499 dev scores + copy decoding, learning curve, results/limits/cost); "Grounded QA" nav link. Every number from `site_data.py` → `site.json` `raft` | `next build` passes. Checked in a real browser at 1360 px and 390 px (no horizontal scroll), light and dark. Live deploy: 8 chapters render; the playground refuses the near-miss preset through the public API |

**Found and fixed during the browser check:**
- the stacked-bar legends printed percentages twice;
- the τ histogram faded the wrong side and was unreadable on a linear scale (now square-root);
- the τ slider wrapped;
- the learning-curve ticks were thinned in a narrow card, and the memorisation bars had no room;
- the loss-mask view hid the target below a scroll;
- the cost labels were internal step numbers;
- an earlier model-card example didn't reproduce (two short passages; the model cited the wrong one). It
  was replaced with a 4-passage example whose real output, including one wrong answer, is printed in the card.

**Base model card updated (2026-10-23):** both links in the live `Deepanshu027/slm-125m-base` card now point to the new API URL (Hub commit `afc25c29`). Only those two strings changed. Verified from a fresh download and a 200 from the new `/docs`.