# Phase 5 — Pretrain the 125M model

**Compute:** GPU (H100) · **Cost:** the only real spend in the project ·
**Status:** ⛔ **GPU steps BLOCKED pending explicit approval** — Step 1 is $0

This is the approval document. Spend is approved against what is written here,
so every number below is derived from measured artifacts, not assumptions.

---

## 0. Approved parameters (2026-10-03)

| Parameter | Value |
|---|---|
| Epochs | **4** (one continuous cosine schedule, never restarted) |
| Budget cap | **`--max-usd 50`** (`config.BUDGET_CAP_USD = 50.0`) |
| Full-run hardware | **8 × H100** (H100 SXM5, $0.001097 / GPU-second) |
| Publish target | `mdeepanshu/slm-125m-base` |

## 1. Preflight — verified

`verify_model.py`, run `ap-kln3YfqkknDnP9BaFJTS9V`, exits non-zero on any violation.
Result: **ALL INVARIANTS HOLD.**

| Invariant | Measured |
|---|---|
| Exact parameter count | **125,848,320** = `config.MODEL.approx_params()` |
| Embedding / non-embedding | 12,582,912 / 113,265,408 |
| Embeddings tied | True |
| Master weight dtype | `torch.float32` |
| Token ids (tokenizer = `model.py`) | bos 0 · eos 1 · pad 2 · unk 3 — all match |
| `<\|bos\|>` occurrences in packed data | **0** across all 2,059,674,624 tokens |
| `<\|eos\|>` occurrences | 663,297 train + 6,810 val = 670,107 (≈ the 670,124 corpus docs) |
| Val bytes/token (`skip_special_tokens=True`) | **4.666731** (case-law 4.485 · sec 5.027 · fineweb 4.279) |

## 2. Run size

| | Value |
|---|---|
| Train tokens per epoch | 2,039,072,768 (from `/data/tokens/index.json`) |
| Tokens seen (4 epochs) | **8,156,291,072** |
| Global batch | 524,288 tokens = 32 seq × 1024 × 8 GPUs × **2 grad-accum** |
| Optimizer steps | **15,556** — `config.pretrain_steps()`, derived from the index |
| Warmup | 200M tokens = **381 steps** (2.4% of the run) |
| LR | 6e-4 peak → 6e-5 floor, cosine over all 15,556 steps |
| Tokens / parameter | **64.8** (Chinchilla-optimal is ~20) |

Training length comes from one place: `PRETRAIN_EPOCHS × train_tokens` via
`pretrain_steps()`. `TARGET_TOKENS` is now commented as a Chinchilla reference only.

4 epochs sits at the edge of the useful range for repeated data: Muennighoff et al.
(2023, *Scaling Data-Constrained Language Models*) find up to ~4 epochs is nearly
as good as fresh tokens, with returns falling off sharply beyond. Since this corpus
is fixed at 2.04B unique tokens, 4 passes is a reasonable way to spend the budget —
but it makes an honest validation set more important, not less (see §11).

## 3. Cost — read this before approving

Training FLOPs ≈ `6 × N × D` = 6 × 125,848,320 × 8,156,291,072 = **6.159 × 10¹⁸**.
Because GPU-seconds = FLOPs ÷ (MFU × per-GPU peak), **total cost depends only on
MFU, not on GPU count** — more GPUs buy wall-clock, not savings:

```
cost(4 epochs) ≈ $7.855 / MFU      (H100 $0.001097/GPU-s, × 1.15 CPU/memory overhead)
```

| MFU | Cost | Wall-clock on 8×H100 | vs $50 cap |
|---|---|---|---|
| 10% | $78.56 | 130 min | ❌ over |
| 15% | $52.37 | 87 min | ❌ over |
| **15.7%** | **$50.00** | 83 min | **break-even** |
| **16.2%** *(reference repo, measured)* | **$48.49** | 80 min | ⚠️ **3% margin** |
| 20% | $39.28 | 65 min | ✅ 21% margin |
| 25% | $31.42 | 52 min | ✅ |
| 30% | $26.19 | 43 min | ✅ |

**The honest read:** at the only *measured* MFU we have (16.2%, from the reference
repo), 4 epochs fits under $50 with a 3% margin. A 1-point MFU shortfall pushes it
over and the budget guard would stop the run partway through. **The run is
affordable only if throughput is at or above the reference** — which is why the
performance stack (§4) and the calibration step (§10) decide this phase.

Cost per hour on 8×H100 including overhead: **$36.33**.

## 4. Performance stack — recommendation and why

**Recommended: SDPA attention + fused AdamW + TF32 matmuls as the baseline, with
`torch.compile` A/B-tested during calibration and kept if it is faster and resumes
cleanly. Skip flash-attn 2.**

Where the FLOPs go, per layer, forward, seq 1024:

| Block | GFLOP | Share |
|---|---|---|
| MLP (SwiGLU) | 14.50 | **64.3%** |
| QKVO projections | 4.83 | 21.4% |
| Attention scores + AV | 3.22 | **14.3%** |

- **Skip flash-attn 2.** Attention is only 14% of FLOPs, and PyTorch's SDPA already
  dispatches to a fused flash kernel on H100. The remaining gain is small, and
  building the flash-attn wheel is the most fragile step an image build can have.
- **Try `torch.compile`.** At hidden size 768 the matmuls are cheap, so the unfused
  elementwise work around them — RMSNorm, the SiLU gate, RoPE, residual adds — is a
  large share of step time. Fusing that is exactly what compile does, which makes its
  usual 1.2–1.8× gain unusually valuable for a model this narrow:

  | Gain over 16.2% | MFU | 4-epoch cost | Margin under $50 |
  |---|---|---|---|
  | 1.0× (no gain) | 16.2% | $48.49 | 3% |
  | 1.2× | 19.4% | $40.41 | 19% |
  | **1.3×** | **21.1%** | **$37.30** | **25%** |
  | 1.5× | 24.3% | $32.33 | 35% |

  Its risks — 1–3 minutes of compile time, and occasional recompiles on resume —
  are exactly what the < $1 calibration can expose before any real money is spent.
- **Fused AdamW** (`fused=True`): in a 125M model the optimizer step is a
  non-trivial share of each step; the fused kernel removes per-parameter launches.
- **TF32** for any fp32 matmul that escapes autocast.

## 5. Precision

- **Never** `model.to(torch.bfloat16)`. Master weights stay **fp32**; the AdamW
  state is fp32.
- Forward and backward run under
  `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`.
- **Why it matters:** bf16 has an 8-bit mantissa, so near 1.0 its spacing is
  2⁻⁷ ≈ 0.0078. An AdamW update of lr × ~1 = 6e-4 is far below half that spacing,
  so on bf16 weights it rounds away to nothing — RMSNorm gains initialised at 1.0
  would stay frozen at exactly 1.0 for the entire run, and every other weight would
  lose most small updates the same way. fp32 master weights keep the updates.

## 6. Token ids and prompting

- `config.json` and `generation_config.json` set **`bos_token_id=0`, `eos_token_id=1`,
  `pad_token_id=2`** explicitly from `model.py` — never LlamaConfig defaults
  (Llama defaults are bos 1 / eos 2, which would be off by one here).
- **Prompts start with `<|eos|>`, not `<|bos|>`** (`model.PROMPT_PREFIX_TOKEN_ID`).
  Measured above: BOS never appears in the training data, so its input embedding
  is untrained. Every training document begins immediately after an EOS, so EOS is
  the prefix that matches what the model actually learned. `config.json` still
  records `bos_token_id=0` because that is the true vocabulary entry; the sample
  generator (Phase 5) and the server (Phase 6) prepend EOS explicitly. This is the
  same convention GPT-2 uses, where `<|endoftext|>` is both separator and prefix.

## 7. Metrics

Logged every 20 steps; evaluated on the full val split every 1,000 steps.

| Metric | Formula | Why |
|---|---|---|
| loss | mean token cross-entropy, nats | the optimised quantity |
| perplexity | `exp(loss)` | conventional, but **tokenizer-dependent** |
| **bits-per-byte** | `loss × 0.309145` | `loss / (ln 2 × 4.666731)` — **tokenizer-independent**, comparable to models with any vocabulary |
| tokens/sec, MFU | `6 × N × tok/s ÷ (n_gpus × 989e12)` | the cost driver |
| spend, projected spend | `config.projected_cost_usd()` | budget guard |

The 4.666731 constant is the measured val-split ratio. If the val split is rebuilt
(§11), it must be re-measured — `verify_model.py` does that.

## 8. Checkpoint, resume and schedule continuity

Modal preempted containers **twice during Phase 4 alone**, so preemption is
designed for, not hoped against.

- `/data/checkpoints/ckpt.pt` every **500 steps**: model, optimizer, step, data
  position (epoch + offset), and RNG states. Written to a temp file and renamed
  (atomic), then `volume.commit()`.
- **The LR is a pure function of the global step**, so a resume after preemption
  continues the *same* cosine schedule from the same point. This satisfies the
  "one continuous schedule, no re-warming" rule automatically: a preemption is a
  pause, not a restart.
- Per-epoch shuffle is seeded `seed + epoch`, so resuming mid-epoch reproduces the
  same order and skips exactly the windows already consumed.
- Worst-case loss per preemption: 500 steps ≈ 262M tokens ≈ 2.6 min of 8×H100 ≈ $1.6.
- Final artefact: `/data/checkpoints/base/` — `config.json`, `model.safetensors`,
  `generation_config.json`, tokenizer files. Metrics: `/data/checkpoints/metrics.jsonl`.

## 9. Budget guard (`--max-usd 50`)

From step 20 onward the trainer projects total spend from **live** throughput:
`projected = spent_so_far + remaining_steps × measured_sec_per_step × rate`. If
`projected > max_usd` it **checkpoints first, then exits cleanly** with the
projection in the log. It also hard-stops if actual spend crosses the cap. Because
the projection is live, a run that will not fit is stopped within its first few
dollars rather than near the end.

## 10. Execution plan — three steps

### Step 1 — Local smoke test · Mac M2 Pro · **$0**

**Goal:** prove the data loader, forward/backward, loss, logging and
checkpoint save → resume round-trip before any GPU exists.

- Full model geometry, **20 steps**, micro-batch 2, run on MPS (or CPU), **fp32**.
- Data: 2 val shards copied locally (~3 MB each; same `.bin` format as train).
- Kill at step 10, resume, and confirm step, LR and data position continue exactly.
- Pass: loss finite and falling, checkpoint reloads bit-exact, resume is seamless,
  saved `config.json` carries bos 0 / eos 1 / pad 2.

**What it cannot test:** CUDA bf16 autocast, fused AdamW, NCCL/DDP and compile-on-CUDA
exist only on GPU. Those are first exercised in Step 2.

### Step 2 — Calibration · 1 × H100 on Modal · **< $1**

**Goal:** replace the assumed MFU with a measured one.

- **50 steps**, real recipe, bf16 autocast — run twice: without and with
  `torch.compile` (~$0.30 each).
- Records tokens/sec, MFU, peak memory, compile time, and a resume test.
- **Decision rule.** Single-GPU MFU is an *upper bound* for 8 GPUs (DDP adds
  communication), so allow ~15% for scaling loss:

  | 1×H100 projected 4-epoch cost | Action |
  |---|---|
  | ≤ $42 | proceed to Step 3 (scaling loss still fits under $50) |
  | $42 – $50 | stop and report — too close to call |
  | > $50 | **do not launch**; return with options (3 epochs, higher cap, more tuning) |

### Step 3 — Full 4-epoch pretraining · 8 × H100 on Modal · **~$37–48**

- **15,556 steps**, 8.156B tokens, one cosine schedule over the whole run.
- `--max-usd 50` with the live guard from §9, so the real 8-GPU throughput is
  checked within the first minutes — if it does not fit, it stops early,
  checkpointed, not at $49.
- Expected: ~43–80 min wall-clock depending on measured MFU.
- On completion: write HF `base/`, run final val eval, and generate samples from a
  legal and a financial prefix, each prompted with `<|eos|>`.

## 11. Open decision — document-level validation split (recommended)

Phase 4 sends every 100th **window** to val. Documents span windows, so the rest of
each val document is in training:

| Source | Avg windows per document | What a val window's siblings look like |
|---|---|---|
| sec | 18.8 | ~18 sibling windows of the same filing are in train |
| case-law | 3.4 | ~2–3 sibling windows in train |
| fineweb-edu | 1.1 | mostly self-contained |

This follows the spec and matches the reference, so perplexity stays comparable
to it. But it makes val perplexity optimistic, and "held-out" overstates it.

**4 epochs makes this materially worse.** With 4 passes over the ~18 sibling
windows of each SEC val filing, the model increasingly *recalls* that specific
filing — names, figures, phrasing — so SEC val loss measures recall rather than
generalisation. And the signal multi-epoch training most needs is the one this
destroys: overfitting shows up as val loss rising while train loss falls, but a
val set that shares documents with train tends to fall *with* it. A leaky val set
would hide exactly the failure 4 epochs risks.

**Recommendation: rebuild Phase 4 with a document-level split before Step 3.**
~10 minutes of CPU, ~$0; the tokenizer is unchanged. Each document goes wholly to
val when `blake2b(text) % 100 == 0`, train and val pack into separate buffers, and
`pretrain_steps()` re-derives the step count from the new index automatically.
(It must use `blake2b`, not Python's `hash()`, which is salted per process — the
same hazard noted for `dedup.word_ngrams`.)

Trade-off: it modifies `modal_app.py`, one of the four verbatim GUIDE files, and
replaces the recorded Phase 4 artefacts — so it needs an explicit yes.

## 12. Validation criteria

1. Loss falls without divergence; grad-norm stays bounded; no NaN/Inf.
2. Val loss, perplexity and BPB reported at each 1,000-step eval; final BPB is the
   headline, tokenizer-independent number. (Not MMLU — near-random at 125M.)
3. Train/val gap tracked across all 4 epochs as the overfitting signal.
4. A resume from `ckpt.pt` demonstrated to continue at the correct step and LR.
5. `/data/checkpoints/base/` loads with `AutoModelForCausalLM.from_pretrained` and
   carries bos 0 / eos 1 / pad 2.
6. Coherent domain samples from legal and financial prefixes, prompted with EOS.
7. Total spend ≤ $50.

## 13. Files

| File | State |
|---|---|
| `config.py` | ✅ amended: 4 epochs, $50 cap, GPU rates, `pretrain_steps()`, `projected_cost_usd()`, exact `approx_params()` |
| `model.py` | ✅ written: factory, explicit token ids, `PROMPT_PREFIX_TOKEN_ID = EOS` |
| `verify_model.py` | ✅ written and run: preflight invariants, fails loudly |
| `data.py` | to write — memory-mapped windows, per-epoch shuffle, resumable, rank-disjoint |
| `train.py` | to write — fp32 master + bf16 autocast, fused AdamW, cosine, BPB, guard, atomic ckpt |
| `pretrain_app.py` | to write — Modal GPU function, Volume, `--max-usd` |

---

## ⛔ Approval required

- **Step 1** ($0) — no approval needed; proceeds on the go-ahead to build.
- **Step 2** (< $1, 1 × H100) — needs explicit approval.
- **Step 3** (~$37–48, 8 × H100, `--max-usd 50`) — needs explicit approval *after*
  Step 2's measured MFU is reported against the decision rule in §10.
- **§11** document-level split — needs a yes or no.

## Results

_(filled in after execution)_
