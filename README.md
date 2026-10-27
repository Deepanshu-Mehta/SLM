# SLM-125M: a legal & financial language model, pretrained from scratch and taught to cite

A **125.8M-parameter** Llama-style language model trained from scratch, with a fresh tokenizer and
fresh weights, on US court opinions, SEC filings and educational web text. A grounded fine-tune of it
(**SLM-125M-RAFT**) then learned to answer **only from passages it is given**: it names the passage,
quotes it word for word and answers, or refuses. A scale study then put the same grounded-QA training
on three other models to separate **model size** from **pretraining scale** (Part 3). Finally,
preference optimization with AI feedback (RLAIF → DPO) was tried on the best models of each kind
(Part 4, round 1). Everything runs on [Modal](https://modal.com), and every number below comes from a run
artifact.

| | **SLM-125M (base)** | **SLM-125M-RAFT (grounded)** |
|---|---|---|
| What it does | Continues legal / financial text | Answers from ≤ 4 passages with a citation and a verbatim quote, or refuses |
| Hugging Face | [`Deepanshu027/slm-125m-base`](https://huggingface.co/Deepanshu027/slm-125m-base) | [`Deepanshu027/slm-125m-raft`](https://huggingface.co/Deepanshu027/slm-125m-raft) |
| Try it | [Playground](https://slm-125m-legal-sandy.vercel.app) | [Grounded QA demo](https://slm-125m-legal-sandy.vercel.app/grounded) |
| How it was built, visually | [/build](https://slm-125m-legal-sandy.vercel.app/build) | [/grounded](https://slm-125m-legal-sandy.vercel.app/grounded#problem) |
| API (Modal CPU, scale-to-zero) | `POST /generate`, `/generate/stream`, `/score` | `POST /answer` |
| Compute cost | ≤ $25.32 (8 × H100, 44.9 min) | ≈ $14.70 (teacher APIs + 1 × H100) |

API docs: <https://deepanshu-m23csai--slm-125m-serve-api.us-east.modal.run/docs>

> Neither model is legal or financial advice. The base model **invents facts and citations**. The
> grounded model cites the right passage 90% of the time but reads the answer out of it correctly
> only about a third of the time: **always check its quote against the passage.**

---

## Part 1: pretraining SLM-125M

![Training and held-out loss over 8.16B tokens](docs/05-loss-curve.png)

| | |
|---|---|
| Parameters | **125,848,320**, all trainable (12 layers, 768 wide, 12 heads, ctx 1,024, vocab 16,384) |
| Held-out loss / perplexity | **2.127** nats/token / **8.39**, on 19.56M tokens of *whole documents* never trained on |
| **Held-out bits per byte** | **0.657**: tokenizer-independent, the number to compare across models |
| By source (bits/byte) | SEC filings **0.463** · case law **0.687** · web **1.017** |
| Training | 8.16B tokens: 4 epochs over 2.04B unique, 15,564 steps × 524,288 tokens |
| Hardware / time | 8 × H100 SXM, **44.9 min**, 3.36M tokens/s, 32.1% MFU |
| Cost | data pipeline $0 (CPU) · calibration ≤ $0.54 · pretraining ≤ $24.78 |

Held-out loss improved at all 16 evaluations, so 4 epochs did not overfit. Each epoch gained about
half as much as the previous one, so the corpus is close to exhausted at this size.

### The data, and why it is mixed this way

| Source | Dataset | Train tokens | Share |
|---|---|---|---|
| US case law | [`HFforLegal/case-law`](https://huggingface.co/datasets/HFforLegal/case-law) (`us`) | 715.3M | 35.1% |
| SEC filings | [`PleIAs/SEC`](https://huggingface.co/datasets/PleIAs/SEC) | 860.6M | 42.2% |
| Educational web | [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`) | 464.2M | 22.8% |

The obvious recipe, 70% case law / 20% SEC / 10% web at 10B tokens, is **impossible**. Measured
before downloading anything, case law holds about 0.81B clean tokens, while 70% of 10B would need
7B. So the mix is *legal-first*: every clean legal token, plus a capped web slice. That gives 77% legal.

### Pipeline

| Phase | What happens | Command | Output on the Modal Volume |
|---|---|---|---|
| 0 | Smoke test (10 docs/source) and yield measurement | `modal run modal_app.py::main` then `::measure` | — |
| 1 | Stream parquet shards (20 workers, one per shard), 6-rule cleaning chain | `modal run --detach modal_app.py::clean --fineweb-shards 5` | `/data/clean/` |
| 2 | MinHash-LSH near-dedup, exact dedup, CaseHOLD 13-gram decontamination | `modal run --detach modal_app.py::dedup` | `/data/corpus/` |
| 3 | Train a 16,384-token byte-level BPE tokenizer | `modal run --detach modal_app.py::tokenizer` | `/data/tokenizer/` |
| 4 | Tokenize, pack into 1,024-token `uint16` windows, **document-level** 99/1 split | `modal run --detach modal_app.py::tokenize` | `/data/tokens/` |
| 5 | Pretrain (calibrate on 1 × H100, then 8 × H100) | see *Reproduce* | `/data/checkpoints/` |
| 6 | Publish to the Hub, serve on Modal CPU | see *Reproduce* | — |

**Funnel:** 718,780 documents streamed → 697,958 after cleaning → **670,124** after removing:
- 1,606 near-duplicates;
- 2,051 exact duplicates;
- 24,177 benchmark-contaminated documents (10.3% of case law).

Each phase is documented, with its measured results, in [`docs/`](docs/).

---

## Part 2: SLM-125M-RAFT, answer only from the passages

Retrieval-Augmented Fine-Tuning (RAFT): every training example puts **4 passages** in the prompt.
In answerable examples one passage holds the answer and the other three are hard distractors (BM25
look-alikes and passages from the same document). In unanswerable ones, none does.

**Format.**
```
<|eos|><|user|>Passage 1: …\nPassage 2: …\nPassage 3: …\nPassage 4: …\nQuestion: …<|assistant|>Source:
 Passage 3\nQuote: "<verbatim span of passage 3>"\nAnswer: <short answer><|eos|>
 none\nAnswer: The passages do not contain the answer.<|eos|>                        (refusal)
```
- **The first generated token is the decision.** With s = P(" none") / (P(" none") + P(" Pass")),
  the model refuses iff **s ≥ τ = 0.7**.
- **Copy-constrained decoding** ([`raft_decode.py`](raft_decode.py)) then keeps the quote a verbatim
  span of the cited passage and forces the `Answer:` line. The same file runs in evaluation, in the
  API and in the Hugging Face repo.

### Use it

```python
import importlib.util, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "Deepanshu027/slm-125m-raft"
tok = AutoTokenizer.from_pretrained(repo)
net = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32).eval()
spec = importlib.util.spec_from_file_location("raft_decode", hf_hub_download(repo, "raft_decode.py"))
rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
print(rd.answer(net, tok, passages=["…", "…", "…", "…"], question="…"))
# {'decision': 'answer'|'refuse', 's': …, 'passage': k, 'quote': '…', 'answer': '…', …}
```
or over HTTP:
```bash
curl -s -X POST https://deepanshu-m23csai--slm-125m-serve-api.us-east.modal.run/answer \
  -H 'content-type: application/json' \
  -d '{"passages": ["…", "…"], "question": "…", "copy_constrained": true, "tau": 0.7}'
```

### How it was built

1. **Teacher-written data.**
   - `deepseek-flash` (off-peak) wrote questions, verbatim quotes and answers from training-split
     passages only.
   - Every candidate passed model-free checks (verbatim quote, no copied question wording, no vague
     references) and a **blind re-check**: the checker sees only the passages and question and must
     agree on the passage, quote and answer.
   - Near-miss questions also passed a dedicated "could a careful reader still answer this?" check.
   - A 5% audit by a second model family (`gemini-3.8-flash`) disagreed on ≤ 7.3% per kind.
2. **Five kinds of example** (20,340 in total):

   | Kind | Rows |
   |---|---|
   | Answerable (A) | 9,600 |
   | Oracle removed (B1) | 4,222 |
   | Near miss (B2) | 2,198 |
   | No or unrelated passages (B3) | 720 |
   | Grounded tasks (C: summarise, list, yes/no) | 3,600 |

   About 35% of examples end in a refusal.
3. **Leakage control.**
   - Dev (499) and test (1,000) come only from held-out documents, scanned for 13-gram overlap
     with the whole pretraining corpus.
   - No training passage comes from a dev/test document.
   - Every dev/test near miss was read by hand.
4. **Training.**
   - Full fine-tune from the fp32 pretraining checkpoint: LR 5e-5 with cosine decay, 3 epochs,
     batch 64 × 1,024, 1 × H100.
   - Loss = SFT loss on the target tokens only + 1.0 × replay loss on raw pretraining text, each
     averaged over its own tokens.
   - The chat-token embeddings (never seen in pretraining) were initialised from the statistics of
     the trained rows.
5. **Checkpoint choice.** The best dev score under copy-constrained decoding, among checkpoints whose
   held-out bits/byte rose by at most 0.02.

### Results (frozen test set, held-out documents, scored once)

Targets were set before any training and were not lowered afterwards. Answers were judged by
`gemini-3.8-flash` against the reference.

| Metric | Target | **SLM-125M-RAFT** | Plain greedy decoding |
|---|---|---|---|
| Hallucination rate | ≤ 15% | **32.5%** ❌ | 53.9% |
| Refusal recall (unanswerable) | ≥ 80% | **78.0%** ❌ | 78.0% |
| Over-refusal (answerable) | ≤ 20% | **4.8%** ✅ | 4.8% |
| Grounded accuracy (correct + right passage) | ≥ 60% | **31.6%** ❌ | 33.6% |
| Citation accuracy | ≥ 80% | **90.3%** ✅ | 89.9% |
| Quote fidelity (verbatim) | ≥ 90% | **99.8%** ✅ | 39.1% |
| Format validity | ≥ 99% | **100.0%** ✅ | 99.8% |
| Held-out bits/byte (forgetting) | ≤ +0.02 | **+0.009** ✅ | |

- **Decision quality:** answer-or-refuse AUROC 0.923.
- **Refusal recall by kind:** passage removed 85.3%, near miss 56.0%, no or unrelated passages 100%.
- **CaseHOLD:** zero-shot PMI accuracy fell 4.8 points. That is reported rather than gated, after a
  breakdown showed the context-attributable margin unchanged.
- **Learning curve:** judged grounded accuracy on dev went 21.5% → 27.3% → 28.7% → 34.9% at 2K → 5K →
  10K → 20K examples, roughly +4 points per doubling.
- **What it means:** the decision, the citation and the verbatim quote work. Reading the right answer
  out of the cited passage does not, at 125M parameters, and more data alone won't close that gap.
- Full write-up, including every deviation from the plan and every correction:
  [`docs/08-raft.md`](docs/08-raft.md) §13. Release: [`docs/09-raft-release.md`](docs/09-raft-release.md).

---

## Part 3: what scale buys (the same grounded-QA training on other models)

SLM-125M-RAFT cites the right passage 90% of the time but reads the right answer out of it only a third of
the time. To find out whether that is the data or the model, the **same 20,340 training rows, the same frozen
1,000-item test set and the same judge** were used on three more models. They were chosen so that size and
pretraining change one at a time.

| Test (judged, copy-constrained decoding) | Target | **SLM-125M** (ours) | **SLM-500M** (same corpus) | **Gemma 3 270M** | **Gemma 2 2B** |
|---|---|---|---|---|---|
| Parameters (outside the embeddings) | | 126M (113M) | 518M (476M) | 268M (100M) | 2.6B (2.0B) |
| Pretraining | | 2.0B tokens × 4 epochs, our corpus | 2.2B × ~5, the same three sources | ~6T tokens, general | ~2T tokens, general |
| Fine-tuning (rows) | | full + replay (20K) | full + replay (20K) | full (20K) | LoRA r 32 (5K) |
| Grounded accuracy | ≥ 60% | 31.6% ❌ | 34.6% ❌ | **64.4%** ✅ | **91.2%** ✅ |
| Hallucination | ≤ 15% | 32.5% ❌ | 30.7% ❌ | 20.1% ❌ | **8.6%** ✅ |
| Refusal recall (near misses) | ≥ 80% | 78.0% (56%) ❌ | 75.0% (47%) ❌ | 89.0% (79%) ✅ | 90.0% (76%) ✅ |
| Citation / verbatim quote / format | | 90.3 / 99.8 / 100% | 89.5 / 99.8 / 100% | 97.3 / 98.2 / 99.9% | 98.0 / 99.1 / 99.3% |
| Held-out bits/byte change (≤ +0.02) | | +0.009 ✅ | +0.018 ✅ | **+0.308** ❌ | +0.020 ✅ |
| CaseHOLD change (reported) | | −4.8 | −1.5 | −9.6 | −0.1 |

**What it shows:**
- **4× the parameters on the same ≈ 2B-token corpus barely helps:** +3 points grounded accuracy (about one
  standard error).
- **About the same size with ≈ 750× more pretraining doubles it:** Gemma 3 270M has 100M parameters outside
  its embeddings (ours: 113M) and gains +33 points. The rule fixed before the run said ≥ 60% means
  "pretraining scale is the lever even at this size".
- **Reading comes from pretraining; fine-tuning mostly teaches refusing.** Gemma 2 2B, instruction-tuned
  and with no fine-tuning, already scores 83.8% grounded. But it refuses only 25% of unanswerable
  questions (3% of near misses). LoRA on 5,000 of our rows took refusal recall to 90% and hallucination to
  8.6%.
- **The data and labels are sound:** Gemma 2 2B meets all seven targets on our rows.
- **Forgetting is the price of adapting a general model.** No tested Gemma 3 setting kept its held-out
  bits/byte within +0.02, so its result is reported with that miss. Gemma 2 2B passed only at a low LoRA
  learning rate.
- **Fair test:**
  - The 500M's pipeline splits validation by window, so it probably saw our test documents. Measured
    against text neither model saw, it showed no extra familiarity, and it answered under 1% of test
    questions with no passages.
  - The Gemma models' closed-book accuracy is small: excluding those items barely changes their scores.
- **Copy-constrained decoding:** it rescues small models' quotes (39% → 99.8% verbatim for ours). Models
  that already quote well score a few points higher without it.

Write-ups: [`docs/10-gemma.md`](docs/10-gemma.md) (Gemma 2 2B),
[`docs/11-slm500m-raft.md`](docs/11-slm500m-raft.md) (500M, overlap checks),
[`docs/12-gemma3-270m.md`](docs/12-gemma3-270m.md) (Gemma 3 270M). None of these models is published from
this project.

---

## Part 4: preference optimization with AI feedback (RLAIF → DPO), round 1

The fine-tuned models already get the format right. What remains is **choosing**: whether to answer, and
which span to answer from. Preference optimization teaches choices, but only between options the model
already produces. That was measured first.

**Method** (following the course's [RLHF](https://slm-rlhf.vercel.app) and
[preference-data](https://slm-rlhf-data.vercel.app) walk-throughs, with DPO in place of reward model + PPO):
1. **Prompts:** 1,600 verified questions never used in training (answerable, near misses, no relevant
   passages, tasks).
2. **Samples:** 4 on-policy samples per prompt from each fine-tuned model.
   - The answer/refuse decision is *sampled* from the model.
   - Quotes stay verbatim (copy-constrained).
   - A format constraint keeps the syntax rigid, so temperature 0.8 acts on content only: 99.9% of
     samples parse.
3. **AI feedback:** Gemini Flash scores every distinct valid sample 1–5 against a written constitution:
   - refuse only when the passages lack the answer;
   - every claim must be in the cited passage, with a quote that backs it;
   - the answer must match the reference;
   - concise, not long.
4. **Pairs:** samples whose scores differ by ≥ 2 become chosen/rejected.
   - Length bias is checked on answer pairs (chosen/rejected length ratio 1.09 and 1.06, limit 1.15).
   - Judge agreement with the construction labels is ≥ 93% on every question kind.
5. **DPO:** against the fine-tuned model as the frozen reference, plus a small SFT term on the chosen answer
   and pretraining replay for the 125M; β ∈ {0.1, 0.3}. Checkpoints, τ and the pick follow pre-registered
   rules.

**Models:** SLM-125M (ours; the only one with forgetting headroom) and Gemma 3 270M (the best small
model; it missed only the hallucination target). The 500M had no forgetting headroom, and Gemma 2 2B
already met every target.

**Headroom check (go/no-go), dev:** the best of 8 samples is grounded on **56.4%** of questions for
SLM-125M (greedy 34.9%) and **80.4%** for Gemma 3 (greedy 65.8%). Both passed.

**Round 1 results (dev, judged; the test set has not been used yet):**

| Dev, copy-constrained | Grounded | Hallucination | Near-miss refusal | Over-refusal | Δ bits/byte vs SFT |
|---|---|---|---|---|---|
| SLM-125M-RAFT (SFT) | 34.9% | 30.9% | 66.7% | 1.8% | |
| **+ DPO β 0.3 (pick)** | **36.4%** | **29.3%** | 69.4% | 2.9% | +0.0002 |
| + DPO β 0.1 | 29.1% | 22.6% | 91.7% | 23.6% ❌ | +0.0002 |
| Gemma 3 270M RAFT (SFT) | 65.8% | 19.2% | 80.6% | 2.5% | |
| **+ DPO β 0.1 (pick)** | 65.5% | **17.6%** | **88.9%** | 6.2% | +0.016 |

**What it shows:**
- **Preference optimization moved the decision more than the answers.** Near-miss refusal rose by up to 8
  points, and hallucination fell by 1.6 points for both models.
- **That misses the pre-registered bar** of a 5-point hallucination drop with no loss of grounded accuracy.
- **At β 0.1 the 125M over-learned refusing** (it refused 24% of answerable questions), so β 0.3 was picked.
- **The reason is visible in the data.** Most samples scored 2/5, and equal scores make no pair. So right
  answers were rarely contrasted with wrong ones (1,316 and 872 pairs, about half the expected number);
  most pairs taught refusing.
- **A second round would target that:** more samples per answerable question, and pairs only from clearly
  right vs clearly wrong answers.
- **Judge overlap is a known risk:** the labeller and the evaluation judge are both Gemini Flash (different
  prompts). Deterministic metrics, an independent judge and human spot-checks are planned for the final
  test pass.
- **Cost of round 1:** ≈ $7.45 (GPU $2.56, Gemini $4.89).
- Write-up: [`docs/13-rlaif.md`](docs/13-rlaif.md).

---

## Reproduce

**Prerequisites:**
- a Modal account;
- a Hugging Face write token (only for publishing);
- for Part 2, a DeepSeek and a Gemini API key;
- Python 3.12.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cat > .env.local <<'ENV'            # git-ignored; never commit it
MODAL_TOKEN_ID=ak-...
MODAL_TOKEN_SECRET=as-...
HUGGINGFACE_TOKEN=hf_...
DEEPSEEK_API_KEY=sk-...              # Part 2 only
GEMINI_API_KEY=...                   # Part 2 only
ENV
set -a && source .env.local && set +a
modal volume create slm-125m
python3 config.py                   # model: 125,848,320 params (~125.8M) | vocab 16384 | 12L/768d/12h kv=12
```

### Part 1 (base model)

Phases 0–4 are the CPU commands in the pipeline table above (≈ $0). Use `--detach` on long runs, so
a dropped local connection does not stop the job. **GPU steps cost money:** every GPU entrypoint
refuses to start without `--confirm`.

```bash
python smoke_local.py                                                  # $0, CPU: 31 checks, incl. bit-exact resume and 2-process DDP
modal run --detach pretrain_app.py::calibrate --confirm               # ~$0.50, 1 x H100: measures MFU, eager vs torch.compile
modal run --detach pretrain_app.py::pretrain --confirm --max-usd 49 --compile   # ~$25, 8 x H100
modal volume get slm-125m /checkpoints/base .smoke/final/              # + metrics.jsonl, run_status.json, samples.json
python publish_hf.py stats .smoke/final/base && python publish_hf.py model .smoke/final/base
modal deploy serve_modal.py                                            # API on Modal CPU; $0 while idle
```
- **Local data for `smoke_local.py`:** the tokenizer files in `.smoke/tokenizer/`, plus any two
  shards from `/data/tokens/val/` placed in `.smoke/tokens/{train,val}/`.
- **Budget guard:** the trainer projects spend from live throughput and stops itself, checkpointing
  first, before exceeding `--max-usd`.
- **Read-only verifiers:** `modal run verify_tokens.py::check`, `modal run verify_model.py::run`.

### Part 2 (SLM-125M-RAFT): every paid step needs `--confirm`

```bash
python smoke_raft.py --sft                                            # $0, CPU: data, pipeline, trainer, decoding, scoring checks
modal run --detach raft_app.py::prepare --confirm                     # passage pools, leakage scan, BM25 (~$0.2, CPU fan-out)
# teacher data, per stage (trial -> pilot -> scale): sources, generate, candidates, assemble, check, select, upload
python raft_pipeline.py sources    --stage scale --confirm
python raft_pipeline.py generate   --stage scale --writer deepseek --workers 32 --confirm
python raft_pipeline.py candidates --stage scale
python raft_pipeline.py assemble   --stage scale --confirm
python raft_pipeline.py check      --stage scale --checker ds_v3  --confirm
python raft_pipeline.py check      --stage scale --checker ds_off --confirm
python raft_pipeline.py select     --stage scale && python raft_pipeline.py audit --stage scale --confirm
python raft_pipeline.py upload     --stage scale
modal run --detach raft_gpu_app.py::full_check --confirm --max-usd 2.0    # 5K/10K/20K curve + test pass, 1 x H100
python raft_pipeline.py judge --stage scale --confirm && python raft_pipeline.py score --stage scale
python publish_raft.py card && python publish_raft.py upload && python publish_raft.py verify
```
- **Pilot experiments** (`sweep`, `lambda_check`, `confirm_check`, `casehold_check`,
  `pilot_eval_check` in `raft_gpu_app.py`) are documented step by step in `docs/08-raft.md` §13.
- **Spend guards:** the teacher stages charge a per-stage spend ledger, capped by
  `config.RAFT_STAGE_CAPS_USD`, and refuse DeepSeek's peak-price hours.

### Part 3 (the scale study): every GPU step needs `--confirm` and stays under its stage cap

```bash
# Gemma 2 2B (LoRA): weights, probe, baselines, pilot, the 5K test pass
modal run gemma_app.py::fetch_check
modal run --detach gemma_app.py::probe_check --confirm --max-usd 0.5
modal run --detach gemma_app.py::pilot_check --confirm --low-lr --max-usd 1.8
modal run --detach gemma_app.py::test5k_check --confirm --max-usd 0.98
# The 500M (same corpus): weights, replay store, probe, pilot, 20K; then the overlap test (local CPU)
modal run slm500_app.py::fetch_check && modal run slm500_app.py::replay_check --model base500
modal run --detach slm500_app.py::pilot_check --confirm && modal run --detach slm500_app.py::full_check --confirm --lr 2e-5
python membership_check.py --n 200
# Gemma 3 270M (full fine-tuning; needs transformers >= 4.55: .venv-g3 for local checks)
modal run gemma3_app.py::fetch_check
modal run --detach gemma3_app.py::pilot_check --confirm && modal run --detach gemma3_app.py::full_check --confirm --lr 2e-5
# judging (Gemini batch): stage gemma | slm500 | gemma3
python raft_pipeline.py judge --stage gemma3 --confirm && python raft_pipeline.py score --stage gemma3
```
- **Keep the laptop awake** (`caffeinate -i`). Idle sleep makes Modal cancel a running job after about
  two minutes, even with `--detach`. Part 4 avoids this by deploying its app and spawning jobs.

### Part 4 (RLAIF → DPO): GPU jobs run on a deployed private app, so a sleeping laptop can't cancel them

```bash
python -c "import rlaif_data as D; D.write_jsonl(D.LOCAL/'prompts.jsonl', D.build_prompts())"   # spare verified prompts
modal deploy rlaif_app.py && modal run rlaif_app.py::upload_prompts
python rlaif_launch.py spawn headroom_slm 0.3 --confirm && python rlaif_launch.py spawn headroom_g3 0.3 --confirm
python rlaif_pipeline.py fetch --track slm125 --what headroom      # and gemma3
python rlaif_pipeline.py headroom-judge --confirm && python rlaif_pipeline.py headroom-score   # go / no-go
python rlaif_launch.py spawn sample_slm 0.5 --confirm               # and sample_g3; then fetch --what samples
python rlaif_pipeline.py label --track slm125 --confirm && python rlaif_pipeline.py pairs --track slm125
python rlaif_launch.py spawn dpo_slm 0.75 --confirm                 # and dpo_g3; `status` shows progress
python rlaif_pipeline.py dpo-judge --track slm125 --confirm
```

## Repository layout

| Path | Role |
|---|---|
| `config.py` | Single source of truth: model geometry, data mix, budgets and stage caps, paths, training recipes, URLs |
| `cleaning.py`, `dedup.py`, `modal_app.py` | Phases 0–4 (from [`GUIDE.md`](GUIDE.md); `modal_app.py` amended for the document-level split) |
| `data.py`, `model.py`, `train.py`, `pretrain_app.py` | Phase 5: window store, model factory, training loop, Modal GPU launcher |
| `smoke_local.py`, `verify_tokens.py`, `verify_model.py` | Local smoke test and read-only verifiers |
| `serve/`, `serve_modal.py` | Inference API (FastAPI; `/generate`, `/score`, `/answer`) and its Modal deployment |
| `publish_hf.py`, `plot_pretrain.py` | Base model card, training stats, Hub upload; loss-curve figure |
| `raft_app.py` | Phase 7 on Modal CPU: passage pools, leakage scan, BM25, context assembly (one worker per shard) |
| `raft_data.py`, `raft_prompts.py`, `teacher.py`, `raft_pipeline.py` | Teacher data: format, prompts, DeepSeek/Gemini clients with spend ledgers, stage pipeline, J1 judging and scoring |
| `sft.py`, `eval_raft.py`, `eval_forgetting.py`, `raft_gpu_app.py`, `smoke_raft.py` | Grounded SFT trainer, evaluation, forgetting guards, GPU jobs, local checks |
| `raft_decode.py`, `publish_raft.py` | Shared grounded decoding (eval + API + HF; SLM and Gemma prompt formats) and the SLM-125M-RAFT release |
| `gemma_sft.py`, `gemma_app.py`, `smoke_gemma.py` | Part 3, Gemma 2 2B: trainer (LoRA or full fine-tuning; logits only at target positions for the 256K vocabulary), Modal jobs, local checks |
| `slm500_app.py`, `smoke_slm500.py`, `membership_check.py` | Part 3, the 500M: RAFT jobs (replay re-tokenized from our corpus), local checks, the train/test overlap test |
| `gemma3_app.py`, `smoke_gemma3.py` | Part 3, Gemma 3 270M: Modal jobs (transformers 4.57.6, separate venv `.venv-g3`), local checks |
| `rlaif_data.py`, `dpo.py`, `rlaif_app.py`, `rlaif_launch.py`, `rlaif_pipeline.py`, `smoke_rlaif.py` | Part 4: prompts, constitution judge and pairs; DPO trainer; Modal jobs (deployed) and launcher; judging/headroom/labels CLI; local checks |
| `migrate_app.py` | One-off: copied the Modal Volume to a new Modal account (volume to volume) |
| `docs/` | `MASTER_PLAN.md`, one document per phase with measured results (`08-raft.md` = Phase 7, `09-raft-release.md` = Phase 8, `10`–`12` = the scale study, `13-rlaif.md` = preference optimization) |
| `EXPERIMENTS.md` | Experiment / decision / correction log (the E-, D- and C- IDs the docs cite) |
| `GUIDE.md`, `INITIAL.md` | The original specifications |

The website's source is not in this repository; the live site is linked above.

## Design notes worth knowing

- **Prompts start with `<|eos|>`, not `<|bos|>`.** Packing puts an `<|eos|>` after every document and
  never inserts a BOS, so `<|bos|>` occurs 0 times in the 2.06B training tokens.
- **Validation is split by document**, not by window. A window split would leave ~18 sibling windows of
  each SEC validation filing in training.
- **`torch.compile` decided the pretraining budget.** Eager mode measured 15.2% MFU (≈ $51.67 for 4
  epochs). Compiled reached 32.4% (≈ $24.28), with identical losses.
- **Near-duplicate removal is conservative.** 32 MinHash permutations gave LSH 3 bands × 10 rows, which
  flags only 29% of pairs at Jaccard 0.8. So 1,606 is a floor.
- **Fine-tuning forgetting peaks mid-run and recovers as the LR decays.** The learning rate, not the
  replay weight, was the lever: at LR 5e-5 the change stayed under +0.01 bits/byte.
- **A small pilot memorises.** 2,000 examples × 8 epochs scored 95% on its own training questions and
  23% on new ones. The fix was more unique data and choosing checkpoints on answer quality, not on the
  answer/refuse decision.
- **Copy-constrained decoding is free and decisive for faithfulness** (verbatim quotes 39% → 99.8%). It
  cannot fix a wrong span.
- **Pre-registered readings.** Every comparison had its decision rule written down before the run. When a
  forgetting gate failed, the run stopped and the choice (lower LR, LoRA, or report it) was made explicitly
  and logged (`EXPERIMENTS.md` D-30–D-35).
- **Measure headroom before preference optimization.** Best-of-8 against greedy decided whether DPO had
  anything to choose from. Then a format constraint kept sampled answers syntactically valid, so no pair
  ever penalized good grounding for a syntax slip.
- **Judging with a partial set of verdicts is wrong, not low.** A missing verdict counts against the model, so
  `raft_pipeline.py score` records `j1_coverage`, and incomplete sets are never reported.

## Limitations

- **Base model:** fluent in the domain register, unreliable on facts, especially general knowledge.
  It may reproduce passages from public court opinions and SEC filings, including names.
- **Training data:** US-only case law, partly OCR'd; SEC text skews toward 1990s 10-K filings.
- **Grounded model:**
  - misses three of its seven pre-set targets (above);
  - weak on near-miss questions;
  - limited to 4 passages of about 160 tokens by the 1,024-token context.
- **No safety tuning.** Preference optimization (Part 4) is one round, evaluated on dev only so far; it
  missed its pre-registered bar.
- **Scale study:** one seed per setting. Gemma 2 2B was tested at 5K rows (it met every target there).
  Gemma 3 270M's forgetting gate was not met. Gemma models carry Google's Gemma terms.

## License

No license has been chosen yet. Until one is added, all rights are reserved by default. The training
data keeps its own terms: see each dataset's page (fineweb-edu is ODC-By 1.0). Teacher-generated
training data for Part 2 came from DeepSeek (whose terms allow training on outputs) and was checked by
Gemini.
