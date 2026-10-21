# SLM-125M — a legal & financial language model, pretrained from scratch

A **125.8M-parameter** Llama-style language model trained from scratch, with a fresh tokenizer
and fresh weights, on US court opinions, SEC filings and educational web text. The whole
pipeline runs on [Modal](https://modal.com): streaming the data, cleaning, deduplication,
benchmark decontamination, tokenizer training, packing, 4-epoch pretraining on 8 × H100,
and serving. Compute cost: **≤ $25.32**.

**[Try it in the browser](https://slm-125m-legal-sandy.vercel.app)** ·
**[Model on Hugging Face](https://huggingface.co/Deepanshu027/slm-125m-base)** ·
**[API docs](https://deepanshu-m23csai--slm-125m-serve-api.us-east.modal.run/docs)** ·
**[How it was built, visually](https://slm-125m-legal-sandy.vercel.app/build)**

> **This is a base model.** It continues text; it does not follow instructions or chat. It writes
> fluent legal and financial prose and **invents facts and citations**. Not legal or financial advice.

![Training and held-out loss over 8.16B tokens](docs/05-loss-curve.png)

## Results

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

## The data, and why it is mixed this way

| Source | Dataset | Train tokens | Share |
|---|---|---|---|
| US case law | [`HFforLegal/case-law`](https://huggingface.co/datasets/HFforLegal/case-law) (`us`) | 715.3M | 35.1% |
| SEC filings | [`PleIAs/SEC`](https://huggingface.co/datasets/PleIAs/SEC) | 860.6M | 42.2% |
| Educational web | [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`) | 464.2M | 22.8% |

The obvious recipe, 70% case law / 20% SEC / 10% web at 10B tokens, is **impossible**. Measured
before downloading anything, case law holds about 0.81B clean tokens, while 70% of 10B would need
7B. So the mix is *legal-first*: every clean legal token, plus a capped web slice. That gives 77% legal.

## Pipeline

| Phase | What happens | Command | Output on the Modal Volume |
|---|---|---|---|
| 0 | Smoke test (10 docs/source) and yield measurement | `modal run modal_app.py::main` then `::measure` | — |
| 1 | Stream parquet shards (20 workers, one per shard), 6-rule cleaning chain | `modal run --detach modal_app.py::clean --fineweb-shards 5` | `/data/clean/` |
| 2 | MinHash-LSH near-dedup, exact dedup, CaseHOLD 13-gram decontamination | `modal run --detach modal_app.py::dedup` | `/data/corpus/` |
| 3 | Train a 16,384-token byte-level BPE tokenizer | `modal run --detach modal_app.py::tokenizer` | `/data/tokenizer/` |
| 4 | Tokenize, pack into 1,024-token `uint16` windows, **document-level** 99/1 split | `modal run --detach modal_app.py::tokenize` | `/data/tokens/` |
| 5 | Pretrain (calibrate on 1 × H100, then 8 × H100) | see below | `/data/checkpoints/` |
| 6 | Publish to the Hub, serve on Modal CPU | see below | — |

Funnel: 718,780 documents streamed → 697,958 after cleaning → **670,124** after removing 1,606
near-duplicates, 2,051 exact duplicates and 24,177 benchmark-contaminated documents (10.3% of
case law). Each phase is documented, with its measured results, in [`docs/`](docs/).

## Reproduce

**Prerequisites:** a Modal account, a Hugging Face write token (only for publishing), and
Python 3.12.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cat > .env.local <<'ENV'            # git-ignored; never commit it
MODAL_TOKEN_ID=ak-...
MODAL_TOKEN_SECRET=as-...
HUGGINGFACE_TOKEN=hf_...
ENV
set -a && source .env.local && set +a
modal volume create slm-125m
python3 config.py                   # model: 125,848,320 params (~125.8M) | vocab 16384 | 12L/768d/12h kv=12
```

Then run Phases 0–4 in order with the commands above. They are CPU-only and cost about $0. Use
`--detach` on long runs, so a dropped local connection does not stop the job.

**Phase 5: GPU, costs money.** Every GPU entrypoint refuses to start without `--confirm`.

```bash
python smoke_local.py                                                  # $0, CPU: 31 checks, incl. bit-exact resume and 2-process DDP
modal run --detach pretrain_app.py::calibrate --confirm               # ~$0.50, 1 x H100: measures MFU, eager vs torch.compile
modal run --detach pretrain_app.py::pretrain --confirm --max-usd 49 --compile   # ~$25, 8 x H100
```

`smoke_local.py` reads a small local copy of the data: the tokenizer files in `.smoke/tokenizer/`,
plus any two shards from `/data/tokens/val/` placed in `.smoke/tokens/train/` and `.smoke/tokens/val/`
(fetch them with `modal volume get`). The trainer projects spend from live throughput and stops
itself, checkpointing first, if a run would exceed `--max-usd`. Checkpoints are written atomically
every 500 steps, and a preempted container resumes exactly where it stopped.

**Phase 6: publish and serve.**

```bash
modal volume get slm-125m /checkpoints/base .smoke/final/    # + metrics.jsonl, run_status.json, samples.json
python publish_hf.py stats .smoke/final/base                  # training_stats.json + model card
python publish_hf.py model .smoke/final/base                  # upload to config.HF_REPO
modal deploy serve_modal.py                                   # API on Modal CPU; $0 while idle
```

Verification scripts (read-only, CPU): `modal run verify_tokens.py::check` and
`modal run verify_model.py::run`. The latter exits non-zero if the parameter count, tied
embeddings, token ids or bits-per-byte constant are wrong.

## Repository layout

| Path | Role |
|---|---|
| `config.py` | Single source of truth: model geometry, data mix, budgets, paths, training recipe, URLs |
| `cleaning.py`, `dedup.py`, `modal_app.py` | Phases 0–4 (from [`GUIDE.md`](GUIDE.md); `modal_app.py` amended for the document-level split) |
| `data.py`, `model.py`, `train.py`, `pretrain_app.py` | Phase 5: window store, model factory, training loop, Modal GPU launcher |
| `smoke_local.py`, `verify_tokens.py`, `verify_model.py` | Local smoke test and read-only verifiers |
| `serve/`, `serve_modal.py` | Inference API (FastAPI) and its Modal deployment |
| `publish_hf.py`, `plot_pretrain.py` | Model card, training stats, Hub upload; loss-curve figure |
| `docs/` | `MASTER_PLAN.md` plus one document per phase, each with measured results |
| `GUIDE.md`, `INITIAL.md` | The original specifications |

The website's source is not in this repository; the live site is linked above.

## Design notes worth knowing

- **Prompts start with `<|eos|>`, not `<|bos|>`.** Packing puts an `<|eos|>` after every
  document and never inserts a BOS, so `<|bos|>` occurs 0 times in the 2.06B training tokens.
- **Validation is split by document**, not by window. A window split would leave ~18 sibling
  windows of each SEC validation filing in training.
- **`torch.compile` decided the budget.** Eager mode measured 15.2% MFU, which would put 4 epochs
  at about $51.67; compiled reached 32.4%, about $24.28, with identical losses.
- **Near-duplicate removal is conservative.** 32 MinHash permutations gave LSH 3 bands × 10 rows,
  which flags only 29% of pairs at Jaccard 0.8. So 1,606 is a floor.

## Limitations

A 125M base model: fluent in the domain register, unreliable on facts, especially general
knowledge. It may reproduce passages from public court opinions and SEC filings, including
names. US-only case law, partly OCR'd; SEC text skews toward 1990s 10-K filings. No instruction,
preference or safety tuning.

## License

No license has been chosen yet. Until one is added, all rights are reserved by default. The
training data keeps its own terms: see each dataset's page (fineweb-edu is ODC-By 1.0).
