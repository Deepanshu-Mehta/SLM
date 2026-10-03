# Phase 6 — Publish the model and serve it

**Compute:** CPU · **Cost so far:** $0 · **Status:** model published ✅ · API hosting ⛔ blocked on a decision

Scope of this phase, per the 2026-10-03 instruction: push the model to Hugging Face,
serve it, make sure no GPU is left running, and **plan** (not yet build) the Vercel
website. The website plan is `docs/07-website-plan.md`.

## Decisions taken

| Decision | Choice | Why |
|---|---|---|
| Namespace | **`Deepanshu027`** (requested: `mdeepanshu`) | `huggingface.co/mdeepanshu` returns 404, and the write token in `.env.local` belongs to `Deepanshu027`, so a push to `mdeepanshu/…` would be rejected. HF repos can be renamed or moved later. |
| Visibility | **Public** model repo and Space | A website anyone can open must call a public endpoint; a public Space exposes the model anyway. |
| Serving | **HF Space, Docker SDK, free `cpu-basic`** (2 vCPU, 16 GB) | $0 and nothing to switch off. Trade-off: the Space sleeps after inactivity and the first request wakes it. The alternative, a Modal `@asgi_app`, would be warm but bill while running. |
| Weights in the Space | Downloaded from the model repo **at image build time** | One canonical copy of the weights, and no download on a cold start. |
| License | **Left unset — owner's decision** | Not something to choose on your behalf. Sources: fineweb-edu is ODC-By 1.0; check HFforLegal/case-law and PleIAs/SEC terms before picking one. |

## What gets published

**`Deepanshu027/slm-125m-base`** (model): `config.json` (with `architectures`),
`model.safetensors` (bf16, 251.7 MB), `generation_config.json`, the 3 tokenizer
files, `training_stats.json` (machine-readable run facts: architecture, data mix,
eval curve, cost — the website reads these), `README.md` model card,
`assets/loss-curve.png`.

**`Deepanshu027/slm-125m-api`** (Space): `Dockerfile` (Python 3.12 pinned, CPU
torch), `app.py` (FastAPI), `requirements.txt`, `README.md` (Space config).

## API contract (what the Vercel site will call)

| Method | Path | Body → Response |
|---|---|---|
| GET | `/info` | → model facts: params (all trainable), architecture, data mix, val loss/ppl/bits-per-byte, tokens, eval curve |
| POST | `/generate` | `{prompt, max_new_tokens, min_new_tokens, temperature, top_k, top_p, repetition_penalty, seed?}` → `{generated, prompt_tokens, new_tokens, stopped_on_eos, elapsed_ms, tokens_per_s}` |
| POST | `/generate/stream` | same body → Server-Sent Events, one `{"token": …}` per piece, then `{"done": true, …}` |
| POST | `/score` | `{text}` → per-token surprise (nats, probability), perplexity and bits-per-byte of *your* text |
| GET | `/health` | → `{"status": "ok"}` (used by the site's "waking the model" state) |

CORS is open, so a static Vercel site can call it directly.

### Generation rules baked into the server (the Phase 3 and 6 gotchas)

- Every prompt is prefixed with **`<|eos|>` (id 1), not `<|bos|>`**: BOS never
  occurs in the training data (measured 0 times in 2.06B tokens).
- **`min_new_tokens` defaults to 16**, so a base model can't emit EOS after one token.
- **Partial-UTF-8 tokens are suppressed by default.** These are byte-level tokens
  that decode to U+FFFD on their own, the source of "unicode garbage". Valid
  non-ASCII tokens such as `§`, `’` and `—` are kept, since legal text depends on them.
- `temperature = 0` means greedy decoding; `top_k = 0` disables top-k.
- Long prompts are trimmed from the **left**, so prompt + output fit the 1,024-token context.
- One generation at a time, behind a lock (2 vCPUs; parallel requests would only thrash).

## Validation criteria

| # | Check |
|---|---|
| 1 | No Modal app or container is running (no GPU billing) |
| 2 | Server passes a local test against the verified artifact before any upload |
| 3 | Model repo loads via `AutoModelForCausalLM.from_pretrained("Deepanshu027/slm-125m-base")` from the Hub |
| 4 | Space builds and reaches `RUNNING` on free `cpu-basic` hardware |
| 5 | Live `/info`, `/generate`, `/generate/stream`, `/score` all answer correctly from the public URL, including a CORS preflight |

## Results (2026-10-03)

| # | Check | Result |
|---|---|---|
| 1 | No Modal app or container running | ✅ 0 live apps, 0 containers; nothing is billing |
| 2 | Server tested locally against the verified artifact | ✅ all endpoints (below) |
| 3 | Model loads from the Hub | ✅ **https://huggingface.co/Deepanshu027/slm-125m-base**: public, `from_pretrained` works, `architectures` present, bos/eos/pad 0/1/2, **weights SHA-256 identical** to the locally verified file |
| 4 | Space builds on free `cpu-basic` | ⛔ **blocked**: `402 Payment Required` — *"Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription."* Nothing was created |
| 5 | Live endpoints from a public URL | ⏸ pending the hosting decision (`docs/07-website-plan.md` §9) |

### Local API test (`serve/app.py` against the downloaded `base/`)

- `/info`: 125,848,320 params, all trainable; val bits/byte 0.6572; 133 partial-UTF-8
  tokens suppressed.
- `/generate` (seed 0): *"…’s motion to quash was properly denied and the case was remanded
  for a new trial…"*, at 32 tok/s on the Mac CPU.
- `/generate` greedy on *"Item 7. Management's Discussion and Analysis…"*: *"The following
  discussion should be read in conjunction with the consolidated financial statements…"*
- Same seed twice gives identical output. `/generate/stream` sends 25 SSE events and then `done`.
- `/score`: legal sentence ppl **5.86** (0.455 bits/byte) vs nonsense ppl **488** (2.55).
- Bad input returns 422 (empty prompt, temperature 9). The CORS preflight returns
  `access-control-allow-origin: *`.
- The partial-UTF-8 filter keeps every character legal text needs (`§ ’ “ ” — – é ¶ °`
  are single tokens). Only `½ € £ ¢` become unreachable, since they need two partial
  byte tokens; the filter can be switched off per request.

### Hub check of the published model

Fresh-cache download, then a sample: *"The Company believes that its existing cash balances,
together with anticipated cash flows from operations, will be sufficient to meet the
Company's anticipated requirements for working capital and capital expenditures…"*

### Files

| File | Role |
|---|---|
| `publish_hf.py` | builds `training_stats.json` and the model card from run artifacts; `model` / `space` upload subcommands |
| `serve/app.py`, `serve/Dockerfile`, `serve/requirements.txt`, `serve/README.md` | the inference API (FastAPI) and its Space packaging |
| `docs/07-website-plan.md` | the Vercel website plan |
