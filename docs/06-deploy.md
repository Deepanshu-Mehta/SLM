# Phase 6 — Publish the model and serve it

**Compute:** CPU · **Cost:** ~$0 (idle = $0) · **Status:** ✅ **DONE**: model, API and website all live

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

## Hosting decision (2026-10-03): Modal CPU, scale-to-zero

The HF Docker Space is blocked by the PRO requirement (402). Chosen instead: serve the same,
already-tested `serve/app.py` as a Modal web endpoint (`serve_modal.py`):

| Setting | Value | Why |
|---|---|---|
| Hardware | **CPU only**, 2 physical cores, 3 GiB | No GPU anywhere in serving |
| `min_containers` | **0** | **$0 while idle**: nothing runs between visits |
| `scaledown_window` | 180 s | The container shuts itself off 3 minutes after the last request |
| `max_containers` | **1** | Hard cap: worst case is one CPU container, ≈ **$0.12/hour** |
| Rate limit | 30 POSTs per IP per minute → HTTP 429 | A public site can't be hammered into a big bill |
| `@modal.concurrent` | 16 inputs | `/health` and `/info` answer even while a generation holds the lock |
| Weights | baked into the image from `Deepanshu027/slm-125m-base` at build | One canonical copy; no download on a cold start |

Take it down entirely with `modal app stop slm-125m-serve`.

**Validation:** live endpoints answer from the public URL (including CORS and the 429), and
**after the idle window the container count returns to 0**, proving nothing keeps billing.

License: **left unset by decision** (2026-10-03).

## Results (2026-10-03)

| # | Check | Result |
|---|---|---|
| 1 | No Modal app or container running | ✅ 0 live apps, 0 containers; nothing is billing |
| 2 | Server tested locally against the verified artifact | ✅ all endpoints (below) |
| 3 | Model loads from the Hub | ✅ **https://huggingface.co/Deepanshu027/slm-125m-base**: public, `from_pretrained` works, `architectures` present, bos/eos/pad 0/1/2, **weights SHA-256 identical** to the locally verified file |
| 4 | Space builds on free `cpu-basic` | ⛔ **blocked → replaced by Modal CPU (decision 2026-10-03)**: `402 Payment Required` — *"Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription."* Nothing was created |
| 5 | Live endpoints from a public URL | ✅ **https://mdeepanshu2706--slm-125m-serve-api.modal.run** (Modal CPU, scale-to-zero): see *Live test* below |

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

### Live test of the Modal endpoint (2026-10-03)

`https://mdeepanshu2706--slm-125m-serve-api.modal.run`, deployed with `modal deploy serve_modal.py`.

| Check | Result |
|---|---|
| Cold start (no container → first `/health`) | **11.9 s**; warm `/health` 0.8 s |
| `/info` | live run facts: 125,848,320 trainable params; val 2.1269 / 8.389 / 0.6572 bits/byte; 8,160,018,432 tokens; 16-point eval curve |
| `/generate` speed (2 CPU cores) | **11.8–17.5 tokens/s**; a 60-token completion in 3.4–5.1 s |
| `/generate/stream` | 27 SSE events then `{"done": true}` |
| `/score` | legal ppl **5.856** vs nonsense **488.297**, identical to the local run |
| CORS preflight | HTTP 200, `access-control-allow-origin: *` |
| Rate limit | after exactly **30** POSTs in a minute, every further POST gets **429** |
| **Scale to zero** | 1 container at 17:04:33 → **0 containers at 17:08:11** (218 s after the last request). App state `deployed`, **0 running tasks = $0 while idle** |

Live samples:
- *"We review the district court's grant of summary judgment de novo"* → *". Ventura v. City
  & Cnty., 2015 MT 4, ¶ 10, 378 Mont. 212, 343 P.3d 558; Gatz v. State, 2014 MT 268, ¶ 11…"*.
  The citation format is perfect (and `¶` survives the character filter), but **the
  citations are invented**. That's the base-model limitation the website must disclose.
- *"Risk Factors. An investment in our common stock involves"* → *" several risks, including
  the following: - the potential for: - dilution of earnings and earnings per share…"*

**Not verified:** whether Modal forwards each visitor's IP in `x-forwarded-for`. All test
requests came from one machine, so the 429 proves the cap works but not whether it is
per-visitor or shared. Either way, it bounds load.

### Bug fixed during deploy

The first deploy crash-looped with `ModuleNotFoundError: No module named 'config'`, and
every request was redirected (HTTP 303) after 150 s. `serve_modal.py` imports `config`,
which wasn't in the image. Fix: `.add_local_python_source("config")`. This is the same
class of mistake as the earlier `verify_tokens.py` crash, and the lesson is the same: a
Modal image must contain every module the *entrypoint file* imports, not just the server.

### Stream fix (2026-10-03)

`/generate/stream` events are **text chunks, not tokens**. `TextIteratorStreamer` flushes at
word boundaries, so 30 tokens arrive as about 27 events. The final `done` event now carries
the true `new_tokens`, `prompt_tokens` and `stopped_on_eos`, read from the generate output.
Verified live: 27 events, then `{"done": true, "new_tokens": 30, ...}`.

## Website (2026-10-03)

Live at **https://slm-125m-legal-sandy.vercel.app**. Plan, build notes and the full verification record are in
`docs/07-website-plan.md` §11–12. The site code (`site/`) is kept out of git by request.

## Phase 6 deliverables

| Deliverable | URL |
|---|---|
| HF model repo | https://huggingface.co/Deepanshu027/slm-125m-base |
| Inference API (Modal CPU, scale-to-zero) | https://deepanshu-m23csai--slm-125m-serve-api.us-east.modal.run (`/docs` for OpenAPI). Moved from `https://mdeepanshu2706--slm-125m-serve-api.modal.run` on 2026-10-21 (new Modal account, D-22) |
| Website (Vercel) | https://slm-125m-legal-sandy.vercel.app |

### Model card correction (2026-10-03)

The first published card linked to `Deepanshu027/slm-125m-api`, a Space that was never created
(the 402). It was regenerated by `publish_hf.py stats`, which now links the live website and
Modal API from `config.SITE_URL` / `config.API_URL`. Only `README.md` was re-uploaded
(Hub commit `1ba0a91`). The weights are unchanged: `model.safetensors` still has hash
`e080e475…`, the value verified at publish time.
