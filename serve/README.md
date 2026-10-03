---
title: SLM-125M API
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Inference API for a 125M legal/financial base model
---

# SLM-125M — inference API

FastAPI server for [`Deepanshu027/slm-125m-base`](https://huggingface.co/Deepanshu027/slm-125m-base),
a 125.8M-parameter Llama-style model pretrained from scratch on US case law, SEC filings
and educational web text. It is a **base completer**: give it the start of a passage and it
continues it. It does not follow instructions or chat.

| Method | Path | Purpose |
|---|---|---|
| GET | `/info` | model facts: parameters, data mix, held-out loss / perplexity / bits-per-byte, eval curve |
| POST | `/generate` | `{prompt, max_new_tokens, min_new_tokens, temperature, top_k, top_p, repetition_penalty, seed}` |
| POST | `/generate/stream` | same body, Server-Sent Events |
| POST | `/score` | per-token surprise and perplexity of your own text |
| GET | `/docs` | interactive OpenAPI docs |

Live deployment: **Modal CPU, scale-to-zero** (`serve_modal.py` at the repo root), at
`https://mdeepanshu2706--slm-125m-serve-api.modal.run`. The `Dockerfile` and the front matter
above package the same server as a Hugging Face Docker Space, which requires an HF PRO account.

```bash
curl -X POST https://mdeepanshu2706--slm-125m-serve-api.modal.run/generate \
  -H 'content-type: application/json' \
  -d '{"prompt": "The court held that the defendant", "max_new_tokens": 60}'
```

It runs on CPU and scales to zero when idle, so the first request after a quiet period takes
about 12 seconds. Rate limit: 30 POST requests per minute per client.
