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

```bash
curl -X POST https://deepanshu027-slm-125m-api.hf.space/generate \
  -H 'content-type: application/json' \
  -d '{"prompt": "The court held that the defendant", "max_new_tokens": 60}'
```

Runs on free CPU hardware. After a period of inactivity the Space sleeps, and the first
request wakes it (this can take a few tens of seconds).
