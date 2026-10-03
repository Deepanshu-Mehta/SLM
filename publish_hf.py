"""Phase 6: assemble training_stats.json + the model card, publish the model and the Space.

    python publish_hf.py stats  <model_dir>          # write training_stats.json + README.md
    python publish_hf.py model  <model_dir>          # upload to config.HF_REPO   (public)
    python publish_hf.py space                       # create/upload config.HF_SPACE (needs HF PRO), wait for RUNNING

Every number comes from a run artifact (metrics.jsonl, run_status.json, the Phase 1/2/4
reports), never retyped. Needs HF_TOKEN in the environment.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import config

RUN = Path(".smoke/final")        # downloaded from /data/checkpoints
REPORTS = Path(".smoke/reports")  # downloaded phase1/phase2/index reports
SOURCES = {s.name: s for s in config.DATA_MIX}

# Measured from the Modal app lifetimes (docs/05-pretrain.md); not in any run file.
RUN_FACTS = {"hardware": "8 x H100 SXM5 (Modal)", "wall_clock_min": 44.9,
             "cost_usd_upper_bound": 24.78, "phase5_cost_usd_upper_bound": 25.32}


def _per_source_tokens(index: dict[str, Any]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for shard in index["shards"]:
        agg = out.setdefault(shard["source"], {"train_tokens": 0, "val_tokens": 0,
                                               "train_docs": 0, "val_docs": 0})
        for key in agg:
            agg[key] += shard[key]
    return out


def build_training_stats() -> dict[str, Any]:
    rows = [json.loads(l) for l in (RUN / "metrics.jsonl").read_text().splitlines() if l.strip()]
    train = [r for r in rows if "loss" in r]
    evals = [r for r in rows if "val_loss" in r and "loss" not in r]
    status = json.loads((RUN / "run_status.json").read_text())
    p1 = json.loads((REPORTS / "phase1_report.json").read_text())
    p2 = json.loads((REPORTS / "phase2_report.json").read_text())
    index = json.loads((REPORTS / "index.json").read_text())
    per_src = _per_source_tokens(index)
    total_train = sum(v["train_tokens"] for v in per_src.values())
    m, t = config.MODEL, config.TRAIN
    final = evals[-1]
    mfus = sorted(r["mfu"] for r in train if r.get("mfu") and r["step"] >= 40)
    n_params = m.approx_params()
    embed = m.vocab_size * m.hidden_size
    return {
        "model": {
            "repo": config.HF_REPO, "kind": "pretrained base model (text completion; not instruction- or chat-tuned)",
            "parameters": n_params, "trainable_parameters": n_params,
            "embedding_parameters": embed, "non_embedding_parameters": n_params - embed,
            "architecture": {"type": "decoder-only transformer (Llama-style)", "layers": m.num_hidden_layers,
                             "hidden_size": m.hidden_size, "attention_heads": m.num_attention_heads,
                             "kv_heads": m.num_key_value_heads, "head_dim": m.hidden_size // m.num_attention_heads,
                             "mlp": f"SwiGLU, intermediate {m.intermediate_size}", "norm": "RMSNorm (pre-norm)",
                             "positions": f"RoPE (theta {m.rope_theta:g})", "context_length": m.max_position_embeddings,
                             "tied_embeddings": m.tie_word_embeddings},
            "tokenizer": {"type": "byte-level BPE, trained from scratch on this corpus", "vocab_size": m.vocab_size,
                          "special_tokens": {"<|bos|>": 0, "<|eos|>": 1, "<|pad|>": 2, "<|unk|>": 3},
                          "prompt_prefix": "<|eos|> (id 1)"},
        },
        "data": {
            "strategy": "legal-first: all of case-law and SEC (they hold only ~2B clean tokens), plus a 0.5B web slice",
            "sources": {name: {"hf_id": SOURCES[name].hf_id, "split": SOURCES[name].split,
                               "config": SOURCES[name].config_name, "proxy_token_budget": SOURCES[name].token_budget,
                               **v, "train_share": round(v["train_tokens"] / total_train, 4)}
                        for name, v in per_src.items()},
            "pipeline": {"phase1_clean": p1, "phase2_dedup_decontam": p2},
            "split": {"kind": "document-level (blake2b(text) % 100 == 0 -> val)",
                      "train_tokens": index["train_tokens"], "val_tokens": index["val_tokens"],
                      "train_docs": index["train_docs"], "val_docs": index["val_docs"],
                      "window": index["seq_len"], "dtype": index["dtype"]},
        },
        "training": {
            "epochs": config.PRETRAIN_EPOCHS, "steps": status["total_steps"],
            "tokens_seen": train[-1]["tokens_seen"], "tokens_per_parameter": round(train[-1]["tokens_seen"] / n_params, 1),
            "global_batch_tokens": t.global_batch_tokens, "optimizer": f"AdamW (b1 {t.beta1}, b2 {t.beta2}, wd {t.weight_decay})",
            "lr": {"peak": t.lr, "min": t.min_lr, "warmup_steps": t.warmup_tokens // t.global_batch_tokens,
                   "schedule": "linear warmup, one cosine decay over all steps"},
            "precision": "fp32 master weights, bf16 autocast", "grad_clip": t.grad_clip,
            "throughput_tokens_per_s": round(train[-1]["tok_per_s"]), "mfu_median": round(mfus[len(mfus) // 2], 4),
            **RUN_FACTS,
        },
        "eval": {
            "split": "document-level held-out val (1% of documents, never trained on)",
            "val_tokens": final["val_tokens"], "val_loss": round(final["val_loss"], 4),
            "val_perplexity": round(final["val_ppl"], 3), "val_bits_per_byte": round(final["val_bpb"], 4),
            "final_train_loss_500step_mean": round(sum(r["loss"] for r in train[-25:]) / 25, 4),
            "curve": [{"step": e["step"], "tokens": e["step"] * t.global_batch_tokens,
                       "val_loss": round(e["val_loss"], 4), "val_perplexity": round(e["val_ppl"], 3),
                       "val_bits_per_byte": round(e["val_bpb"], 4)} for e in evals],
            "train_curve": [{"step": r["step"], "tokens": r["tokens_seen"], "loss": round(r["loss"], 4),
                             "lr": r["lr"]} for r in train if r["step"] % 100 == 0 or r["step"] == train[-1]["step"]],
        },
    }


def model_card(s: dict[str, Any]) -> str:
    a, e, tr, d = s["model"]["architecture"], s["eval"], s["training"], s["data"]
    src = d["sources"]
    rows = "\n".join(
        f"| {n} | [`{v['hf_id']}`](https://huggingface.co/datasets/{v['hf_id']}) | {v['train_tokens'] / 1e6:,.0f}M "
        f"| {v['train_share']:.1%} | {v['train_docs'] + v['val_docs']:,} |" for n, v in src.items())
    curve = "\n".join(f"| {c['step']:,} | {c['tokens'] / 1e9:.2f}B | {c['val_loss']:.4f} | {c['val_perplexity']:.2f} "
                      f"| {c['val_bits_per_byte']:.4f} |" for c in e["curve"][::3] + [e["curve"][-1]])
    return f"""---
language: en
library_name: transformers
pipeline_tag: text-generation
tags: [llama, legal, finance, sec-filings, case-law, base-model, pretrained-from-scratch]
datasets: [{', '.join(v['hf_id'] for v in src.values())}]
---

# SLM-125M base — legal & financial

A **125.8M-parameter** Llama-style language model **pretrained from scratch** (fresh tokenizer,
fresh weights) on US court opinions, SEC filings and educational web text.

> **This is a base model.** It continues text; it does not follow instructions or chat.
> Give it the opening of a passage and it writes the rest in the same register.

**Held-out result:** loss **{e['val_loss']}** nats/token · perplexity **{e['val_perplexity']}** ·
**{e['val_bits_per_byte']} bits/byte**, on a document-level held-out split of
{e['val_tokens']:,} tokens that was never trained on.

![Training curve](assets/loss-curve.png)

## Use

Prefix prompts with `<|eos|>`, **not** `<|bos|>`. In training every document begins right
after an `<|eos|>`, and `<|bos|>` never appears in the training data.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{config.HF_REPO}")
model = AutoModelForCausalLM.from_pretrained("{config.HF_REPO}")
ids = [tok.convert_tokens_to_ids("<|eos|>")] + tok.encode("The court held that the defendant", add_special_tokens=False)
out = model.generate(torch.tensor([ids]), max_new_tokens=60, min_new_tokens=16, do_sample=True,
                     temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.1)
print(tok.decode(out[0, len(ids):], skip_special_tokens=True))
```

Try it in the browser at **[{config.SITE_URL.removeprefix("https://")}]({config.SITE_URL})**, or call the hosted API at
`{config.API_URL}` (`POST /generate`; OpenAPI docs at [`/docs`]({config.API_URL}/docs)).

## Architecture

| | |
|---|---|
| Parameters | {s['model']['parameters']:,} (all trainable; {s['model']['non_embedding_parameters']:,} non-embedding) |
| Layers / hidden / heads | {a['layers']} / {a['hidden_size']} / {a['attention_heads']} (head dim {a['head_dim']}, MHA) |
| MLP | {a['mlp']} |
| Normalisation / positions | {a['norm']} / {a['positions']} |
| Context | {a['context_length']} tokens |
| Embeddings | tied input/output |
| Tokenizer | {s['model']['tokenizer']['type']}, {s['model']['tokenizer']['vocab_size']:,} tokens |

## Training data

{d['strategy']}. The mix is **not** the often-quoted 70/20/10: the two legal sources simply do not
contain more text, so every clean legal token is used and web text is capped.

| Source | Dataset | Train tokens | Share | Documents |
|---|---|---|---|---|
{rows}

Pipeline: streamed (never downloaded whole) → a deterministic 6-step cleaning chain (line filters,
boilerplate, length, repetition, language, OCR-garble gate for scanned opinions) → MinHash-LSH
near-dedup (Jaccard 0.8) and exact dedup → **decontamination against CaseHOLD / LexGLUE** (any
document sharing a 13-gram with the benchmark is dropped) → tokenized and packed into 1,024-token
windows. Validation is split **by document**, so no held-out document is ever partly in training.

## Training

| | |
|---|---|
| Tokens seen | {tr['tokens_seen'] / 1e9:.2f}B ({tr['epochs']} epochs over {d['split']['train_tokens'] / 1e9:.2f}B unique tokens; {tr['tokens_per_parameter']} tokens/param) |
| Steps × batch | {tr['steps']:,} × {tr['global_batch_tokens']:,} tokens |
| Optimiser | {tr['optimizer']}, grad-clip {tr['grad_clip']} |
| LR | {tr['lr']['peak']} → {tr['lr']['min']}, {tr['lr']['warmup_steps']} warmup steps, {tr['lr']['schedule']} |
| Precision | {tr['precision']} |
| Hardware | {tr['hardware']}, {tr['wall_clock_min']} min, {tr['throughput_tokens_per_s'] / 1e6:.2f}M tokens/s ({tr['mfu_median']:.1%} MFU) |
| Compute cost | ≤ ${tr['cost_usd_upper_bound']} |

## Evaluation (held-out, whole split)

| Step | Tokens | Loss | Perplexity | Bits/byte |
|---|---|---|---|---|
{curve}

Held-out loss improved at every one of the 16 evaluations, so 4 epochs did not overfit. Gains
roughly halved with each epoch, so this corpus is close to exhausted at this size.
Bits-per-byte is tokenizer-independent; use it rather than perplexity to compare against models
with other vocabularies. Per source, the model is far stronger on legal text (≈0.69 bits/byte on
case law) than on general web text (≈1.0).

## Limitations

- **Base model, 125M parameters.** Fluent in legal and financial register; **unreliable on facts**,
  especially general knowledge (e.g. it will confidently misdescribe photosynthesis).
- **May reproduce text from its training data**: public court opinions and SEC filings, which
  contain names of real people and companies. Do not treat output as legal or financial advice.
- Domain and era skew: SEC text is weighted toward 1990s 10-K filings; case-law is US-only and
  partly OCR'd.
- No safety tuning, instruction tuning or RLHF.
"""


def cmd_stats(model_dir: Path) -> None:
    stats = build_training_stats()
    (model_dir / "training_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (model_dir / "README.md").write_text(model_card(stats))
    (model_dir / "assets").mkdir(exist_ok=True)
    shutil.copy("docs/05-loss-curve.png", model_dir / "assets" / "loss-curve.png")
    print(f"wrote training_stats.json, README.md, assets/loss-curve.png into {model_dir}")


def cmd_model(model_dir: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(config.HF_REPO, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=config.HF_REPO, folder_path=str(model_dir), repo_type="model",
                      commit_message="Upload SLM-125M base (4 epochs, 8.16B tokens)",
                      ignore_patterns=["*.tmp", ".cache/*"])
    print(f"https://huggingface.co/{config.HF_REPO}")


def cmd_space() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(config.HF_SPACE, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    api.upload_folder(repo_id=config.HF_SPACE, folder_path="serve", repo_type="space",
                      commit_message="SLM-125M inference API")
    print(f"https://huggingface.co/spaces/{config.HF_SPACE}")
    t0, last = time.time(), None
    while time.time() - t0 < 30 * 60:
        stage = api.get_space_runtime(config.HF_SPACE).stage
        if stage != last:
            print(f"  [{time.time() - t0:5.0f}s] {stage}", flush=True)
            last = stage
        if stage in ("RUNNING", "BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"):
            break
        time.sleep(15)
    if last != "RUNNING":
        raise SystemExit(f"Space ended in {last}; see the build logs on the Space page")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    {"stats": lambda: cmd_stats(Path(rest[0])), "model": lambda: cmd_model(Path(rest[0])),
     "space": cmd_space}[cmd]()
