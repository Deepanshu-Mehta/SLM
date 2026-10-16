"""Forgetting guards for Phase 7 (docs/08-raft.md §8.2, §8.4).

1. CaseHOLD (LexGLUE `case_hold`, test split, 3,600 items): zero-shot, each of the five
   candidate holdings scored by its log-likelihood after the citing context. Reports
   accuracy by total log-prob (acc) and by log-prob per UTF-8 byte (acc_norm, as in
   lm-eval-harness; token counts are tokenizer-specific), with Wilson 95% intervals.
2. Held-out bits/byte on a fixed, seeded subset of validation windows: the same number of
   windows from each source's `-000` shard, each source scored with its OWN bytes/token
   (EXPERIMENTS C-05), plus a pooled figure.

The functions take an in-memory model, so the GPU container can run them before and after
SFT on identical numerics. The CLI is the local sanity run of step 7.1.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import config
import model as M

CASEHOLD_ROWS_URL = ("https://datasets-server.huggingface.co/rows?dataset=coastalcph/lex_glue"
                     "&config=case_hold&split=test&offset={offset}&length={length}")
CASEHOLD_TEST_ROWS = 3_600
PLACEHOLDER = "<HOLDING>"
SOURCES = tuple(name for name, _ in config.RAFT.source_mix)

Autocast = Callable[[], contextlib.AbstractContextManager]


def no_autocast() -> contextlib.AbstractContextManager:
    return contextlib.nullcontext()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


# ------------------------------------------------------------------------------ CaseHOLD
def fetch_casehold(cache: Path) -> list[dict]:
    """The test split as a list of {context, endings, label}; downloaded once, then cached."""
    if cache.exists():
        rows = [json.loads(ln) for ln in cache.read_text().splitlines() if ln.strip()]
        if len(rows) == CASEHOLD_TEST_ROWS:
            return rows
    rows: list[dict] = []
    for offset in range(0, CASEHOLD_TEST_ROWS, 100):
        url = CASEHOLD_ROWS_URL.format(offset=offset, length=100)
        for attempt in range(8):
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    page = json.load(resp)
                break
            except urllib.error.HTTPError as e:      # 429: honour Retry-After, else back off
                if attempt == 7 or e.code not in (429, 500, 502, 503, 504):
                    raise
                time.sleep(float(e.headers.get("Retry-After") or 0) or min(60, 5 * 2 ** attempt))
            except OSError:
                if attempt == 7:
                    raise
                time.sleep(min(60, 5 * 2 ** attempt))
        rows.extend(r["row"] for r in page["rows"])
        time.sleep(1.0)                              # stay well under the public rate limit
    if len(rows) != CASEHOLD_TEST_ROWS:
        raise RuntimeError(f"expected {CASEHOLD_TEST_ROWS} CaseHOLD rows, got {len(rows)}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def _continuation_ids(tok, prefix: str, ending: str, max_len: int) -> tuple[list[int], int]:
    """Token ids for <|eos|> + prefix + ending, and the index where the ending starts.

    Uses character offsets of the joint encoding, so a token spanning the boundary counts
    as part of the ending. The prefix is left-truncated to fit max_len.
    """
    text = prefix + ending
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    start = next((i for i, (_, end) in enumerate(offsets) if end > len(prefix)), len(ids))
    room = max_len - 1 - (len(ids) - start)            # tokens of prefix that fit
    if room < 1:
        raise ValueError("ending alone exceeds the context window")
    keep_from = max(0, start - room)
    ids = [M.PROMPT_PREFIX_TOKEN_ID] + ids[keep_from:]
    return ids, 1 + (start - keep_from)


@torch.no_grad()
def score_casehold(net, tok, items: list[dict], device: torch.device,
                   autocast: Autocast = no_autocast, items_per_batch: int = 4) -> dict:
    """Zero-shot CaseHOLD accuracy (acc and byte-normalised acc_norm)."""
    net.eval()
    weight = net.get_output_embeddings().weight
    hits = hits_norm = 0
    for b in range(0, len(items), items_per_batch):
        batch = items[b:b + items_per_batch]
        seqs, starts, nbytes = [], [], []
        for it in batch:
            prefix = it["context"].split(PLACEHOLDER)[0]
            for ending in it["endings"]:
                ids, start = _continuation_ids(tok, prefix, ending, config.SEQ_LEN)
                seqs.append(ids)
                starts.append(start)
                nbytes.append(len(ending.encode("utf-8")))
        width = max(len(s) for s in seqs)
        x = torch.full((len(seqs), width), M.PAD_TOKEN_ID, dtype=torch.long)
        for i, s in enumerate(seqs):        # right padding + causal attention: no mask needed
            x[i, :len(s)] = torch.tensor(s)
        x = x.to(device)
        with autocast():
            hidden = net.model(input_ids=x, use_cache=False).last_hidden_state
        scores = []
        for i, s in enumerate(seqs):
            pos = torch.arange(starts[i] - 1, len(s) - 1, device=device)   # predict start..end
            logits = hidden[i, pos].float() @ weight.float().T
            logp = torch.log_softmax(logits, dim=-1)
            targets = x[i, pos + 1]
            scores.append(float(logp.gather(1, targets[:, None]).sum()))
        for j, it in enumerate(batch):
            s = scores[5 * j:5 * j + 5]
            nb = nbytes[5 * j:5 * j + 5]
            hits += int(np.argmax(s)) == it["label"]
            hits_norm += int(np.argmax([v / n for v, n in zip(s, nb)])) == it["label"]
    n = len(items)
    return {"n": n, "acc": hits / n, "acc_ci95": wilson(hits, n),
            "acc_norm": hits_norm / n, "acc_norm_ci95": wilson(hits_norm, n), "chance": 0.2}


# -------------------------------------------------------------------- held-out bits/byte
def heldout_subset(val_dir: Path, per_source: int = config.RAFT.eval_windows_per_source,
                   seed: int = config.RAFT.seed) -> dict:
    """Seeded window indices from each source's `-000` val shard, with the shard's sha256."""
    subset = {}
    for k, src in enumerate(SOURCES):
        path = val_dir / f"{src}-000.bin"
        n = path.stat().st_size // (config.SEQ_LEN * 2)
        idx = np.sort(np.random.default_rng([seed, k]).choice(n, per_source, replace=False))
        subset[src] = {"shard": path.name, "windows_in_shard": int(n),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "indices": idx.tolist()}
    return subset


@torch.no_grad()
def heldout_bpb(net, tok, val_dir: Path, subset: dict, device: torch.device,
                autocast: Autocast = no_autocast, batch: int = 4) -> dict:
    """Per-source and pooled loss / bits-per-byte on the subset windows."""
    net.eval()
    out, tot_nats, tot_bytes = {}, 0.0, 0
    for src, spec in subset.items():
        path = val_dir / spec["shard"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise RuntimeError(f"{path} differs from the shard the subset was drawn from")
        arr = np.memmap(path, dtype=np.uint16, mode="r").reshape(-1, config.SEQ_LEN)
        windows = np.asarray(arr[spec["indices"]], dtype=np.int64)
        nats = 0.0
        n_pred = 0
        for b in range(0, len(windows), batch):
            x = torch.from_numpy(windows[b:b + batch]).to(device)
            with autocast():
                loss = net(input_ids=x, labels=x, use_cache=False).loss
            k = x.shape[0] * (x.shape[1] - 1)
            nats += float(loss) * k
            n_pred += k
        n_bytes = sum(len(tok.decode(w.tolist(), skip_special_tokens=True).encode("utf-8"))
                      for w in windows)
        n_tokens = windows.size
        loss_mean = nats / n_pred
        bpt = n_bytes / n_tokens
        out[src] = {"windows": len(windows), "loss": loss_mean, "bytes_per_token": bpt,
                    "bpb": loss_mean / (math.log(2) * bpt)}
        tot_nats += loss_mean * n_tokens
        tot_bytes += n_bytes
    out["pooled_bpb"] = tot_nats / (math.log(2) * tot_bytes)
    return out


# ------------------------------------------------------------------------------------ CLI
def main() -> None:
    p = argparse.ArgumentParser(description="Phase 7 forgetting-guard baselines (local sanity run).")
    p.add_argument("--model-dir", default=".smoke/final/base")
    p.add_argument("--val-dir", default=".smoke/valshards")
    p.add_argument("--casehold-cache", default=".smoke/casehold_test.jsonl")
    p.add_argument("--out", default=".smoke/reports/raft_baseline_local.json")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=0, help="score only the first N CaseHOLD items")
    p.add_argument("--skip-casehold", action="store_true")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    net = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float32).to(device)
    report: dict = {"model_dir": args.model_dir, "device": args.device, "dtype": "float32"}

    t0 = time.time()
    subset = heldout_subset(Path(args.val_dir))
    report["heldout"] = heldout_bpb(net, tok, Path(args.val_dir), subset, device)
    report["heldout_subset"] = {k: {kk: v[kk] for kk in ("shard", "windows_in_shard", "sha256")}
                                for k, v in subset.items()}
    print(f"[heldout] {time.time() - t0:.0f}s " + json.dumps(report["heldout"], indent=1), flush=True)

    if not args.skip_casehold:
        items = fetch_casehold(Path(args.casehold_cache))
        if args.limit:
            items = items[:args.limit]
        t0 = time.time()
        report["casehold"] = score_casehold(net, tok, items, device)
        report["casehold"]["seconds"] = round(time.time() - t0, 1)
        print(f"[casehold] {json.dumps(report['casehold'])}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
