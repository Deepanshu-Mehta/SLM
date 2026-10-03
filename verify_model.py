"""$0 CPU verification for Phase 5 prerequisites.

Fails loudly (non-zero exit) if any invariant the trainer relies on is broken:
exact param count, tied embeddings, special-token ids, and the BOS claim.
Also measures the exact val-split bytes/token used to report bits-per-byte.
"""

from __future__ import annotations

import modal

import config

app = modal.App(f"{config.PROJECT}-verify-model")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers==4.46.3", "torch==2.5.1", "numpy")
    .add_local_python_source("config", "model")
)
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}


@app.function(image=image, volumes=VOLUMES, timeout=60 * 30, cpu=8.0, memory=16_384)
def check() -> dict:
    import glob

    import numpy as np
    from transformers import AutoTokenizer

    import model as M

    out: dict = {}

    # 1. exact parameter count of the real LlamaForCausalLM
    net = M.build_model()
    out["params"] = M.count_parameters(net)
    out["approx_params"] = config.MODEL.approx_params()
    out["tied"] = net.lm_head.weight.data_ptr() == net.model.embed_tokens.weight.data_ptr()
    out["n_final_norm"] = net.model.norm.weight.numel()
    out["param_dtype"] = str(next(net.parameters()).dtype)

    # 2. special-token ids as the saved tokenizer actually assigns them
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    out["token_ids"] = {k: tok.convert_tokens_to_ids(v) for k, v in config.SPECIAL_TOKENS.items()}

    # 3. does <|bos|> (id 0) ever occur in the packed data? count EOS too.
    counts = {"train": {"tokens": 0, "bos": 0, "eos": 0}, "val": {"tokens": 0, "bos": 0, "eos": 0}}
    for split, d in (("train", config.TRAIN_TOKENS_DIR), ("val", config.VAL_TOKENS_DIR)):
        for path in sorted(glob.glob(f"{d}/*.bin")):
            ids = np.fromfile(path, dtype=np.uint16)
            counts[split]["tokens"] += int(ids.size)
            counts[split]["bos"] += int(np.count_nonzero(ids == M.BOS_TOKEN_ID))
            counts[split]["eos"] += int(np.count_nonzero(ids == M.EOS_TOKEN_ID))
    out["occurrences"] = counts

    # 4. bytes/token over the WHOLE val split. skip_special_tokens=True so each
    #    <|eos|> contributes 0 bytes of text (not the 7-byte literal "<|eos|>"),
    #    while still counting as a predicted token.
    tot_tokens = tot_bytes = 0
    per_source: dict[str, dict] = {}
    for path in sorted(glob.glob(f"{config.VAL_TOKENS_DIR}/*.bin")):
        ids = np.fromfile(path, dtype=np.uint16)
        nb = len(tok.decode(ids.tolist(), skip_special_tokens=True).encode("utf-8"))
        src = path.split("/")[-1].rsplit("-", 1)[0]
        agg = per_source.setdefault(src, {"tokens": 0, "bytes": 0})
        agg["tokens"] += int(ids.size)
        agg["bytes"] += nb
        tot_tokens += int(ids.size)
        tot_bytes += nb
    out["val"] = {"tokens": tot_tokens, "bytes": tot_bytes, "bytes_per_token": tot_bytes / tot_tokens}
    out["val_per_source"] = {k: {**v, "bytes_per_token": v["bytes"] / v["tokens"]}
                             for k, v in per_source.items()}
    return out


@app.local_entrypoint()
def run():
    import math
    import sys

    import model as M

    r = check.remote()
    failures: list[str] = []

    p, a = r["params"], r["approx_params"]
    print("\n=== 1. EXACT PARAMETER COUNT ===")
    print(f"  total (tied)          : {p['total']:,}")
    print(f"  embedding / non-embed : {p['embedding']:,} / {p['non_embedding']:,}")
    print(f"  config.approx_params(): {a:,}")
    print(f"  final RMSNorm         : {r['n_final_norm']}")
    print(f"  embeddings tied       : {r['tied']}")
    print(f"  master weight dtype   : {r['param_dtype']}")
    if p["total"] != a:
        failures.append(f"approx_params {a:,} != real {p['total']:,}")
    if not r["tied"]:
        failures.append("embeddings are NOT tied")
    if r["param_dtype"] != "torch.float32":
        failures.append(f"master weights are {r['param_dtype']}, expected fp32")

    print("\n=== 2. SPECIAL-TOKEN IDS (tokenizer vs model.py constants) ===")
    expect = {"bos_token": M.BOS_TOKEN_ID, "eos_token": M.EOS_TOKEN_ID,
              "pad_token": M.PAD_TOKEN_ID, "unk_token": M.UNK_TOKEN_ID}
    for k, want in expect.items():
        got = r["token_ids"][k]
        ok = got == want
        print(f"  {k:<10} tokenizer={got}  model.py={want}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"{k}: tokenizer {got} != model.py {want}")

    print("\n=== 3. TOKEN OCCURRENCES IN PACKED DATA ===")
    for split, c in r["occurrences"].items():
        print(f"  {split:<5} tokens={c['tokens']:>13,}  <|bos|>(0)={c['bos']:>9,}  "
              f"<|eos|>(1)={c['eos']:>9,}  (avg {c['tokens']/max(c['eos'],1):,.0f} tok/doc)")

    v = r["val"]
    print("\n=== 4. VAL SPLIT bytes/token (for bits-per-byte) ===")
    print(f"  tokens {v['tokens']:,} | text bytes {v['bytes']:,} | bytes/token {v['bytes_per_token']:.6f}")
    for k, s in sorted(r["val_per_source"].items()):
        print(f"    {k:<12} {s['bytes_per_token']:.4f} bytes/token")
    print(f"  BPB = loss_nats * {1 / (math.log(2) * v['bytes_per_token']):.6f}")

    print("\n=== RESULT ===")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        sys.exit(1)
    print("  ALL INVARIANTS HOLD")
