"""Phase 4 verification (read-only). Not part of the GUIDE's four verbatim files.

Checks the packed token artifacts are structurally valid and that the token count
is explained by tokenizer efficiency rather than by lost documents.
"""

from __future__ import annotations

import modal

import config

# Self-contained: this verifier must not depend on modal_app's image, whose
# add_local_python_source list does not include modal_app itself.
app = modal.App(f"{config.PROJECT}-verify")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers==4.46.3", "numpy")
    .add_local_python_source("config")
)
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}


@app.function(image=image, volumes=VOLUMES, timeout=60 * 20, cpu=4.0, memory=8_192)
def verify() -> dict:
    import glob
    import os

    import numpy as np
    from transformers import AutoTokenizer

    seq_bytes = config.SEQ_LEN * 2  # uint16
    report: dict = {}

    # 1. every .bin is an exact multiple of one window
    bad: list[str] = []
    totals = {"train": 0, "val": 0}
    for split in ("train", "val"):
        for path in sorted(glob.glob(f"{config.TOKENS_DIR}/{split}/*.bin")):
            size = os.path.getsize(path)
            totals[split] += size
            if size % seq_bytes:
                bad.append(f"{os.path.basename(path)}: {size} bytes")
    report["window_alignment_violations"] = bad
    report["train_tokens_on_disk"] = totals["train"] // 2
    report["val_tokens_on_disk"] = totals["val"] // 2

    # 2. decode the first window of one shard per source -> must be readable text
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    samples: dict[str, str] = {}
    for src in ("case-law", "sec", "fineweb-edu"):
        hits = sorted(glob.glob(f"{config.TRAIN_TOKENS_DIR}/{src}-*.bin"))
        if not hits:
            continue
        ids = np.fromfile(hits[0], dtype=np.uint16, count=config.SEQ_LEN)
        samples[src] = tok.decode(ids.tolist())[:300]
    report["samples"] = samples

    # 3. measure REAL chars/token on a corpus sample per source
    eff: dict[str, dict] = {}
    for src in ("case-law", "sec", "fineweb-edu"):
        paths = sorted(glob.glob(f"{config.CORPUS_DIR}/{src}/*.txt"))
        chars = ntok = ndocs = 0
        batch: list[str] = []
        with open(paths[0], encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                batch.append(line)
                chars += len(line)
                ndocs += 1
                if ndocs >= 500:
                    break
        for ids in tok(batch, add_special_tokens=False)["input_ids"]:
            ntok += len(ids) + 1  # +1 for the appended <|eos|>
        eff[src] = {
            "docs": ndocs,
            "chars": chars,
            "tokens": ntok,
            "chars_per_token": round(chars / ntok, 3),
        }
    report["efficiency"] = eff
    return report


@app.local_entrypoint()
def check():
    r = verify.remote()

    print("\n=== 1. WINDOW ALIGNMENT (every .bin must be a multiple of 2048 bytes) ===")
    v = r["window_alignment_violations"]
    print(f"  violations: {len(v)}" + (f" -> {v}" if v else "  [PASS]"))
    print(f"  train tokens on disk: {r['train_tokens_on_disk']:,}")
    print(f"  val   tokens on disk: {r['val_tokens_on_disk']:,}")

    print("\n=== 2. DECODED FIRST WINDOW PER SOURCE ===")
    for src, text in r["samples"].items():
        print(f"\n[{src}]\n  {text[:280]}")

    print("\n=== 3. REAL chars/token BY SOURCE (500 docs each) ===")
    tot_c = tot_t = 0
    for src, e in r["efficiency"].items():
        tot_c += e["chars"]
        tot_t += e["tokens"]
        proxy = e["chars"] / config.CHARS_PER_TOKEN
        print(f"  {src:<12} {e['chars']:>10,} chars -> {e['tokens']:>9,} tokens "
              f"| {e['chars_per_token']:.3f} chars/tok "
              f"| proxy predicted {proxy/1e3:.0f}K, real {e['tokens']/1e3:.0f}K "
              f"({e['tokens']/proxy - 1:+.1%})")
    print(f"  {'OVERALL':<12} {tot_c/tot_t:.3f} chars/token")
