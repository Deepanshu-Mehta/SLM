"""$0 local membership test for Phase 10 (docs/11-slm500m-raft.md §2, steps 10.0 and 10.1, E-40/E-41).

Did slm-500m-base (and the author's 125M, same corpus and window-level split) train on our dev/test passages?

Per-passage bits/byte (prefix <|eos|>, all passage tokens scored, nats / UTF-8 bytes) for three models:
  ours125   SLM-125M base: saw our TRAIN documents only (document-level split)
  their125  thesreedath/slm-125m-base: same corpus/split as the 500M
  their500  thesreedath/slm-500m-base
on four passage sets:
  train     oracle passages of our RAFT train rows (all models saw these documents)
  devtest   oracle passages of our dev+test rows (ours125 did not see them; the others probably did)
  unseen    fineweb-edu sample-10BT file 013 (beyond both corpora's ~0.5B-token prefix: nobody saw it),
            cleaned and chunked with our pipeline, clean-tagged, one passage per document

Difference-in-differences on web text, with ours125 as the reference that saw neither devtest nor unseen:
  DiD_M = [bpb_M(devtest) / bpb_M(unseen)] / [bpb_ours(devtest) / bpb_ours(unseen)]
  ~1 if M did not see devtest; ~(1 - s_M) if it did, s_M = M's seen-vs-unseen reduction.
Positive control for s_M: bpb_M(train) / bpb_M(unseen) on web text (M saw train; includes set differences).
95% intervals by bootstrap over passages.

    python membership_check.py [--n 200]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import cleaning
import raft_data as R

D = Path(".smoke/raft/scale/datasets")
UNSEEN = Path(".smoke/slm500m/fineweb_unseen_013.jsonl")
MODELS = {"ours125": ".smoke/final/base", "their125": ".smoke/slm500m/their125", "their500": ".smoke/slm500m/base"}
SOURCES = ("case-law", "sec", "fineweb-edu")


def passage_sets(n: int, seed: int = 0) -> dict[str, list[tuple[str, str]]]:
    rng = random.Random(seed)
    rows = {g: [json.loads(ln) for ln in open(D / f"{g}.jsonl")] for g in ("train", "dev", "test")}

    def oracle(rs: list[dict], src: str) -> list[str]:
        seen, out = set(), []
        for r in rs:
            if r["bucket"] == "A" and r["source"] == src:
                p = r["passages"][r["k"] - 1]
                if p not in seen:
                    seen.add(p)
                    out.append(p)
        return out

    sets = {"train": [], "devtest": []}
    for src in SOURCES:
        tr, dt = oracle(rows["train"], src), oracle(rows["dev"] + rows["test"], src)
        sets["train"] += [(src, p) for p in rng.sample(tr, min(n, len(tr)))]
        sets["devtest"] += [(src, p) for p in rng.sample(dt, min(n, len(dt)))]
    tok = AutoTokenizer.from_pretrained(MODELS["ours125"])
    unseen = []
    for ln in open(UNSEEN):
        doc = json.loads(ln)["text"]
        res = cleaning.clean_document(doc)
        if not res.kept:
            continue
        cands = [t for t, _ in R.chunk_document(res.text, tok) if not R.passage_tags(t, "fineweb-edu")]
        if cands:
            unseen.append(("fineweb-edu", rng.choice(cands)))
    sets["unseen"] = rng.sample(unseen, min(n, len(unseen)))
    return sets


@torch.no_grad()
def score(path: str, items: list[tuple[str, str]]) -> list[float]:
    tok = AutoTokenizer.from_pretrained(path)
    net = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
    eos = tok.convert_tokens_to_ids("<|eos|>")
    out = []
    for _, text in items:
        ids = [eos] + tok.encode(text, add_special_tokens=False)
        x = torch.tensor([ids])
        nats = float(net(input_ids=x, labels=x).loss) * (len(ids) - 1)
        out.append(nats / (math.log(2) * len(text.encode("utf-8"))))
    return out


def boot(fn, arrays: list[np.ndarray], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    vals = [fn(*[a[rng.integers(0, len(a), len(a))] for a in arrays]) for _ in range(n)]
    return float(fn(*arrays)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="passages per source per set")
    a = ap.parse_args()
    torch.set_num_threads(6)
    sets = passage_sets(a.n)
    print({k: len(v) for k, v in sets.items()}, flush=True)
    bpb = {}
    for m, path in MODELS.items():
        for s, items in sets.items():
            bpb[(m, s)] = np.array(score(path, items))
        print("scored", m, flush=True)
    src = {s: np.array([x for x, _ in items]) for s, items in sets.items()}
    rep = {"n": {k: len(v) for k, v in sets.items()}, "mean_bpb": {}, "did_web": {}, "train_vs_unseen_web": {}}
    print(f"\n{'model':9s} {'source':12s} {'train':>7s} {'devtest':>8s} {'unseen':>7s}")
    for m in MODELS:
        for so in SOURCES:
            row = [bpb[(m, s)][src[s] == so].mean() if (src[s] == so).any() else float("nan") for s in ("train", "devtest", "unseen")]
            rep["mean_bpb"][f"{m}|{so}"] = row
            print(f"{m:9s} {so:12s} " + " ".join(f"{v:7.3f}" for v in row))
    web = lambda m, s: bpb[(m, s)][src[s] == "fineweb-edu"]
    print("\nweb text, 95% bootstrap intervals:")
    for m in ("their125", "their500"):
        did = boot(lambda md, mu, od, ou: (md.mean() / mu.mean()) / (od.mean() / ou.mean()),
                   [web(m, "devtest"), web(m, "unseen"), web("ours125", "devtest"), web("ours125", "unseen")])
        rep["did_web"][m] = did
        print(f"  DiD {m} (devtest vs unseen, ref ours125): {did[0]:.3f} [{did[1]:.3f}, {did[2]:.3f}]   (1 = not seen)")
    for m in MODELS:
        tv = boot(lambda t, u: t.mean() / u.mean(), [web(m, "train"), web(m, "unseen")])
        dv = boot(lambda t, u: t.mean() / u.mean(), [web(m, "devtest"), web(m, "unseen")])
        rep["train_vs_unseen_web"][m] = {"train/unseen": tv, "devtest/unseen": dv}
        print(f"  {m:9s} train/unseen {tv[0]:.3f} [{tv[1]:.3f}, {tv[2]:.3f}]   devtest/unseen {dv[0]:.3f} [{dv[1]:.3f}, {dv[2]:.3f}]")
    for m in ("their125", "their500"):            # legal sources: ratio to ours125, train vs devtest
        for so in ("case-law", "sec"):
            f = lambda s: (bpb[(m, s)] / bpb[("ours125", s)])[src[s] == so]
            d = boot(lambda t, v: v.mean() / t.mean(), [f("train"), f("devtest")])
            rep.setdefault("legal_ratio_shift", {})[f"{m}|{so}"] = d
            print(f"  {m:9s} {so:9s} ratio-to-ours devtest/train {d[0]:.3f} [{d[1]:.3f}, {d[2]:.3f}]   (<1 = extra advantage on devtest)")
    Path(".smoke/slm500m/membership_v2.json").write_text(json.dumps(
        {**rep, "bpb": {f"{m}|{s}": v.tolist() for (m, s), v in bpb.items()},
         "sources": {s: v.tolist() for s, v in src.items()}}, indent=1))


if __name__ == "__main__":
    main()
