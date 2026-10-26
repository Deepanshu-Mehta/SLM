"""$0 local membership check: does slm-500m-base treat our dev/test passages like training text?
Per-passage bits/byte (prefix <|eos|>, score the passage tokens) for our 125M and the 500M, on oracle
passages from our TRAIN rows (both models trained on these docs) and our DEV/TEST rows (our 125M never saw
these docs; the 500M did if its window-level split kept them)."""
import json, math, random, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(6)
D = ".smoke/raft/scale/datasets"
rows = {g: [json.loads(l) for l in open(f"{D}/{g}.jsonl")] for g in ("train", "dev", "test")}
rng = random.Random(0)
def pick(rs, n):
    out = []
    for src in ("case-law", "sec", "fineweb-edu"):
        pool = [r for r in rs if r["bucket"] == "A" and r["source"] == src]
        out += [(src, r["passages"][r["k"] - 1]) for r in rng.sample(pool, n)]
    return out
sets = {"train": pick(rows["train"], 40), "devtest": pick(rows["dev"] + rows["test"], 40)}
res = {}
for name, path in (("ours125", ".smoke/final/base"), ("their500", ".smoke/slm500m/base")):
    tok = AutoTokenizer.from_pretrained(path)
    net = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
    eos = tok.convert_tokens_to_ids("<|eos|>")
    for sname, items in sets.items():
        for src, text in items:
            ids = [eos] + tok.encode(text, add_special_tokens=False)
            x = torch.tensor([ids])
            with torch.no_grad():
                loss = float(net(input_ids=x, labels=x).loss)
            nats = loss * (len(ids) - 1)
            res.setdefault((name, sname, src), []).append(nats / (math.log(2) * len(text.encode())))
    del net
    print("done", name, flush=True)
print(f"{'model':9s} {'source':12s} {'train':>7s} {'devtest':>8s} {'gap':>7s}  (bits/byte, mean of 40 passages; gap = devtest - train)")
summary = {}
for name in ("ours125", "their500"):
    for src in ("case-law", "sec", "fineweb-edu", "ALL"):
        get = lambda s: np.mean(sum((res[(name, s, x)] for x in ("case-law", "sec", "fineweb-edu")), [])) if src == "ALL" else np.mean(res[(name, s, src)])
        tr, dt = get("train"), get("devtest")
        summary[(name, src)] = (tr, dt)
        print(f"{name:9s} {src:12s} {tr:7.3f} {dt:8.3f} {dt - tr:+7.3f}")
# paired view: per-passage 500M bpb relative to the 125M reference (log-ratio), devtest vs train
for s in ("train", "devtest"):
    a = np.array(sum((res[("their500", s, x)] for x in ("case-law", "sec", "fineweb-edu")), []))
    b = np.array(sum((res[("ours125", s, x)] for x in ("case-law", "sec", "fineweb-edu")), []))
    print(f"500M/125M bits-per-byte ratio on {s}: {np.mean(a / b):.3f} (median {np.median(a / b):.3f})")
json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res.items()}, open(".smoke/slm500m/membership.json", "w"))
