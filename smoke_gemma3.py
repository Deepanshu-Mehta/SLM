"""$0 local checks for Phase 11 (docs/12-gemma3-270m.md, step 11.1). Run in the separate venv:

    .venv-g3/bin/python smoke_gemma3.py          # transformers 4.57.6 (gemma3_text needs >= 4.55)

1. The Gemma 3 tokenizer resolves to the GEMMA format; all 21,839 rows encode (one <bos>, <eos> last, the
   forced decision tokens, <= 1,024 tokens).
2. A tiny random Gemma 3 (real 262K tokenizer, CPU): target-only CE == the HF forward's CE; teacher-forced s
   == HF; FULL fine-tuning through gemma_sft.train_run (keep='select', state round trip); generation forces
   the decision; copy-constraint walks on Gemma 3 tokens end in verbatim quotes.
"""

from __future__ import annotations

import math
import random
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM, Gemma3TextConfig

import config
import gemma_sft as GS
import raft_data as R
import raft_decode as RD
import sft

TOK = ".smoke/gemma3/gemma-3-270m"
DATA = Path(".smoke/raft/scale/datasets")
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def rows(g: str) -> list[dict]:
    return sft.load_rows(DATA / f"{g}.jsonl")


def tiny(tok) -> Gemma3ForCausalLM:
    torch.manual_seed(0)
    cfg = Gemma3TextConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=32, query_pre_attn_scalar=32,
                           max_position_embeddings=2048, sliding_window=512, layer_types=["sliding_attention", "sliding_attention", "full_attention"],
                           pad_token_id=0, eos_token_id=1, bos_token_id=2, tie_word_embeddings=True)
    cfg._attn_implementation = "eager"
    return Gemma3ForCausalLM(cfg).float().eval()


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TOK)
    print("format")
    d = RD.decision_ids(tok)
    check("Gemma 3 resolves to the GEMMA format", d["format"] == "gemma" and d["none"] == 7293 and d["pass"] == 113155
          and d["pad"] == 0 and d["bos"] == 2 and d["stop"][0] == 1, str(d))
    check("single-token decodes keep the leading space", tok.decode([d["none"]]) == " none" and tok.decode([d["pass"]]) == " Passage")
    bad = longest = n = 0
    for g in ("train", "dev", "test"):
        for r in rows(g):
            n += 1
            x, labels = R.encode_example(tok, RD.GEMMA.convert(r["prompt"]), RD.GEMMA.convert(r["target"]))
            longest = max(longest, len(x))
            tg = x[labels.count(-100):]
            want = [d["none"]] if r["bucket"] in sft.REFUSAL_BUCKETS else [d["pass"], *d["forced"], d["digits"][r["k"]]]
            bad += (x[0] != 2 or x.count(2) != 1 or x[-1] != 1 or tg[:len(want)] != want or len(x) > config.GEMMA.max_len)
    check(f"all {n} rows: one <bos>, <eos> last, decision tokens, <= {config.GEMMA.max_len}", bad == 0, f"{bad} bad, longest {longest}")

    print("tiny random Gemma 3 (CPU)")
    net = tiny(tok)
    dev_rows = rows("dev")
    pick = [r for r in dev_rows if r["bucket"] == "A"][:6] + [r for r in dev_rows if r["bucket"] in ("B1", "B2")][:4]
    data = GS.encode_rows(pick, tok)
    x, m = GS.pad_batch(data.seqs[:1], data.plen[:1].tolist(), d["pad"])      # one full row: targets sit at the end
    with torch.no_grad():
        r = GS.loss_sums(net, x, m, d)
        logits = net(input_ids=x, use_cache=False).logits.float()
        mm = m[:, 1:]
        ref = torch.nn.functional.cross_entropy(logits[:, :-1][mm], x[:, 1:][mm], reduction="sum")
    check("target-only CE == HF forward CE", float(ref) > 0 and abs(float(r["sft"]) - float(ref)) < 1e-3 * float(ref),
          f"{float(r['sft']):.3f} vs {float(ref):.3f} over {int(mm.sum())} target tokens")
    del logits
    tf = GS.teacher_forced(net, data, d, torch.device("cpu"), batch=2)
    with torch.no_grad():
        want = [float(RD.decision_score(net(input_ids=torch.from_numpy(s)[None], use_cache=False).logits[0, p - 1], d))
                for s, p in zip(data.seqs, data.plen)]
    check("teacher_forced s == decision_score of the HF forward", max(abs(a - b) for a, b in zip(tf["s"], want)) < 1e-4)

    init = GS.trainable_state(net)
    calls = []

    def extra() -> dict:
        calls.append(1)
        return {"heldout_pooled_bpb": 1.0, "gate_ok": True, "select_score": float(len(calls) == 2)}
    train = GS.encode_rows(pick * 6, tok)
    res = GS.train_run(net, init, train, data, d, torch.device("cpu"), lr=3e-3, epochs=3, n_evals=3, extra_eval=extra,
                       log=lambda *_: None, micro_tokens=4_096)
    h = res["history"]
    check("full fine-tuning: completes, 3 evals, every weight trainable, loss falls",
          res["status"] == "completed" and len(h) == 3 and all(p.requires_grad for p in net.parameters())
          and h[-1]["dev_target_loss"] < h[0]["dev_target_loss"], f"{h[0]['dev_target_loss']:.2f} -> {h[-1]['dev_target_loss']:.2f}")
    check("keep='select' picks the best-scoring eval (the 2nd)", res["best"]["step"] == h[1]["step"], f"best {res['best']['step']}")
    xs = x[:1, :64]
    with torch.no_grad():
        GS.load_trainable(net, res["best_state"]); a = net(input_ids=xs).logits
        GS.load_trainable(net, init); b = net(input_ids=xs).logits
        GS.load_trainable(net, res["best_state"]); c = net(input_ids=xs).logits
    check("full-state round trip (best / init / best)", torch.equal(a, c) and not torch.allclose(a, b))

    prompts, cps = [r["prompt"] for r in data.rows], [r["passages"] or [] for r in data.rows]
    for tau in (0.0, 1.01):
        g = RD.generate_batch(net, tok, prompts, [len(c) for c in cps], torch.device("cpu"), tau=tau, batch=5, max_new=20, copy_passages=cps)
        want = "none" if tau == 0.0 else "pass"
        ok = all(o["choice"] == want for o in g)
        if want == "pass":
            ok &= all(o["text"].startswith(" Passage ") and 1 <= int(o["text"][9]) <= len(c) for o, c in zip(g, cps))
        check(f"generate_batch tau={tau}: decision '{want}' forced", ok, repr(g[0]["text"][:30]))

    print("LoRA on the tiny Gemma 3 (D-34)")
    base = tiny(tok).to(torch.bfloat16)
    pm, linit = GS.add_lora(base)
    calls.clear()
    res = GS.train_run(pm, linit, train, data, d, torch.device("cpu"), lr=3e-3, epochs=3, n_evals=3, extra_eval=extra,
                       log=lambda *_: None, micro_tokens=4_096)
    trainable = [n for n, p in pm.named_parameters() if p.requires_grad]
    check("LoRA: only adapters train (fp32), embeddings frozen, run completes",
          res["status"] == "completed" and trainable and all("lora_" in n for n in trainable)
          and all(p.dtype == torch.float32 for n, p in pm.named_parameters() if p.requires_grad), f"{len(trainable)} adapter tensors")
    GS.load_trainable(pm, res["best_state"])
    with torch.no_grad():
        before = pm(input_ids=xs).logits.float()
        merged = pm.merge_and_unload()
        after = merged(input_ids=xs).logits.float()
    check("LoRA merge matches the adapter model (bf16)", torch.allclose(before, after, atol=0.1, rtol=0.05),
          f"max {float((before - after).abs().max()):.3f}")

    print("copy constraint on Gemma 3 tokens (random walks)")
    vocab = [tok.decode([j]) for j in range(len(tok))]
    rng = random.Random(0)
    ok_n = 0
    walks = [r for r in dev_rows if r["bucket"] == "A"][:10]
    for r in walks:
        prompt = tok.encode(RD.GEMMA.convert(r["prompt"]), add_special_tokens=False)
        cc = RD.CopyConstraint(tok, len(prompt), [r["passages"]], vocab, eos_id=tok.eos_token_id)
        gen = tok.encode(f' Passage {r["k"]}\nQuote: "', add_special_tokens=False)
        for _ in range(200):
            scores = torch.randn(1, len(tok), generator=torch.Generator().manual_seed(rng.randrange(10**9)))
            out = cc(torch.tensor([prompt + gen]), scores)
            gen.append(rng.choice(torch.nonzero(out[0] > -math.inf).flatten().tolist()))
            text = tok.decode(gen, skip_special_tokens=True)
            if "\nAnswer:" in text:
                break
        mt = re.match(r'^ Passage (\d)\nQuote: "(.*?)"\nAnswer:', text, re.S)
        ok_n += bool(mt) and mt.group(2) in r["passages"][r["k"] - 1]
    check(f"{len(walks)} walks end in a verbatim quote + the forced scaffold", ok_n == len(walks), f"{ok_n}")

    print("gates")
    import gemma3_app as A
    for args, name in (((False, 0.4, "11.2-probe"), "no --confirm"), ((True, 9.0, "11.2-probe"), "over the cap")):
        try:
            A._gate(*args)
            check(f"_gate refuses: {name}", False)
        except SystemExit:
            check(f"_gate refuses: {name}", True)
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED: {failures}'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
