"""$0 local checks for Phase 12 (docs/13-rlaif.md, step 12.0). Exits non-zero on any failure.

1. Sampling: decision sampling follows s; FormatConstraint forces the refusal text and the openers.
2. rlaif_data: valid_sample, judge jobs (distinct + valid only), score-gap pairs, length statistics, agreement.
3. dpo: the DPO term equals a hand computation; target log-probs equal the HF forward's; a tiny model trained on
   toy pairs raises its pair accuracy; reference log-probs equal the policy's at step 0.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import dpo as DPO
import raft_decode as RD
import rlaif_data as D

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def tiny(tok) -> LlamaForCausalLM:
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=1024,
                                        tie_word_embeddings=True, bos_token_id=0, eos_token_id=1, pad_token_id=2)).float()


def main() -> None:
    tok = AutoTokenizer.from_pretrained(".smoke/final/base")
    rows = {r["id"]: r for r in D.read_jsonl(D.LOCAL / "prompts.jsonl")}
    a = next(r for r in rows.values() if r["bucket"] == "A")
    print("sampling")
    ids = RD.decision_ids(tok)
    logits = torch.zeros(1, len(tok))
    logits[0, ids["none"]], logits[0, ids["pass"]] = math.log(0.3), math.log(0.7)
    g = torch.Generator().manual_seed(0)
    n_none = 0
    for _ in range(2000):
        rc = RD.RaftConstraint(ids, 0, [4], tau=0.5, sample_decision=True, generator=g)
        rc(torch.zeros((1, 0), dtype=torch.long), logits.clone())
        n_none += rc.choice[0] == "none"
    check("sampled decision refuses with probability s (s = 0.3)", abs(n_none / 2000 - 0.3) < 0.03, f"{n_none / 2000:.3f}")
    vocab = [tok.decode([j]) for j in range(len(tok))]
    fc = RD.FormatConstraint(tok, 0, vocab, ids)
    seq = [ids["none"]]
    for _ in range(len(fc.refusal_tail)):
        out = fc(torch.tensor([seq]), torch.zeros(1, len(tok)))
        seq.append(int(torch.argmax(out[0])))
    check("refusal branch is forced to the training text + stop",
          tok.decode(seq[:-1]) == " none\nAnswer: " + RD.REFUSAL and seq[-1] == RD.EOS_ID, repr(tok.decode(seq)))
    body = tok.encode(" Passage 2", add_special_tokens=False)
    out = fc(torch.tensor([body]), torch.zeros(1, len(tok)))
    allowed = [vocab[i] for i in torch.nonzero(out[0] > -math.inf).flatten().tolist()]
    check("after ` Passage k` only `\\nQuote: \"` / `\\nAnswer:` openers are allowed",
          allowed and all(("\nQuote: \"".startswith(t) or "\nAnswer:".startswith(t) or t.startswith("\nAnswer:")) for t in allowed), f"{len(allowed)} tokens")
    ans = tok.encode(' Passage 2\nQuote: "x"\nAnswer: abc', add_special_tokens=False)
    out = fc(torch.tensor([ans]), torch.zeros(1, len(tok)))
    check("in the answer: no newline tokens, stop allowed",
          all(out[0, i] == -math.inf for i in fc.newline[:50]) and out[0, RD.EOS_ID] > -math.inf)

    print("rlaif_data")
    good = f' Passage {a["k"]}\nQuote: "{a["quote"]}"\nAnswer: {a["answer"]}'
    check("valid_sample: the reference output is valid; a broken quote and a bad format are not",
          D.valid_sample(a, good) and not D.valid_sample(a, f' Passage {a["k"]}\nQuote: "zzz not there"\nAnswer: x')
          and not D.valid_sample(a, " Passage 1 and then"))
    samples = {a["id"]: [{"text": good, "choice": "pass"}, {"text": good, "choice": "pass"},
                         {"text": " none\nAnswer: " + RD.REFUSAL, "choice": "none"}, {"text": " garbage", "choice": "pass"}]}
    jobs = D.judge_jobs(rows, samples, "r1")
    check("judge jobs: distinct, valid samples only (2 of 4)", len(jobs) == 2, f"{len(jobs)}")
    check("constitution shows the reference for A, never the bucket", "Reference answer:" in jobs[0].user and "bucket" not in jobs[0].user.lower())
    verdicts = {f"{a['id']}|r1|0": {"score": 5, "should_refuse": False}, f"{a['id']}|r1|2": {"score": 2, "should_refuse": False}}
    sc = D.scored(rows, samples, verdicts, "r1")
    pr, st = D.pairs(rows, sc, tok)
    check("pairs: one pair, answer (5) > refusal (2); duplicates collapse", len(pr) == 1 and pr[0]["chosen"] == good
          and pr[0]["kind"] == "answer>refuse", str(st["kinds"]))
    check("length statistics present", "length_ratio" in st and "length_bias_flag" in st, str(st))
    check("agreement vs construction label", D.agreement(rows, verdicts) == {"A": 1.0})

    print("dpo")
    lc, lr_, rc, rr = torch.tensor([-10.0]), torch.tensor([-12.0]), torch.tensor([-11.0]), torch.tensor([-11.0])
    d, m = DPO.dpo_terms(lc, lr_, rc, rr, beta=0.1)
    want = -math.log(1 / (1 + math.exp(-0.1 * ((-10 + 11) - (-12 + 11)))))
    check("DPO term == hand computation", abs(float(d) - want) < 1e-6 and abs(float(m) - 0.2) < 1e-6, f"{float(d):.6f} vs {want:.6f}")
    net = tiny(tok)
    enc = DPO.encode_pairs(pr * 8, rows, tok)
    with torch.no_grad():
        lp, n = DPO.target_logps(net, [enc[0]["c"][0]], [enc[0]["c"][1]], 2, torch.device("cpu"))
        x = torch.from_numpy(enc[0]["c"][0])[None]
        logits = net(input_ids=x).logits[0].float()
        p0 = enc[0]["c"][1]
        ref = float(torch.log_softmax(logits[p0 - 1:-1], -1).gather(1, x[0, p0:, None]).sum())
    check("target_logps == HF forward sum over target tokens", abs(float(lp[0]) - ref) < 1e-3, f"{float(lp[0]):.3f} vs {ref:.3f} ({int(n[0])} tokens)")
    DPO.reference_logps(net, enc, 2, torch.device("cpu"), lambda: torch.autocast("cpu", enabled=False))
    with torch.no_grad():
        lp2, _ = DPO.target_logps(net, [enc[0]["c"][0]], [enc[0]["c"][1]], 2, torch.device("cpu"))
    check("reference log-probs equal the policy's before training", abs(enc[0]["ref_c"] - float(lp2[0])) < 1e-4)
    res = DPO.train_dpo(net, enc, torch.device("cpu"), beta=0.5, lr=1e-3, epochs=3, pairs_per_step=8, micro=4, alpha=0.0,
                        log=lambda *_: None)
    with torch.no_grad():
        lc2, _ = DPO.target_logps(net, [p["c"][0] for p in enc[:1]], [p["c"][1] for p in enc[:1]], 2, torch.device("cpu"))
        lr2, _ = DPO.target_logps(net, [p["r"][0] for p in enc[:1]], [p["r"][1] for p in enc[:1]], 2, torch.device("cpu"))
    margin = (float(lc2[0]) - enc[0]["ref_c"]) - (float(lr2[0]) - enc[0]["ref_r"])
    check("DPO training raises the chosen-vs-rejected margin above 0", res["status"] == "completed" and margin > 0, f"margin {margin:.3f}")
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED: {failures}'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
