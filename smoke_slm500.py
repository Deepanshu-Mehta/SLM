"""$0 local checks for Phase 10 (docs/11-slm500m-raft.md §7, step 10.2). Exits non-zero on any failure.

1. Both author models' tokenizers give the SLM format; all 21,839 stored rows encode within 1,024 tokens
   with targets starting with the forced decision tokens.
2. Replay packing (doc + <|eos|> windows) round-trips to the source text.
3. eval_forgetting.heldout_bpb_text equals a hand computation.
4. The unchanged Phase 7 trainer (sft.train_one, keep='select', replay, micro 16 and 32) runs a tiny random
   Llama with the 500M tokenizer on CPU; the loss falls.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import config
import eval_forgetting as EF
import eval_raft as E
import raft_data as R
import raft_decode as RD
import sft
import slm500_app as A
from data import WindowStore

TOK = {"base500": ".smoke/slm500m/base", "control125": ".smoke/slm500m/their125"}
DATA = Path(".smoke/raft/scale/datasets")
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def rows(g: str) -> list[dict]:
    return sft.load_rows(DATA / f"{g}.jsonl")


def format_checks() -> None:
    print("format")
    allrows = rows("train") + rows("dev") + rows("test")
    for key, path in TOK.items():
        tok = AutoTokenizer.from_pretrained(path)
        d = RD.decision_ids(tok)
        check(f"{key}: SLM format, our special-token ids", d["format"] == "slm" and d["suppress"] == list(RD.SUPPRESS)
              and d["stop"] == [RD.EOS_ID] and d["pad"] == RD.PAD_ID and d["bos"] == RD.BOS_ID, str(d))
        bad = longest = 0
        for r in allrows:
            x, labels = R.encode_example(tok, r["prompt"], r["target"])
            longest = max(longest, len(x))
            tg = x[labels.count(-100):]
            want = [d["none"]] if r["bucket"] in sft.REFUSAL_BUCKETS else [d["pass"], *d["forced"], d["digits"][r["k"]]]
            bad += (tg[:len(want)] != want or x[0] != RD.EOS_ID or x[-1] != RD.EOS_ID or x.count(RD.EOS_ID) != 2
                    or len(x) > config.SEQ_LEN)
        check(f"{key}: all {len(allrows)} rows: <|eos|> first and last, decision tokens, <= {config.SEQ_LEN}", bad == 0,
              f"{bad} bad, longest {longest}")
        data = sft.tokenize(rows("dev")[:8], tok, config.SEQ_LEN)
        check(f"{key}: sft.tokenize pads with <|pad|> and masks the target", int(data.ids[0, data.length[0]]) == RD.PAD_ID
              and bool(data.mask[0, data.decision_pos[0]]) and not data.mask[0, data.decision_pos[0] - 1])


def replay_checks() -> None:
    print("replay packing")
    tok = AutoTokenizer.from_pretrained(TOK["base500"])
    docs = [json.loads(ln)["text"] for ln in open(".smoke/slm500m/fineweb_unseen_013.jsonl")][:30]
    w = A.pack_windows(docs, tok)
    flat = tok.decode(w.reshape(-1).tolist(), skip_special_tokens=False)
    joined = "".join(d + "<|eos|>" for d in docs)
    check("windows are uint16, 1,024 wide, and decode to a prefix of doc + <|eos|> ...",
          w.dtype == np.uint16 and w.shape[1] == config.SEQ_LEN and joined.startswith(flat[:-50]) and len(flat) > 1000,
          f"{w.shape[0]} windows")
    with tempfile.TemporaryDirectory() as d:
        w.tofile(Path(d) / "x.bin")
        st = WindowStore.open(d, config.SEQ_LEN)
        check("WindowStore reads the store", len(st) == w.shape[0] and np.array_equal(st.gather(np.array([0])), w[:1]))
    check("replay sample is deterministic and independent of the RAFT pool salt",
          A.replay_sample("abc", 0.5) == A.replay_sample("abc", 0.5)
          and sum(A.replay_sample(f"d{i}", 0.015) for i in range(20000)) in range(200, 420))


def bpb_checks() -> None:
    print("tokenizer-free held-out bits/byte")
    tok = AutoTokenizer.from_pretrained(".smoke/final/base")
    net = LlamaForCausalLM(LlamaConfig(vocab_size=32768, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                       num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=2048,
                                       tie_word_embeddings=True)).eval()
    t500 = AutoTokenizer.from_pretrained(TOK["base500"])
    val = Path(".smoke/valshards")
    subset = {k: {**v, "indices": v["indices"][:2]} for k, v in EF.heldout_subset(val).items()}
    got = EF.heldout_bpb_text(net, t500, tok, val, subset, torch.device("cpu"), batch=2)
    spec = subset["case-law"]
    arr = np.memmap(val / spec["shard"], dtype=np.uint16, mode="r").reshape(-1, config.SEQ_LEN)
    nats = nb = 0.0
    with torch.no_grad():
        for wdw in np.asarray(arr[spec["indices"]]):
            t = tok.decode(wdw.tolist(), skip_special_tokens=True)
            ids = torch.tensor([[RD.EOS_ID] + t500(t, add_special_tokens=False, split_special_tokens=True)["input_ids"]])
            nats += float(net(input_ids=ids, labels=ids).loss) * (ids.shape[1] - 1)
            nb += len(t.encode("utf-8"))
    want = nats / (math.log(2) * nb)
    check("heldout_bpb_text == hand computation (<|eos|> prefix, per UTF-8 byte)", abs(got["case-law"]["bpb"] - want) < 1e-3,
          f"{got['case-law']['bpb']:.4f} vs {want:.4f}")


def trainer_checks() -> None:
    print("Phase 7 trainer on a tiny random Llama with the 500M tokenizer")
    tok = AutoTokenizer.from_pretrained(TOK["base500"])
    dec = E.decision_ids(tok)
    pick = [r for r in rows("dev") if r["bucket"] == "A"][:20] + [r for r in rows("dev") if r["bucket"] in ("B1", "B2")][:12]
    data = sft.tokenize(pick, tok, config.SEQ_LEN)
    docs = [json.loads(ln)["text"] for ln in open(".smoke/slm500m/fineweb_unseen_013.jsonl")][:60]
    with tempfile.TemporaryDirectory() as d:
        A.pack_windows(docs, tok).tofile(Path(d) / "r.bin")
        store = WindowStore.open(d, config.SEQ_LEN)
        for micro in (16, 32):
            torch.manual_seed(0)
            net = LlamaForCausalLM(LlamaConfig(vocab_size=32768, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                               num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=1024,
                                               tie_word_embeddings=True, bos_token_id=0, eos_token_id=1, pad_token_id=2))
            sft.init_chat_tokens(net, config.SFT.seed)
            calls = []

            def extra() -> dict:
                calls.append(1)
                return {"heldout_pooled_bpb": 1.0, "gate_ok": True, "select_score": float(len(calls))}
            cfg = sft.RunConfig(lr=3e-3, epochs=4, micro=micro, keep="select", log_every=10_000, eval_every=2, seq_len=config.SEQ_LEN)
            res = sft.train_one(cfg, net, net, data, data, store, dec, torch.device("cpu"), None, log=lambda *_: None, extra_eval=extra)
            h = res["history"]
            plan = res["plan"]
            check(f"micro {micro}: completes with replay ({plan['sft_micro']} SFT + {plan['replay_micro']} replay x {plan['accum']}), "
                  "keep='select' takes the last (best) eval, target loss falls",
                  res["status"] == "completed" and res["best"]["step"] == res["steps"] and h[-1]["dev_target_loss"] < h[0]["dev_target_loss"],
                  f"{h[0]['dev_target_loss']:.2f} -> {h[-1]['dev_target_loss']:.2f}")


def misc_checks() -> None:
    print("probe and gates")
    shots = [{"question": "Q1?", "answer": "A1"}]
    p = A.closed_book_prompt(shots, "Q2?")
    check("closed-book prompt: <|eos|> + few-shot, no passages", p == "<|eos|>Question: Q1?\nAnswer: A1\n\nQuestion: Q2?\nAnswer:")
    for args, name in (((False, 0.5, "10.4-probe"), "no --confirm"), ((True, 9.0, "10.4-probe"), "over the cap")):
        try:
            A._gate(*args)
            check(f"_gate refuses: {name}", False)
        except SystemExit:
            check(f"_gate refuses: {name}", True)


def main() -> None:
    format_checks()
    replay_checks()
    bpb_checks()
    trainer_checks()
    misc_checks()
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED: {failures}'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
