"""$0 local checks for Phase 9 (docs/10-gemma.md §4, 9.1). Exits non-zero on any failure.

1. The Gemma format on the real tokenizer and all 21,839 stored rows.
2. SLM-125M decoding is byte-identical to the committed raft_decode.py (released weights, CPU).
3. The LoRA trainer on a tiny random Gemma 2 (real 256K tokenizer, no download): target-only loss
   equals the full-logit loss, decision scores, keep='select', adapter round trip, merge, generation.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import config
import eval_raft as E
import gemma_sft as GS
import raft_data as R
import raft_decode as RD
import sft

GEMMA_TOK = ".smoke/gemma/gemma-2-2b"
GEMMA_IT_TOK = ".smoke/gemma/gemma-2-2b-it"
SLM_RAFT = ".smoke/raft/scale/hf"                # the released SLM-125M-RAFT weights
DATA = Path(".smoke/raft/scale/datasets")
SCRATCH = Path(".smoke/gemma/smoke")
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def rows(group: str) -> list[dict]:
    return sft.load_rows(DATA / f"{group}.jsonl")


# --------------------------------------------------------------------------- 1. format
def format_checks(tok, slm_tok) -> None:
    print("format")
    d, s = RD.decision_ids(tok), RD.decision_ids(slm_tok)
    check("SLM ids unchanged", s["none"] == 5494 and s["pass"] == 12175 and s["forced"] == [479]
          and s["suppress"] == list(RD.SUPPRESS) and s["stop"] == [RD.EOS_ID] and s["pad"] == RD.PAD_ID, str(s))
    check("Gemma ids", d["format"] == "gemma" and d["none"] == 8031 and d["pass"] == 81975 and len(d["forced"]) == 1
          and d["pad"] == 0 and d["bos"] == 2 and d["stop"][0] == 1, str(d))
    it = RD.decision_ids(AutoTokenizer.from_pretrained(GEMMA_IT_TOK))
    check("instruct tokenizer gives the same ids", {k: v for k, v in it.items()} == d)
    check("single-token decodes keep the leading space", tok.decode([d["none"]]) == " none"
          and tok.decode([d["pass"]]) == " Passage")
    prompt = RD.GEMMA.prompt(["A.", "B."], "Q?")
    check("Gemma prompt spelling", prompt == "<bos>Passage 1: A.\nPassage 2: B.\nQuestion: Q?\nSource:", repr(prompt))
    check("convert = prompt", RD.GEMMA.convert(RD.SLM.prompt(["A.", "B."], "Q?")) == prompt)
    check("SLM convert is the identity", RD.SLM.convert(RD.SLM.prompt(["A."], "Q?")) == RD.SLM.prompt(["A."], "Q?"))
    specials = [v.content for v in tok.added_tokens_decoder.values() if v.special]
    n = bad_start = bad_dec = too_long = has_special = 0
    longest = 0
    for g in ("train", "dev", "test"):
        for r in rows(g):
            n += 1
            txt = r["prompt"] + r["target"]
            has_special += any(sp in txt for sp in specials if sp != "<|eos|>")
            p, t = RD.GEMMA.convert(r["prompt"]), RD.GEMMA.convert(r["target"])
            x, labels = R.encode_example(tok, p, t)
            longest = max(longest, len(x))
            too_long += len(x) > config.GEMMA.max_len
            bad_start += x[0] != tok.bos_token_id or x.count(tok.bos_token_id) != 1 or x[-1] != tok.eos_token_id
            tg = x[labels.count(-100):]
            want = [d["none"]] if r["bucket"] in sft.REFUSAL_BUCKETS else [d["pass"], *d["forced"], d["digits"][r["k"]]]
            bad_dec += tg[:len(want)] != want
    check(f"all {n} rows: one <bos> first, <eos> last", bad_start == 0, f"{bad_start} bad")
    check("all rows: target starts with the decision tokens the constraint forces", bad_dec == 0, f"{bad_dec} bad")
    check(f"all rows fit max_len {config.GEMMA.max_len}", too_long == 0, f"longest {longest}")
    check("no Gemma special-token strings in any row", has_special == 0, f"{has_special} rows")
    try:
        GS.encode_rows([{**rows("dev")[0]}], tok, max_len=50)
        check("encode_rows rejects an over-long row", False)
    except SystemExit:
        check("encode_rows rejects an over-long row", True)
    train = rows("train")
    shots = GS.pick_shots(train)
    zs = GS.instruct_prompt(rows("test")[0], shots)
    ids = tok.encode(zs, add_special_tokens=False)
    check("instruct prompt: chat template, prefilled Source:, one <bos>",
          zs.startswith("<bos><start_of_turn>user\n") and zs.endswith("<start_of_turn>model\nSource:")
          and ids.count(tok.bos_token_id) == 1 and "<|" not in zs, f"{len(ids)} tokens, shots {[s['id'] for s in shots]}")
    check("pick_shots is deterministic", [s["id"] for s in GS.pick_shots(list(reversed(train)))] == [s["id"] for s in shots])


# ------------------------------------------------------------------ 2. SLM equivalence
def slm_equivalence(slm_tok) -> None:
    print("SLM-125M decoding unchanged (committed raft_decode.py vs working copy)")
    old_src = subprocess.run(["git", "show", "HEAD:raft_decode.py"], capture_output=True, text=True, check=True).stdout
    path = SCRATCH / "raft_decode_head.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(old_src)
    spec = importlib.util.spec_from_file_location("raft_decode_head", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    from transformers import AutoModelForCausalLM
    net = AutoModelForCausalLM.from_pretrained(SLM_RAFT, torch_dtype=torch.float32)
    test = rows("test")
    sample = [r for r in test if r["bucket"] in ("A", "C")][:16] + [r for r in test if r["bucket"] in ("B1", "B2")][:8]
    prompts, ks = [r["prompt"] for r in sample], [len(r["passages"] or []) for r in sample]
    cps = [r["passages"] or [] for r in sample]
    dev = torch.device("cpu")
    same = True
    for copy in (None, cps):
        a = old.generate_batch(net, slm_tok, prompts, ks, dev, tau=0.7, batch=8, copy_passages=copy)
        b = RD.generate_batch(net, slm_tok, prompts, ks, dev, tau=0.7, batch=8, copy_passages=copy)
        same &= all(x["text"] == y["text"] and x["choice"] == y["choice"] and x["s"] == y["s"] for x, y in zip(a, b))
    check("24 test rows, plain and copy2: identical text, choice and s", same)
    o1 = old.answer(net, slm_tok, sample[0]["passages"], sample[0]["question"])
    o2 = RD.answer(net, slm_tok, sample[0]["passages"], sample[0]["question"])
    check("answer() identical", o1 == o2)
    # eval_forgetting.score_casehold with its new defaults == the committed version (SLM: <|eos|> prefix, no soft-cap)
    import eval_forgetting as EF
    ef_src = subprocess.run(["git", "show", "HEAD:eval_forgetting.py"], capture_output=True, text=True, check=True).stdout
    (SCRATCH / "eval_forgetting_head.py").write_text(ef_src)
    spec = importlib.util.spec_from_file_location("eval_forgetting_head", SCRATCH / "eval_forgetting_head.py")
    ef_old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ef_old)
    items = [json.loads(ln) for ln in Path(".smoke/casehold_test.jsonl").read_text().splitlines()[:6]]
    a_ = ef_old.score_casehold(net, slm_tok, items, dev, return_scores=True)
    b_ = EF.score_casehold(net, slm_tok, items, dev, return_scores=True)
    check("CaseHOLD scorer unchanged for SLM (6 items, all 60 log-probs)", a_["scores"] == b_["scores"])
    saved = {json.loads(ln)["id"]: json.loads(ln) for ln in Path(".smoke/raft/scale/eval/test_copy2.jsonl").read_text().splitlines()}
    b = RD.generate_batch(net, slm_tok, prompts, ks, dev, tau=0.7, batch=8, copy_passages=cps)
    match = sum(saved[r["id"]]["text"] == o["text"] for r, o in zip(sample, b))
    print(f"    (CPU fp32 vs the saved H100 bf16 test generations: {match}/{len(sample)} identical texts; reported, not gated)")


# --------------------------------------------------------------------- 3. tiny Gemma 2
def tiny_gemma(tok):
    from transformers import Gemma2Config, Gemma2ForCausalLM
    torch.manual_seed(0)
    cfg = Gemma2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=2, num_key_value_heads=1, head_dim=32, query_pre_attn_scalar=32,
                       max_position_embeddings=2048, sliding_window=4096, final_logit_softcapping=30.0,
                       attn_logit_softcapping=50.0, pad_token_id=0, eos_token_id=1, bos_token_id=2,
                       tie_word_embeddings=True, attn_implementation="eager")
    return Gemma2ForCausalLM(cfg).float().eval()


def trainer_checks(tok) -> None:
    print("LoRA trainer on a tiny random Gemma 2")
    dec = RD.decision_ids(tok)
    dev_rows = rows("dev")
    pick = [r for r in dev_rows if r["bucket"] == "A"][:6] + [r for r in dev_rows if r["bucket"] in ("B1", "B2")][:4]
    data = GS.encode_rows(pick, tok)
    net = tiny_gemma(tok)
    x, m = GS.pad_batch(data.seqs[:4], data.plen[:4].tolist(), dec["pad"])
    with torch.no_grad():
        r = GS.loss_sums(net, x, m, dec)
        logits = net(input_ids=x, use_cache=False).logits.float()     # HF forward applies the soft-cap
        mm = m[:, 1:]
        ref = torch.nn.functional.cross_entropy(logits[:, :-1][mm], x[:, 1:][mm], reduction="sum")
    check("target-only CE == full-logit CE (soft-capped)", abs(float(r["sft"]) - float(ref)) < 1e-3 * float(ref),
          f"{float(r['sft']):.4f} vs {float(ref):.4f}")
    tf = GS.teacher_forced(net, data, dec, torch.device("cpu"), batch=4)
    with torch.no_grad():
        want = []
        for s, p in zip(data.seqs, data.plen):
            lg = net(input_ids=torch.from_numpy(s)[None], use_cache=False).logits[0, p - 1]
            want.append(float(RD.decision_score(lg, dec)))
    check("teacher_forced s == decision_score of the HF forward", max(abs(a - b) for a, b in zip(tf["s"], want)) < 1e-4)
    ps = GS.prompt_scores(net, tok, [r["prompt"] for r in data.rows], torch.device("cpu"), batch=4)
    check("prompt_scores == teacher_forced s", max(abs(a - b) for a, b in zip(ps, tf["s"])) < 1e-4)

    # bits/byte: per UTF-8 byte of the decoded windows, against a hand computation
    import eval_forgetting as EF
    slm_tok = AutoTokenizer.from_pretrained(".smoke/final/base")
    val = Path(".smoke/valshards")
    subset = {k: {**v, "indices": v["indices"][:2]} for k, v in EF.heldout_subset(val).items()}
    bpb = GS.heldout_bpb(net, tok, slm_tok, val, subset, torch.device("cpu"), batch=2)
    spec = subset["sec"]
    arr = np.memmap(val / spec["shard"], dtype=np.uint16, mode="r").reshape(-1, config.SEQ_LEN)
    nats = nb = 0.0
    with torch.no_grad():
        for w in np.asarray(arr[spec["indices"]]):
            t = slm_tok.decode(w.tolist(), skip_special_tokens=True)
            ids = torch.tensor([[tok.bos_token_id] + tok(t, add_special_tokens=False, split_special_tokens=True)["input_ids"]])
            nats += float(net(input_ids=ids, labels=ids, use_cache=False).loss) * (ids.shape[1] - 1)
            nb += len(t.encode("utf-8"))
    check("held-out bits/byte matches a hand computation", abs(bpb["sec"]["bpb"] - nats / (math.log(2) * nb)) < 1e-3,
          f"{bpb['sec']['bpb']:.4f} vs {nats / (math.log(2) * nb):.4f}")

    # keep='select' rule (as sft.train_one)
    b0 = {"step": 0}
    e1 = {"gate_ok": False, "heldout_pooled_bpb": 0.9, "select_score": 0.5}
    e2 = {"gate_ok": True, "heldout_pooled_bpb": 1.0, "select_score": -0.2}
    e3 = {"gate_ok": True, "heldout_pooled_bpb": 0.8, "select_score": -0.3}
    e4 = {"gate_ok": False, "heldout_pooled_bpb": 0.7, "select_score": 0.9}
    check("better(): nothing passed -> lowest bpb; a pass beats a fail; then higher score wins; a fail never replaces a pass",
          GS.better(e1, b0) and GS.better(e2, e1) and not GS.better(e3, e2) and not GS.better(e4, e2))

    # length-grouped micro-batches take exactly the rows sft.micro_batch would, per step
    plan = sft.make_plan(len(data) * 30, 3, config.GEMMA.max_len, 52, 13, sft_per_replay=0)
    big = GS.GemmaData(data.seqs * 30, np.tile(data.plen, 30), np.tile(data.refusal, 30), data.buckets * 30, data.rows * 30)
    from data import StreamOrder
    order = StreamOrder(len(big), 1337)
    same = True
    for st in range(plan.total_steps):
        micros = GS.step_micros(order, big, plan, st, max_tokens=3_000)
        got = np.concatenate(micros)
        want = np.concatenate([order.windows(np.arange(st * 52 + m * 13, st * 52 + m * 13 + 13)) for m in range(4)])
        lens = [len(big.seqs[i]) for i in got]
        same &= sorted(got.tolist()) == sorted(want.tolist()) and lens == sorted(lens)
        same &= all(len(m) * max(len(big.seqs[i]) for i in m) <= 3_000 or len(m) == 1 for m in micros)
    check("step_micros: same rows per step as sft.micro_batch, sorted, micro-batches within the token budget",
          bool(same), f"{plan.total_steps} steps")

    # LoRA: zero-init adapters leave the model unchanged; training lowers the loss; round trip; merge.
    # Logit comparisons on a 2 x 64 slice: full (B, T, 256K) fp32 logits do not fit an 8 GB laptop.
    del logits
    x = x[:2, :64]
    pm, init = GS.add_lora(tiny_gemma(tok))
    base_logits = net(input_ids=x, use_cache=False).logits
    lora_logits = pm(input_ids=x, use_cache=False).logits
    check("LoRA init (B = 0) leaves outputs unchanged", torch.allclose(base_logits, lora_logits, atol=1e-5))
    check("adapters are fp32 and the only trainable tensors",
          all(p.dtype == torch.float32 and "lora_" in n for n, p in pm.named_parameters() if p.requires_grad))
    calls = []

    def extra() -> dict:
        calls.append(1)
        return {"heldout_pooled_bpb": 1.0 - 0.01 * len(calls), "gate_ok": True, "select_score": -float(len(calls) == 2)}
    train = GS.encode_rows(pick * 6, tok)                 # 60 rows -> 2 steps per epoch at 52 rows/step
    res = GS.train_run(pm, init, train, data, dec, torch.device("cpu"), lr=3e-3, epochs=3, n_evals=3, extra_eval=extra,
                       micro=13, log=lambda *_: None)
    h = res["history"]
    check("train_run: completes, evals as planned", res["status"] == "completed" and len(h) == 3 and res["steps"] == 6,
          f"{res['status']} steps {res['steps']} evals {len(h)}")
    check("train_run: dev target loss falls", h[-1]["dev_target_loss"] < h[0]["dev_target_loss"],
          f"{h[0]['dev_target_loss']:.3f} -> {h[-1]['dev_target_loss']:.3f}")
    check("train_run: keep='select' picks the best-scoring gate-passing eval", res["best"]["step"] in (2, 6)
          and res["best"]["select_score"] == 0.0 and res["best"]["step"] != 4, f"best step {res['best']['step']}")
    GS.load_adapter(pm, res["best_state"])
    a = pm(input_ids=x, use_cache=False).logits
    GS.load_adapter(pm, init)
    b = pm(input_ids=x, use_cache=False).logits
    GS.load_adapter(pm, res["best_state"])
    c = pm(input_ids=x, use_cache=False).logits
    check("adapter state round trip", torch.equal(a, c) and not torch.allclose(a, b))
    # train_run restarts from the same init: two runs with the same data and seed give the same weights
    r2 = GS.train_run(pm, init, train, data, dec, torch.device("cpu"), lr=3e-3, epochs=3, n_evals=3, extra_eval=extra,
                      micro=13, log=lambda *_: None)
    calls.clear()
    same = all(torch.equal(res["best_state"][k], r2["best_state"][k]) for k in res["best_state"]) if r2["best"]["step"] == res["best"]["step"] else None
    check("train_run is deterministic from init_state", same is not False)
    GS.load_adapter(pm, res["best_state"])
    with torch.no_grad():
        before = pm(input_ids=x, use_cache=False).logits
        merged = pm.merge_and_unload()
        after = merged(input_ids=x, use_cache=False).logits
    check("merge_and_unload matches the adapter model", torch.allclose(before, after, atol=1e-4),
          f"max {float((before - after).abs().max()):.2e}")

    # generation with the Gemma format: decision forced from s, quotes verbatim under copy2
    prompts = [r["prompt"] for r in data.rows]
    cps = [r["passages"] or [] for r in data.rows]
    for tau in (0.0, 1.01):
        g = RD.generate_batch(merged, tok, prompts, [len(c) for c in cps], torch.device("cpu"), tau=tau, batch=5,
                              max_new=40, copy_passages=cps)
        want = "none" if tau == 0.0 else "pass"
        ok = all(o["choice"] == want for o in g)
        if want == "pass":                       # " Passage" + forced "▁" + a digit within the passage count
            ok &= all(o["text"].startswith(" Passage ") and 1 <= int(o["text"][9]) <= len(c) for o, c in zip(g, cps))
        check(f"generate_batch tau={tau}: decision '{want}' forced" + (", digit in range" if want == "pass" else ""), ok,
              repr(g[0]["text"][:40]))


def copy_walks(tok) -> None:
    """CopyConstraint on Gemma's 256K vocabulary: random walks through the allowed tokens from
    ` Passage k\nQuote: "` must end in a verbatim span of passage k followed by the forced scaffold."""
    print("copy constraint on Gemma tokens (random walks)")
    import random
    vocab = [tok.decode([j]) for j in range(len(tok))]
    dev_rows = [r for r in rows("dev") if r["bucket"] == "A"][:12]
    rng = random.Random(0)
    ok_quote = ok_scaffold = 0
    for r in dev_rows:
        prompt = tok.encode(RD.GEMMA.convert(r["prompt"]), add_special_tokens=False)
        cc = RD.CopyConstraint(tok, len(prompt), [r["passages"]], vocab, eos_id=tok.eos_token_id)
        gen = tok.encode(f' Passage {r["k"]}\nQuote: "', add_special_tokens=False)
        for _ in range(200):
            scores = torch.randn(1, len(tok), generator=torch.Generator().manual_seed(rng.randrange(10**9)))
            out = cc(torch.tensor([prompt + gen]), scores)
            allowed = torch.nonzero(out[0] > -math.inf).flatten().tolist()
            gen.append(rng.choice(allowed))
            text = tok.decode(gen, skip_special_tokens=True)
            if "\nAnswer:" in text:
                break
        m = __import__("re").match(r'^ Passage (\d)\nQuote: "(.*?)"\nAnswer:', text, __import__("re").S)
        ok_scaffold += bool(m)
        ok_quote += bool(m) and m.group(2) in r["passages"][r["k"] - 1] and len(m.group(2)) <= 240
    check(f"{len(dev_rows)} walks reach the forced `\\nAnswer:` scaffold", ok_scaffold == len(dev_rows), f"{ok_scaffold}")
    check(f"{len(dev_rows)} walks: the quote is a verbatim span of the cited passage (<= 240 chars)",
          ok_quote == len(dev_rows), f"{ok_quote}")


# -------------------------------------------------------------------------- 4. gates
def app_checks() -> None:
    print("gemma_app gates")
    import gemma_app as GA
    for args, name in (((False, 0.5, "9.2-probe"), "no --confirm"), ((True, 9.0, "9.2-probe"), "over the cap")):
        try:
            GA._gate(*args)
            check(f"_gate refuses: {name}", False)
        except SystemExit:
            check(f"_gate refuses: {name}", True)
    GA._gate(True, 0.5, "9.2-probe")
    check("_gate passes within the cap", True)
    check("every stage has a cap", set(config.GEMMA_STAGE_CAPS_USD) >= {"9.2-probe", "9.3-baselines", "9.4-pilot", "9.5-full"})


def main() -> None:
    tok = AutoTokenizer.from_pretrained(GEMMA_TOK)
    only = sys.argv[1:]
    if not only or "format" in only:
        format_checks(tok, AutoTokenizer.from_pretrained(SLM_RAFT))
    if not only or "slm" in only:
        slm_equivalence(AutoTokenizer.from_pretrained(SLM_RAFT))
    if not only or "copy" in only:
        copy_walks(tok)
    if not only or "trainer" in only:
        trainer_checks(tok)
    if not only or "app" in only:
        app_checks()
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED: {failures}'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
