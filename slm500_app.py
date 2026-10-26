"""Phase 10 jobs on Modal: RAFT on thesreedath/slm-500m-base (docs/11-slm500m-raft.md §3, §6).

    modal run slm500_app.py::fetch_check                                       # pinned weights -> Volume (CPU, cents)
    modal run slm500_app.py::replay_check --model base500                      # 10.3 replay windows (CPU, one worker per shard)
    modal run --detach slm500_app.py::probe_check --confirm --max-usd 0.8      # 10.4, 1 x H100
    modal run --detach slm500_app.py::pilot_check --confirm --max-usd 3.0      # 10.5, 1 x H100
    modal run --detach slm500_app.py::full_check --confirm --lr <10.5 pick> --max-usd 4.5   # 10.6, 1 x H100
    modal run --detach slm500_app.py::control_check --confirm --max-usd 1.5    # 10.7, 1 x H100 (author's 125M)

The RAFT recipe is SLM-125M's (raft_gpu_app.full_run, D-23/D-27): full fine-tuning, fp32 master weights +
bf16 autocast, torch.compile, replay lambda 1.0, 35% refusal, 3 epochs, keep='select' (best dev copy2 proxy
among evals with held-out bits/byte <= +0.02), the pre-registered tau rule, copy2. The models share our
special-token layout, so sft.py / eval_raft.py / raft_decode.py run unchanged. What differs:
  - rows, replay windows and CaseHOLD use the model's own tokenizer; replay = OUR train-corpus documents
    (the same documents as its corpus) re-tokenized and packed doc + <|eos|> into 1,024-token windows;
  - the bits/byte gate is tokenizer-free (eval_forgetting.heldout_bpb_text), relative to the model's own base.
Every GPU function projects spend from its wall clock (+60 s) and stops before a run or pass that would
cross --max-usd. Results are appended to the Volume per run, so a restarted container resumes.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import modal

import config

G = config.SLM500
app = modal.App(f"{config.PROJECT}-slm500")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("gcc")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "numpy==2.1.3", "safetensors==0.4.5")
    .add_local_python_source("config", "model", "data", "train", "cleaning", "raft_data", "raft_decode",
                             "eval_raft", "eval_forgetting", "sft")
)
cpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("transformers==4.46.3", "numpy==2.1.3", "huggingface_hub")
             .add_local_python_source("config", "model", "cleaning", "raft_data", "slm500_app"))
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
SECRETS = [modal.Secret.from_name(config.HF_SECRET_NAME)]
ROOT = Path(config.SLM500_DIR)
MODELS = {"base500": (G.base, G.base_revision), "control125": (G.control, G.control_revision)}
SCALE_DIR = Path(config.RAFT_DIR) / "datasets" / "scale"
CASEHOLD = Path(config.RAFT_DIR) / "eval" / "casehold_test.jsonl"
LOCAL = Path(".smoke/slm500m")


def model_dir(key: str) -> Path:
    return ROOT / "models" / key


def replay_dir(key: str) -> Path:
    return ROOT / "replay" / key


def _rate() -> float:
    return config.GPU_RATE_PER_SEC[G.gpu] * config.COST_OVERHEAD_FACTOR


def _clock(log_path: Path):
    t0 = time.time()
    spent = lambda: (time.time() - t0 + 60.0) * _rate()

    def log(m: str) -> None:
        line = f"[{time.time() - t0:7.0f}s ${spent():.2f}] {m}"
        print(line, flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")
    return spent, log


# ------------------------------------------------------------------------- pure core
def replay_sample(text: str, rate: float) -> bool:
    """Deterministic document sample for the replay store (its own salt, independent of the RAFT pools)."""
    import raft_data as R
    return R._hash_mod(text + "|replay500", 1_000_000) < int(rate * 1_000_000)


def pack_windows(docs, tok, seq_len: int = config.SEQ_LEN):
    """doc + <|eos|> concatenated and cut into seq_len windows (the pretraining format); remainder dropped."""
    import numpy as np
    eos = tok.convert_tokens_to_ids("<|eos|>")
    buf: list[int] = []
    out = []
    for d in docs:
        buf += tok.encode(d, add_special_tokens=False) + [eos]
        while len(buf) >= seq_len:
            out.append(buf[:seq_len])
            buf = buf[seq_len:]
    if max(len(tok), 1) > 65_536:
        raise ValueError("vocabulary does not fit uint16")
    return np.asarray(out, dtype=np.uint16).reshape(-1, seq_len)


# ------------------------------------------------------------------------------ CPU jobs
@app.function(image=cpu_image, volumes=VOLUMES, secrets=SECRETS, cpu=4.0, memory=16_384, timeout=60 * 30)
def fetch() -> dict:
    """Pinned weights of the 500M and the author's 125M onto the Volume. Idempotent."""
    import os
    from huggingface_hub import snapshot_download
    out = {}
    for key, (repo, rev) in MODELS.items():
        d = model_dir(key)
        if not (d / "model.safetensors").exists():
            snapshot_download(repo, revision=rev, local_dir=str(d), token=os.environ["HF_TOKEN"],
                              allow_patterns=["*.json", "*.safetensors"])
        out[key] = {p.name: p.stat().st_size for p in sorted(d.iterdir()) if p.is_file()}
    (ROOT / "models" / "manifest.json").write_text(json.dumps({k: {"repo": r, "revision": v} for k, (r, v) in MODELS.items()}, indent=2))
    volume.commit()
    return out


@app.local_entrypoint()
def fetch_check():
    print(json.dumps(fetch.remote(), indent=2))


@app.function(image=cpu_image, volumes=VOLUMES, cpu=2.0, memory=8_192, timeout=60 * 30)
def replay_build(shard: str, key: str, rate: float) -> dict:
    """10.3, one worker per corpus shard: sampled TRAIN documents (never a val document) -> packed windows."""
    from transformers import AutoTokenizer
    import raft_data as R
    tok = AutoTokenizer.from_pretrained(model_dir(key))
    docs, n_docs = [], 0
    with open(shard, encoding="utf-8") as fh:
        for line in fh:
            text = line.rstrip("\n")
            if text and not R.is_val_doc(text) and replay_sample(text, rate):
                docs.append(text)
    n_docs = len(docs)
    win = pack_windows(docs, tok)
    src, stem = Path(shard).parent.name, Path(shard).stem
    out = replay_dir(key) / f"{src}-{stem}.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    win.tofile(out)
    volume.commit()
    return {"shard": f"{src}/{stem}", "docs": n_docs, "windows": int(win.shape[0])}


@app.local_entrypoint()
def replay_check(model: str = "base500"):
    """10.3 (Modal CPU, cents): replay store for `model`, fanned out one worker per corpus shard."""
    if model not in MODELS:
        raise SystemExit(f"--model must be one of {list(MODELS)}")
    shards = list(_list_shards.remote())                                # corpus shards on the Volume
    rate = G.replay_tokens / 2_040_088_576                              # config: our train tokens (index.json)
    res = list(replay_build.starmap([(s, model, rate) for s in shards]))
    tot = sum(r["windows"] for r in res)
    print(json.dumps(res, indent=1))
    print(f"{model}: {tot} windows = {tot * config.SEQ_LEN / 1e6:.1f}M tokens from {sum(r['docs'] for r in res)} train docs")


@app.function(image=cpu_image, volumes=VOLUMES, cpu=1.0, timeout=120)
def _list_shards() -> list[str]:
    import glob
    return sorted(glob.glob(f"{config.CORPUS_DIR}/*/*.txt"))


# ------------------------------------------------------------------------------ GPU jobs
def _load(key: str, device):
    """Tokenizer, decision ids, fp32 model (+ base state), rows, dev data, replay store, guard inputs."""
    import torch
    from transformers import AutoTokenizer

    import eval_forgetting as EF
    import eval_raft as E
    import sft
    from data import WindowStore
    tok = AutoTokenizer.from_pretrained(model_dir(key))
    dec = E.decision_ids(tok)
    if dec["format"] != "slm":
        raise SystemExit(f"{key}: expected the SLM special-token layout, got {dec['format']}")
    raw = sft.load_init(str(model_dir(key))).to(device)
    local = Path(f"/tmp/replay/{key}")
    shutil.copytree(replay_dir(key), local, dirs_exist_ok=True)
    store = WindowStore.open(str(local), config.SEQ_LEN)
    pool = sft.load_rows(SCALE_DIR / "train.jsonl")
    dev = sft.tokenize(sft.load_rows(SCALE_DIR / "dev.jsonl"), tok, config.SEQ_LEN)
    src_tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    subset = EF.heldout_subset(Path(config.VAL_TOKENS_DIR))
    items = [json.loads(ln) for ln in CASEHOLD.read_text().splitlines() if ln.strip()]
    return {"tok": tok, "dec": dec, "raw": raw, "store": store, "pool": pool, "dev": dev, "src_tok": src_tok,
            "subset": subset, "items": items, "base_state": {k: v.detach().clone() for k, v in raw.state_dict().items()}}


def _guards(raw, S, device, ac, casehold: bool = True) -> dict:
    import eval_forgetting as EF
    g = {"heldout": EF.heldout_bpb_text(raw, S["tok"], S["src_tok"], Path(config.VAL_TOKENS_DIR), S["subset"], device, ac,
                                        batch=8)}
    if casehold:
        g["casehold"] = EF.score_casehold(raw, S["tok"], S["items"], device, ac, items_per_batch=16)
    return g


def _gens(rows, out) -> list[dict]:
    return [{"id": r["id"], "bucket": r["bucket"], "s": o["s"], "choice": o["choice"], "text": o["text"]}
            for r, o in zip(rows, out)]


CLOSED_BOOK_SHOTS = 3


def closed_book_prompt(shots: list[dict], question: str) -> str:
    """Few-shot closed-book Q/A for a BASE model (no passages): knowledge of the test facts, not reading."""
    ex = "".join(f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots)
    return f"<|eos|>{ex}Question: {question}\nAnswer:"


@app.function(image=image, gpu=G.gpu, volumes=VOLUMES, cpu=8.0, memory=65_536, timeout=60 * 25)
def probe(max_usd: float) -> dict:
    """10.4: step time and memory (micro 32 and 16, compiled), eval timings, the base guards, and a few-shot
    closed-book probe of all three BASE models on the 500 test A questions (overlap at the knowledge level)."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import eval_raft as E
    import sft
    from train import autocast
    spent, log = _clock(ROOT / "probe" / "log.txt")
    device = torch.device("cuda")
    ac = lambda: autocast(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    S = _load("base500", device)
    raw, dec = S["raw"], S["dec"]
    rep: dict = {"replay_windows": len(S["store"])}
    sft.init_chat_tokens(raw, config.SFT.seed)
    train = sft.tokenize(sft.select_mix(S["pool"], 2_000, config.SFT.refusal_share, config.SFT.seed), S["tok"], config.SEQ_LEN)
    net = torch.compile(raw)
    for micro in (32, 16):                         # 16 only if 32 runs out of memory
        if spent() > max_usd * 0.6 or rep.get("micro32", {}).get("s_per_step"):
            break
        raw.load_state_dict(S["base_state"])
        torch.cuda.reset_peak_memory_stats()
        cfg = sft.RunConfig(epochs=1, max_steps=8, micro=micro, log_every=1, keep="final", gpu=G.gpu)
        t = time.time()
        try:
            sft.train_one(cfg, raw, net, train, None, S["store"], dec, device, None, log=lambda *_: None)
            torch.cuda.synchronize()
            rep[f"micro{micro}"] = {"s_8_steps_incl_compile": time.time() - t, "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
            cfg = sft.RunConfig(epochs=1, max_steps=6, micro=micro, log_every=1, keep="final", gpu=G.gpu)
            t = time.time()
            sft.train_one(cfg, raw, net, train, None, S["store"], dec, device, None, log=lambda *_: None)
            torch.cuda.synchronize()
            rep[f"micro{micro}"]["s_per_step"] = (time.time() - t) / 6
        except torch.cuda.OutOfMemoryError:
            rep[f"micro{micro}"] = {"oom": True}
            torch.cuda.empty_cache()
        log(f"micro {micro}: {rep[f'micro{micro}']}")
    raw.load_state_dict(S["base_state"])
    dev = S["dev"]
    t = time.time()
    tf = E.teacher_forced(raw, [r[:m].tolist() for r, m in zip(dev.ids, dev.length)], dev.decision_pos.tolist(),
                          list(dev.mask), device, dec, 16, ac)
    rep["teacher_forced_dev_s"] = time.time() - t
    cps = [r["passages"] or [] for r in dev.rows[:64]]
    t = time.time()
    g = E.generate_batch(raw, S["tok"], [r["prompt"] for r in dev.rows[:64]], [len(c) for c in cps], device, tau=0.5,
                         batch=64, autocast=ac, copy_passages=cps)
    rep["gen64_copy2_s"] = time.time() - t
    import eval_forgetting as EF
    t = time.time()
    rep["base_bpb"] = _guards(raw, S, device, ac, casehold=False)["heldout"]["pooled_bpb"]
    rep["heldout_bpb_s"] = time.time() - t
    t = time.time()                                # timing only; the full base CaseHOLD runs in 10.5
    rep["casehold200_pmi"] = EF.score_casehold(raw, S["tok"], S["items"][:200], device, ac, items_per_batch=16)["acc_pmi"]
    rep["casehold200_s"] = time.time() - t
    log(f"timings {rep}")
    # closed-book probe of the three base models (knowledge of the test facts without passages)
    test_a = [r for r in sft.load_rows(SCALE_DIR / "test.jsonl") if r["bucket"] == "A"]
    shots = [r for r in sft.load_rows(SCALE_DIR / "train.jsonl") if r["bucket"] == "A" and len(r["answer"]) < 40][:CLOSED_BOOK_SHOTS]
    del net, raw
    torch.cuda.empty_cache()
    cb = {}
    for key, path in (("ours125", config.BASE_CKPT_DIR), ("control125", str(model_dir("control125"))),
                      ("base500", str(model_dir("base500")))):
        if spent() > max_usd * 0.9:
            log(f"budget: stopping before the closed-book probe of {key}")
            break
        tok = AutoTokenizer.from_pretrained(path)
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to(device).eval()
        eos, nl = tok.convert_tokens_to_ids("<|eos|>"), tok.encode("\n", add_special_tokens=False)
        out = []
        for i in range(0, len(test_a), 32):
            part = test_a[i:i + 32]
            enc = [tok.encode(closed_book_prompt(shots, r["question"]), add_special_tokens=False) for r in part]
            w = max(len(e) for e in enc)
            x = torch.full((len(enc), w), 2, dtype=torch.long)
            mask = torch.zeros_like(x)
            for j, e in enumerate(enc):
                x[j, w - len(e):] = torch.tensor(e)
                mask[j, w - len(e):] = 1
            with torch.no_grad(), ac():
                y = m.generate(input_ids=x.to(device), attention_mask=mask.to(device), do_sample=False, max_new_tokens=32,
                               eos_token_id=[eos] + nl[-1:], pad_token_id=2, suppress_tokens=[0, 2, 3, 4, 5, 6])
            for r, row in zip(part, y[:, w:].tolist()):
                out.append({"id": r["id"], "question": r["question"],
                            "answer": tok.decode(row, skip_special_tokens=True).split("\n")[0].strip()})
        cb[key] = out
        (ROOT / "probe" / f"closed_book_{key}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))
        volume.commit()
        del m
        torch.cuda.empty_cache()
        log(f"closed-book {key}: {len(out)} answers")
    rep["closed_book_shots"] = [s["id"] for s in shots]
    rep["spent_usd"] = spent()
    (ROOT / "probe" / "probe.json").write_text(json.dumps(rep, indent=2))
    volume.commit()
    return {**rep, "closed_book": cb}


@app.local_entrypoint()
def probe_check(confirm: bool = False, max_usd: float = 0.8):
    """10.4 (paid: 1 x H100). Requires --confirm."""
    _gate(confirm, max_usd, "10.4-probe")
    rep = probe.remote(max_usd)
    d = LOCAL / "probe"
    d.mkdir(parents=True, exist_ok=True)
    for key, out in rep.pop("closed_book").items():
        (d / f"closed_book_{key}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))
    (d / "probe.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


def _runs(max_usd: float, key: str, which: str, runs: list[dict], test: bool) -> dict:
    """raft_gpu_app.full_run for another base: each run from the same base weights, keep='select', tau, dev
    generations; with `test`, the test pass on the best run (each decoding mode saved as it finishes),
    teacher-forced test decisions, guards and the HF export."""
    import torch

    import eval_raft as E
    import sft
    from train import autocast
    if not (G.probe_step_s and G.probe_eval_s):
        raise SystemExit("run the 10.4 probe first and record probe_step_s / probe_eval_s in config.SLM500")
    run_dir = ROOT / "runs" / key / which
    run_dir.mkdir(parents=True, exist_ok=True)
    spent, log = _clock(run_dir / "log.txt")
    device = torch.device("cuda")
    ac = lambda: autocast(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    S = _load(key, device)
    raw, dec, tok, dev = S["raw"], S["dec"], S["tok"], S["dev"]
    gpath = ROOT / "runs" / key / "guards_base.json"
    if gpath.exists():
        base = json.loads(gpath.read_text())
    else:
        base = _guards(raw, S, device, ac)
        gpath.write_text(json.dumps(base, default=str))
        volume.commit()
    base_bpb = base["heldout"]["pooled_bpb"]
    log(f"staged {key}: pool {len(S['pool'])}, dev {len(dev)}, replay {len(S['store'])} windows; base bpb {base_bpb:.4f}, "
        f"CaseHOLD PMI {base['casehold']['acc_pmi']:.4f}")
    prompts, ks = [r["prompt"] for r in dev.rows], [len(r["passages"] or []) for r in dev.rows]
    cps = [r["passages"] or [] for r in dev.rows]
    net = torch.compile(raw)

    def extra() -> dict:
        bpb = _guards(raw, S, device, ac, casehold=False)["heldout"]["pooled_bpb"]
        g = E.generate_batch(raw, tok, prompts, ks, device, tau=config.SFT.tau_pilot, batch=64, autocast=ac, copy_passages=cps)
        m = E.proxy_metrics(dev.rows, g)
        return {"heldout_pooled_bpb": bpb, "bpb_delta": bpb - base_bpb, "gate_ok": bpb - base_bpb <= config.RAFT.max_bpb_increase,
                "select_score": m["selection_score"], **{f"proxy_{k}": v for k, v in m.items()}}

    res_path = run_dir / "results.jsonl"
    done = {json.loads(ln)["key"] for ln in res_path.read_text().splitlines() if ln.strip()} if res_path.exists() else set()
    for run in runs:
        rk = f"n{run['n']}-lr{run['lr']:g}"
        if rk in done:
            continue
        plan = sft.make_plan(run["n"], config.SFT.epochs, config.SEQ_LEN, micro=G.micro_batch)
        est = (plan.total_steps * G.probe_step_s + (run["evals"] + 2) * G.probe_eval_s + 90) * _rate()
        if spent() + est > max_usd:
            log(f"budget: run {rk} (est ${est:.2f}) would cross --max-usd {max_usd}; stopping")
            break
        t1 = time.time()
        raw.load_state_dict(S["base_state"])
        sft.init_chat_tokens(raw, config.SFT.seed)
        train = sft.tokenize(sft.select_mix(S["pool"], run["n"], config.SFT.refusal_share, config.SFT.seed), tok, config.SEQ_LEN)
        cfg = sft.RunConfig(lr=run["lr"], epochs=config.SFT.epochs, gpu=G.gpu, max_usd=max_usd, log_every=10_000,
                            keep=config.SFT.keep, micro=G.micro_batch, eval_every=max(1, round(plan.total_steps / run["evals"])))
        log(f"run {rk}: {plan.total_steps} steps, eval every {cfg.eval_every}, micro {G.micro_batch}")
        res = sft.train_one(cfg, raw, net, train, dev, S["store"], dec, device, None, log=lambda *_: None, extra_eval=extra)
        raw.load_state_dict(res["best_state"])
        torch.save(res["best_state"], run_dir / f"{rk}.pt")
        tf = E.teacher_forced(raw, [r[:m].tolist() for r, m in zip(dev.ids, dev.length)], dev.decision_pos.tolist(),
                              list(dev.mask), device, dec, 16, ac)
        tau = E.choose_tau(E.tau_table(tf["s"], dev.refusal.tolist(), dev.buckets))
        gens, prox = {}, {}
        for mode in ("plain", "copy2"):
            g = E.generate_batch(raw, tok, prompts, ks, device, tau=tau["tau"], batch=64, autocast=ac,
                                 copy_passages=cps if mode == "copy2" else None)
            gens[mode], prox[mode] = _gens(dev.rows, g), E.proxy_metrics(dev.rows, g)
        best = res["best"]
        row = {"key": rk, **run, "steps": res["steps"], "selected_step": best["step"], "gate_ok": best.get("gate_ok"),
               "bpb_delta": best.get("bpb_delta"), "select_score": best.get("select_score"),
               "dev_macro_f1": best.get("dev_macro_f1"), "dev_auroc": best.get("dev_auroc"), "tau": tau, "proxies": prox,
               "history": [{k: h.get(k) for k in ("step", "dev_macro_f1", "dev_target_loss", "bpb_delta", "select_score", "gate_ok")}
                           for h in res["history"]], "gens": gens, "seconds": round(time.time() - t1, 1)}
        with res_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        volume.commit()
        log(f"{rk}: step {best['step']}/{res['steps']} gate {best.get('gate_ok')} bpb {best.get('bpb_delta', 0):+.4f} | tau {tau['tau']} | "
            f"copy2 score {prox['copy2']['selection_score']:+.3f} grounded {prox['copy2']['proxy_grounded_acc']:.3f} ({row['seconds']}s)")

    rows = [json.loads(ln) for ln in res_path.read_text().splitlines() if ln.strip()] if res_path.exists() else []
    out = {"status": "completed", "runs": rows, "base_guards": {"bpb": base_bpb, "casehold_pmi": base["casehold"]["acc_pmi"]}}
    if not test or not rows:
        return {**out, "spent_usd": spent()}
    top = max(rows, key=lambda r: r["proxies"]["copy2"]["selection_score"])
    final_path = run_dir / "final.json"
    if final_path.exists():
        return {**out, "final": json.loads(final_path.read_text()), "spent_usd": spent()}
    raw.load_state_dict(torch.load(run_dir / f"{top['key']}.pt", map_location=device, weights_only=False))
    test_rows = sft.load_rows(SCALE_DIR / "test.jsonl")
    tdata = sft.tokenize(test_rows, tok, config.SEQ_LEN)
    tcps = [r["passages"] or [] for r in test_rows]
    tgens = {}
    for mode in ("plain", "copy2"):
        part = run_dir / f"test_{mode}.jsonl"
        if part.exists():
            tgens[mode] = [json.loads(ln) for ln in part.read_text().splitlines() if ln.strip()]
            continue
        if spent() + 2 * G.probe_eval_s * _rate() > max_usd:
            log(f"budget: stopping before test {mode}")
            return {**out, "status": "budget_stop", "spent_usd": spent()}
        g = E.generate_batch(raw, tok, [r["prompt"] for r in test_rows], [len(c) for c in tcps], device, tau=top["tau"]["tau"],
                             batch=64, autocast=ac, copy_passages=tcps if mode == "copy2" else None)
        tgens[mode] = _gens(test_rows, g)
        part.write_text("".join(json.dumps(x) + "\n" for x in tgens[mode]))
        volume.commit()
        log(f"test {mode}: done")
    tft = E.teacher_forced(raw, [r[:m].tolist() for r, m in zip(tdata.ids, tdata.length)], tdata.decision_pos.tolist(),
                           list(tdata.mask), device, dec, 16, ac)
    guards = _guards(raw, S, device, ac)
    sft.export(raw, None, run_dir / "hf", str(model_dir(key)), tau=top["tau"]["tau"])
    final = {"key": top["key"], "tau": top["tau"]["tau"], "test_gens": tgens,
             "test_decision": E.decision_metrics(tft["s"], tdata.refusal.tolist(), tdata.buckets, top["tau"]["tau"]),
             "guards": guards, "bpb_delta_pooled": guards["heldout"]["pooled_bpb"] - base_bpb,
             "bpb_delta_by_source": {s: guards["heldout"][s]["bpb"] - base["heldout"][s]["bpb"] for s in ("case-law", "sec", "fineweb-edu")},
             "casehold_pmi_delta_points": 100 * (guards["casehold"]["acc_pmi"] - base["casehold"]["acc_pmi"])}
    final_path.write_text(json.dumps(final, default=str))
    volume.commit()
    log(f"final {top['key']}: tau {final['tau']} | test AUROC {final['test_decision']['auroc']:.3f} | "
        f"bpb {final['bpb_delta_pooled']:+.4f} | casehold {final['casehold_pmi_delta_points']:+.1f}")
    return {**out, "final": final, "spent_usd": spent()}


@app.function(image=image, gpu=G.gpu, volumes=VOLUMES, cpu=8.0, memory=65_536, timeout=60 * 90)
def pilot(max_usd: float) -> dict:
    return _runs(max_usd, "base500", "pilot", [{"n": G.pilot_size, "lr": lr, "evals": G.pilot_evals} for lr in G.pilot_lrs], False)


@app.function(image=image, gpu=G.gpu, volumes=VOLUMES, cpu=8.0, memory=65_536, timeout=60 * 120)
def full(max_usd: float, lr: float) -> dict:
    return _runs(max_usd, "base500", "full", [{"n": G.full_size, "lr": lr, "evals": G.full_evals}], True)


@app.function(image=image, gpu=G.gpu, volumes=VOLUMES, cpu=8.0, memory=65_536, timeout=60 * 60)
def control(max_usd: float) -> dict:
    """10.7: the author's 125M with SLM-125M's exact 20K recipe (LR 5e-5, D-23): the overlap held equal."""
    return _runs(max_usd, "control125", "full", [{"n": G.full_size, "lr": config.SFT.lr, "evals": G.full_evals}], True)


@app.local_entrypoint()
def pilot_check(confirm: bool = False, max_usd: float = 3.0):
    _gate(confirm, max_usd, "10.5-pilot")
    _report("pilot", pilot.remote(max_usd))


@app.local_entrypoint()
def full_check(confirm: bool = False, lr: float = 0.0, max_usd: float = 4.5):
    _gate(confirm, max_usd, "10.6-full")
    if lr not in G.pilot_lrs:
        raise SystemExit(f"--lr must be one of {G.pilot_lrs} (the 10.5 pre-registered pick)")
    _report("full", full.remote(max_usd, lr))


@app.local_entrypoint()
def control_check(confirm: bool = False, max_usd: float = 1.5):
    _gate(confirm, max_usd, "10.7-control")
    _report("control", control.remote(max_usd))


def _gate(confirm: bool, max_usd: float, stage: str) -> None:
    if not confirm:
        raise SystemExit(f"Step {stage} uses a GPU and costs money. Re-run with --confirm once approved.")
    cap = config.SLM500_STAGE_CAPS_USD[stage]
    if max_usd > cap:
        raise SystemExit(f"--max-usd {max_usd} exceeds the approved cap ${cap} for {stage}")


def _report(which: str, rep: dict) -> None:
    d = LOCAL / which
    d.mkdir(parents=True, exist_ok=True)
    for r in rep.get("runs", []):
        for mode, g in r.pop("gens").items():
            (d / f"dev_{r['key']}_{mode}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in g))
    if rep.get("final"):
        for mode, g in rep["final"].pop("test_gens").items():
            (d / f"test_{mode}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in g))
    (d / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    for r in rep.get("runs", []):
        print(f"{r['key']}: step {r['selected_step']}/{r['steps']} gate {r['gate_ok']} | tau {r['tau']['tau']} | "
              f"copy2 {r['proxies']['copy2']['selection_score']:+.3f} grounded {r['proxies']['copy2']['proxy_grounded_acc']:.3f}")
    print(f"status {rep['status']}; spent ${rep['spent_usd']:.2f} -> {d}")
