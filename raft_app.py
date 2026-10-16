"""Phase 7 Modal app (docs/08-raft.md). Step 7.2: passage pools, dev/test leakage scan, BM25.

Layout on the Volume (config.RAFT_DIR = /data/raft):
  passages/{train,val}/{source}/{shard}.jsonl   pass 1: val = every val doc; train = a
                                                 deterministic document sample (distractor pool)
  devtest/candidates.jsonl                       dev/test source candidates (clean val passages)
  devtest/source_pool.jsonl                      candidates that pass the leakage scan
  bm25/{split}-{source}/                         bm25s index + pids.json
  reports/7.2.json                               everything measured

The core logic lives in plain functions (`build_passages`, `select_candidates`, `leak_scan`,
`build_bm25`) so `smoke_raft.py` can test them locally. Modal functions are thin wrappers,
one worker per corpus shard (fan-out), driven by a remote `prepare_driver` so a dropped
local connection can't stall the pipeline halfway.

Run (paid, needs approval): modal run --detach raft_app.py::prepare --confirm
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import modal

import config

app = modal.App(f"{config.PROJECT}-raft")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wamerican")          # /usr/share/dict/words, as in Phase 1's OCR gate
    .pip_install("transformers==4.46.3", "numpy==2.1.3", "scipy==1.14.1", "bm25s==0.3.13")
    .add_local_python_source("config", "cleaning", "dedup", "raft_data")
)
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}

RAFT = Path(config.RAFT_DIR)
SOURCES = tuple(name for name, _ in config.RAFT.source_mix)
MEAN_PASSAGE_TOKENS = 139            # measured on 6,345 real passages (smoke_raft.py)
DEVTEST_CANDIDATES = 3_000           # ~1,330 source passages needed for dev+test, x buffer
MAX_SOURCES_PER_DOC = 2              # dev/test diversity
CPU_RATE, MEM_RATE = 0.0000131, 0.00000222   # Modal $/core/s and $/GiB/s (modal.com/pricing)


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def _write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# ============================================================== pure core (testable)
def pool_rate(train_tokens: int, target: int = config.RAFT.distractor_pool_per_source) -> float:
    """Document sample rate that yields ~target train passages for a source."""
    return min(1.0, target * MEAN_PASSAGE_TOKENS / max(1, train_tokens))


def build_passages(corpus_file: Path, source: str, rate: float, tokenizer, out_dir: Path) -> dict:
    """Pass 1 for one corpus shard: chunk every val doc and the sampled train docs."""
    import raft_data as R

    stem = corpus_file.stem
    rows: dict[str, list[dict]] = {"train": [], "val": []}
    stats = Counter()
    tags_seen: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    elig: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    hist: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    with corpus_file.open(encoding="utf-8") as fh:
        for doc_idx, line in enumerate(fh):
            text = line.rstrip("\n")
            if not text:
                continue
            stats["docs"] += 1
            split = "val" if R.is_val_doc(text) else "train"
            if split == "train" and not R.in_pool_sample(text, rate):
                continue
            stats[f"{split}_docs"] += 1
            passages = R.chunk_document(text, tokenizer)
            did = R.doc_id(text)
            devtest = ("dev" if R.is_dev_doc(text) else "test") if split == "val" else None
            for i, (p, n) in enumerate(passages):
                tags = R.passage_tags(p, source)
                e = R.source_eligibility(tags, source)
                tags_seen[split].update(tags or ["none"])
                elig[split][e or "not_source"] += 1
                hist[split][min(n // 20 * 20, 160)] += 1
                rows[split].append({
                    "pid": f"{source}/{stem}/{doc_idx}/{i}", "doc": did, "chunk": i,
                    "n_chunks": len(passages), "tokens": n, "tags": tags, "elig": e,
                    "devtest": devtest, "text": p})
    out = {"source": source, "shard": stem, "rate": rate, **stats}
    for split in ("train", "val"):
        out[f"{split}_passages"] = _write_jsonl(out_dir / split / source / f"{stem}.jsonl", rows[split])
        out[f"{split}_tags"] = dict(tags_seen[split])
        out[f"{split}_elig"] = dict(elig[split])
        out[f"{split}_token_hist"] = {str(k): v for k, v in sorted(hist[split].items())}
    return out


def select_candidates(val_rows: list[dict], n_total: int = DEVTEST_CANDIDATES,
                      seed: int = config.RAFT.seed,
                      per_doc_cap: dict[str, int] | None = None) -> list[dict]:
    """Stratified dev/test source candidates: source mix 40/40/20, dev:test 1:2 by document,
    a global per-document cap (default MAX_SOURCES_PER_DOC), and SEC 'numeric' passages
    filled first up to ~10% of SEC picks so numeric questions survive."""
    import random

    rng = random.Random(seed)
    caps = per_doc_cap or {}
    picked: list[dict] = []
    per_doc: Counter = Counter()        # global: a document never exceeds its cap
    for source, share in config.RAFT.source_mix:
        cap = caps.get(source, MAX_SOURCES_PER_DOC)
        for devtest, frac in (("dev", 1 / 3), ("test", 2 / 3)):
            want = round(n_total * share * frac)
            pool = [r for r in val_rows if r["pid"].startswith(source + "/")
                    and r["devtest"] == devtest and r["elig"]]
            rng.shuffle(pool)
            quota_numeric = round(want * 0.10) if source == "sec" else 0
            numeric = [r for r in pool if r["elig"] == "numeric"]
            clean = [r for r in pool if r["elig"] == "clean"]
            got: list[dict] = []
            for group, limit in ((numeric, quota_numeric), (clean, want)):
                taken = 0
                for r in group:
                    if len(got) >= want or taken >= limit:
                        break
                    if per_doc[r["doc"]] >= cap:
                        continue
                    per_doc[r["doc"]] += 1
                    got.append(r)
                    taken += 1
            picked.extend(got)
    return picked


def leak_scan(corpus_file: Path, candidates: list[dict], n: int = config.RAFT.leak_ngram) -> dict:
    """Pass 2 for one shard: which of each candidate's n-grams occur in this shard's TRAIN
    documents. Hashes are process-local, so the candidate n-grams are hashed here too and
    only positions (not hashes) leave the process."""
    import raft_data as R

    index: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for c, row in enumerate(candidates):
        for pos, h in enumerate(R.ngram_hash_list(R.word_list(row["text"]), n)):
            index[h].append((c, pos))
    keys = set(index)
    hits: dict[int, int] = defaultdict(int)          # bitmask of hit n-gram positions
    train_docs = 0
    with corpus_file.open(encoding="utf-8") as fh:
        for line in fh:
            text = line.rstrip("\n")
            if not text or R.is_val_doc(text):
                continue
            train_docs += 1
            for h in keys.intersection(R.ngram_hash_list(R.word_list(text), n)):
                for c, pos in index[h]:
                    hits[c] |= 1 << pos
    return {"train_docs": train_docs, "hits": dict(hits)}


def leak_fractions(candidates: list[dict], shard_results: list[dict],
                   n: int = config.RAFT.leak_ngram) -> list[float]:
    import raft_data as R

    union: dict[int, int] = defaultdict(int)
    for res in shard_results:
        for c, mask in res["hits"].items():
            union[int(c)] |= mask
    fracs = []
    for c, row in enumerate(candidates):
        total = max(1, len(R.word_list(row["text"])) - n + 1)
        fracs.append(union.get(c, 0).bit_count() / total)
    return fracs


def build_bm25(rows: list[dict], out_dir: Path | None, probes: int = 500,
               seed: int = config.RAFT.seed) -> dict:
    """Index passages with BM25 and run the sanity probe: a passage's longest sentence
    (>= 8 words) must retrieve top-1 that passage, or a passage containing that exact
    sentence (SEC filings repeat sentences across sections and filings)."""
    import random

    import bm25s

    import raft_data as R

    texts = [r["text"] for r in rows]
    t0 = time.time()
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(texts, stopwords="en", show_progress=False),
                    show_progress=False)
    build_s = time.time() - t0
    if out_dir is not None:
        retriever.save(str(out_dir))
        (out_dir / "pids.json").write_text(json.dumps([r["pid"] for r in rows]))
    rng = random.Random(seed)
    queries, targets = [], []
    for i in rng.sample(range(len(rows)), min(probes, len(rows))):
        sents = [texts[i][a:b].strip() for a, b in R.sentence_spans(texts[i])]
        sents = [s for s in sents if len(s.split()) >= 8]
        if sents:
            queries.append(max(sents, key=len))
            targets.append(i)
    docs, _ = retriever.retrieve(bm25s.tokenize(queries, stopwords="en", show_progress=False),
                                 k=1, show_progress=False)
    hit = sum(texts[int(docs[j, 0])] == texts[t] or queries[j] in texts[int(docs[j, 0])]
              for j, t in enumerate(targets))
    return {"passages": len(rows), "build_seconds": round(build_s, 1),
            "probes": len(targets), "top1_hit_rate": hit / max(1, len(targets))}


# ================================================================== Modal wrappers
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)


def _corpus_shards() -> list[tuple[str, str]]:
    return [(s, p.name) for s in SOURCES
            for p in sorted(Path(config.CORPUS_DIR, s).glob("*.txt"))]


@app.function(image=image, volumes=VOLUMES, timeout=60 * 40, cpu=2.0, memory=8_192)
def passages_shard(source: str, shard_name: str) -> dict:
    t0 = time.time()
    index = json.loads(Path(config.TOKENS_DIR, "index.json").read_text())
    train_tokens = sum(s["train_tokens"] for s in index["shards"] if s["source"] == source)
    res = build_passages(Path(config.CORPUS_DIR, source, shard_name), source,
                         pool_rate(train_tokens), _tokenizer(), RAFT / "passages")
    volume.commit()
    return {**res, "seconds": time.time() - t0, "cpu": 2.0, "mem_gib": 8.0}


@app.function(image=image, volumes=VOLUMES, timeout=60 * 40, cpu=2.0, memory=16_384)
def leak_scan_shard(source: str, shard_name: str,
                    candidates_file: str = "devtest/candidates.jsonl") -> dict:
    t0 = time.time()
    volume.reload()
    candidates = _jsonl(RAFT / candidates_file)
    res = leak_scan(Path(config.CORPUS_DIR, source, shard_name), candidates)
    return {**res, "source": source, "shard": shard_name,
            "seconds": time.time() - t0, "cpu": 2.0, "mem_gib": 16.0}


@app.function(image=image, volumes=VOLUMES, timeout=60 * 40, cpu=2.0, memory=16_384)
def bm25_index(split: str, source: str) -> dict:
    t0 = time.time()
    volume.reload()
    rows = [r for f in sorted((RAFT / "passages" / split / source).glob("*.jsonl"))
            for r in _jsonl(f)]
    out_dir = RAFT / "bm25" / f"{split}-{source}"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = build_bm25(rows, out_dir)
    volume.commit()
    return {**res, "split": split, "source": source,
            "seconds": time.time() - t0, "cpu": 2.0, "mem_gib": 16.0}


@app.function(image=image, volumes=VOLUMES, timeout=60 * 120, cpu=1.0, memory=8_192)
def prepare_driver() -> dict:
    """Step 7.2 end to end, run remotely so a local disconnect can't stop it mid-way."""
    t0 = time.time()
    shards = _corpus_shards()
    report: dict = {"started": t0, "corpus_shards": len(shards)}

    pass1 = list(passages_shard.starmap(shards))
    report["pass1"] = pass1
    volume.reload()

    val_rows = [r for s in SOURCES for f in sorted((RAFT / "passages" / "val" / s).glob("*.jsonl"))
                for r in _jsonl(f)]
    candidates = select_candidates(val_rows)
    _write_jsonl(RAFT / "devtest" / "candidates.jsonl", candidates)
    volume.commit()
    report["candidates"] = {"total": len(candidates),
                            "by_source_split": Counter(f"{r['pid'].split('/')[0]}:{r['devtest']}"
                                                       for r in candidates),
                            "sec_numeric": sum(r["elig"] == "numeric" for r in candidates)}

    pass2 = list(leak_scan_shard.starmap(shards))
    fracs = leak_fractions(candidates, pass2)
    limit = config.RAFT.leak_max_fraction
    pool = [{**r, "leak_fraction": f} for r, f in zip(candidates, fracs) if f < limit]
    _write_jsonl(RAFT / "devtest" / "source_pool.jsonl", pool)
    volume.commit()
    bins = Counter("0" if f == 0 else "<1%" if f < 0.01 else "<10%" if f < limit else ">=10%"
                   for f in fracs)
    report["leak"] = {"limit": limit, "fraction_bins": dict(bins), "kept": len(pool),
                      "dropped_by_source": Counter(r["pid"].split("/")[0]
                                                   for r, f in zip(candidates, fracs) if f >= limit),
                      "shard_seconds": [r["seconds"] for r in pass2]}

    jobs = [(split, s) for split in ("train", "val") for s in SOURCES]
    report["bm25"] = list(bm25_index.starmap(jobs))

    worker_cost = sum(r["seconds"] * (r["cpu"] * CPU_RATE + r["mem_gib"] * MEM_RATE)
                      for r in pass1 + pass2 + report["bm25"])
    driver_cost = (time.time() - t0) * (1.0 * CPU_RATE + 8.0 * MEM_RATE)
    report["cost_usd_estimate"] = round(worker_cost + driver_cost, 4)
    report["wall_seconds"] = round(time.time() - t0, 1)
    (RAFT / "reports").mkdir(parents=True, exist_ok=True)
    (RAFT / "reports" / "7.2.json").write_text(json.dumps(report, indent=2, default=dict))
    volume.commit()
    return json.loads(json.dumps(report, default=dict))


# ------------------------------------------------- 7.2 diagnostics (after the first run)
@app.function(image=image, volumes=VOLUMES, timeout=60 * 30, cpu=2.0, memory=16_384)
def bm25_miss_diag(split: str, source: str, probes: int = 500) -> dict:
    """Why do BM25 probes miss? For each miss: the target's rank in the top 10, and the
    word-5-gram Jaccard between the target and the top-1 passage (near-duplicate or not)."""
    import random

    import bm25s

    import raft_data as R

    t0 = time.time()
    rows = [r for f in sorted((RAFT / "passages" / split / source).glob("*.jsonl")) for r in _jsonl(f)]
    texts = [r["text"] for r in rows]
    retriever = bm25s.BM25.load(str(RAFT / "bm25" / f"{split}-{source}"))
    rng = random.Random(config.RAFT.seed)
    queries, targets = [], []
    for i in rng.sample(range(len(rows)), probes):
        sents = [texts[i][a:b].strip() for a, b in R.sentence_spans(texts[i])]
        sents = [x for x in sents if len(x.split()) >= 8]
        if sents:
            queries.append(max(sents, key=len))
            targets.append(i)
    docs, _ = retriever.retrieve(bm25s.tokenize(queries, stopwords="en", show_progress=False),
                                 k=10, show_progress=False)
    sh = lambda t: set(R.ngram_hash_list(R.word_list(t), 5))
    misses = []
    for j, t in enumerate(targets):
        top = int(docs[j, 0])
        if texts[top] == texts[t] or queries[j] in texts[top]:
            continue
        ranks = [int(d) for d in docs[j]]
        misses.append({"rank": ranks.index(t) + 1 if t in ranks else None,
                       "jaccard_top1": round(R.jaccard(sh(texts[t]), sh(texts[top])), 3),
                       "same_doc": rows[t]["doc"] == rows[top]["doc"],
                       "query": queries[j][:160], "top1": texts[top][:160]})
    return {"split": split, "source": source, "probes": len(targets), "misses": misses,
            "seconds": time.time() - t0, "cpu": 2.0, "mem_gib": 16.0}


@app.function(image=image, volumes=VOLUMES, timeout=60 * 60, cpu=1.0, memory=16_384)
def diagnose_driver() -> dict:
    """Leak fractions for EVERY source-eligible SEC val passage (plus the existing case-law
    and web candidates), and the SEC BM25 misses."""
    t0 = time.time()
    rows = [r for f in sorted((RAFT / "passages" / "val" / "sec").glob("*.jsonl")) for r in _jsonl(f)
            if r["elig"]]
    old = [r for r in _jsonl(RAFT / "devtest" / "candidates.jsonl") if not r["pid"].startswith("sec/")]
    cands = rows + old
    _write_jsonl(RAFT / "devtest" / "leak_diag_candidates.jsonl", cands)
    volume.commit()
    shards = _corpus_shards()
    res = list(leak_scan_shard.starmap([(s, n, "devtest/leak_diag_candidates.jsonl")
                                        for s, n in shards]))
    fracs = leak_fractions(cands, res)
    _write_jsonl(RAFT / "devtest" / "leak_diag.jsonl",
                 ({"pid": r["pid"], "doc": r["doc"], "devtest": r["devtest"], "elig": r["elig"],
                   "tokens": r["tokens"], "leak_fraction": f} for r, f in zip(cands, fracs)))
    bm = [bm25_miss_diag.remote("val", "sec"), bm25_miss_diag.remote("train", "sec")]
    volume.commit()
    cost = sum(r["seconds"] * (r["cpu"] * CPU_RATE + r["mem_gib"] * MEM_RATE) for r in res + bm)
    cost += (time.time() - t0) * (CPU_RATE + 16 * MEM_RATE)
    return {"candidates": len(cands), "sec_candidates": len(rows), "bm25": bm,
            "cost_usd_estimate": round(cost, 4), "wall_seconds": round(time.time() - t0, 1)}


@app.local_entrypoint()
def diagnose(confirm: bool = False):
    """7.2 diagnostics (paid, within the approved 7.2 cap). Requires --confirm."""
    if not confirm:
        raise SystemExit("Costs money (Modal CPU). Re-run with --confirm once approved.")
    out = diagnose_driver.remote()
    Path(".smoke/reports/raft_7.2_diag.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "bm25"}, indent=1))


@app.local_entrypoint()
def prepare(confirm: bool = False):
    """Step 7.2 (paid: Modal CPU). Requires --confirm."""
    if not confirm:
        raise SystemExit("Step 7.2 costs money (Modal CPU). Re-run with --confirm once approved.")
    report = prepare_driver.remote()
    out = Path(".smoke/reports/raft_7.2.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    p1 = report["pass1"]
    print(f"pass 1: {sum(r['val_passages'] for r in p1):,} val + "
          f"{sum(r['train_passages'] for r in p1):,} train passages from {len(p1)} shards")
    print(f"candidates: {report['candidates']}")
    print(f"leak: {report['leak']}")
    for b in report["bm25"]:
        print(f"bm25 {b['split']}-{b['source']}: {b['passages']:,} passages, "
              f"top-1 probe {b['top1_hit_rate']:.1%}")
    print(f"wall {report['wall_seconds']}s, estimated cost ${report['cost_usd_estimate']}"
          f" -> {out}")
