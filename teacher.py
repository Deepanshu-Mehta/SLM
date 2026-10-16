"""Teacher / checker API clients for Phase 7 (docs/08-raft.md §5.5). Runs locally.

Keys come from `.env.local` (never printed or logged). Spend is guarded per stage:
before every request (or Gemini batch) the worst-case cost is *reserved* against the stage
cap from config.RAFT_STAGE_CAPS_USD, and the reservation is replaced by the real cost from
the response's usage. Results are appended to JSONL as they arrive, keyed, so a crash or
re-run never pays twice for the same request.

- DeepSeek (`deepseek-flash`): concurrent chat completions, JSON mode, thinking switched
  per request. Refuses to run in DeepSeek's peak hours unless allow_peak. A no-think call
  that reports reasoning aborts the run (thinking is DeepSeek's default and triples cost).
- Gemini (`gemini-3.8-flash`): Batch API with inline requests; the job name is saved so an
  interrupted poll can re-attach.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import config

T = config.TEACHER
CHARS_PER_TOKEN_EST = 3.0          # conservative (over-estimates tokens) for worst-case reservations


class BudgetExceeded(RuntimeError):
    pass


class ThinkingLeak(RuntimeError):
    pass


def load_env(path: str = ".env.local") -> None:
    """KEY=VALUE lines into os.environ (existing values win). Values are never printed."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def is_peak(now: dt.datetime | None = None) -> bool:
    now = now or dt.datetime.now(dt.timezone.utc)
    return now.weekday() < 5 and any(a <= now.hour < b for a, b in T.peak_hours_utc)


def est_tokens(*texts: str) -> int:
    return int(sum(len(t) for t in texts) / CHARS_PER_TOKEN_EST) + 16


# -------------------------------------------------------------------------- ledger
class Ledger:
    """Thread-safe spend ledger for one stage, persisted to JSON."""

    def __init__(self, path: Path, cap_usd: float):
        self.path, self.cap = Path(path), cap_usd
        self.lock = threading.Lock()
        saved = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.spent = float(saved.get("spent_usd", 0.0))
        self.calls = dict(saved.get("calls", {}))
        self.tokens = dict(saved.get("tokens", {}))
        self.reserved = 0.0

    def reserve(self, worst_usd: float) -> None:
        with self.lock:
            if self.spent + self.reserved + worst_usd > self.cap:
                raise BudgetExceeded(f"stage cap ${self.cap:.2f}: spent ${self.spent:.4f} + "
                                     f"reserved ${self.reserved:.4f} + next ${worst_usd:.4f}")
            self.reserved += worst_usd

    def settle(self, worst_usd: float, actual_usd: float, kind: str, tokens: dict) -> None:
        with self.lock:
            self.reserved -= worst_usd
            self.spent += actual_usd
            self.calls[kind] = self.calls.get(kind, 0) + 1
            agg = self.tokens.setdefault(kind, {})
            for k, v in tokens.items():
                agg[k] = agg.get(k, 0) + int(v or 0)
            self._write()

    def release(self, worst_usd: float) -> None:
        with self.lock:
            self.reserved -= worst_usd

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"cap_usd": self.cap, "spent_usd": round(self.spent, 6),
                                   "calls": self.calls, "tokens": self.tokens,
                                   "updated": time.time()}, indent=2))
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------- results I/O
def done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(ln)["key"] for ln in path.read_text().splitlines() if ln.strip()}


class JsonlAppender:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def write(self, row: dict) -> None:
        with self.lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class Job:
    key: str
    system: str
    user: str
    max_tokens: int
    thinking: bool = False


# ------------------------------------------------------------------------- DeepSeek
class DeepSeek:
    def __init__(self, ledger: Ledger, allow_peak: bool = False, workers: int = 16):
        from openai import OpenAI

        load_env()
        if "DEEPSEEK_API_KEY" not in os.environ:
            raise SystemExit("DEEPSEEK_API_KEY missing from .env.local")
        self.client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=T.writer_base_url,
                             timeout=180, max_retries=0)
        self.ledger, self.allow_peak, self.workers = ledger, allow_peak, workers

    def _prices(self) -> tuple[float, float, float]:
        mult = T.writer_peak_multiplier if is_peak() else 1.0
        return T.writer_in * mult, T.writer_cached_in * mult, T.writer_out * mult

    def call(self, job: Job) -> dict:
        if is_peak() and not self.allow_peak:
            raise BudgetExceeded("DeepSeek peak hours (price x2): refusing without allow_peak")
        p_in, p_cached, p_out = self._prices()
        worst = (est_tokens(job.system, job.user) * p_in + job.max_tokens * p_out) / 1e6
        self.ledger.reserve(worst)
        extra = {"thinking": {"type": "enabled" if job.thinking else "disabled"}}
        if job.thinking:
            extra["reasoning_effort"] = T.check_thinking_effort
        last_err = ""
        for attempt in range(5):
            t0 = time.time()
            try:
                r = self.client.chat.completions.create(
                    model=T.writer, max_tokens=job.max_tokens,
                    messages=[{"role": "system", "content": job.system},
                              {"role": "user", "content": job.user}],
                    response_format={"type": "json_object"}, extra_body=extra)
            except Exception as e:                      # rate limit, 5xx, timeout, network
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
                time.sleep(min(60, 4 * 2 ** attempt))
                continue
            u = r.usage
            hit = int(getattr(u, "prompt_cache_hit_tokens", 0) or 0)
            miss = int(getattr(u, "prompt_cache_miss_tokens", None) or (u.prompt_tokens - hit))
            details = getattr(u, "completion_tokens_details", None)
            reasoning = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
            msg = r.choices[0].message
            if not job.thinking and (reasoning or getattr(msg, "reasoning_content", None)):
                self.ledger.release(worst)
                raise ThinkingLeak(f"{job.key}: no-think call returned {reasoning} reasoning tokens")
            cost = (miss * p_in + hit * p_cached + u.completion_tokens * p_out) / 1e6
            tokens = {"prompt_hit": hit, "prompt_miss": miss, "completion": u.completion_tokens,
                      "reasoning": reasoning}
            content = (msg.content or "").strip()
            if not content and attempt < 4:            # known DeepSeek JSON-mode quirk: retry
                self.ledger.settle(0.0, cost, "deepseek_empty", tokens)
                last_err = "empty_content"
                continue
            self.ledger.settle(worst, cost, "deepseek_think" if job.thinking else "deepseek", tokens)
            try:
                parsed, err = json.loads(content), ""
            except json.JSONDecodeError:
                parsed, err = None, "bad_json"
            return {"key": job.key, "ok": parsed is not None, "json": parsed, "error": err,
                    "finish": r.choices[0].finish_reason, "tokens": tokens, "cost": cost,
                    "seconds": round(time.time() - t0, 2), "thinking": job.thinking,
                    "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    "peak": is_peak()}
        self.ledger.release(worst)
        return {"key": job.key, "ok": False, "json": None, "error": last_err, "tokens": {},
                "cost": 0.0, "thinking": job.thinking}

    def run(self, jobs: Iterable[Job], out_path: Path) -> dict:
        """Run jobs concurrently, appending results; skips keys already in out_path."""
        out = JsonlAppender(out_path)
        todo = [j for j in jobs if j.key not in done_keys(out_path)]
        stats = {"submitted": len(todo), "ok": 0, "failed": 0}
        with ThreadPoolExecutor(self.workers) as pool:
            futures = {pool.submit(self.call, j): j for j in todo}
            pending = set(futures)
            while pending:
                finished, pending = wait(pending, return_when=FIRST_EXCEPTION)
                for f in finished:
                    if f.exception() is not None:
                        for p in pending:
                            p.cancel()
                        raise f.exception()
                    row = f.result()
                    out.write(row)
                    stats["ok" if row["ok"] else "failed"] += 1
        return stats


# --------------------------------------------------------------------------- Gemini
class GeminiBatch:
    def __init__(self, ledger: Ledger):
        from google import genai

        load_env()
        if "GEMINI_API_KEY" not in os.environ:
            raise SystemExit("GEMINI_API_KEY missing from .env.local")
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.ledger = ledger

    def run(self, jobs: list[Job], out_path: Path, name: str, poll_s: int = 30,
            max_wait_s: int = 26 * 3600) -> dict:
        from google.genai import types

        out_path = Path(out_path)
        state_path = out_path.with_suffix(".job.json")
        todo = [j for j in jobs if j.key not in done_keys(out_path)]
        if not todo:
            return {"submitted": 0}
        worst = sum(est_tokens(j.system, j.user) * T.checker_batch_in
                    + j.max_tokens * T.checker_batch_out for j in todo) / 1e6
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        if state is None:
            self.ledger.reserve(worst)
            level = types.ThinkingLevel.LOW if T.checker_thinking == "low" else types.ThinkingLevel.MINIMAL
            reqs = [types.InlinedRequest(
                contents=[types.Content(role="user", parts=[types.Part(text=j.user)])],
                metadata={"key": j.key},
                config=types.GenerateContentConfig(
                    system_instruction=j.system, response_mime_type="application/json",
                    max_output_tokens=j.max_tokens,
                    thinking_config=types.ThinkingConfig(thinking_level=level)))
                for j in todo]
            job = self.client.batches.create(model=T.checker, src=reqs,
                                             config={"display_name": name})
            state = {"job": job.name, "keys": [j.key for j in todo], "worst_usd": worst,
                     "created": time.time()}
            state_path.write_text(json.dumps(state))
        else:
            self.ledger.reserve(state["worst_usd"])      # re-attach: hold the same reservation
        terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
                    "JOB_STATE_EXPIRED", "JOB_STATE_PARTIALLY_SUCCEEDED"}
        t0 = time.time()
        while True:
            job = self.client.batches.get(name=state["job"])
            if job.state.name in terminal:
                break
            if time.time() - t0 > max_wait_s:
                self.ledger.release(state["worst_usd"])
                raise TimeoutError(f"Gemini batch {state['job']} still {job.state.name}")
            time.sleep(poll_s)
        responses = job.dest.inlined_responses if job.dest else []
        out = JsonlAppender(out_path)
        total = 0.0
        stats = {"state": job.state.name, "ok": 0, "failed": 0}
        for key, resp in zip(state["keys"], responses or []):
            meta_key = (resp.metadata or {}).get("key", key)
            if resp.error or resp.response is None:
                out.write({"key": meta_key, "ok": False, "json": None, "error": str(resp.error)[:300]})
                stats["failed"] += 1
                continue
            um = resp.response.usage_metadata
            tokens = {"prompt": um.prompt_token_count or 0, "output": um.candidates_token_count or 0,
                      "thoughts": um.thoughts_token_count or 0}
            cost = (tokens["prompt"] * T.checker_batch_in
                    + (tokens["output"] + tokens["thoughts"]) * T.checker_batch_out) / 1e6
            total += cost
            try:
                parsed, err = json.loads(resp.response.text or ""), ""
            except (json.JSONDecodeError, ValueError):
                parsed, err = None, "bad_json"
            out.write({"key": meta_key, "ok": parsed is not None, "json": parsed, "error": err,
                       "tokens": tokens, "cost": cost})
            stats["ok" if parsed is not None else "failed"] += 1
            self.ledger.settle(0.0, 0.0, "gemini_batch", tokens)
        self.ledger.settle(state["worst_usd"], total, "gemini_batch_job", {})
        state_path.unlink()
        stats["cost"] = total
        return stats
