"""Inference API for Deepanshu027/slm-125m-base, a 125M legal/financial base completer, and
(POST /answer) Deepanshu027/slm-125m-raft, its grounded fine-tune: answer only from the given
passages with a cited passage and a verbatim quote, or refuse (Phase 8, docs/09-raft-release.md).

This is a *base* model: it continues text, it does not follow instructions. Every
prompt is prefixed with <|eos|> (id 1), never <|bos|>: training documents always
start right after an EOS, and <|bos|> never occurs in the training data.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterator

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))
RAFT_DIR = Path(os.environ.get("RAFT_DIR", "/app/model-raft"))
EOS_ID, PAD_ID = 1, 2
MAX_CONTEXT = 1024
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).eval()
STATS: dict[str, Any] = json.loads((MODEL_DIR / "training_stats.json").read_text())
# Byte-level tokens that are an incomplete UTF-8 sequence decode to U+FFFD on their
# own; sampling them is what produces "unicode garbage". Valid characters (§ ’ —) stay.
PARTIAL_UTF8_IDS = [i for i in range(len(tokenizer)) if "�" in tokenizer.decode([i])]
LOCK = threading.Lock()  # one generation at a time; parallel ones would only thrash the CPU
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "30"))
_hits: dict[str, deque[float]] = defaultdict(deque)
_hits_lock = threading.Lock()

app = FastAPI(title="SLM-125M API", version="1.0",
              description="Pretrained 125M base model (legal + SEC + web). Completion only.")


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Per-IP sliding window on POSTs, so a public site can't be run up into a bill."""
    if request.method == "POST":
        forwarded = request.headers.get("x-forwarded-for", "")
        ip = forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")
        now = time.time()
        with _hits_lock:
            hits = _hits[ip]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= RATE_LIMIT_PER_MIN:
                retry = int(61 - (now - hits[0]))
                return JSONResponse({"detail": f"rate limit: {RATE_LIMIT_PER_MIN} requests/minute"},
                                    status_code=429, headers={"Retry-After": str(retry)})
            hits.append(now)
            if len(_hits) > 10_000:  # forget idle clients
                for key in [k for k, v in _hits.items() if not v or now - v[-1] > 60]:
                    del _hits[key]
    return await call_next(request)


# Added last = outermost, so even a 429 carries CORS headers and the browser can read it.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                   allow_headers=["*"], expose_headers=["Retry-After"])


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int = Field(80, ge=1, le=256)
    min_new_tokens: int = Field(16, ge=0, le=256)
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="0 = greedy")
    top_k: int = Field(50, ge=0, le=1000, description="0 = disabled")
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    suppress_partial_utf8: bool = True


class ScoreRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


class AnswerRequest(BaseModel):
    passages: list[str] = Field(..., max_length=4, description="0-4 passages (~160 tokens each in training)")
    question: str = Field(..., min_length=1, max_length=600)
    copy_constrained: bool = Field(True, description="keep the quote a verbatim span of the cited passage")
    tau: float = Field(0.7, ge=0.05, le=0.95, description="refuse iff s = P(none)/(P(none)+P(Pass)) >= tau")


_raft: dict[str, Any] = {}


def raft_model() -> tuple[Any, Any, Any]:
    """Loaded on the first /answer call, so the base model's cold start is unchanged."""
    if not _raft:
        import raft_decode
        _raft.update(rd=raft_decode, tok=AutoTokenizer.from_pretrained(RAFT_DIR),
                     net=AutoModelForCausalLM.from_pretrained(RAFT_DIR, torch_dtype=torch.float32).eval())
    return _raft["rd"], _raft["tok"], _raft["net"]


def encode_prompt(prompt: str, max_new_tokens: int) -> torch.Tensor:
    """<|eos|> + prompt, trimmed from the left so prompt + output fit the context."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    keep = MAX_CONTEXT - max_new_tokens - 1
    return torch.tensor([[EOS_ID] + ids[-keep:]])


def generation_kwargs(req: GenerateRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": req.max_new_tokens,
        "min_new_tokens": min(req.min_new_tokens, req.max_new_tokens),
        "repetition_penalty": req.repetition_penalty,
        "pad_token_id": PAD_ID,
        "eos_token_id": EOS_ID,
        "do_sample": req.temperature > 0,
    }
    if req.temperature > 0:
        kwargs.update(temperature=req.temperature, top_k=req.top_k, top_p=req.top_p)
    else:  # greedy: neutralise sampling defaults from generation_config.json
        kwargs.update(temperature=None, top_k=None, top_p=None)
    if req.suppress_partial_utf8:
        kwargs["suppress_tokens"] = PARTIAL_UTF8_IDS
    return kwargs


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, Any]:
    return {"model": STATS["model"]["repo"], "docs": "/docs",
            "grounded_model": "Deepanshu027/slm-125m-raft",
            "endpoints": ["/info", "/generate", "/generate/stream", "/score", "/answer", "/health"]}


@app.get("/info")
def info() -> dict[str, Any]:
    return {**STATS, "serving": {
        "hardware": "HF Spaces cpu-basic (2 vCPU)", "dtype": "float32", "context": MAX_CONTEXT,
        "prompt_prefix": "<|eos|> (id 1)", "suppressed_partial_utf8_tokens": len(PARTIAL_UTF8_IDS)}}


@app.post("/generate")
def generate(req: GenerateRequest) -> dict[str, Any]:
    x = encode_prompt(req.prompt, req.max_new_tokens)
    t0 = time.time()
    with LOCK, torch.no_grad():
        if req.seed is not None:
            torch.manual_seed(req.seed)
        y = model.generate(x, attention_mask=torch.ones_like(x), **generation_kwargs(req))
    new = y[0, x.shape[1]:].tolist()
    elapsed = time.time() - t0
    return {"prompt": req.prompt, "generated": tokenizer.decode(new, skip_special_tokens=True),
            "prompt_tokens": x.shape[1] - 1, "new_tokens": len(new),
            "stopped_on_eos": bool(new) and new[-1] == EOS_ID,
            "elapsed_ms": round(elapsed * 1000), "tokens_per_s": round(len(new) / elapsed, 1)}


@app.post("/generate/stream")
def generate_stream(req: GenerateRequest) -> StreamingResponse:
    x = encode_prompt(req.prompt, req.max_new_tokens)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    result: dict[str, Any] = {}
    finished = threading.Event()

    def run() -> None:
        try:
            with LOCK, torch.no_grad():
                if req.seed is not None:
                    torch.manual_seed(req.seed)
                out = model.generate(x, attention_mask=torch.ones_like(x), streamer=streamer,
                                     **generation_kwargs(req))
            new = out[0, x.shape[1]:].tolist()
            result.update(new_tokens=len(new), stopped_on_eos=bool(new) and new[-1] == EOS_ID)
        except Exception:  # never leave the client waiting on a dead generation
            streamer.end()
            raise
        finally:
            finished.set()

    def events() -> Iterator[str]:
        # Streamed pieces are text chunks (the streamer flushes at word boundaries), not
        # tokens; the true token count arrives in the final "done" event.
        t0 = time.time()
        threading.Thread(target=run, daemon=True).start()
        for piece in streamer:
            if piece:
                yield f"data: {json.dumps({'token': piece})}\n\n"
        finished.wait(timeout=10)
        yield f"data: {json.dumps({'done': True, 'elapsed_ms': round((time.time() - t0) * 1000), 'prompt_tokens': x.shape[1] - 1, **result})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/score")
def score(req: ScoreRequest) -> dict[str, Any]:
    """How surprised the model is by each token of *your* text."""
    ids = [EOS_ID] + tokenizer.encode(req.text, add_special_tokens=False)[: MAX_CONTEXT - 1]
    x = torch.tensor([ids])
    with LOCK, torch.no_grad():
        logprobs = torch.log_softmax(model(x).logits[0, :-1].float(), dim=-1)
    targets = x[0, 1:]
    nll = -logprobs[torch.arange(len(targets)), targets]
    n_bytes = len(tokenizer.decode(ids[1:]).encode("utf-8"))
    mean = float(nll.mean())
    return {
        "n_tokens": len(targets),
        "mean_nll": round(mean, 4),
        "perplexity": round(math.exp(mean), 3),
        "bits_per_byte": round(float(nll.sum()) / (math.log(2) * n_bytes), 4),
        "tokens": [{"token": tokenizer.decode([t]), "nll": round(float(n), 3),
                    "prob": round(math.exp(-float(n)), 4)} for t, n in zip(targets.tolist(), nll)],
    }


@app.post("/answer")
def answer(req: AnswerRequest) -> JSONResponse:
    """Grounded answer from SLM-125M-RAFT: a cited passage, a verbatim quote and an answer, or a refusal."""
    passages = [p.strip() for p in req.passages if p.strip()]
    if any(len(p) > 2000 for p in passages):
        return JSONResponse({"detail": "each passage must be at most 2,000 characters"}, status_code=422)
    with LOCK:
        rd, tok, net = raft_model()
        n_prompt = len(tok.encode(rd.format_prompt(passages, req.question.strip()), add_special_tokens=False))
        if n_prompt + rd.MAX_NEW_TOKENS > MAX_CONTEXT:
            return JSONResponse({"detail": f"passages + question are {n_prompt} tokens; the limit is "
                                           f"{MAX_CONTEXT - rd.MAX_NEW_TOKENS} (1,024 context minus the answer)"},
                                status_code=422)
        t0 = time.time()
        with torch.no_grad():
            out = rd.answer(net, tok, passages, req.question.strip(), tau=req.tau, copy=req.copy_constrained)
    return JSONResponse({**out, "tau": req.tau, "copy_constrained": req.copy_constrained,
                         "prompt_tokens": n_prompt, "elapsed_ms": round((time.time() - t0) * 1000)})
