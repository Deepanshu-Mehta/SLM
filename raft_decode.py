"""Grounded answering with SLM-125M-RAFT: prompt format, answer/refuse decision, copy-constrained
decoding. Standalone (torch + transformers only), so the same code runs in evaluation
(eval_raft.py), in the Modal API (serve/) and from the Hugging Face model repo.

    tok = AutoTokenizer.from_pretrained(repo); net = AutoModelForCausalLM.from_pretrained(repo)
    out = answer(net, tok, ["passage one ...", "passage two ..."], "Who signed the agreement?")

Format: `<|eos|><|user|>Passage 1: ...\nPassage k: ...\nQuestion: ...<|assistant|>Source:`. The first
generated token is the decision: s = P(" none") / (P(" none") + P(" Pass")); refuse iff s >= tau.
After " Passage k", copy-constrained decoding keeps the quote a verbatim span of passage k and
forces the `\nAnswer:` scaffold (docs/08-raft.md §13, 7.6-7.7).
"""

from __future__ import annotations

import math
import re

import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

BOS_ID, EOS_ID, PAD_ID, UNK_ID = 0, 1, 2, 3
K_PASSAGES = 4
MAX_NEW_TOKENS = 128
TAU = 0.7                                         # 7.8 tau for the released 20K model
REFUSAL = "The passages do not contain the answer."


def format_prompt(passages: list[str], question: str) -> str:
    body = "\n".join(f"Passage {i}: {p}" for i, p in enumerate(passages, 1))
    return "<|eos|><|user|>" + (body + "\n" if body else "") + f"Question: {question}" + "<|assistant|>Source:"


SUPPRESS = (BOS_ID, PAD_ID, UNK_ID, 4, 5, 6)          # <|bos|> <|pad|> <|unk|> <|user|> <|assistant|> <|system|>
GREEDY = GenerationConfig(do_sample=False, num_beams=1, min_new_tokens=0, max_new_tokens=MAX_NEW_TOKENS,
                          eos_token_id=EOS_ID, pad_token_id=PAD_ID, bos_token_id=BOS_ID)


def decision_ids(tok) -> dict:
    """{'none', 'pass', 'age', 'digits': {k: id}}; fails loudly if the tokenizer disagrees."""
    none = tok.encode(" none", add_special_tokens=False)
    passage = tok.encode(" Passage", add_special_tokens=False)
    digits = {k: tok.encode(f" {k}", add_special_tokens=False) for k in range(1, K_PASSAGES + 1)}
    if len(none) != 1 or len(passage) != 2 or any(len(v) != 1 for v in digits.values()):
        raise RuntimeError(f"unexpected tokenisation: none={none} passage={passage} digits={digits}")
    return {"none": none[0], "pass": passage[0], "age": passage[1],
            "digits": {k: v[0] for k, v in digits.items()}}


def decision_score(logits: torch.Tensor, ids: dict) -> torch.Tensor:
    """s = P(none) / (P(none) + P(Pass)) from next-token logits (..., V)."""
    two = logits[..., [ids["none"], ids["pass"]]].float()
    return torch.softmax(two, dim=-1)[..., 0]


class RaftConstraint(LogitsProcessor):
    """Per-row constrained decoding for a left-padded batch (all prompts end with `Source:`)."""

    def __init__(self, ids: dict, prompt_width: int, ks: list[int], tau: float | None):
        self.ids, self.width, self.ks, self.tau = ids, prompt_width, ks, tau
        self.scores: list[float] = [float("nan")] * len(ks)
        self.choice: list[str | None] = [None] * len(ks)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        step = input_ids.shape[1] - self.width
        scores = scores.clone()
        scores[:, list(SUPPRESS)] = -math.inf
        if self.tau is None:                         # unconstrained: specials suppressed only
            return scores
        neg = torch.full_like(scores, -math.inf)
        for b in range(scores.shape[0]):
            if step == 0:
                s = float(decision_score(scores[b], self.ids))
                self.scores[b] = s
                self.choice[b] = "none" if s >= self.tau else "pass"
                keep = [self.ids["none"] if self.choice[b] == "none" else self.ids["pass"]]
            elif self.choice[b] == "pass" and step == 1:
                keep = [self.ids["age"]]
            elif self.choice[b] == "pass" and step == 2:
                keep = [self.ids["digits"][k] for k in range(1, max(1, self.ks[b]) + 1)]
            else:
                continue
            row = neg[b].clone()
            row[keep] = scores[b, keep]
            scores[b] = row
        return scores


_QUOTE_OPEN = re.compile(r'^ Passage (\d)\nQuote: "(.*)$', re.S)
_YES_NO = (" Yes", " No", " Yes.", " No.")


class CopyConstraint(LogitsProcessor):
    """Copy-constrained quoting: while generation is inside `Quote: "...`, only tokens that keep
    the quote an exact substring of the cited passage (or the closing `"`) are allowed; once the
    quote is closed, the `\nAnswer:` scaffold is forced. The answer/refuse decision is untouched;
    tokens are ranked by the model, best first.

    7.7 format fixes (dev format validity 0.966; 17 failures read):
    - a bare `"` closed the quote and the model wrote on instead of `\nAnswer:` (10 of 17): the
      scaffold is now forced after the close (the tokenizer has no `"\n` token);
    - when no top candidate continued the copy, the constraint let go and the model looped until
      max_new_tokens (4): it now forces the close instead;
    - a 300-char quote plus the answer overflowed 128 tokens (3): cap 240 chars (train quotes:
      p99 222, max 260)."""

    SCAFFOLD = "\nAnswer:"

    def __init__(self, tok, prompt_width: int, passages: list[list[str]], vocab: list[str], top: int = 256,
                 max_quote_chars: int = 240, answer_copy: bool = False, max_answer_chars: int = 200,
                 eos_id: int = 1):
        self.tok, self.width, self.passages, self.vocab, self.top = tok, prompt_width, passages, vocab, top
        self.max_quote_chars = max_quote_chars
        self.answer_copy, self.max_answer_chars, self.eos_id = answer_copy, max_answer_chars, eos_id
        self.close_id = vocab.index('"')
        self.fallbacks = 0                         # forced closes: nothing continued the copy

    def _restrict(self, scores: torch.FloatTensor, b: int, allowed: list[int]) -> None:
        keep = scores[b, allowed].clone()
        scores[b] = -math.inf
        scores[b, allowed] = keep

    def _answer(self, scores: torch.FloatTensor, b: int, a: str, src: str) -> None:
        """7.8b answer copy: the answer is a case-insensitive span of the cited passage, or Yes/No;
        <|eos|> only once it is non-empty and valid."""
        low = src.lower()
        done = bool(a.strip()) and (a.strip().rstrip(".").lower() in low or a in _YES_NO)
        if len(a) >= self.max_answer_chars:
            self._restrict(scores, b, [self.eos_id])
            return
        allowed = [self.eos_id] if done else []
        cands = torch.topk(scores[b], self.top).indices.tolist()
        for pool in (cands, range(len(self.vocab))):     # full vocab only if the top candidates fail
            for i in pool:
                t = self.vocab[i]
                if not t or "\n" in t or "<|" in t:
                    continue
                x = a + t
                if (x.lstrip() or x == " ") and (x.lstrip().lower() in low or any(y.startswith(x) for y in _YES_NO)):
                    allowed.append(i)
            if len(allowed) > (1 if done else 0):
                break
        self._restrict(scores, b, allowed or [self.eos_id])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] - self.width < 3:
            return scores
        for b in range(scores.shape[0]):
            text = self.tok.decode(input_ids[b, self.width:].tolist(), skip_special_tokens=True)
            m = _QUOTE_OPEN.match(text)
            if not m:
                continue
            k, rest = int(m.group(1)), m.group(2)
            if '"' in rest:                        # closed: force the `\nAnswer:` scaffold
                tail = rest.split('"', 1)[1]
                if tail.startswith(self.SCAFFOLD):
                    if self.answer_copy and 1 <= k <= len(self.passages[b]):
                        self._answer(scores, b, tail[len(self.SCAFFOLD):], self.passages[b][k - 1])
                    continue
                if not self.SCAFFOLD.startswith(tail):
                    continue
                need = self.SCAFFOLD[len(tail):]
                allowed = [i for i, t in enumerate(self.vocab) if t and need.startswith(t)]
                self._restrict(scores, b, allowed)
                continue
            ps = self.passages[b]
            if not 1 <= k <= len(ps):
                continue
            q, src = rest, ps[k - 1]
            must_close = len(q) >= self.max_quote_chars
            allowed = []
            for i in torch.topk(scores[b], self.top).indices.tolist():
                t = self.vocab[i]
                if (i == self.close_id and q and q in src) or \
                        (not must_close and t and '"' not in t and (q + t) in src):
                    allowed.append(i)
            if not allowed and not q:              # empty quote: any token that starts a span of the passage
                allowed = [i for i, t in enumerate(self.vocab) if t and '"' not in t and t in src]
            if not allowed:                        # nothing continues the copy: close it
                self.fallbacks += 1
                allowed = [self.close_id]
            self._restrict(scores, b, allowed)
        return scores


@torch.no_grad()
def generate_batch(net, tok, prompts: list[str], ks: list[int], device: torch.device,
                   tau: float | None = 0.5, max_new: int = MAX_NEW_TOKENS,
                   batch: int = 32, autocast=None, copy_passages: list[list[str]] | None = None,
                   answer_copy: bool = False) -> list[dict]:
    """Greedy constrained generation. Returns [{'s', 'choice', 'text'}] where text starts after
    `Source:` (prepend 'Source:' to parse)."""
    ids = decision_ids(tok)
    net.eval()
    out = []
    for i in range(0, len(prompts), batch):
        enc = [tok.encode(p, add_special_tokens=False) for p in prompts[i:i + batch]]
        width = max(len(e) for e in enc)
        x = torch.full((len(enc), width), PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(enc), width), dtype=torch.long)
        for j, e in enumerate(enc):                  # left padding for generation
            x[j, width - len(e):] = torch.tensor(e)
            mask[j, width - len(e):] = 1
        proc = RaftConstraint(ids, width, ks[i:i + batch], tau)
        procs = [proc]
        if copy_passages is not None:
            if not hasattr(generate_batch, "_vocab"):
                generate_batch._vocab = [tok.decode([j]) for j in range(len(tok))]
            procs.append(CopyConstraint(tok, width, copy_passages[i:i + batch], generate_batch._vocab,
                                        answer_copy=answer_copy, eos_id=EOS_ID))
        ctx = autocast() if autocast else torch.autocast("cpu", enabled=False)
        with ctx:
            # An explicit config: the base model's saved one has min_new_tokens=16 and sampling,
            # which blocks <|eos|> after the 15-token refusal (found by the 7.5 smoke test).
            y = net.generate(input_ids=x.to(device), attention_mask=mask.to(device),
                             generation_config=GREEDY.__class__(**{**GREEDY.to_dict(), "max_new_tokens": max_new}),
                             logits_processor=LogitsProcessorList(procs))
        for j in range(len(enc)):
            gen = y[j, width:].tolist()
            if EOS_ID in gen:
                gen = gen[:gen.index(EOS_ID)]
            out.append({"s": proc.scores[j], "choice": proc.choice[j],
                        "text": tok.decode(gen, skip_special_tokens=False)})
    return out



_OUT = re.compile(r'^Source: (none|Passage (\d))\n(?:Quote: "(.*)"\n)?Answer: (.*)$', re.S)


def parse_output(text_after_source: str) -> dict:
    """Parse 'Source:' + generated text. valid=False when the format is broken."""
    m = _OUT.match("Source:" + text_after_source)
    if not m:
        return {"valid": False, "refusal": None, "k": None, "quote": None, "answer": None}
    refusal = m.group(1) == "none"
    answer = m.group(4).strip()
    valid = (answer == REFUSAL) if refusal else True
    return {"valid": valid, "refusal": refusal, "k": None if refusal else int(m.group(2)),
            "quote": m.group(3), "answer": answer}


def answer(net, tok, passages: list[str], question: str, tau: float = TAU, copy: bool = True,
           device: torch.device | None = None) -> dict:
    """One grounded answer: {'decision': 'answer'|'refuse', 's', 'passage', 'quote', 'answer', 'raw'}."""
    if not 0 <= len(passages) <= K_PASSAGES:
        raise ValueError(f"at most {K_PASSAGES} passages")
    device = device or next(net.parameters()).device
    o = generate_batch(net, tok, [format_prompt(passages, question)], [len(passages)], device, tau=tau,
                       batch=1, copy_passages=[passages] if copy else None)[0]
    p = parse_output(o["text"])
    return {"decision": "refuse" if o["choice"] == "none" else "answer", "s": o["s"],
            "passage": p["k"], "quote": p["quote"], "answer": p["answer"] if p["valid"] or p["answer"] else None,
            "valid": p["valid"], "raw": "Source:" + o["text"]}
