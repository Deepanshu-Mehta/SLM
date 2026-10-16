"""Pure, unit-tested helpers for Phase 7 data (docs/08-raft.md §4.1).

Nothing here touches Modal, the network or the Volume, so it runs the same locally and in
containers. Splits reuse the exact document rules of Phase 4 (`modal_app._is_val_doc`).
"""

from __future__ import annotations

import hashlib
import re
from bisect import bisect_left
from collections import Counter
from typing import Iterable

import config
from cleaning import nonword_ratio

# ------------------------------------------------------------------------------ splits
def _hash_mod(text: str, mod: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % mod


def is_val_doc(text: str) -> bool:
    """Identical to modal_app._is_val_doc: the Phase 4 document-level split."""
    return _hash_mod(text, config.VAL_ONE_IN_N_DOCS) == 0


def is_dev_doc(text: str) -> bool:
    """Within val documents: dev vs test, by document (no document feeds both)."""
    return _hash_mod(text + "|dev", config.RAFT.dev_hash_mod) == 0


def in_pool_sample(text: str, rate: float) -> bool:
    """Deterministic document sample for the train distractor pool."""
    return _hash_mod(text + "|pool", 1_000_000) < int(rate * 1_000_000)


def doc_id(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


# --------------------------------------------------------------------- sentence split
# Words that end with "." without ending a sentence (legal + financial text).
_ABBREV = frozenset("""
v vs no nos inc corp co ltd llc lp plc u.s id ibid e.g i.e mr mrs ms dr st jr sr art sec secs
cf etc fig al jan feb mar apr jun jul aug sep sept oct nov dec app supp cir ct dist cal tex
fla ill mass mich ohio pa n.y n.j d.c para paras pp p ex rev stat ann ch cl n.d s.d e.d w.d
f l.ed s.ct u.s.c c.f.r fed reg dept govt assn bros mfg nat'l int'l
""".split())
_SENT_END = re.compile(r"[.!?][\"'”’)\]]*\s+(?=[\"“'(\[]?[A-Z0-9§])")
_LAST_WORD = re.compile(r"([A-Za-z][A-Za-z.']*)\.?[\"'”’)\]]*$")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences. Conservative: a wrong merge only lengthens a passage."""
    spans, start = [], 0
    for m in _SENT_END.finditer(text):
        before = text[start:m.start() + 1]
        w = _LAST_WORD.search(before)
        word = w.group(1).lower().rstrip(".") if w else ""
        if word in _ABBREV or (len(word) == 1 and word.isalpha()):   # "v.", initials "J."
            continue
        spans.append((start, m.end()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(a, b) for a, b in spans if text[a:b].strip()]


# ---------------------------------------------------------------------------- chunking
def chunk_document(text: str, tokenizer, min_tokens: int = config.RAFT.passage_min_tokens,
                   max_tokens: int = config.RAFT.passage_max_tokens) -> list[tuple[str, int]]:
    """Sentence-packed passages of [min_tokens, max_tokens] tokens: [(text, n_tokens)].

    One tokenizer pass with offsets gives token counts for any character span. Sentences
    longer than max_tokens are cut at token boundaries. A short tail merges into the
    previous passage when it fits, otherwise it is dropped.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    starts = [a for a, _ in enc["offset_mapping"]]

    def ntok(a: int, b: int) -> int:
        return bisect_left(starts, b) - bisect_left(starts, a)

    out: list[list] = []                 # [char_start, char_end, n_tokens]
    cur: list | None = None

    def flush() -> None:
        nonlocal cur
        if cur is None:
            return
        if cur[2] >= min_tokens:
            out.append(cur)
        elif out and out[-1][2] + cur[2] <= max_tokens and out[-1][1] <= cur[0]:
            out[-1] = [out[-1][0], cur[1], out[-1][2] + cur[2]]
        cur = None

    for a, b in sentence_spans(text):
        n = ntok(a, b)
        if n > max_tokens:               # hard-cut an over-long sentence
            flush()
            i0 = bisect_left(starts, a)
            i1 = bisect_left(starts, b)
            for i in range(i0, i1, max_tokens):
                j = min(i + max_tokens, i1)
                ca = starts[i]
                cb = starts[j] if j < len(starts) else b
                cur = [ca, min(cb, b), j - i]
                flush()
            continue
        if cur is not None and cur[2] + n > max_tokens:
            flush()
        cur = [a, b, n] if cur is None else [cur[0], b, cur[2] + n]
    flush()
    return [(text[a:b].strip(), n) for a, b, n in out if text[a:b].strip()]


# ------------------------------------------------------------------------ quality tags
_CITATION = re.compile(
    r"\b\d+\s+(?:U\.\s?S\.|S\.\s?Ct\.|L\.\s?Ed\.(?:\s?2d)?|F\.(?:\s?Supp\.)?(?:\s?\d[a-z]{1,2})?|"
    r"(?:N\.[EWY]\.|S\.[EW]\.|So\.|A\.|P\.|Cal\.(?:\s?App\.)?)(?:\s?\d[a-z]{1,2})?)\s+\d+")
_BOILERPLATE = re.compile(
    r"indicate by check mark|securities and exchange commission\s+washington|"
    r"forward[- ]looking statements|pursuant to the requirements of the securities|"
    r"table of contents|intentionally left blank|all rights reserved|commission file number|"
    r"incorporated (?:herein )?by reference to exhibit|large accelerated filer", re.IGNORECASE)


_STRICT_OCR = frozenset(s.name for s in config.DATA_MIX if s.strict_ocr)


def passage_tags(text: str, source: str) -> list[str]:
    """Reasons a passage is a poor *source* for questions (empty list = clean).

    The OCR check applies only to sources Phase 1 OCR-gated (case law). On SEC and web text
    the dictionary ratio flags names and jargon, not OCR damage.
    """
    tags = []
    nonspace = re.sub(r"\s+", "", text)
    n = max(1, len(nonspace))
    letters = sum(c.isalpha() for c in nonspace)
    digits = sum(c.isdigit() for c in nonspace)
    if letters / n < 0.6:
        tags.append("low_alpha")
    if digits / n > 0.15:
        tags.append("numeric")
    if len(_CITATION.findall(text)) >= 3:
        tags.append("citations")
    if _BOILERPLATE.search(text):
        tags.append("boilerplate")
    if source in _STRICT_OCR and nonword_ratio(text) > config.CLEAN.nonword_ratio_max:
        tags.append("ocr")
    return tags


def source_eligibility(tags: Iterable[str], source: str) -> str:
    """'clean' = a normal source passage; 'numeric' = SEC table/number passage kept on
    purpose (quota ~10% of SEC sources); '' = not a source (still fine as a distractor)."""
    tags = set(tags)
    if not tags:
        return "clean"
    if source == "sec" and tags <= {"low_alpha", "numeric"}:
        return "numeric"
    return ""


# ------------------------------------------------------------------------- n-gram sets
_WORD = re.compile(r"[a-z0-9]+")


def word_list(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def ngram_hash_list(words: list[str], n: int) -> list[int]:
    """Process-local hashes of word n-grams, in position order. Like dedup.word_ngrams,
    Python's hash() is salted per process: only compare hashes made in the same process."""
    return [hash(tuple(words[i:i + n])) for i in range(len(words) - n + 1)]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


# ================================================================ teacher-output checks
# docs/08-raft.md §5.2. Every function is model-free and returns a reason string on failure
# ("" = pass), so drop reasons can be counted per bucket and source.
_CANON = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'",
                        "“": '"', "”": '"', "„": '"', "‟": '"',
                        "–": "-", "—": "-", "‒": "-", "―": "-",
                        "−": "-", " ": " "})


def _canon_with_map(text: str) -> tuple[str, list[int]]:
    """Canonical form (NFKC, straight quotes, plain dashes, single spaces) and, for each
    canonical character, the index of the original character it came from."""
    import unicodedata

    out, idx = [], []
    prev_space = False
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch.translate(_CANON)).translate(_CANON):
            if c.isspace():
                if prev_space:
                    continue
                c, prev_space = " ", True
            else:
                prev_space = False
            out.append(c)
            idx.append(i)
    return "".join(out), idx


def canon(text: str) -> str:
    return _canon_with_map(text)[0].strip()


def recut_quote(quote: str, passage: str) -> tuple[str, tuple[int, int]] | None:
    """The exact passage substring matching `quote` after canonicalisation, with its
    character span in the passage; None if the quote is not verbatim."""
    q = canon(quote).strip(" \"'")
    if not q:
        return None
    cp, idx = _canon_with_map(passage)
    at = cp.find(q)
    if at < 0:
        return None
    a, b = idx[at], idx[at + len(q) - 1] + 1
    return passage[a:b], (a, b)


_HEDGE = re.compile(r"\b(not (?:stated|mentioned|provided|specified|given)|does not (?:say|state|"
                    r"specify|mention|indicate)|unclear|cannot be determined|unknown)\b", re.I)
_STOP = frozenset("""a an the of and or to in on at by for with from as is are was were be been
that this these those it its which who whom what when where how did does do has have had not
no yes than then into over under about after before between""".split())
FACTOID_TYPES = frozenset({"fact", "number_or_date", "party_or_entity", "definition"})


def _content(words: list[str]) -> list[str]:
    return [w for w in words if w not in _STOP]


def check_question(question: str, passage: str) -> str:
    words = question.split()
    if not 5 <= len(words) <= 40:
        return "question_length"
    # no run of >= 6 consecutive words copied from the passage, not counting name tokens
    toks = [None if (i > 0 and w[:1].isupper()) else w.lower().strip(".,;:?!()\"'")
            for i, w in enumerate(words)]
    pw = set(ngram_hash_list(word_list(passage), 6))
    run: list[str] = []
    for t in toks + [None]:
        if t is None or not t:
            if len(run) >= 6 and pw & set(ngram_hash_list(word_list(" ".join(run)), 6)):
                return "question_copies_passage"
            run = []
        else:
            run.append(t)
    return ""


def check_answer(answer: str, quote: str, passage: str, qtype: str) -> str:
    words = answer.split()
    if not 1 <= len(words) <= 30:
        return "answer_length"
    if _HEDGE.search(answer):
        return "answer_hedges"
    cp = canon(passage).lower()
    for i, w in enumerate(words):
        tok = w.strip(".,;:?!()\"'")
        if any(ch.isdigit() for ch in tok) and canon(tok).lower() not in cp:
            return "answer_number_not_in_passage"
        if (i > 0 and tok[:1].isupper() and tok.lower() not in ("yes", "no")
                and tok.lower() not in cp):
            return "answer_name_not_in_passage"
    if qtype in FACTOID_TYPES:
        content = _content(word_list(answer))
        if content:
            in_quote = set(word_list(quote))
            if sum(w in in_quote for w in content) / len(content) < 0.5:
                return "answer_not_in_quote"
    return ""


def check_g1_item(item: dict, passage: str) -> tuple[str, dict]:
    """A G1 question/answer: returns (reason, cleaned item with the re-cut quote and span)."""
    if not all(isinstance(item.get(k), str) and item[k].strip()
               for k in ("question", "answer", "quote", "type")):
        return "schema", {}
    cut = recut_quote(item["quote"], passage)
    if cut is None:
        return "quote_not_verbatim", {}
    quote, span = cut
    if len(quote.split()) > 40:
        return "quote_too_long", {}
    reason = (check_question(item["question"], passage)
              or check_answer(item["answer"], quote, passage, item["type"]))
    if reason:
        return reason, {}
    return "", {"question": item["question"].strip(), "answer": item["answer"].strip(),
                "quote": quote, "quote_span": span, "type": item["type"]}


def check_g2_item(item: dict, passage: str) -> tuple[str, dict]:
    q = item.get("question")
    if not isinstance(q, str) or not q.strip():
        return "no_question", {}
    reason = check_question(q, passage)
    if reason:
        return reason, {}
    names = {w.strip(".,;:?!()\"'").lower() for i, w in enumerate(q.split()) if i and w[:1].isupper()}
    cp = canon(passage).lower()
    if not any(n and n in cp for n in names) and len(set(_content(word_list(q)))
                                                      & set(word_list(passage))) < 2:
        return "near_miss_off_topic", {}
    return "", {"question": q.strip(), "missing_detail": str(item.get("missing_detail", ""))}


def check_g3_item(item: dict, passage: str, task: str) -> tuple[str, dict]:
    ins, resp = item.get("instruction"), item.get("response")
    if not (isinstance(ins, str) and isinstance(resp, str) and ins.strip() and resp.strip()):
        return "schema", {}
    if not re.search(r"\bthe passage\b", ins, re.I):
        return "no_passage_placeholder", {}
    out = {"instruction": ins.strip(), "response": resp.strip(), "task": task}
    if task == "summarize":
        if len(resp.split()) > 35:
            return "summary_too_long", {}
    elif task == "extract_list":
        cp = canon(passage).lower()
        items = [x.strip() for x in resp.split(";") if x.strip()]
        if not items or any(canon(x).lower().rstrip(".") not in cp for x in items):
            return "list_item_not_verbatim", {}
    elif task == "yes_no_check":
        m = re.match(r"\s*(Yes|No)\.\s*(.*)$", resp, re.S)
        if not m:
            return "yes_no_format", {}
        cut = recut_quote(item.get("quote") or m.group(2), passage)
        if cut is None or len(cut[0].split()) > 30:
            return "quote_not_verbatim", {}
        out.update({"yes_no": m.group(1), "quote": cut[0], "response": f"{m.group(1)}."})
    return "", out


def answers_agree(a: str, b: str) -> bool:
    """Numbers/names identical, or token-F1 >= 0.5 (checker vs writer answer)."""
    def digits(s: str) -> set[str]:
        return {canon(t).strip(".,;:()") for t in s.split() if any(c.isdigit() for c in t)}
    da, db = digits(a), digits(b)
    if da and da == db:
        return True
    wa, wb = _content(word_list(a)), _content(word_list(b))
    if not wa or not wb:
        return canon(a).lower() == canon(b).lower()
    common = sum((Counter(wa) & Counter(wb)).values())
    if common == 0:
        return False
    p, r = common / len(wb), common / len(wa)
    return 2 * p * r / (p + r) >= 0.5


def spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


# ======================================================================= student format
# docs/08-raft.md §4. `Source:` is the last prompt token; the target starts with the
# decision token (` none` or ` Passage`).
REFUSAL_TARGET = f" none\nAnswer: {config.RAFT.refusal}"


def format_prompt(passages: list[str], question: str | None = None, task: str | None = None) -> str:
    body = "\n".join(f"Passage {i}: {p}" for i, p in enumerate(passages, 1))
    ask = f"Question: {question}" if question is not None else f"Task: {task}"
    return "<|eos|><|user|>" + (body + "\n" if body else "") + ask + "<|assistant|>Source:"


def format_target(k: int | None, answer: str | None = None, quote: str | None = None) -> str:
    """k=None -> refusal. Quote is included for QA and yes/no tasks."""
    if k is None:
        return REFUSAL_TARGET + "<|eos|>"
    out = f" Passage {k}"
    if quote is not None:
        out += f'\nQuote: "{quote}"'
    return out + f"\nAnswer: {answer}<|eos|>"


def encode_example(tokenizer, prompt: str, target: str) -> tuple[list[int], list[int]]:
    """(input ids, labels) with labels = -100 on the prompt; equal to whole-string encoding."""
    p = tokenizer.encode(prompt, add_special_tokens=False)
    t = tokenizer.encode(target, add_special_tokens=False)
    return p + t, [-100] * len(p) + t
