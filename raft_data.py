"""Pure, unit-tested helpers for Phase 7 data (docs/08-raft.md §4.1).

Nothing here touches Modal, the network or the Volume, so it runs the same locally and in
containers. Splits reuse the exact document rules of Phase 4 (`modal_app._is_val_doc`).
"""

from __future__ import annotations

import hashlib
import re
from bisect import bisect_left
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
