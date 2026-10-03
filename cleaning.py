"""The fixed, rule-based, deterministic cleaning pipeline (pure functions)."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from config import CLEAN

_BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*form\s+10[-\s]?[kq]\b.*$",
        r"^\s*page\s+\d+\s+of\s+\d+\s*$",
        r"^\s*table\s+of\s+contents\s*$",
        r"^\s*/s/\s*.*$",
        r"^\s*all\s+rights\s+reserved.*$",
        r"^\s*united\s+states\s+securities\s+and\s+exchange\s+commission\s*$",
        r"^\s*securities\s+and\s+exchange\s+commission\s*$",
        r"^\s*washington,?\s+d\.?\s?c\.?\s+\d{5}\s*$",
        r"^\s*\[?\s*x\s*\]?\s*$",
    )
)

_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z]+")
_ALNUM = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True)
class CleanResult:
    kept: bool
    text: str
    reason: str
    raw_chars: int
    clean_chars: int


def _nonalnum_ratio(line: str) -> float:
    if not line:
        return 1.0
    alnum = sum(1 for c in line if _ALNUM.match(c))
    return 1.0 - alnum / len(line)


def filter_lines(text: str) -> str:
    out: list[str] = []
    for raw in text.splitlines():
        line = _WHITESPACE.sub(" ", raw).strip()
        if len(line) < CLEAN.min_line_chars:
            continue
        if _nonalnum_ratio(line) > CLEAN.max_nonalnum_ratio:
            continue
        out.append(line)
    return "\n".join(out)


def strip_boilerplate(text: str) -> str:
    return "\n".join(
        line
        for line in text.splitlines()
        if not any(p.match(line) for p in _BOILERPLATE_PATTERNS)
    )


def is_repetitive(text: str) -> bool:
    words = text.split()
    n = CLEAN.ngram_n
    if len(words) < n * 2:
        return False
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return False
    counts = Counter(grams)
    top = sum(c for _, c in counts.most_common(CLEAN.repetition_top_k))
    return top / len(grams) > CLEAN.max_repetition_ratio


def _ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) < 128) / len(text)


def is_english(text: str) -> bool:
    """ASCII-ratio first; langdetect only on the ambiguous 90-99% band."""
    sample = text[: CLEAN.lang_sample_chars]
    ratio = _ascii_ratio(sample)
    if ratio >= 0.99:
        return True
    if ratio < 0.90:
        return False
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(sample) == "en"
    except Exception:
        return ratio > 0.95


_OCR_TOKEN = re.compile(r"[A-Za-z]{3,}")
_ENGLISH_WORDS: frozenset[str] | None = None


def _english_words() -> frozenset[str]:
    global _ENGLISH_WORDS
    if _ENGLISH_WORDS is None:
        try:
            with open(CLEAN.dict_path, encoding="utf-8", errors="ignore") as fh:
                _ENGLISH_WORDS = frozenset(
                    w.strip().lower() for w in fh if w.strip().isalpha()
                )
        except OSError:
            _ENGLISH_WORDS = frozenset()
    return _ENGLISH_WORDS


def nonword_ratio(text: str) -> float:
    words = _english_words()
    if not words:
        return 0.0
    toks = [t.lower() for t in _OCR_TOKEN.findall(text)]
    if len(toks) < CLEAN.ocr_min_tokens:
        return 0.0
    nonword = sum(1 for t in toks if t not in words)
    return nonword / len(toks)


def is_ocr_garble(text: str) -> bool:
    return nonword_ratio(text) > CLEAN.nonword_ratio_max


def clean_document(text: str, *, strict_ocr: bool = False) -> CleanResult:
    """Run one document through the full deterministic chain."""
    raw_chars = len(text)
    step1 = filter_lines(text)
    step2 = strip_boilerplate(step1)
    if len(step2) < CLEAN.min_doc_chars:
        return CleanResult(False, "", "too_short", raw_chars, len(step2))
    if is_repetitive(step2):
        return CleanResult(False, "", "repetitive", raw_chars, len(step2))
    if not is_english(step2):
        return CleanResult(False, "", "non_english", raw_chars, len(step2))
    if strict_ocr and is_ocr_garble(step2):
        return CleanResult(False, "", "ocr", raw_chars, len(step2))
    return CleanResult(True, step2, "kept", raw_chars, len(step2))
