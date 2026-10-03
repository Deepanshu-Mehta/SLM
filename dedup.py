"""Pure helpers for Phase 2 (dedup + contamination strip)."""

from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def words(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


def exact_hash(text: str) -> str:
    return hashlib.blake2b(normalize(text).encode("utf-8"), digest_size=16).hexdigest()


def word_ngrams(tokens: list[str], n: int) -> set[int]:
    """Fast native hash of word n-grams (contam set and doc grams share a process)."""
    if len(tokens) < n:
        return set()
    return {hash(tuple(tokens[i : i + n])) for i in range(len(tokens) - n + 1)}


def shingles(tokens: list[str], k: int) -> set[bytes]:
    if len(tokens) < k:
        return {" ".join(tokens).encode("utf-8")} if tokens else set()
    return {" ".join(tokens[i : i + k]).encode("utf-8") for i in range(len(tokens) - k + 1)}
