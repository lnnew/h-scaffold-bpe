"""Regex pre-tokenisation and word counting.

All tokenisers in this repo are byte-level: base tokens 0..255 are raw bytes, a
pretoken is a `bytes` object, and merges are learned inside pretokens only.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import regex

# GPT-2 style pattern, extended so that (a) combining marks stay attached to their
# letters (matters for Indic scripts) and (b) each CJK ideograph / kana is its own
# pretoken (otherwise a whole sentence is one pretoken).  Identical to GPT-2 on
# plain English text apart from the \p{M} addition.
PATTERN = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d"""
    r"""| ?[\p{Han}\p{Hiragana}\p{Katakana}]"""
    r"""| ?[\p{L}\p{M}]+| ?\p{N}+| ?[^\s\p{L}\p{M}\p{N}]+|\s+(?!\S)|\s+"""
)


# The team's (and GPT-2's) pattern, used when a tokenizer trained by `hscaffold_bpe` is loaded so that
# pretokenization is bit-identical to the one the tokenizer was trained with.
PATTERN_GPT2 = regex.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
PATTERNS = {"margin": PATTERN, "gpt2": PATTERN_GPT2}


def pretokenize(text: str, pattern: str = "margin") -> list[bytes]:
    return [m.encode("utf-8") for m in PATTERNS[pattern].findall(text)]


def count_words(texts: Iterable[str], pattern: str = "margin") -> Counter:
    c: Counter = Counter()
    for t in texts:
        c.update(pretokenize(t, pattern))
    return c
