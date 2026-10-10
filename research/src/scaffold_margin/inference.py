"""Alternative inference over a FIXED visible vocabulary (survey §3.6 "general inference algorithms").

Scaffold-BPE encodes by full expanded-vocabulary BPE followed by demolition of scaffold tokens.  The survey
(Cognetta et al. 2026, §3.1.3) notes that this can leave segmentations that the visible vocabulary could
encode with fewer tokens.  These segmenters keep the visible vocabulary (and model ids) of a Tokenizer and
change only how a pretoken is segmented:

  canonical  the tokenizer's own encoding (BPE / Scaffold demolition / H-Scaffold re-merge)
  longest    left-to-right maximal munch over visible tokens (MaxMatch/FLOTA-style greedy)
  dp         minimum number of visible tokens (PathPiece-style); ties -> longest first token, recursively
             (deterministic: the lexicographically longest-first minimal segmentation)

All three produce model ids in the same id space, so LM streams are directly comparable.
"""
from __future__ import annotations

import numpy as np

from .bpe import Tokenizer
from .pretok import pretokenize

METHODS = ("canonical", "longest", "dp")


class VisibleSegmenter:
    def __init__(self, tok: Tokenizer, method: str):
        assert method in METHODS
        self.tok, self.method = tok, method
        vis = np.nonzero(tok.levels == 0)[0]
        self.vocab = {}
        for t in vis:
            b = tok.token_bytes[t]
            if b in self.vocab:
                raise ValueError(f"two visible tokens share surface bytes {b!r}")
            self.vocab[b] = int(tok.to_model[t])
        self.maxlen = max(map(len, self.vocab))
        self._cache: dict = {}

    # -- per pretoken -------------------------------------------------------------------
    def _longest(self, w: bytes) -> list:
        out, i, n, V, L = [], 0, len(w), self.vocab, self.maxlen
        while i < n:
            for j in range(min(n, i + L), i, -1):
                v = V.get(w[i:j])
                if v is not None:
                    out.append(v)
                    i = j
                    break
        return out

    def _dp(self, w: bytes) -> list:
        n, V, L = len(w), self.vocab, self.maxlen
        suf = [0] * (n + 1)                       # suf[i] = min #tokens for w[i:]
        for i in range(n - 1, -1, -1):
            best = 1 << 30
            for j in range(i + 1, min(n, i + L) + 1):
                if suf[j] + 1 < best and w[i:j] in V:
                    best = suf[j] + 1
            suf[i] = best
        out, i = [], 0
        while i < n:                              # longest first token consistent with optimality
            for j in range(min(n, i + L), i, -1):
                if suf[j] + 1 == suf[i] and w[i:j] in V:
                    out.append(V[w[i:j]])
                    i = j
                    break
        return out

    def encode_word(self, w: bytes) -> tuple:
        hit = self._cache.get(w)
        if hit is not None:
            return hit
        if self.method == "canonical":
            ids = self.tok.encode_word(w)
        elif self.method == "longest":
            ids = self._longest(w)
        else:
            ids = self._dp(w)
        ids = tuple(ids)
        self._cache[w] = ids
        return ids

    def encode(self, text: str) -> list:
        out: list = []
        for w in pretokenize(text, self.tok.pattern):
            out.extend(self.encode_word(w))
        return out
