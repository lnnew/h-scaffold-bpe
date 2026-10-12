"""Alternative inference over a FIXED visible vocabulary (survey §3.6 "general inference algorithms").

Scaffold-BPE encodes by full expanded-vocabulary BPE followed by demolition of scaffold tokens.  The survey
(Cognetta et al. 2026, §3.1.3) notes that this can leave segmentations that the visible vocabulary could
encode with fewer tokens.  These segmenters keep the visible vocabulary (and model ids) of a Tokenizer and
change only how a pretoken is segmented:

  canonical  the tokenizer's own encoding (BPE / Scaffold demolition / H-Scaffold re-merge)
  longest    left-to-right maximal munch over visible tokens (MaxMatch/FLOTA-style greedy)
  dp         minimum number of visible tokens (PathPiece-style); ties -> longest first token, recursively
             (deterministic: the lexicographically longest-first minimal segmentation)
  constrained  minimise  #tokens + lam * #boundary changes vs canonical  (lam=0 -> dp count, lam>=1 -> canonical);
             a continuum between "change nothing" (canonical / H-Scaffold-like) and "maximal compression" (dp)

All three produce model ids in the same id space, so LM streams are directly comparable.
"""
from __future__ import annotations

import numpy as np

from .bpe import Tokenizer
from .pretok import pretokenize

METHODS = ("canonical", "longest", "dp")
ALL_METHODS = METHODS + ("constrained",)


class VisibleSegmenter:
    def __init__(self, tok: Tokenizer, method: str, lam: float = 0.0):
        assert method in ALL_METHODS
        self.tok, self.method, self.lam = tok, method, lam
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

    def _constrained(self, w: bytes) -> list:
        """Exact DP over (cost = tokens + lam * boundary symmetric difference vs canonical); ties -> fewer changes,
        then longest token ending at each position (deterministic)."""
        n, V, L, lam = len(w), self.vocab, self.maxlen, self.lam
        canon = self.tok.encode_word(w)
        cb, pos = set(), 0
        for t in canon[:-1]:
            pos += len(self.tok.token_bytes[self.tok.from_model[t]])
            cb.add(pos)
        inner = [0] * (n + 1)                    # inner[j] = #canonical boundaries < j  (prefix count)
        for j in range(1, n + 1):
            inner[j] = inner[j - 1] + ((j - 1) in cb and j - 1 > 0)
        INF = (float("inf"), 0)
        best = [(0.0, 0)] + [INF] * n            # (cost, changes)
        back = [0] * (n + 1)
        for j in range(1, n + 1):
            end_change = 0 if (j == n or j in cb) else 1
            for i in range(j - 1, max(0, j - L) - 1, -1):
                if best[i][0] == INF[0] or w[i:j] not in V:
                    continue
                removed = inner[j] - inner[i + 1]          # canonical boundaries strictly inside (i, j)
                ch = best[i][1] + removed + end_change
                cand = (best[i][0] + 1 + lam * (removed + end_change), ch)
                if cand < best[j]:
                    best[j], back[j] = cand, i
        out, j = [], n
        while j > 0:
            i = back[j]
            out.append(V[w[i:j]])
            j = i
        return out[::-1]

    def encode_word(self, w: bytes) -> tuple:
        hit = self._cache.get(w)
        if hit is not None:
            return hit
        if self.method == "canonical":
            ids = self.tok.encode_word(w)
        elif self.method == "longest":
            ids = self._longest(w)
        elif self.method == "constrained":
            ids = self._constrained(w)
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


class CausalScaffoldFeatures:
    """Input-only feature for Scaffold-BPE: at position t, the id of the longest scaffold (hidden, level>=1) token whose
    demolition equals EXACTLY the last k>=2 model tokens ending at t.  It is a function of tokens <= t only (no access
    to future tokens), so it adds no information about the prediction target beyond the observed prefix.

    ids: 0 = none, 1..S = scaffold tokens (sorted by expanded id).  `perm` (control) maps them to a fixed random
    permutation, keeping where features fire and the parameter count but destroying their identity."""

    def __init__(self, tok: Tokenizer, shuffle_seed=None):
        hidden = [t for t in range(256, len(tok.levels)) if tok.levels[t] >= 1]
        self.size = len(hidden) + 1
        self.table = {}
        for k, t in enumerate(hidden, 1):
            leaves = tuple(int(x) for x in tok.to_model[tok._demolish([t], 1)])
            if len(leaves) >= 2:
                prev = self.table.get(leaves)
                if prev is None or len(tok.token_bytes[hidden[prev - 1]]) < len(tok.token_bytes[t]):
                    self.table[leaves] = k
        self.maxk = max(map(len, self.table)) if self.table else 0
        self.last = {leaves[-1] for leaves in self.table}
        self.perm = np.arange(self.size)
        if shuffle_seed is not None:
            self.perm[1:] = np.random.default_rng(shuffle_seed).permutation(np.arange(1, self.size))

    def features(self, ids) -> list:
        out = [0] * len(ids)
        T, last, maxk = self.table, self.last, self.maxk
        for t in range(1, len(ids)):
            if ids[t] not in last:
                continue
            for k in range(min(maxk, t + 1), 1, -1):
                v = T.get(tuple(ids[t - k + 1 : t + 1]))
                if v is not None:
                    out[t] = int(self.perm[v])
                    break
        return out
