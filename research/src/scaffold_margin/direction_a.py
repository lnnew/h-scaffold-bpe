"""Direction A: marginal-utility replacement-token selection for Scaffold-BPE.

Idea (kept deliberately small): Scaffold-BPE frees vocabulary slots and refills them with the
next most *frequent* merges.  Frequency is a stand-in for "how many bits does this slot save".
Here we replace that stand-in by a model-based estimate of the same quantity, leaving the
merge table, scaffold machinery and encoding procedure untouched.

Pool construction
  * Run Scaffold-BPE to K_pool = K + P normal tokens.  The expanded merge table is kept.
  * Choose which K of the pool's normal tokens stay in the model vocabulary; every other
    pool token becomes scaffold (kept in the merge table, demolished at encoding).
    Any subset works because demolition always bottoms out in bytes.

Token value (first-order, per token t, over a held-out scoring corpus)
    V(t) = sum over occurrences of [NLL(text with t demolished into pieces) - NLL(text with t)]
         ~ seen(t) * mean_delta_bits(t)       (empirical-Bayes shrunk when few occurrences)
  Frequency-ranking is the special case mean_delta_bits(t) = const.  Compression-ranking
  uses mean_delta_bits(t) = const * (pieces(t) - 1).  The LM-ranking additionally sees how
  hard the neighbouring tokens become to predict.

First-order means: interactions between simultaneously removed tokens are ignored (a removed
token's pieces may themselves be removed).  Teacher bias applies: see utility.py.
"""
from __future__ import annotations

import numpy as np

from .bpe import Tokenizer, TrainResult, assign_levels
from .utility import CandidateAlts, collect_edits, score_edits, summarize


def pool_tokenizer(res: TrainResult, k_pool: int) -> Tokenizer:
    lv = assign_levels(res, tau=0.5, target_normal=k_pool)
    lv = np.where(lv > 0, 2, 0).astype(np.int8)
    meta = dict(kind="apool", k_pool=k_pool, q=res.q, merge_freq=res.merge_freq, resid=res.resid.tolist())
    return Tokenizer(res.merges, lv, mode="scaffold", meta=meta)


def with_selected(pool: Tokenizer, keep_ids, kind: str) -> Tokenizer:
    """Same merge table; only `keep_ids` (+ bytes) are normal.  Everything else is scaffold."""
    lv = np.full(len(pool.levels), 2, dtype=np.int8)
    lv[:256] = 0
    lv[list(keep_ids)] = 0
    return Tokenizer(pool.merges, lv, mode="scaffold", pattern=pool.pattern, meta=dict(pool.meta, kind=kind))


def n_pieces(pool: Tokenizer, t: int) -> int:
    """Number of normal tokens t decomposes into when removed from the pool vocabulary."""
    st, n = list(pool.merges[t - 256]), 0
    while st:
        x = st.pop()
        if pool.levels[x] >= 1:
            a, b = pool.merges[x - 256]
            st += [a, b]
        else:
            n += 1
    return n


def pool_candidates(pool: Tokenizer) -> np.ndarray:
    return np.nonzero(pool.levels[256:] == 0)[0] + 256


def token_value_labels(pool: Tokenizer, dev_words: list, scorer, ctx_words=12, max_per_token=16, chunk=20000):
    """Bits SAVED by keeping each pool-normal token, estimated on `dev_words` (list of pretoken
    lists): mean over sampled occurrences of NLL(token demolished) - NLL(token kept).
    Positive = the token is worth its slot.  (collect_edits measures before-after with
    before = token kept, so the sign is flipped here.)  Returns (summary dict, EditSet.seen)."""
    cands = set(int(t) for t in pool_candidates(pool))
    index: dict = {}
    for w in {w for ws in dev_words for w in ws}:
        for t in set(pool._merge_loop(list(w))):
            if t in cands:
                index.setdefault(t, []).append(w)
    alts = CandidateAlts()
    memo: dict = {}
    for t, ws in index.items():
        def fn(w, t=t):
            k = (w, t)
            if k not in memo:
                memo[k] = tuple(int(x) for x in pool.to_model[pool.encode_ablated(w, {t})])
            return memo[k]
        alts.add(t, fn, ws)
    edits = collect_edits(dev_words, pool.encode_word, alts, ctx_words=ctx_words, max_per_key=max_per_token)
    deltas = score_edits(scorer, edits, pool.eos_id, chunk=chunk)
    return summarize({t: -v for t, v in deltas.items()}), edits.seen
