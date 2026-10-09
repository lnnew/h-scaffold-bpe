"""Tiny synthetic corpora for smoke tests (no network, no datasets needed).

Words are built from a Zipf-distributed morpheme inventory so that BPE produces the
nested intermediate tokens (the 'Ari'+'zona' situation) that Scaffold-BPE targets.
"""
from __future__ import annotations

import itertools
import random

_SYL = ["ka", "to", "ri", "ma", "ne", "so", "lu", "pe", "da", "vi", "zo", "ar", "en", "ok", "ie",
        "th", "sh", "ing", "tion", "er", "co", "ro", "ax", "um", "on", "al", "is", "ex"]


def make_corpus(n_docs=300, words_per_doc=120, n_stems=600, seed=0, alphabet=None, stem_seed=None):
    """`stem_seed` fixes the word inventory independently of the document sampling `seed`, so that
    train/dev/test splits can share vocabulary (default: stems depend on `seed`, as before)."""
    rng = random.Random(seed)
    rs = rng if stem_seed is None else random.Random(stem_seed)
    syl = alphabet or _SYL
    stems = ["".join(rs.choice(syl) for _ in range(rs.randint(1, 4))) for _ in range(n_stems)]
    suffixes = ["", "", "", "s", "ed", "ing", "er", "ly", "ion"]
    cum = list(itertools.accumulate(1.0 / (i + 1) for i in range(n_stems)))
    docs = []
    for _ in range(n_docs):
        ws = []
        for _ in range(words_per_doc):
            s = rng.choices(stems, cum_weights=cum)[0] + rng.choice(suffixes)
            ws.append(s)
        docs.append(" ".join(ws) + ".\n")
    return docs
