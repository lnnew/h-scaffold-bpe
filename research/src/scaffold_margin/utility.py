"""Model-oriented utility estimation (shared by Directions A and C).

The common primitive is an *edit*: the same raw text window represented by two token
sequences, `before` and `after`.  Under a teacher LM,

    delta_bits = (NLL(before) - NLL(after)) / ln 2        (positive: `after` is better)

Both sequences cover identical bytes, so delta_bits is the change in the numerator of
bits-per-byte (the denominator is unchanged).  Windows are built from whole pretokens
(`ctx_words` on either side of the edited pretoken), so the edit is local and the
rest of the window is shared.

Caveat that every consumer must keep in mind: this is a *proxy*.  A fixed teacher
trained on one segmentation can prefer that segmentation (teacher bias), and a
likelihood change under a fixed model is not the BPB change after retraining.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Iterable

import numpy as np

from .pretok import pretokenize

LN2 = math.log(2.0)


def doc_word_lists(docs: Iterable[str], pattern: str = "margin") -> list:
    return [pretokenize(d, pattern) for d in docs]


class EditSet(dict):
    """{key: [(before, after)]} plus `.seen[key]` = total occurrences before reservoir sampling."""

    seen: dict


def collect_edits(
    docs_words: list,
    base_enc: Callable[[bytes], tuple],
    alt_fns: dict,
    ctx_words: int = 12,
    max_per_key: int = 32,
    seed: int = 0,
    min_ctx_words: int = 0,
) -> dict:
    """Build (before, after) windows.

    base_enc : pretoken -> tuple of model ids under the *reference* tokenizer.
    alt_fns  : {key: fn(word) -> tuple|None}; fn returns the alternative model-id tuple for
               that pretoken, or None if the key does not apply to this pretoken.
               A shared pre-filter keeps this cheap: keys are tried only for words in
               `alt_fns.candidates[word]` when that attribute exists (see `index_candidates`).
    Returns {key: list[(before_ids, after_ids)]} with at most `max_per_key` windows per key
    (reservoir-sampled, so no ordering bias).
    """
    rng = np.random.default_rng(seed)
    seen = defaultdict(int)
    out = EditSet()
    cand = getattr(alt_fns, "candidates", None)
    for words in docs_words:
        if len(words) < 1 + min_ctx_words:
            continue
        enc = [base_enc(w) for w in words]
        for i, w in enumerate(words):
            keys = cand.get(w) if cand is not None else alt_fns.keys()
            if not keys:
                continue
            for k in keys:
                alt = alt_fns[k](w)
                if alt is None or alt == enc[i]:
                    continue
                seen[k] += 1
                n = seen[k]
                lst = out.setdefault(k, [])
                if len(lst) < max_per_key:
                    slot = len(lst)
                else:
                    slot = int(rng.integers(0, n))
                    if slot >= max_per_key:
                        continue  # reservoir says: drop; avoid building the window at all
                lo, hi = max(0, i - ctx_words), min(len(words), i + ctx_words + 1)
                before = [t for e in enc[lo:hi] for t in e]
                after = [t for e in enc[lo:i] for t in e] + list(alt) + [t for e in enc[i + 1 : hi] for t in e]
                if slot == len(lst):
                    lst.append((before, after))
                else:
                    lst[slot] = (before, after)
    out.seen = dict(seen)
    return out


class CandidateAlts(dict):
    """dict key -> fn plus an inverted index word -> keys (for fast pre-filtering)."""

    def __init__(self):
        super().__init__()
        self.candidates: dict = {}

    def add(self, key, fn, words: Iterable[bytes]):
        self[key] = fn
        for w in words:
            self.candidates.setdefault(w, []).append(key)


def score_edits(scorer, edits: dict, eos_id: int, chunk: int = 20000) -> dict:
    """{key: [(before, after)]} -> {key: np.array of delta_bits}."""
    flat, owners = [], []
    for k, lst in edits.items():
        for b, a in lst:
            flat.append(b)
            flat.append(a)
            owners.append(k)
    deltas = []
    for i in range(0, len(flat), chunk):
        nll = scorer.score(flat[i : i + chunk], eos_id)
        deltas.append(nll)
    nll = np.concatenate(deltas) if deltas else np.zeros(0)
    d = (nll[0::2] - nll[1::2]) / LN2
    res: dict = defaultdict(list)
    for k, v in zip(owners, d):
        res[k].append(v)
    return {k: np.array(v) for k, v in res.items()}


def summarize(deltas: dict, shrink_k: float = 4.0) -> dict:
    """Per-key mean / sem / n and an empirical-Bayes shrunk mean (toward the global mean)."""
    allv = np.concatenate(list(deltas.values())) if deltas else np.zeros(1)
    g = float(allv.mean()) if len(allv) else 0.0
    out = {}
    for k, v in deltas.items():
        n = len(v)
        m = float(v.mean())
        out[k] = dict(n=n, mean=m, sem=float(v.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
                      shrunk=(n * m + shrink_k * g) / (n + shrink_k), frac_pos=float((v > 0).mean()))
    return out


# ---- predictability analysis ------------------------------------------------------------
def predictability(features: np.ndarray, names: list, labels: np.ndarray, seed: int = 0, folds: int = 5) -> dict:
    """Does a cheap feature vector predict the LM-based utility label?

    * per-feature Spearman correlation with the label
    * cross-validated (rule-level, out-of-fold) Spearman and AUC (label > 0) of a
      standardised ridge/logistic combination
    Returns plain dicts; callers decide whether the signal justifies a learned selector.
    """
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    ok = np.isfinite(features).all(1) & np.isfinite(labels)
    X, y = features[ok], labels[ok]
    res: dict = dict(n=int(ok.sum()), per_feature={}, cv={})
    for j, nm in enumerate(names):
        if X[:, j].std() == 0:
            continue
        rho, p = spearmanr(X[:, j], y)
        res["per_feature"][nm] = dict(rho=float(rho), p=float(p))
    if len(y) < 20:
        res["cv"] = dict(note="too few rules for a learned selector (n<20)")
        return res
    oof_r = np.zeros(len(y))
    oof_p = np.zeros(len(y))
    kf = KFold(folds, shuffle=True, random_state=seed)
    yb = (y > 0).astype(int)
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        oof_r[te] = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
        if yb[tr].min() != yb[tr].max():
            oof_p[te] = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X[tr]), yb[tr]).predict_proba(sc.transform(X[te]))[:, 1]
        else:
            oof_p[te] = yb[tr].mean()
    res["cv"]["spearman"] = float(spearmanr(oof_r, y)[0])
    res["cv"]["auc"] = float(roc_auc_score(yb, oof_p)) if yb.min() != yb.max() else float("nan")
    res["cv"]["base_rate_pos"] = float(yb.mean())
    res["oof_ridge"] = oof_r
    res["ok_mask"] = ok
    return res


def pick_matched(keys: list, scores: np.ndarray, m: int) -> list:
    """Top-m keys by score (ties broken by key order)."""
    order = np.argsort(-scores, kind="stable")[:m]
    return [keys[i] for i in order]


def pick_random(keys: list, m: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    return [keys[i] for i in rng.choice(len(keys), size=min(m, len(keys)), replace=False)]


def pick_random_traffic(keys: list, n_occ: np.ndarray, target: float, seed: int) -> list:
    """Random rule subset whose total corpus traffic (sum n_occ) first reaches `target`.

    Rule-count matching alone is confounded: frequency-based selection picks high-traffic
    rules, so it changes many more tokens than a random subset of equal size.  This control
    matches the *amount of segmentation change* instead."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    out, tot = [], 0.0
    for i in order:
        if tot >= target:
            break
        out.append(keys[i])
        tot += n_occ[i]
    return out
