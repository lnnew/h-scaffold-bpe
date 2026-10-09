"""Re-merge event extraction (Phase 0 audit + Phase 1 candidate dataset).

Definitions (all token-weighted: a pretoken type counts as often as it occurs):
  attempted : full-BPE output contains an L2 token, so L2 demolition + a second pass ran
  accepted  : the second pass performed >= 1 merge
  changed   : final H-Scaffold output differs from Scaffold-BPE output
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def remerge_provenance(h, word: bytes) -> list:
    """Second-pass merges of one pretoken with provenance.  Returns [(a, b, c, cross_boundary)].

    cross_boundary (the report's 'Cross-boundary' column): the merged token is built from pieces of
    >= 2 different full-BPE tokens, i.e. the re-merge creates a boundary the first pass could not.
    Final ids are identical to `Tokenizer.trace(word)['pass2']` (tested)."""
    full = h._merge_loop(list(word))
    ids, origin = [], []
    for i, t in enumerate(full):
        pieces = h._demolish([t], 2)
        ids += pieces
        origin += [frozenset([i])] * len(pieces)
    out = []
    if len(ids) == len(full):          # nothing demolished -> nothing can re-merge
        return out
    ok, rank, merges = h._remerge_ok, h.rank, h.merges
    while len(ids) > 1:
        best = None
        for i in range(len(ids) - 1):
            r = rank.get((ids[i], ids[i + 1]))
            if r is not None and ok[r] and (best is None or r < best):
                best = r
        if best is None:
            break
        a, b = merges[best]
        c = 256 + best
        n_ids, n_or, i = [], [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                u = origin[i] | origin[i + 1]
                out.append((a, b, c, len(u) > 1))
                n_ids.append(c); n_or.append(u); i += 2
            else:
                n_ids.append(ids[i]); n_or.append(origin[i]); i += 1
        ids, origin = n_ids, n_or
    return out


def event_stats(h, word_counts: dict) -> dict:
    """h: hscaffold Tokenizer.  Returns corpus-level attempted/accepted/changed rates."""
    tot_words = att = acc = chg = 0
    n_merge_events = n_cross_events = 0
    tot_tok_s = tot_tok_h = 0
    n_types = att_t = acc_t = chg_t = 0
    for w, c in word_counts.items():
        tr = h.trace(w)
        s = h.to_model[tr["scaffold"]]
        hh = h.to_model[tr["hscaffold"]]
        tot_words += c
        n_types += 1
        tot_tok_s += c * len(s)
        tot_tok_h += c * len(hh)
        a = tr["has_l2"]
        b = bool(tr["remerges"])
        if b:
            prov = remerge_provenance(h, w)
            n_merge_events += c * len(prov)
            n_cross_events += c * sum(1 for x in prov if x[3])
        d = not np.array_equal(s, hh)
        att += c * a; acc += c * b; chg += c * d
        att_t += a; acc_t += b; chg_t += d
    return dict(
        pretokens=tot_words, types=n_types,
        attempted_rate=att / tot_words, accepted_rate=acc / tot_words, changed_rate=chg / tot_words,
        attempted_types=att_t, accepted_types=acc_t, changed_types=chg_t,
        merge_events=n_merge_events, cross_boundary_events=n_cross_events,
        tokens_scaffold=tot_tok_s, tokens_hscaffold=tot_tok_h,
        fertility_delta_pct=100.0 * (tot_tok_h - tot_tok_s) / tot_tok_s,
    )


def level_counts(h) -> dict:
    return {f"L{l}": int((h.levels[256:] == l).sum()) for l in range(3)}


def extract_rules(h, word_counts: dict, max_examples: int = 4) -> dict:
    """Candidate re-merge rules (a,b)->c that fire in pass 2 on `word_counts`.

    Per-rule marginal effect is measured by enabling *only that rule* (singleton
    allow-list) and comparing with Scaffold-BPE's output of the same pretoken.  A rule
    that only fires after another re-merge shows no effect in isolation
    (`chain_only`).
    """
    rules: dict = {}
    r_resid = np.array(h.meta["resid"]) if "resid" in h.meta else None
    ratio = np.array(h.meta["ratio"]) if "ratio" in h.meta else None
    mfreq = h.meta.get("merge_freq")
    q = h.meta.get("q", 0)
    for w, cnt in word_counts.items():
        tr = h.trace(w)
        if not tr["remerges"]:
            continue
        base = tr["scaffold"]
        l2_ratio = float("nan")
        if ratio is not None:
            l2 = [ratio[t] for t in tr["full"] if h.levels[t] == 2]
            if l2:
                l2_ratio = float(np.mean(l2))   # r/f(Qhead) at the L2 token's last judgement
        for (a, b, c) in set(tr["remerges"]):
            st = rules.setdefault((a, b), dict(
                a=a, b=b, c=c, n_occ=0, n_types=0, dtok_sum=0, changed=0, boundary_only=0,
                capitalized=0, l2_ratio_sum=0.0, examples=[]))
            alt = h.encode_with_rules(w, [(a, b)])
            d = len(alt) - len(base)
            ch = alt != base
            st["n_occ"] += cnt
            st["n_types"] += 1
            st["dtok_sum"] += d * cnt
            st["changed"] += cnt * ch
            st["boundary_only"] += cnt * (ch and d == 0)
            st["capitalized"] += cnt * w.lstrip(b" ")[:1].isupper()
            if l2_ratio == l2_ratio:
                st["l2_ratio_sum"] += l2_ratio * cnt
            if len(st["examples"]) < max_examples:
                st["examples"].append(w.decode("utf-8", "replace"))
    out = {}
    for k, st in rules.items():
        n = st["n_occ"]
        c = st["c"]
        rc = float(r_resid[c]) if r_resid is not None else float("nan")
        mf = float(mfreq[c - 256]) if mfreq is not None else float("nan")
        out[k] = dict(
            a=st["a"], b=st["b"], c=c, c_level=int(h.levels[c]),
            c_text=h.token_bytes[c].decode("utf-8", "replace"),
            n_occ=n, n_types=st["n_types"],
            log_n_occ=float(np.log1p(n)),
            log_r_c=float(np.log1p(rc)) if rc == rc else float("nan"),
            collapse=rc / mf if mf == mf and mf > 0 else float("nan"),
            merge_rank=c - 256,
            c_len=len(h.token_bytes[c]),
            dtok_per_event=st["dtok_sum"] / n,
            changed_frac=st["changed"] / n,
            boundary_only_frac=st["boundary_only"] / n,
            capitalized_frac=st["capitalized"] / n,
            l2_ratio=st["l2_ratio_sum"] / n if st["l2_ratio_sum"] else float("nan"),
            chain_only=bool(st["changed"] == 0),
            examples=st["examples"],
        )
    return out


def build_rule_alts(h, rule_keys, words):
    """CandidateAlts mapping each rule (a,b) -> fn(pretoken) giving the model-id tuple obtained
    when *only that rule* is enabled, restricted (via an inverted index) to pretokens in
    `words` on which the rule fires under full H-Scaffold."""
    from .utility import CandidateAlts

    keyset = set(rule_keys)
    fire: dict = {}
    for w in set(words):
        for (a, b, _c) in set(h.trace(w)["remerges"]):
            if (a, b) in keyset:
                fire.setdefault((a, b), []).append(w)
    alts = CandidateAlts()
    for key, ws in fire.items():
        alts.add(key, lambda w, key=key: tuple(int(x) for x in h.to_model[h.encode_with_rules(w, [key])]), ws)
    return alts
