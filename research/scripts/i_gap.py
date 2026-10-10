#!/usr/bin/env python
"""E1 inference gap + E0 leaf-vs-hide (tokenizer-only, CPU).

E1  For the Scaffold visible vocabulary, how many tokens do canonical Scaffold (demolish), H-Scaffold
    (L2 re-merge), longest-match and minimum-token DP need?  How much of the Scaffold->DP gap does
    H-Scaffold close?  BPE (its own vocabulary) is reported for scale.
E0  Freeing k visible slots from a BPE vocabulary: leaf pruning (Purason et al. 2026) vs hiding
    (Scaffold-style demolition of the k least-exposed tokens).  Exposure = BPE emission counts on the
    tokenizer-training word counts; compression measured on held-out val/test.

Usage: i_gap.py --tok_dir DIR --text_dir DIR --word_counts PKL --out JSON [--K 32768 65536]
"""
import argparse
import heapq
import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scaffold_margin.adapter import find_tokenizer, load_team_tokenizer  # noqa: E402
from scaffold_margin.data import read_jsonl  # noqa: E402
from scaffold_margin.inference import VisibleSegmenter  # noqa: E402
from scaffold_margin.pretok import pretokenize  # noqa: E402


def split_words(text_dir, split, pattern):
    c = Counter()
    for t, _ in read_jsonl(os.path.join(text_dir, f"{split}.jsonl")):
        c.update(pretokenize(t, pattern))
    return c


def e1(K, a, splits):
    path = lambda n: find_tokenizer(a.tok_dir, n, K)  # noqa: E731
    bpe = load_team_tokenizer(path("bpe"))
    sc = load_team_tokenizer(path("scaffold"))
    hs = load_team_tokenizer(path("hscaffold"), second_pass="full")
    segs = {m: VisibleSegmenter(sc, m) for m in ("canonical", "longest", "dp")}
    res = {}
    for split, wc in splits.items():
        tot, n_better, n_worse_longest = Counter(), 0, 0
        examples = []
        for w, n in wc.items():
            c = dict(bpe=len(bpe.encode_word(w)), scaffold=len(segs["canonical"].encode_word(w)),
                     hscaffold=len(hs.encode_word(w)), longest=len(segs["longest"].encode_word(w)),
                     dp=len(segs["dp"].encode_word(w)))
            assert c["dp"] <= min(c["scaffold"], c["hscaffold"], c["longest"])
            for k, v in c.items():
                tot[k] += n * v
            if c["dp"] < c["scaffold"]:
                n_better += n
                examples.append((n, w, c))
            n_worse_longest += n * (c["longest"] > c["scaffold"])
        P = sum(wc.values())
        gap = tot["scaffold"] - tot["dp"]
        examples.sort(key=lambda x: -x[0])

        def show(seg, w):
            return [seg.tok.token_bytes[seg.tok.from_model[i]].decode("utf-8", "replace") for i in seg.encode_word(w)]

        res[split] = dict(
            pretokens=P, types=len(wc), tokens=dict(tot),
            rel_vs_scaffold={k: tot[k] / tot["scaffold"] - 1 for k in tot},
            pretokens_dp_better=n_better, pretokens_dp_better_frac=n_better / P,
            pretokens_longest_worse_frac=n_worse_longest / P,
            hscaffold_gap_closed=(tot["scaffold"] - tot["hscaffold"]) / gap if gap else None,
            top_examples=[dict(count=n, word=w.decode("utf-8", "replace"), scaffold=show(segs["canonical"], w),
                               dp=show(segs["dp"], w), hscaffold_len=c["hscaffold"]) for n, w, c in examples[:25]])
        r = res[split]["rel_vs_scaffold"]
        print(f"[E1 {K} {split}] " + " ".join(f"{k}={100*v:+.3f}%" for k, v in r.items())
              + f" dp_better={100*n_better/P:.3f}% hs_closed={res[split]['hscaffold_gap_closed']}", flush=True)
    return res


def e0(K, a, train_wc, splits, ks):
    bpe = load_team_tokenizer(find_tokenizer(a.tok_dir, "bpe", K), cache=False)
    M = len(bpe.merges)
    N = 256 + M
    memo = {}

    def full(w):
        r = memo.get(w)
        if r is None:
            r = memo[w] = bpe._merge_loop(list(w))
        return r

    exposure = np.zeros(N, np.int64)
    for w, n in train_wc.items():
        for t in full(w):
            exposure[t] += n
    childcount = np.zeros(N, np.int64)
    for x, y in bpe.merges:
        childcount[x] += 1
        childcount[y] += 1
    merge_tok = np.arange(256, N)
    key = lambda t: (exposure[t], -t)  # noqa: E731   least exposed first; ties -> later (rarer) merge

    def count(wc, ok=None, hidden=None):
        tot = 0
        for w, n in wc.items():
            ids = full(w) if ok is None else bpe._merge_loop(list(w), ok=ok)
            if hidden is not None:
                out, stack = 0, list(ids)
                while stack:
                    t = stack.pop()
                    if hidden[t]:
                        stack += list(bpe.merges[t - 256])
                    else:
                        out += 1
                tot += n * out
            else:
                tot += n * len(ids)
        return tot

    base = {s: count(wc) for s, wc in splits.items()}
    stats = dict(merge_tokens=M, leaves=int((childcount[256:] == 0).sum()),
                 zero_exposure=int((exposure[256:] == 0).sum()),
                 zero_exposure_nonleaf=int(((exposure[256:] == 0) & (childcount[256:] > 0)).sum()),
                 exposure_lt100=int((exposure[256:] < 100).sum()),
                 exposure_lt100_nonleaf=int(((exposure[256:] < 100) & (childcount[256:] > 0)).sum()))
    rows = []
    for k in ks:
        cc = childcount.copy()
        removed = np.zeros(N, bool)
        heap = [(key(t), t) for t in merge_tok if cc[t] == 0]
        heapq.heapify(heap)
        left = k
        while left and heap:
            _, t = heapq.heappop(heap)
            removed[t] = True
            left -= 1
            for c in bpe.merges[t - 256]:
                cc[c] -= 1
                if c >= 256 and cc[c] == 0:
                    heapq.heappush(heap, (key(c), c))
        hidden = np.zeros(N, bool)
        hidden[sorted(merge_tok, key=key)[:k]] = True
        for name, kw, sel in (("leaf", dict(ok=~removed[256:]), removed), ("hide", dict(hidden=hidden), hidden)):
            row = dict(k=k, method=name, nonleaf_frac=float((childcount[sel] > 0).mean()),
                       exposure_removed=int(exposure[sel].sum()),
                       **{f"{s}_rel": count(wc, **kw) / base[s] - 1 for s, wc in splits.items()})
            rows.append(row)
            print(f"[E0 {K}] " + json.dumps(row), flush=True)
    return dict(stats=stats, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok_dir", required=True)
    ap.add_argument("--text_dir", required=True)
    ap.add_argument("--word_counts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, nargs="+", default=[32768, 65536])
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--e0_k", type=int, nargs="+", default=[500, 1000, 2600, 5000, 10000])
    a = ap.parse_args()
    t0 = time.time()
    splits = {s: split_words(a.text_dir, s, "gpt2") for s in a.splits}
    train_wc = pickle.load(open(a.word_counts, "rb"))
    out = dict(args=vars(a), e1={}, e0={})
    for K in a.K:
        out["e1"][str(K)] = e1(K, a, splits)
        out["e0"][str(K)] = e0(K, a, train_wc, splits, a.e0_k)
        json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    out["seconds"] = time.time() - t0
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
