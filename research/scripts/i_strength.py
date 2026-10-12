#!/usr/bin/env python
"""Scaffold strength sweep: train one tokenizer with scaffold factor c (c=0 -> plain BPE) and report intrinsic stats.

A token is scaffold iff f(x) < c * f(Qhead) (c=1 = Scaffold-BPE).  Visible vocabulary size is fixed at --K.
Writes <out>/tok_c<c>.json and <out>/tok_c<c>.stats.json (scaffold count, merges, token counts on val/test,
training-exposure histogram of visible tokens).
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scaffold_margin.bpe import Tokenizer, assign_levels, train_bpe  # noqa: E402
from scaffold_margin.data import read_jsonl  # noqa: E402
from scaffold_margin.pretok import pretokenize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word_counts", required=True)
    ap.add_argument("--text_dir", required=True)
    ap.add_argument("--c", type=float, required=True)
    ap.add_argument("--K", type=int, default=32767, help="visible (normal) vocabulary size; EOS is added on top")
    ap.add_argument("--pattern", default="gpt2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = f"c{a.c:g}"
    wc = pickle.load(open(a.word_counts, "rb"))
    t0 = time.time()
    if a.c == 0:
        res = train_bpe(wc, num_merges=a.K - 256, log_every=5000)
        tok = Tokenizer(res.merges, assign_levels(res), mode="bpe", pattern=a.pattern, meta=dict(kind="bpe", c=0.0))
    else:
        res = train_bpe(wc, target_normal=a.K, max_merges=12 * a.K, scaffold_factor=a.c, log_every=5000)
        lv = assign_levels(res, tau=0.5, target_normal=a.K)
        tok = Tokenizer(res.merges, lv, mode="scaffold", pattern=a.pattern,
                        meta=dict(kind="scaffold", c=a.c, q=res.q))
    train_s = time.time() - t0
    assert tok.n_normal == a.K, f"visible vocab {tok.n_normal} != {a.K} (raise max_merges?)"
    tok.save(os.path.join(a.out, f"tok_{tag}.json"))

    # training exposure of every visible token (encode the training word counts)
    exposure = np.zeros(tok.n_normal + 1, dtype=np.int64)
    for w, n in wc.items():
        for t in tok.encode_word(w):
            exposure[t] += n
    vis = exposure[: tok.n_normal]
    stats = dict(c=a.c, K=a.K, train_seconds=train_s, merges=len(tok.merges),
                 hidden=int((tok.levels >= 1).sum()), train_tokens=int(vis.sum()),
                 exposure_zero=int((vis == 0).sum()), exposure_lt10=int((vis < 10).sum()),
                 exposure_lt100=int((vis < 100).sum()), exposure_lt1000=int((vis < 1000).sum()),
                 exposure_p01=float(np.percentile(vis, 1)), exposure_median=float(np.median(vis)))
    for split in ("val", "test"):
        c = Counter()
        nb = 0
        for t, _ in read_jsonl(os.path.join(a.text_dir, f"{split}.jsonl")):
            c.update(pretokenize(t, a.pattern))
            nb += len(t.encode())
        n = sum(k * len(tok.encode_word(w)) for w, k in c.items())
        stats[f"{split}_tokens"], stats[f"{split}_bytes_per_token"] = n, nb / n
    np.save(os.path.join(a.out, f"tok_{tag}.exposure.npy"), exposure)
    json.dump(stats, open(os.path.join(a.out, f"tok_{tag}.stats.json"), "w"), indent=1)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
