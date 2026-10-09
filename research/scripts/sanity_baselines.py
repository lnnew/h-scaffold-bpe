#!/usr/bin/env python
"""BPB floor: unigram and bigram token models scored with the SAME BPB code path as the LMs.

Any trained LM must beat these comfortably.  If a trained LM's BPB is near or above the unigram floor,
suspect (a) an undertrained model / broken optimisation, or (b) a BPB accounting problem, BEFORE
comparing tokenizers.  Run on every tokenizer's data dir.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scaffold_margin.data import bpb_from_token_nll, open_stream
from scaffold_margin.scorer import BigramScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--splits", nargs="+", default=["val", "test"])
    a = ap.parse_args()
    meta = json.load(open(f"{a.data}/meta.json")); tb = np.load(f"{a.data}/token_bytes.npy")
    V = meta["n_normal"] + 1
    tr = np.asarray(open_stream(f"{a.data}/train.bin"), dtype=np.int64)
    bg = BigramScorer(V).fit([tr])
    out = {}
    for sp in a.splits:
        s = np.asarray(open_stream(f"{a.data}/{sp}.bin"), dtype=np.int64)
        tg = s[1:]
        uni = -np.log(bg.p_uni[tg])
        big = -bg._logp(s[:-1], s[1:])
        out[sp] = dict(unigram_bpb=bpb_from_token_nll(uni, tg, tb)[0], bigram_bpb=bpb_from_token_nll(big, tg, tb)[0],
                       bytes_per_token=float(tb[tg].sum() / len(tg)))
        print(sp, out[sp])
    json.dump(out, open(f"{a.data}/baseline_bpb.json", "w"), indent=1)


if __name__ == "__main__":
    main()
