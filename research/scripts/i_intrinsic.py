#!/usr/bin/env python
"""Intrinsic tokenizer metrics for the strength sweep (survey §7.2): computed on the encoded train/test streams.

Per tokenizer: Shannon entropy and Renyi efficiency (alpha=2.5, Zouhar et al. 2023) of the token distribution,
vocabulary usage, mean token length (bytes) in the vocabulary and in running text, share of word-initial tokens,
and share of test tokens whose type is rare in training (< 100 occurrences).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scaffold_margin.bpe import Tokenizer  # noqa: E402
from scaffold_margin.data import open_stream  # noqa: E402


def renyi_eff(p, alpha, V):
    p = p[p > 0]
    h = np.log(np.sum(p ** alpha)) / (1 - alpha)
    return float(h / np.log(V))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="runs/strength (tok/, data/)")
    ap.add_argument("--cs", nargs="+", default=["0", "0.5", "1", "2", "4", "8"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    for c in a.cs:
        tok = Tokenizer.load(os.path.join(a.root, f"tok/tok_c{c}.json"))
        V = tok.n_normal
        vis_bytes = [tok.token_bytes[t] for t in tok.from_model]
        row = dict(vocab_mean_len=float(np.mean([len(b) for b in vis_bytes])),
                   vocab_word_initial=float(np.mean([b[:1] == b" " for b in vis_bytes])))
        for split in ("train", "test"):
            s = np.asarray(open_stream(os.path.join(a.root, f"data/c{c}/{split}.bin")))
            s = s[s != tok.eos_id]
            cnt = np.bincount(s, minlength=V + 1)[:V].astype(np.float64)
            p = cnt / cnt.sum()
            tb = tok.model_token_bytes[:V]
            row[split] = dict(tokens=int(cnt.sum()), types_used=int((cnt > 0).sum()),
                              entropy_bits=float(-(p[p > 0] * np.log2(p[p > 0])).sum()),
                              renyi_eff_2_5=renyi_eff(p, 2.5, V),
                              shannon_eff=float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(V)),
                              mean_token_bytes=float((cnt * tb).sum() / cnt.sum()),
                              word_initial_share=float(cnt[[i for i, b in enumerate(vis_bytes) if b[:1] == b" "]].sum() / cnt.sum()))
            if split == "train":
                train_cnt = cnt
            else:
                rare = train_cnt < 100
                row["test"]["rare_type_token_share"] = float(cnt[rare].sum() / cnt.sum())
        out[c] = row
        print(c, json.dumps(row), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
