#!/usr/bin/env python
"""Direction B: vocabulary allocation sweep.

For each allocation exponent alpha the tokenizer-training mixture gives language l the share
        p_l  proportional to  n_l ** alpha        (alpha=1: natural/high-resource-heavy,
                                                   alpha=0: uniform language sampling)
Uniform sampling does not imply parity; determine parity from held-out measurements.
Implemented by re-weighting per-language word counts (no corpus duplication).  Trains BPE and
Scaffold-BPE per alpha at one vocabulary size, then reports per-language compression and
parity metrics.  The parity-metrics gate uses val_<lang>.jsonl; test_<lang>.jsonl is stored
for the final report only and is never used to select anything.  The LM training mixture is
held FIXED elsewhere, so that tokenizer allocation is the only thing that changes.
"""
import argparse, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
from scaffold_margin.bpe import build_tokenizers
from scaffold_margin.data import count_words_parallel, read_jsonl

SCALE = 1000


def gini(x):
    x = np.sort(np.asarray(x, dtype=float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def parity(bpt: dict):
    v = np.array(list(bpt.values()))
    return dict(cv=float(v.std() / v.mean()), max_over_min=float(v.max() / v.min()), gini=gini(v), min=float(v.min()), max=float(v.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_dir", required=True); ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--K", type=int, default=32768)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pc = f"{a.out}/lang_word_counts.pkl"
    if os.path.exists(pc):
        wcs, sizes = pickle.load(open(pc, "rb"))
    else:
        wcs, sizes = {}, {}
        for l in a.langs:
            p = f"{a.text_dir}/train_{l}.jsonl"
            texts = [d for d, _ in read_jsonl(p)]
            sizes[l] = sum(len(t.encode()) for t in texts)
            wcs[l] = count_words_parallel(texts, a.workers)
            print(f"{l}: {sizes[l]/1e6:.1f} MB, {len(wcs[l]):,} types")
        pickle.dump((wcs, sizes), open(pc, "wb"))
    tot = sum(sizes.values())
    val_docs = {l: [d for d, _ in read_jsonl(f"{a.text_dir}/val_{l}.jsonl")] for l in a.langs}
    try:
        test_docs = {l: [d for d, _ in read_jsonl(f"{a.text_dir}/test_{l}.jsonl")] for l in a.langs}
    except FileNotFoundError:
        test_docs = {}      # optional: test text is for the final report only
    report = {}
    for alpha in a.alphas:
        w_share = {l: sizes[l] ** alpha for l in a.langs}
        z = sum(w_share.values())
        weight = {l: (w_share[l] / z) / (sizes[l] / tot) for l in a.langs}   # relative to natural mixture
        wc: Counter = Counter()
        wmax = max(weight.values())
        for l in a.langs:
            f = weight[l] / wmax * SCALE
            for w, c in wcs[l].items():
                v = int(round(c * f))
                if v > 0:
                    wc[w] += v
        toks = build_tokenizers(wc, a.K)
        row = dict(share={l: w_share[l] / z for l in a.langs}, per_lang={})
        for kind in ("bpe", "scaffold"):
            toks[kind].save(f"{a.out}/ml_a{alpha}_{kind}_{a.K}.json")
            def bpt(dd):
                return {l: sum(len(d.encode()) for d in dd[l]) / sum(len(toks[kind].encode(d)) for d in dd[l])
                        for l in a.langs}
            pv = bpt(val_docs)
            row["per_lang"][kind] = pv            # gate split (val) — also read by b_analyze
            row[f"parity_{kind}"] = parity(pv)
            if test_docs:
                row[f"per_lang_test_{kind}"] = bpt(test_docs)
                row[f"parity_test_{kind}"] = parity(row[f"per_lang_test_{kind}"])
        report[str(alpha)] = row
        print(f"alpha={alpha}: val bytes/token scaffold =", {l: round(v, 2) for l, v in row["per_lang"]["scaffold"].items()},
              "| val parity", {k: round(v, 3) for k, v in row["parity_scaffold"].items()}, flush=True)
    json.dump(report, open(f"{a.out}/b_tokenizer_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
