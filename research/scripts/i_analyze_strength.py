#!/usr/bin/env python
"""Scaffold strength sweep analysis (A), exposure buckets (B), typo robustness (C).

A: test/val BPB per scaffold factor c (3 seeds), paired seed deltas + document bootstrap vs c=0 (BPE) and c=1.
B/C: read i_eval_extra.py output (runs named c<c>/seed<s> and v1/<arm>/seed<s>).
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="runs/strength")
    ap.add_argument("--extra", required=True, help="i_eval_extra.py output json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(20261012)
    runs = defaultdict(dict)
    for p in glob.glob(os.path.join(a.root, "lm/c*/seed*/results.json")):
        c, s = p.split(os.sep)[-3][1:], int(p.split(os.sep)[-2][4:])
        r = json.load(open(p))
        z = np.load(p.replace("results.json", "eval_test.npz"))
        runs[c][s] = dict(test=r["bpb"]["test"], val=r["bpb"]["val"], nll=z["doc_nll_bits"], by=z["doc_bytes"])
    stats = {c: json.load(open(os.path.join(a.root, f"tok/tok_c{c}.stats.json"))) for c in runs}
    order = sorted(runs, key=float)
    out = dict(arms={}, vs={})
    print(f"{'c':>4} {'hidden':>7} {'test_tok':>9} {'<100':>5} {'test BPB':>9} {'sd':>7} {'val BPB':>9}  seeds")
    for c in order:
        t = np.array([runs[c][s]["test"] for s in sorted(runs[c])])
        v = np.array([runs[c][s]["val"] for s in sorted(runs[c])])
        out["arms"][c] = dict(test_mean=float(t.mean()), test_sd=float(t.std(ddof=1)) if len(t) > 1 else None,
                              val_mean=float(v.mean()), per_seed={s: runs[c][s]["test"] for s in sorted(runs[c])},
                              hidden=stats[c]["hidden"], test_tokens=stats[c]["test_tokens"],
                              exposure_lt100=stats[c]["exposure_lt100"])
        print(f"{c:>4} {stats[c]['hidden']:>7} {stats[c]['test_tokens']:>9} {stats[c]['exposure_lt100']:>5} "
              f"{t.mean():>9.5f} {t.std(ddof=1) if len(t) > 1 else 0:>7.5f} {v.mean():>9.5f}  {np.round(t, 4).tolist()}")
    for ref in ("0", "1"):
        if ref not in runs:
            continue
        for c in order:
            if c == ref:
                continue
            common = sorted(set(runs[c]) & set(runs[ref]))
            d = [runs[c][s]["test"] - runs[ref][s]["test"] for s in common]
            na = np.mean([runs[c][s]["nll"] for s in common], 0)
            nb = np.mean([runs[ref][s]["nll"] for s in common], 0)
            by = runs[c][common[0]]["by"]
            idx = rng.integers(0, len(by), (5000, len(by)))
            bd = (na[idx].sum(1) - nb[idx].sum(1)) / by[idx].sum(1)
            row = dict(seed_deltas=d, delta_mean=float(np.mean(d)), seeds_better=int(sum(x < 0 for x in d)),
                       doc_ci95=[float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))])
            out["vs"][f"c{c}-c{ref}"] = row
            print(f"  c{c} - c{ref}: mean {row['delta_mean']:+.5f} seeds {np.round(d, 5).tolist()} CI {np.round(row['doc_ci95'], 5).tolist()}")

    if os.path.exists(a.extra):
        ex = json.load(open(a.extra))
        groups = defaultdict(list)
        for run, r in ex.items():
            groups[run.rsplit("/seed", 1)[0]].append(r)
        out["extra"] = {}
        print("\n[C typo robustness] BPB clean / typo0.05 / typo0.15 (mean over seeds), and degradation")
        for g, rs in sorted(groups.items()):
            m = {k: float(np.mean([r[k]["bpb"] for r in rs])) for k in ("clean", "typo0.05", "typo0.15")}
            buckets = [dict(lo=b["lo"], bits_per_byte=float(np.mean([r["by_exposure"][i]["bits_per_byte"] for r in rs])),
                            byte_share=float(np.mean([r["by_exposure"][i]["bytes"] / max(1, sum(x["bytes"] for x in r["by_exposure"]))
                                                      for r in rs])))
                       for i, b in enumerate(rs[0]["by_exposure"])]
            out["extra"][g] = dict(n=len(rs), **m, d05=m["typo0.05"] - m["clean"], d15=m["typo0.15"] - m["clean"], buckets=buckets)
            print(f"  {g:>24} n={len(rs)} {m['clean']:.4f} / {m['typo0.05']:.4f} / {m['typo0.15']:.4f}   "
                  f"+{m['typo0.05'] - m['clean']:.4f} / +{m['typo0.15'] - m['clean']:.4f}")
        print("\n[B] bits/byte by training exposure of target token  (byte share in parentheses)")
        for g, e in sorted(out["extra"].items()):
            print(f"  {g:>24} " + "  ".join(f">={b["lo"]:g}:{b['bits_per_byte']:.2f}({100 * b['byte_share']:.2f}%)"
                                             for b in e["buckets"]))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
