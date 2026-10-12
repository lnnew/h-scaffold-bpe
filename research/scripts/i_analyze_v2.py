#!/usr/bin/env python
"""infer_v2 analysis.

(1) lambda frontier: per segmentation, test tokens and BPB of frozen models trained on canonical / dp / longest / mix
    segmentations (infer_v1), relative to the same models on the canonical segmentation.
(2) causal scaffold features: aux vs shuffled-aux control (paired by seed, same hardware) and vs no-feature baseline.
"""
import argparse
import glob
import json
import os

import numpy as np


def boot_ci(na, nb, by, rng, B=5000):
    idx = rng.integers(0, len(by), (B, len(by)))
    d = (na[idx].sum(1) - nb[idx].sum(1)) / by[idx].sum(1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda_dir", required=True)
    ap.add_argument("--aux_runs", nargs="+", required=True, help="dirs containing lm/<arm>/seed*/results.json")
    ap.add_argument("--base_runs", required=True, help="infer_v1 run root (scaffold_canonical baseline)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(20261012)
    out = {}

    # ---- (1) frontier -------------------------------------------------------------------------
    ev = json.load(open(os.path.join(a.lambda_dir, "eval.json")))
    segs = {}
    for d in sorted(glob.glob(os.path.join(a.lambda_dir, "seg_*"))):
        m = json.load(open(os.path.join(d, "meta.json")))
        segs[os.path.basename(d)] = m["splits"]["test"]["tokens"]
    canon_tokens = segs["seg_lam1"]
    frontier = {}
    for arm in ("scaffold_canonical", "scaffold_dp", "scaffold_longest", "scaffold_mix"):
        rows = {}
        for seg, ntok in segs.items():
            per = {}
            for s in (0, 1, 2):
                k, k0 = f"{arm}/seed{s}|{seg}|test", f"{arm}/seed{s}|seg_lam1|test"
                if k in ev and k0 in ev:
                    per[s] = (ev[k]["bpb"], ev[k]["bpb"] - ev[k0]["bpb"])
            if per:
                rows[seg] = dict(tokens=ntok, tokens_rel=ntok / canon_tokens - 1,
                                 bpb_mean=float(np.mean([v[0] for v in per.values()])),
                                 dbpb_vs_canonical_seg=float(np.mean([v[1] for v in per.values()])),
                                 per_seed={s: v[0] for s, v in per.items()})
        frontier[arm] = rows
        print(f"\n[frontier] models trained on {arm}")
        for seg, r in sorted(rows.items(), key=lambda x: -x[1]["tokens"]):
            print(f"  {seg:>15}  tokens {100*r['tokens_rel']:+.3f}%  BPB {r['bpb_mean']:.5f}  "
                  f"dBPB vs canonical seg {r['dbpb_vs_canonical_seg']:+.5f}")
    out["frontier"] = frontier

    # ---- (2) features -------------------------------------------------------------------------
    runs = {}
    for root in a.aux_runs + [a.base_runs]:
        for p in glob.glob(os.path.join(root, "lm/*/seed*/results.json")):
            arm, seed = p.split(os.sep)[-3], int(p.split(os.sep)[-2][4:])
            if arm not in ("scaffold_aux", "scaffold_auxshuf", "scaffold_canonical"):
                continue
            r = json.load(open(p))
            z = np.load(p.replace("results.json", "eval_test.npz"))
            runs.setdefault(arm, {})[seed] = dict(test=r["bpb"]["test"], val=r["bpb"]["val"], nll=z["doc_nll_bits"],
                                                   by=z["doc_bytes"], n_params=r.get("n_params"))
    feat = {arm: dict(test_mean=float(np.mean([v["test"] for v in r.values()])),
                      val_mean=float(np.mean([v["val"] for v in r.values()])),
                      per_seed={s: (v["test"], v["val"]) for s, v in sorted(r.items())},
                      n_params=next(iter(r.values()))["n_params"]) for arm, r in runs.items()}
    comps = {}
    for x, y in (("scaffold_aux", "scaffold_auxshuf"), ("scaffold_aux", "scaffold_canonical"),
                 ("scaffold_auxshuf", "scaffold_canonical")):
        if x in runs and y in runs:
            common = sorted(set(runs[x]) & set(runs[y]))
            if not common:
                continue
            d = [runs[x][s]["test"] - runs[y][s]["test"] for s in common]
            na = np.mean([runs[x][s]["nll"] for s in common], 0)
            nb = np.mean([runs[y][s]["nll"] for s in common], 0)
            comps[f"{x} - {y}"] = dict(seeds=common, seed_deltas=d, delta_mean=float(np.mean(d)),
                                       seeds_better=int(sum(v < 0 for v in d)),
                                       doc_ci95=boot_ci(na, nb, runs[x][common[0]]["by"], rng))
    out["features"] = dict(arms=feat, comparisons=comps)
    print("\n[features]")
    for arm, f in feat.items():
        print(f"  {arm:>20} test {f['test_mean']:.5f} val {f['val_mean']:.5f} params {f['n_params']}")
    for k, c in comps.items():
        print(f"  {k}: {json.dumps(c)}")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
