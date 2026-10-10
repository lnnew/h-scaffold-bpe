#!/usr/bin/env python
"""E2 analysis: same visible vocabulary, different inference.

Layout: <runs>/lm/<arm>/seed<s>/results.json + eval_<split>.npz
Primary: split 'test' (each arm's own training segmentation; mix -> canonical), vs ref arm.
Cross:   test_canonical / test_longest / test_dp for every Scaffold-vocabulary arm (same ids, same bytes).
Uncertainty: paired seed deltas (training noise) and a paired document bootstrap on seed-averaged per-document
bits (test-set noise).  Three seeds: p-values are descriptive only.
"""
import argparse
import glob
import json
import os

import numpy as np


def load(runs):
    res = {}
    for p in sorted(glob.glob(f"{runs}/lm/*/seed*/results.json")):
        arm, seed = p.split(os.sep)[-3], int(p.split(os.sep)[-2][4:])
        r = json.load(open(p))
        ev = {}
        for split, bpb in r["bpb"].items():
            z = np.load(p.replace("results.json", f"eval_{split}.npz"))
            ev[split] = dict(bpb=bpb, nll=z["doc_nll_bits"], by=z["doc_bytes"])
        res.setdefault(arm, {})[seed] = dict(ev=ev, steps=r["total_steps"], exposure_bytes=r["train_exposure_bytes"],
                                             train_time_s=r["train_time_s"])
    return res


def compare(runs_a, runs_b, split, rng, boot):
    common = sorted(s for s in set(runs_a) & set(runs_b) if split in runs_a[s]["ev"] and split in runs_b[s]["ev"])
    if not common:
        return None
    d = np.array([runs_a[s]["ev"][split]["bpb"] - runs_b[s]["ev"][split]["bpb"] for s in common])
    na = np.mean([runs_a[s]["ev"][split]["nll"] for s in common], 0)
    nb = np.mean([runs_b[s]["ev"][split]["nll"] for s in common], 0)
    by = runs_a[common[0]]["ev"][split]["by"]
    assert np.array_equal(by, runs_b[common[0]]["ev"][split]["by"]), "document bytes differ"
    n = len(by)
    idx = rng.integers(0, n, (boot, n))
    bd = (na[idx].sum(1) - nb[idx].sum(1)) / by[idx].sum(1)
    row = dict(n_seeds=len(common), delta_mean=float(d.mean()), seed_deltas=d.tolist(),
               seeds_better=int((d < 0).sum()), doc_ci95=[float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))])
    if len(d) >= 2:
        from scipy import stats

        row["paired_t_p"] = float(stats.ttest_1samp(d, 0).pvalue)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--ref", default="scaffold_canonical")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(20261011)
    R = load(a.runs)
    out = dict(arms={}, primary={}, cross={}, cross_vs_ref={})
    for arm, runs in sorted(R.items()):
        out["arms"][arm] = {
            split: dict(mean=float(np.mean([r["ev"][split]["bpb"] for r in runs.values() if split in r["ev"]])),
                        per_seed={s: r["ev"][split]["bpb"] for s, r in sorted(runs.items()) if split in r["ev"]})
            for split in sorted({k for r in runs.values() for k in r["ev"]})}
        out["arms"][arm]["_train"] = {s: dict(steps=r["steps"], exposure_bytes=r["exposure_bytes"],
                                              train_time_s=r["train_time_s"]) for s, r in runs.items()}
    print(f"{'arm':>20} " + " ".join(f"{s:>15}" for s in ("test", "test_canonical", "test_longest", "test_dp")))
    for arm, v in out["arms"].items():
        print(f"{arm:>20} " + " ".join(f"{v[s]['mean']:>15.5f}" if s in v else f"{'-':>15}"
                                       for s in ("test", "test_canonical", "test_longest", "test_dp")))
    if a.ref in R:
        for arm in R:
            if arm == a.ref:
                continue
            out["primary"][arm] = compare(R[arm], R[a.ref], "test", rng, a.boot)
            print(f"[primary test] {arm} - {a.ref}: {json.dumps(out['primary'][arm])}")
            if arm.startswith("scaffold"):
                out["cross_vs_ref"][arm] = {sp: compare(R[arm], R[a.ref], sp, rng, a.boot)
                                            for sp in ("test_canonical", "test_longest", "test_dp")}
    for arm in R:          # within-model: cost of switching inference at test time
        if arm.startswith("scaffold"):
            base = "test_canonical"
            out["cross"][arm] = {sp: dict(delta_vs_canonical_seg=float(np.mean(
                [r["ev"][sp]["bpb"] - r["ev"][base]["bpb"] for r in R[arm].values() if sp in r["ev"]])))
                for sp in ("test_longest", "test_dp")}
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
