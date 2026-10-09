#!/usr/bin/env python
"""Aggregate LM runs and compare methods with paired statistics.

Expected layout:  <runs>/<setting>/<method>/seed<k>/results.json  (+ eval_<split>.npz)
  setting  e.g. K32768           method  e.g. scaffold, hscaffold, hsel_random0_f50 ...

For each setting and each method vs --ref:
  * per-seed BPB table, mean +- std
  * paired difference over seeds (same init seed + data seed across methods): mean, std, paired t-test,
    Wilcoxon (n is small: treat p-values as descriptive), and a sign count
  * paired bootstrap over EVALUATION DOCUMENTS (seed-averaged per-doc nll; documents have identical
    byte counts under every tokenizer) -> CI on delta BPB attributable to test-set sampling.
  The two uncertainty sources are different: across-seed variance is training noise; the document
  bootstrap is test-set noise.  A claim needs both to clear zero.
  * minimum detectable effect given the observed across-seed std (80% power, two-sided 5%).
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy import stats as st


def load(runs, split):
    res = {}
    for p in sorted(glob.glob(f"{runs}/*/*/seed*/results.json")):
        parts = p.split(os.sep)
        setting, method, seed = parts[-4], parts[-3], int(parts[-2][4:])
        r = json.load(open(p))
        if split not in r["bpb"]:
            continue
        z = np.load(p.replace("results.json", f"eval_{split}.npz"))
        res.setdefault(setting, {}).setdefault(method, {})[seed] = dict(bpb=r["bpb"][split], nll=z["doc_nll_bits"], by=z["doc_bytes"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--ref", default="scaffold")
    ap.add_argument("--split", default="test"); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    data = load(a.runs, a.split)
    rng = np.random.default_rng(0)
    out = {}
    for setting, methods in data.items():
        if a.ref not in methods:
            print(f"[{setting}] reference '{a.ref}' missing"); continue
        ref = methods[a.ref]
        print(f"\n=== {setting}  (split={a.split}, ref={a.ref}) ===")
        out[setting] = {}
        for m, runs in sorted(methods.items()):
            b = np.array([runs[s]["bpb"] for s in sorted(runs)])
            row = dict(n_seeds=len(b), bpb_mean=float(b.mean()), bpb_std=float(b.std(ddof=1)) if len(b) > 1 else None)
            common = sorted(set(runs) & set(ref))
            if m != a.ref and len(common) >= 1:
                d = np.array([runs[s]["bpb"] - ref[s]["bpb"] for s in common])
                row["delta_mean"] = float(d.mean()); row["delta_pct"] = float(100 * d.mean() / np.mean([ref[s]["bpb"] for s in common]))
                row["seeds_better"] = int((d < 0).sum()); row["n_paired"] = len(d)
                if len(d) >= 2:
                    row["delta_std"] = float(d.std(ddof=1))
                    row["paired_t_p"] = float(st.ttest_1samp(d, 0).pvalue)
                    if len(d) >= 5:
                        row["wilcoxon_p"] = float(st.wilcoxon(d).pvalue)
                    # minimum detectable effect (paired design, 80% power, alpha=.05)
                    n = len(d)
                    row["mde_bpb"] = float((st.norm.ppf(0.975) + st.norm.ppf(0.8)) * d.std(ddof=1) / np.sqrt(n))
                # document-level paired bootstrap on seed-averaged per-doc losses
                by = ref[common[0]]["by"]
                dn = np.mean([runs[s]["nll"] - ref[s]["nll"] for s in common], axis=0)
                idx = rng.integers(0, len(by), size=(a.boot, len(by)))
                boots = dn[idx].sum(1) / by[idx].sum(1)
                row["doc_boot_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            if "doc_boot_ci95" in row:
                lo, hi = row["doc_boot_ci95"]
                doc_ok = lo > 0 or hi < 0
                seed_ok = row.get("paired_t_p", 1.0) < 0.05 and row.get("n_paired", 0) >= 3
                same_sign = (lo > 0) == (row["delta_mean"] > 0)
                row["verdict"] = ("SUPPORTED (seed-level and document-level both clear zero)" if (doc_ok and seed_ok and same_sign)
                                  else "NOT SUPPORTED: " + ("fewer than 3 paired seeds; " if row.get("n_paired", 0) < 3 else "")
                                  + ("" if seed_ok or row.get("n_paired", 0) < 3 else "across-seed test does not clear zero; ")
                                  + ("" if doc_ok else "document bootstrap CI contains zero"))
            out[setting][m] = row
            print(m, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
