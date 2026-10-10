#!/usr/bin/env python
"""Direction B analysis: is parity optimal?  (reads LM runs trained on tokenizers from b_build_tokenizers.py)

Run layout:  <runs>/alpha<A>/<kind>/seed<k>/results.json   with results["bpb"] containing keys test_<lang>.

Also reports _paired_vs_alpha: per-language and equal-weight-mean paired seed statistics vs
--ref_alpha (paired t over shared seeds, descriptive for n<5).  Tokenizer parity in the report
is the val-split metric used for the gate; per_lang_test_* is for the final report only.

Reported per alpha (averaged over seeds):
  * BPB per language, equal-weight mean over languages (the 'every language matters' objective)
    and byte-weighted mean (the 'total corpus' objective)
  * tokenizer parity (from b_tokenizer_report.json) next to LM quality
  * a crude direct/indirect split per language relative to the reference alpha:
        delta_BPB_l  =  slope * delta_log(bytes/token)_l  +  residual_l
    where `slope` is fit across (language, alpha) pairs.  The first term is what compression change
    alone predicts; the residual is everything else (cross-lingual transfer, capacity, embedding
    quality...).  Exploratory only: it is a regression on <= languages x alphas points.
Compare the measured validation parity gap with LM quality. No alpha is assumed to achieve parity.
The regression residual cannot establish a cross-lingual transfer effect.
"""
import argparse, glob, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--tok_report", required=True)
    ap.add_argument("--langs", nargs="+", required=True); ap.add_argument("--kind", default="scaffold")
    ap.add_argument("--text_dir", default=None, help="dir with test_<lang>.jsonl; enables the byte-weighted mean")
    ap.add_argument("--ref_alpha", default="1.0"); ap.add_argument("--K", type=int)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tr = json.load(open(a.tok_report))
    wts = None
    if a.text_dir:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from scaffold_margin.data import read_jsonl
        wts = {l: sum(len(t.encode()) for t, _ in read_jsonl(f"{a.text_dir}/test_{l}.jsonl")) for l in a.langs}
    table = {}
    for d in sorted(glob.glob(f"{a.runs}/alpha*/{a.kind}/seed*/results.json")):
        alpha = d.split(os.sep)[-4][5:]
        seed = int(d.split(os.sep)[-2][4:])
        r = json.load(open(d))
        table.setdefault(alpha, {})[seed] = {l: r["bpb"][f"test_{l}"] for l in a.langs}
    rows, per_lang = {}, {}
    for alpha, lst in table.items():
        runs = list(lst.values())
        m = {l: float(np.mean([x[l] for x in runs])) for l in a.langs}
        per_lang[alpha] = m
        rows[alpha] = dict(n_seeds=len(runs), bpb_per_lang=m, mean_equal_weight=float(np.mean(list(m.values()))),
                           mean_byte_weighted=(float(sum(m[l] * wts[l] for l in a.langs) / sum(wts.values())) if wts else None),
                           bytes_per_token=tr[alpha]["per_lang"][a.kind], parity=tr[alpha][f"parity_{a.kind}"])
    if a.ref_alpha in per_lang:
        xs, ys, tags = [], [], []
        for al, m in per_lang.items():
            if al == a.ref_alpha: continue
            for l in a.langs:
                xs.append(np.log(tr[al]["per_lang"][a.kind][l] / tr[a.ref_alpha]["per_lang"][a.kind][l]))
                ys.append(m[l] - per_lang[a.ref_alpha][l]); tags.append((al, l))
        xs, ys = np.array(xs), np.array(ys)
        slope = float((xs @ ys) / (xs @ xs)) if len(xs) and (xs @ xs) > 0 else float("nan")
        split = {f"{al}|{l}": dict(delta_bpb=float(y), direct=float(slope * x), residual=float(y - slope * x)) for (al, l), x, y in zip(tags, xs, ys)}
        rows["_decomposition_vs_alpha" + a.ref_alpha] = dict(slope_bpb_per_log_bytes_per_token=slope, per_alpha_lang=split)
    best = min((k for k in rows if not k.startswith("_")), key=lambda k: rows[k]["mean_equal_weight"], default=None)
    rows["_best_alpha_equal_weight"] = best
    # paired across-seed comparison vs ref_alpha: same init/data seeds across alphas
    if a.ref_alpha in table:
        import scipy.stats as st

        def paired(metric_fn):
            out = {}
            for al, runs in table.items():
                if al == a.ref_alpha:
                    continue
                common = sorted(set(runs) & set(table[a.ref_alpha]))
                if not common:
                    continue
                d = np.array([metric_fn(runs[s]) - metric_fn(table[a.ref_alpha][s]) for s in common])
                e = dict(n_paired=len(d), delta_mean=float(d.mean()))
                if len(d) >= 2:
                    e["delta_std"] = float(d.std(ddof=1))
                    e["paired_t_p"] = float(st.ttest_1samp(d, 0).pvalue)
                    e["supported_at_05"] = bool(e["paired_t_p"] < 0.05 and len(d) >= 3)
                out[al] = e
            return out

        eq = lambda x: float(np.mean([x[l] for l in a.langs]))
        rows["_paired_vs_alpha" + a.ref_alpha] = dict(
            mean_equal_weight=paired(eq),
            per_lang={l: paired(lambda x, l=l: x[l]) for l in a.langs})
    print(json.dumps(rows, indent=1))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
