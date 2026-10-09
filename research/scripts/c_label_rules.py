#!/usr/bin/env python
"""Direction C (assistant-originated): selective re-merging.  Phase 1 + variant construction.

For every candidate re-merge rule (a,b)->c that fires on the tokenizer-training corpus:
  * cheap features (frequency, collapse ratio, merge rank, length, per-event token change, ...)
  * LM label: mean delta_bits when ONLY that rule is enabled, measured on held-out dev text
    (never on test) under one or more teachers  -> label reliability across teachers
  * predictability of the label from the cheap features (out-of-fold)
  * predicted corpus-level BPB shift of full H-Scaffold vs Scaffold under the teacher
    (sum of per-rule delta_bits over every dev occurrence / dev bytes) -- a direct check on
    whether re-merging is even *capable* of the BPB differences previously reported.
Then writes matched-count allow-list variants: random(3 seeds), frequency, compression+support,
utility-predicted (deployable) and dev-oracle (non-deployable upper bound).
"""
import argparse, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scipy.stats import spearmanr
from scaffold_margin.adapter import attach_residuals, find_tokenizer
from scaffold_margin.bpe import Tokenizer
from scaffold_margin.data import open_stream, read_jsonl
from scaffold_margin.pretok import count_words
from scaffold_margin.remerge import build_rule_alts, event_stats, extract_rules
from scaffold_margin.scorer import BigramScorer
from scaffold_margin.utility import (collect_edits, doc_word_lists, pick_matched, pick_random, predictability,
                                     score_edits, summarize, pick_random_traffic)

FEATURES = ["log_n_occ", "log_r_c", "collapse", "merge_rank", "c_len", "dtok_per_event",
            "boundary_only_frac", "capitalized_frac", "l2_ratio"]


def make_scorer(spec, sc, a):
    if spec == "bigram":
        stream = np.asarray(open_stream(a.train_stream), dtype=np.int64)
        return BigramScorer(sc.n_normal + 1).fit([stream])
    if spec.startswith("torch:"):
        from scaffold_margin.scorer import TorchScorer
        return TorchScorer(spec[6:], device=a.device, batch_size=a.score_batch)
    raise SystemExit(f"unknown scorer {spec}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok_dir", required=True); ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--text_dir", required=True)
    ap.add_argument("--word_counts", default=None, help="pickled Counter[bytes,int] (default <tok_dir>/word_counts.pkl; the team's hscaffold_bpe 'count' output also works)")
    ap.add_argument("--scorers", nargs="+", default=["bigram"], help="bigram | torch:<ckpt>; first one defines labels used for selection")
    ap.add_argument("--train_stream", help="uint32 token stream for the bigram teacher (e.g. tok_data/scaffold_K/train.bin)")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--score_batch", type=int, default=256)
    ap.add_argument("--ctx_words", type=int, default=12)
    ap.add_argument("--max_per_rule", type=int, default=64)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--alpha", type=float, default=1.0, help="U_comp weight on token savings")
    ap.add_argument("--beta", type=float, default=1.0, help="U_comp weight on log support")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variants_dir", default=None, help="where hsel_*.json are written (default: --tok_dir; set it when --tok_dir is the team's artifacts dir)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    K = a.K
    sc = Tokenizer.load(find_tokenizer(a.tok_dir, "scaffold", K))
    h = Tokenizer.load(find_tokenizer(a.tok_dir, "hscaffold", K))
    wc = pickle.load(open(a.word_counts or f"{a.tok_dir}/word_counts.pkl", "rb"))
    if "resid" not in h.meta:          # team-format tokenizer: recompute r(t) from the count table
        attach_residuals(h, wc)
    dev_docs = [d for d, _ in read_jsonl(os.path.join(a.text_dir, "dev_sel.jsonl"))]
    dev_bytes = sum(len(d.encode()) for d in dev_docs)
    dev_words = doc_word_lists(dev_docs, h.pattern)

    rules = extract_rules(h, wc)
    cand = sorted(k for k, r in rules.items() if not r["chain_only"])
    chain = [k for k, r in rules.items() if r["chain_only"]]
    print(f"rules: {len(rules)} total, {len(cand)} with isolated effect, {len(chain)} chain-only (never selectable)")
    if not cand:
        raise SystemExit("no candidate rules; nothing to analyse")

    alts = build_rule_alts(h, cand, [w for ws in dev_words for w in ws])
    edits = collect_edits(dev_words, sc.encode_word, alts, ctx_words=a.ctx_words, max_per_key=a.max_per_rule)
    print(f"dev edits: {sum(edits.seen.values())} occurrences over {len(edits)} rules")

    report = dict(K=K, n_rules=len(rules), n_candidates=len(cand), n_chain_only=len(chain),
                  dev_occurrences=int(sum(edits.seen.values())), dev_bytes=dev_bytes, teachers={})
    labels_by_teacher = {}
    for spec in a.scorers:
        sco = make_scorer(spec, sc, a)
        deltas = score_edits(sco, edits, sc.eos_id)
        summ = summarize(deltas)
        labels_by_teacher[spec] = summ
        pred_dbpb = -sum(summ[k]["mean"] * edits.seen[k] for k in summ) / dev_bytes
        report["teachers"][spec] = dict(
            predicted_dBPB_H_minus_Scaffold=pred_dbpb,
            predicted_dBPB_pct_of_bpb_unknown="divide by absolute BPB of the teacher to get a %",
            frac_rules_positive=float(np.mean([v["mean"] > 0 for v in summ.values()])),
            mean_delta_bits_per_event=float(np.mean([v["mean"] for v in summ.values()])))
        print(f"[{spec}] predicted dBPB (H - Scaffold) = {pred_dbpb:+.6f}   rules with positive mean: {report['teachers'][spec]['frac_rules_positive']:.2f}")
    main_spec = a.scorers[0]
    summ = labels_by_teacher[main_spec]
    keys = [k for k in cand if k in summ]
    if len(keys) == 0:
        report["note"] = ("no candidate rule fired on dev_sel text, so there are no labels: the dev text is too small or the re-merge "
                          "events are too rare to study on it (itself a finding about event sparsity). Use a larger dev split.")
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=1, default=float)
        raise SystemExit("STOP: " + report["note"])
    if len(a.scorers) > 1:  # teacher reliability: do labels agree across teachers?
        for s2 in a.scorers[1:]:
            common = [k for k in keys if k in labels_by_teacher[s2]]
            if len(common) > 2:
                rho = spearmanr([summ[k]["mean"] for k in common], [labels_by_teacher[s2][k]["mean"] for k in common])[0]
                report["teachers"][s2]["label_spearman_vs_" + main_spec] = float(rho)
                print(f"label agreement {main_spec} vs {s2}: spearman={rho:.3f} (n={len(common)})")

    X = np.array([[rules[k][f] for f in FEATURES] for k in keys], dtype=float)
    y = np.array([summ[k]["shrunk"] for k in keys])
    pr = predictability(X, FEATURES, y)
    report["predictability"] = {k: v for k, v in pr.items() if k not in ("oof_ridge", "ok_mask")}
    print("predictability:", json.dumps(report["predictability"], indent=1))

    # --- scores for selection over ALL candidate rules ---------------------------------------
    u_freq = np.array([rules[k]["log_r_c"] for k in cand])
    u_comp = np.array([a.alpha * (-rules[k]["dtok_per_event"]) + a.beta * rules[k]["log_r_c"] for k in cand])
    Xall = np.nan_to_num(np.array([[rules[k][f] for f in FEATURES] for k in cand], dtype=float))
    u_pred = np.zeros(len(cand))
    if pr.get("cv", {}).get("spearman") is not None:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        ok = pr["ok_mask"]
        scl = StandardScaler().fit(X[ok])
        rg = Ridge(alpha=1.0).fit(scl.transform(X[ok]), y[ok])
        u_pred = rg.predict(scl.transform(Xall))
    else:
        print("WARNING: too few labelled rules for a learned selector; 'utilpred' falls back to random order")
        u_pred = np.random.default_rng(0).normal(size=len(cand))
    u_orc = np.array([summ[k]["shrunk"] if k in summ else 0.0 for k in cand])

    variants, rows = {}, []
    for frac in a.fracs:
        m = max(1, int(round(frac * len(cand))))
        tag = f"f{int(frac*100)}"
        variants[f"freq_{tag}"] = pick_matched(cand, u_freq, m)
        variants[f"comp_{tag}"] = pick_matched(cand, u_comp, m)
        variants[f"utilpred_{tag}"] = pick_matched(cand, u_pred, m)
        variants[f"oracle_{tag}"] = pick_matched(cand, u_orc, m)
        n_occ = np.array([rules[k]["n_occ"] for k in cand], dtype=float)
        target = float(sum(rules[k]["n_occ"] for k in variants[f"utilpred_{tag}"]))
        for s in range(a.seeds):
            variants[f"random{s}_{tag}"] = pick_random(cand, m, s)               # matched #rules
            variants[f"randomT{s}_{tag}"] = pick_random_traffic(cand, n_occ, target, s)  # matched traffic
    variants["all"] = list(cand)
    wc_dev = count_words(dev_docs, h.pattern)
    for name, rs in variants.items():
        t = h.variant(allowed_remerge=set(rs), meta=dict(h.meta, kind="hselective", variant=name))
        os.makedirs(a.variants_dir or a.tok_dir, exist_ok=True)
        t.save(f"{a.variants_dir or a.tok_dir}/hsel_{name}_{K}.json")   # this repo's format (pattern is stored)
        st = event_stats(t, wc_dev)
        orc = float(np.mean([summ[k]["shrunk"] for k in rs if k in summ])) if rs else float("nan")
        rows.append(dict(variant=name, n_rules=len(rs), changed_rate_dev=st["changed_rate"],
                         fertility_delta_pct=st["fertility_delta_pct"], mean_dev_label_of_selected=orc))
        print(rows[-1])
    report["variants"] = rows
    report["selection_overlap_jaccard_freq_vs_oracle_f50"] = (
        len(set(variants["freq_f50"]) & set(variants["oracle_f50"])) / max(1, len(set(variants["freq_f50"]) | set(variants["oracle_f50"])))
        if "freq_f50" in variants else None)
    # per-rule table (for slides / manual inspection)
    report["rules"] = [dict(rule=list(k), **{f: rules[k][f] for f in FEATURES}, c_text=rules[k]["c_text"], examples=rules[k]["examples"],
                            label=summ.get(k)) for k in cand]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
