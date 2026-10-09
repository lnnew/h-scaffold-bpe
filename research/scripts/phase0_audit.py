#!/usr/bin/env python
"""Phase 0: reproduction / sanity audit of Scaffold-BPE vs H-Scaffold-BPE.

Answers (machine-checkable subset of the brief's Phase-0 list):
  1 identical final normal vocabulary + compatible ids          5 lossless decoding
  2 L1/L2 classification re-derived from stored r(t), q          6 attempted / accepted / changed rates
  3 every output token is L0                                     7 tau ablation (L2 count, second-merge events)
  4 n_normal == K                                                8 fertility / bytes-per-token deltas
Items that cannot be checked from tokenizer files alone (BPB code, seeds) are covered by
tests/test_pipeline.py and by scripts/run_seeds.sh + scripts/stats.py.
"""
import argparse, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scaffold_margin.adapter import attach_residuals, find_tokenizer
from scaffold_margin.bpe import Tokenizer
from scaffold_margin.data import read_jsonl
from scaffold_margin.pretok import count_words, pretokenize
from scaffold_margin.remerge import event_stats, level_counts


# Targets transcribed from hscaffold_bpe_report_v2 (slide 14).  |S| = L1 + L2 (constant across tau).
REPORT_SCAFFOLD = {32768: 2601, 65536: 4889}
REPORT_TAU = {   # K -> tau -> (L2, L1, L2 pretokens, 2nd merge, cross-boundary)
    32768: {0.25: (1322, 1279, 180, 26, 24), 0.5: (2118, 483, 466, 51, 48), 0.75: (2477, 124, 627, 78, 72)},
    65536: {0.25: (2595, 2294, 103, 13, 13), 0.5: (3988, 901, 247, 24, 23), 0.75: (4677, 212, 339, 28, 26)},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok_dir", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--text_dir", required=True)
    ap.add_argument("--eval_split", default="dev_sel", help="NOT test: the audit may inform design decisions")
    ap.add_argument("--repro_split", default="test", help="the report measured events on the WikiText-103 TEST set (~247K pretokens); "
                    "this split is used ONLY for the tau-ablation reproduction table, never for selection")
    ap.add_argument("--domain_files", nargs="*", default=[], help="name=path.jsonl pairs for per-domain fertility (report: WikiText/WMT17/BLiMP)")
    ap.add_argument("--sample_pretokens", type=int, default=247000, help="size of tau-ablation sample (the earlier report used ~247K)")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--tau_files", nargs="*", default=[], help="tau=path.json (hierarchical tokenizers trained separately per tau, e.g. the team's artifacts/ablation/tokenizers/t025/hierarchical-<K>.json); "
                    "required for the tau table when the main tokenizer has no stored ratio")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    K = a.K
    sc = Tokenizer.load(find_tokenizer(a.tok_dir, "scaffold", K))
    h = Tokenizer.load(find_tokenizer(a.tok_dir, "hscaffold", K))
    bpe = Tokenizer.load(find_tokenizer(a.tok_dir, "bpe", K))
    rep = dict(K=K, checks={}, stats={})
    chk = rep["checks"]

    chk["identical_L0_vocab_and_ids"] = bool(np.array_equal(sc.from_model, h.from_model))
    extra = 1 if sc.meta.get("source") == "hscaffold_bpe" else 0     # team's K counts <|endoftext|> as a visible token
    chk["n_normal_equals_K"] = all(t.n_normal + extra == K for t in (sc, h, bpe))
    stored = sc.levels[256:]
    ratio = np.array(sc.meta["ratio"])[256:] if "ratio" in sc.meta else None
    if ratio is not None:
        tau = sc.meta["tau"]
        lv = np.where(stored == 0, 0, np.where(ratio < tau, 2, 1))
        chk["levels_mismatch_vs_rederived"] = int((lv != stored).sum())
        chk["levels_note"] = "re-derived from stored r/f(Qhead) at last judgement; nonzero only if the trainer demoted tokens to hit exactly K"
    else:
        chk["levels_mismatch_vs_rederived"] = None
        chk["levels_note"] = "tokenizer file has no stored judgement ratio (team format): levels taken as given; use --tau_files for the tau table"
    chk["scaffold_rule"] = sc.meta.get("scaffold_rule", sc.meta.get("source"))
    chk["n_scaffold_tokens(L1+L2)"] = int((stored > 0).sum())
    chk["scaffold_count_vs_report"] = REPORT_SCAFFOLD.get(K)

    docs = [d for d, _ in read_jsonl(os.path.join(a.text_dir, f"{a.eval_split}.jsonl"))]
    ok_l0 = lossless = True
    for d in docs[:300]:
        for t in (sc, h, bpe):
            ids = t.encode(d)
            ok_l0 &= max(ids) < t.n_normal
            lossless &= t.decode(ids).decode("utf-8") == d
    chk["all_outputs_L0"], chk["lossless_decode"] = bool(ok_l0), bool(lossless)

    full = h.variant(second_pass="full", cache=False)
    ev = h.variant(second_pass="event", cache=False)
    chk["event_triggered_pass_equals_full_pass"] = all(full.encode(d) == ev.encode(d) for d in docs[:100])
    chk["H_with_empty_allowlist_equals_scaffold"] = all(h.variant(allowed_remerge=set()).encode(d) == sc.encode(d) for d in docs[:100])

    wc_eval = count_words(docs, h.pattern)
    wc_path = f"{a.tok_dir}/word_counts.pkl"
    wc_train = pickle.load(open(wc_path, "rb")) if os.path.exists(wc_path) else None
    rep["levels"] = level_counts(h)
    if wc_train is not None:
        rep["stats"]["train"] = event_stats(h, wc_train)
    rep["stats"][a.eval_split] = event_stats(h, wc_eval)
    rep["stats"][a.eval_split]["note"] = "attempted~'L2 pretokens', accepted~'2nd merge', cross_boundary_events~'Cross-boundary' in the report"

    # fertility / bytes-per-token on eval split
    nb = sum(len(d.encode()) for d in docs)
    nw = sum(len(d.split()) for d in docs)
    for name, t in (("bpe", bpe), ("scaffold", sc), ("hscaffold", h)):
        n = sum(len(t.encode(d)) for d in docs)
        rep["stats"][f"fertility_{name}"] = dict(tokens=n, tokens_per_word=n / nw, bytes_per_token=nb / n)

    # tau ablation on the first N pretoken occurrences of the eval split
    rdocs = [d for d, _ in read_jsonl(os.path.join(a.text_dir, f"{a.repro_split}.jsonl"))]
    occ = []
    for d in rdocs:
        occ.extend(pretokenize(d, h.pattern))
        if len(occ) >= a.sample_pretokens:
            break
    from collections import Counter
    wc_s = Counter(occ[: a.sample_pretokens])
    abl = []
    tau_variants = {}
    if a.tau_files:
        for spec in a.tau_files:
            t_, path = spec.split("=")
            tau_variants[float(t_)] = Tokenizer.load(path)
    elif ratio is not None:
        for t_ in a.taus:
            lv_full = sc.levels.copy(); lv_full[256:] = np.where(sc.levels[256:] == 0, 0, np.where(ratio < t_, 2, 1))
            tau_variants[t_] = h.variant(levels=lv_full)
    for t_, ht in sorted(tau_variants.items()):
        lv_full = ht.levels
        assert ht.n_normal == h.n_normal
        st = event_stats(ht, wc_s)
        abl.append(dict(tau=t_, L2=int((lv_full[256:] == 2).sum()), L1=int((lv_full[256:] == 1).sum()),
                        sample_pretokens=st["pretokens"],
                        L2_pretokens=int(round(st["attempted_rate"] * st["pretokens"])),
                        second_merge_events_occurrences=int(round(st["accepted_rate"] * st["pretokens"])),
                        second_merge_events_types=st["accepted_types"],
                        cross_boundary_events=int(st["cross_boundary_events"]),
                        report_target=dict(zip(("L2", "L1", "L2_pretokens", "second_merge", "cross_boundary"),
                                               REPORT_TAU.get(K, {}).get(t_, ()))) or None,
                        changed_rate=st["changed_rate"], fertility_delta_pct=st["fertility_delta_pct"]))
    rep["tau_ablation"] = abl
    rep["tau_ablation_note"] = (f"measured on split '{a.repro_split}'; report_target is from the team's report. Exact agreement is NOT expected "
                                "(different tokenizer corpus / independent implementation); order of magnitude and the monotone trends are the check")
    if a.domain_files:   # per-domain fertility and its sigma/CV (report's 'Domain sigma / CV')
        dom = {}
        for spec in a.domain_files:
            nm, path = spec.split("=")
            dd = [d for d, _ in read_jsonl(path)]
            nw = sum(len(d.split()) for d in dd)
            dom[nm] = {n: sum(len(t.encode(d)) for d in dd) / nw for n, t in (("scaffold", sc), ("hscaffold", h))}   # tokens per whitespace word
        for n in ("scaffold", "hscaffold"):
            v = np.array([dom[k][n] for k in dom])
            rep.setdefault("domain_fertility", {})[n] = dict(per_domain={k: dom[k][n] for k in dom}, sigma=float(v.std()), cv=float(v.std() / v.mean()))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep["checks"], indent=1))
    s = rep["stats"][a.eval_split]
    print(f"[{a.eval_split}] attempted={s['attempted_rate']:.5f} accepted={s['accepted_rate']:.5f} changed={s['changed_rate']:.5f} "
          f"fertility_delta={s['fertility_delta_pct']:+.4f}%")
    for r_ in abl:
        print(r_)


if __name__ == "__main__":
    main()
