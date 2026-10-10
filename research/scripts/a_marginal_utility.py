#!/usr/bin/env python
"""Direction A (researcher): is token frequency a good proxy for the value of a vocabulary slot?

Stages
  pool    train Scaffold-BPE to K_pool = K*(1+pool_frac) normal tokens; save the pool tokenizer.
          (then: tokenize_corpus.py on it, and optionally train a neural teacher LM on that data)
  select  score every pool token with model-based marginal utility on held-out dev text,
          build matched-size vocabularies (freq / compression / LM-utility / random), save them,
          and print the *tokenizer-only go/no-go gate*.

Gate (decides whether any LM retraining is worthwhile):
  * Spearman(frequency, V) and Jaccard(freq-selected, LM-selected): if the sets are ~identical,
    Direction A is vacuous at this scale.
  * predicted BPB difference of LM-selected vs freq-selected under the teacher (first order):
    if it is far below seed-to-seed noise, it cannot be detected with the planned budget.
"""
import argparse, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scipy.stats import spearmanr
from scaffold_margin.bpe import Tokenizer, train_bpe
from scaffold_margin.data import encode_docs, read_jsonl
from scaffold_margin.direction_a import (matched_random_swaps, n_pieces, pool_candidates, pool_tokenizer,
                                       select_supported, token_value_labels, with_selected)
from scaffold_margin.scorer import BigramScorer
from scaffold_margin.utility import doc_word_lists, pick_matched, pick_random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pool", "select"])
    ap.add_argument("--tok_dir", required=True); ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--pool_frac", type=float, default=0.25)
    ap.add_argument("--text_dir"); ap.add_argument("--scorers", nargs="+", default=["bigram"])
    ap.add_argument("--select_scorer", help="teacher used for apool_lm; defaults to the last listed scorer")
    ap.add_argument("--train_stream", help="pool-tokenised uint32 stream for the bigram teacher")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--score_batch", type=int, default=256)
    ap.add_argument("--max_per_token", type=int, default=16); ap.add_argument("--ctx_words", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--team_repo", default=None, help="path to the team's h-scaffold-bpe checkout: build the pool with ITS trainer "
                    "(queue-based scaffold revival, bytes-keyed tokens) instead of this repo's approximation")
    ap.add_argument("--word_counts", default=None, help="pickled Counter[bytes,int]; default <tok_dir>/word_counts.pkl")
    ap.add_argument("--out")
    a = ap.parse_args()
    K, Kp = a.K, int(round(a.K * (1 + a.pool_frac)))
    pool_path = f"{a.tok_dir}/apool_pool_{Kp}.json"

    if a.stage == "pool":
        wc = pickle.load(open(a.word_counts or f"{a.tok_dir}/word_counts.pkl", "rb"))
        if a.team_repo:
            from scaffold_margin.adapter import team_pool_tokenizer
            pool = team_pool_tokenizer(wc, Kp, a.team_repo)
            pool.save(pool_path)
            print(f"pool tokenizer (team trainer) saved: {pool_path}  (K_pool={Kp}, expanded merges={len(pool.merges)})")
            return
        res = train_bpe(wc, target_normal=Kp)
        pool_tokenizer(res, Kp).save(pool_path)
        print(f"pool tokenizer saved: {pool_path}  (K_pool={Kp}, expanded merges={len(res.merges)})")
        return

    pool = Tokenizer.load(pool_path)
    selected_scorer = a.select_scorer or a.scorers[-1]
    if selected_scorer not in a.scorers or len(set(a.scorers)) != len(a.scorers):
        ap.error("select_scorer must be listed in scorers; duplicate scorers are not allowed")
    cands = pool_candidates(pool)
    r = np.array(pool.meta["resid"], dtype=float)
    dev_docs = [d for d, _ in read_jsonl(os.path.join(a.text_dir, "dev_sel.jsonl"))]
    if len(dev_docs) < 2 or any(not d for d in dev_docs):
        ap.error("dev_sel needs at least two nonempty documents for disjoint selection and validation")
    dev_bytes = sum(len(d.encode()) for d in dev_docs)
    # Labels (used to SELECT) come from even docs; the predicted-BPB check uses odd docs, so the
    # reported benefit is not just the selection criterion grading its own homework.
    words_A, words_B = doc_word_lists(dev_docs[0::2], pool.pattern), doc_word_lists(dev_docs[1::2], pool.pattern)
    bytes_B = sum(len(d.encode()) for d in dev_docs[1::2])

    from scaffold_margin.data import open_stream
    report = dict(K=K, K_pool=Kp, n_candidates=len(cands), selected_scorer=selected_scorer, teachers={})
    keep_n = K - 256
    freq_sel = pick_matched(list(cands), r[cands], keep_n)
    comp_sel = pick_matched(list(cands), np.array([r[t] * (n_pieces(pool, t) - 1) for t in cands]), keep_n)
    variants = {"freq": freq_sel, "comp": comp_sel}
    for s in range(a.seeds):
        variants[f"random{s}"] = pick_random(list(cands), keep_n, s)

    for ti, spec in enumerate(a.scorers):
        if spec == "bigram":
            sco = BigramScorer(pool.n_normal + 1).fit([np.asarray(open_stream(a.train_stream), dtype=np.int64)])
        else:
            if not spec.startswith("torch:"):
                ap.error("scorers must be bigram or torch:<checkpoint>")
            from scaffold_margin.scorer import TorchScorer
            sco = TorchScorer(spec[6:], device=a.device, batch_size=a.score_batch)
            if sco.cfg.vocab_size < pool.n_normal + 1:
                raise ValueError("teacher vocabulary cannot represent the pool; train it on the pool stream")
        summ, seen = token_value_labels(pool, words_A, sco, a.ctx_words, a.max_per_token)
        summB, seenB = token_value_labels(pool, words_B, sco, a.ctx_words, a.max_per_token)
        mean_bits = np.array([summ[t]["shrunk"] if t in summ else 0.0 for t in cands])  # bits saved per occurrence by keeping t
        seen_dev = np.array([seen.get(t, 0) for t in cands], dtype=float)
        V = r[cands] * mean_bits                     # corpus-scale marginal value (train frequency x bits)
        # held-out value on half B (bits saved by keeping t, scaled to B's occurrence counts)
        mean_B = np.array([summB[t]["shrunk"] if t in summB else 0.0 for t in cands])
        V_dev = np.array([seenB.get(t, 0) for t in cands], dtype=float) * mean_B
        V_dev_in = seen_dev * mean_bits              # in-sample (optimistic) version
        lm_sel = select_supported(cands, V, freq_sel, seen_dev)
        variant = f"lm_teacher{ti}"
        variants[variant] = lm_sel
        if spec == selected_scorer:
            variants["lm"] = lm_sel
            for s in range(a.seeds):
                variants[f"randomswap{s}"] = matched_random_swaps(cands, r[cands], freq_sel, lm_sel, s, seen_dev)
        idx = {int(t): i for i, t in enumerate(cands)}
        def dev_value(sel, V=None): return float(sum((V_dev if V is None else V)[idx[t]] for t in sel))
        covered = int(sum(1 for t in cands if t in summ))
        rho_rv = spearmanr(r[cands], V)[0]
        rho_rm = spearmanr(r[cands][[idx[int(t)] for t in cands if t in summ]], mean_bits[[idx[int(t)] for t in cands if t in summ]])[0] if covered > 2 else float("nan")
        jac = len(set(freq_sel) & set(lm_sel)) / len(set(freq_sel) | set(lm_sel))
        rep = dict(tokenizer_variant=f"apool_{variant}", tokens_with_dev_evidence=covered, spearman_freq_vs_V=float(rho_rv),
                   spearman_freq_vs_mean_bits_per_occurrence=float(rho_rm),
                   jaccard_freq_vs_lm=jac, n_swapped_slots=len(set(lm_sel) - set(freq_sel)),
                   unsupported_decisions="frozen to frequency baseline",
                   predicted_dBPB_lm_minus_freq_heldout=-(dev_value(lm_sel) - dev_value(freq_sel)) / bytes_B,
                   predicted_dBPB_lm_minus_freq_insample=-(dev_value(lm_sel, V_dev_in) - dev_value(freq_sel, V_dev_in)) / (dev_bytes - bytes_B),
                   predicted_dBPB_comp_minus_freq_heldout=-(dev_value(comp_sel) - dev_value(freq_sel)) / bytes_B,
                   predicted_dBPB_random_minus_freq_heldout=[-(dev_value(variants[f'random{s}']) - dev_value(freq_sel)) / bytes_B for s in range(a.seeds)],
                   note="first-order, additive; ignores interactions between simultaneous removals; a fixed-teacher proxy, not a retrained-BPB estimate")
        report["teachers"][spec] = rep
        print(f"[{spec}]", json.dumps(rep, indent=1))
        if spec == selected_scorer:
            fs, ls = set(freq_sel), set(lm_sel)
            def row(t): return dict(token=pool.token_bytes[t].decode("utf-8", "replace"), resid=float(r[t]),
                                    bits_saved_per_occ=float(mean_bits[idx[t]]), n_dev=int(seen_dev[idx[t]]))
            report["only_in_lm_selection"] = sorted((row(t) for t in ls - fs), key=lambda d: -d["bits_saved_per_occ"])[:100]
            report["only_in_freq_selection"] = sorted((row(t) for t in fs - ls), key=lambda d: d["bits_saved_per_occ"])[:100]
    for name, sel in variants.items():
        with_selected(pool, sel, f"apool_{name}").save(f"{a.tok_dir}/apool_{name}_{K}.json")
    report["variants"] = {k: len(v) for k, v in variants.items()}
    json.dump(report, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
