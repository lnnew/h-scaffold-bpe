#!/usr/bin/env python
"""Train BPE + Scaffold-BPE (+ H-Scaffold view) at each vocabulary size.

Outputs  <out>/{bpe,scaffold,hscaffold}_<K>.json  and a pickled word-count table so
re-runs with other K / tau skip the (slow) counting stage.
"""
import argparse, os, pickle, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scaffold_margin.bpe import build_tokenizers
from scaffold_margin.data import count_words_parallel, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="tok_train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, nargs="+", default=[32768, 65536])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--scaffold_rule", default="global", choices=["component", "global"],
                    help="global (default) = identical to the team's trainer on synthetic counts; component = literal slide-2 reading, does not match. "
                         "Prefer training with the team's own trainer (hscaffold CLI) and loading its JSON here.")
    ap.add_argument("--pattern", default="margin", choices=["margin", "gpt2"],
                    help="pretokenizer; use gpt2 for counts that will be used with the team's tokenizers")
    ap.add_argument("--counts_only", action="store_true", help="only write word_counts.pkl and stop")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--max_docs", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    wc_path = os.path.join(a.out, "word_counts.pkl")
    if os.path.exists(wc_path):
        wc = pickle.load(open(wc_path, "rb"))
        print("loaded cached word counts:", len(wc), "types")
    else:
        t = time.time()
        wc = count_words_parallel((d for d, _ in read_jsonl(a.text, a.max_docs)), a.workers, pattern=a.pattern)
        pickle.dump(wc, open(wc_path, "wb"))
        print(f"counted {sum(wc.values()):,} pretokens ({len(wc):,} types) in {time.time()-t:.0f}s")
    if a.counts_only:
        return
    for K in a.vocab:
        t = time.time()
        toks = build_tokenizers(wc, K, tau=a.tau, scaffold_rule=a.scaffold_rule, pattern=a.pattern)
        for name, tok in toks.items():
            tok.save(os.path.join(a.out, f"{name}_{K}.json"))
        print(f"K={K}: trained in {time.time()-t:.0f}s; expanded merges={len(toks['scaffold'].merges)}; "
              f"L1={int((toks['scaffold'].levels==1).sum())} L2={int((toks['scaffold'].levels==2).sum())}", flush=True)


if __name__ == "__main__":
    main()
