#!/usr/bin/env python
"""How often does the team's fast (tiktoken) backend disagree with its reference (python) backend, and with
this repo's encoder, on REAL text?  (docs/AUDIT.md item 2 found disagreements on whitespace/newline boundaries
for the hierarchical variant but could only measure them on synthetic strings.)

Run on the held-out text you will actually tokenise (e.g. WikiText-103 test jsonl, one document per line).
Reports line-level and token-level mismatch rates.  Needs the team's repo on the path (--team_repo).
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scaffold_margin.adapter import load_team_tokenizer
from scaffold_margin.data import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team_repo", default=os.path.join(os.path.dirname(__file__), "..", ".."), help="default: this repo's root (research/ lives inside the team repo)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--text", required=True, help="jsonl with a 'text' field"); ap.add_argument("--max_docs", type=int, default=2000)
    ap.add_argument("--split_lines", action="store_true", help="encode each line of each doc separately (WikiText-style)")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.team_repo, "src"))
    from hscaffold_bpe.model import TokenizerModel
    from hscaffold_bpe.tokenizer import HierarchicalTokenizer

    model = TokenizerModel.load(a.tokenizer)
    ref, fast = HierarchicalTokenizer(model, backend="python"), HierarchicalTokenizer(model, backend="tiktoken")
    mine = load_team_tokenizer(a.tokenizer, cache=True)
    docs = [d for d, _ in read_jsonl(a.text, a.max_docs)]
    units = [ln for d in docs for ln in d.split("\n") if ln] if a.split_lines else docs
    b = lambda t, ids: [model.tokens[t._model_to_internal[i]] for i in ids]
    n = bad_fast = bad_mine = tok = diff_tok = 0
    for u in units:
        r, f = ref.encode(u), fast.encode(u)
        m = [mine.token_bytes[mine.from_model[i]] for i in mine.encode(u)]
        n += 1; tok += len(r)
        if r != f:
            bad_fast += 1; diff_tok += abs(len(r) - len(f))
        bad_mine += (b(ref, r) != m)
    print(f"{n} units, {tok} reference tokens")
    print(f"fast(tiktoken) != reference(python): {bad_fast} units ({100*bad_fast/n:.3f}%), |token-count difference| total = {diff_tok}")
    print(f"this repo's encoder != reference(python): {bad_mine} units ({100*bad_mine/n:.3f}%)")


if __name__ == "__main__":
    main()
