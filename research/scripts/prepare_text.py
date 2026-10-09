#!/usr/bin/env python
"""Materialise text splits as jsonl so every later stage is data-source agnostic.

Splits written to --out:
  lm_train.jsonl   LM training text
  tok_train.jsonl  tokenizer-training text (defaults to a copy/symlink of lm_train)
  dev_sel.jsonl    held-out text used ONLY for utility labels / rule or token selection
  val.jsonl        held-out text for monitoring / hyper-parameters
  test.jsonl       final reporting only (never used for selection or tuning)

Sources
  synthetic   offline smoke-test corpus
  wikitext103 HF `wikitext/wikitext-103-raw-v1`  (valid is split 50/50 into dev_sel / val by article)
  files       --train/--valid/--test point at local .txt/.jsonl (jsonl needs a "text" field)

Leakage note: if --tok_source is a *full Wikipedia dump* it contains the WikiText-103 test
articles.  That only touches tokenizer merges (same for every method), not LM training, but
state it in the write-up.  The default (tokenizer text == LM training text) has no such leak.
"""
import argparse, os, re, sys, json, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scaffold_margin.data import write_jsonl, read_jsonl

HEAD = re.compile(r"^ = [^=].* = \n$")


def split_wikitext(lines):
    docs, cur = [], []
    for ln in lines:
        if HEAD.match(ln) and cur:
            docs.append("".join(cur)); cur = []
        cur.append(ln)
    if cur:
        docs.append("".join(cur))
    return [d for d in docs if d.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["synthetic", "wikitext103", "files"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--train"); ap.add_argument("--valid"); ap.add_argument("--test")
    ap.add_argument("--tok_source", default=None, help="optional jsonl (or HF wikipedia spec 'hf:wikimedia/wikipedia:20231101.en:N_DOCS') for tokenizer training")
    ap.add_argument("--hf_cache", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.source == "synthetic":
        from scaffold_margin.synth import make_corpus
        tr, va, te = (make_corpus(1500, seed=1, stem_seed=0), make_corpus(200, seed=2, stem_seed=0),
                      make_corpus(200, seed=3, stem_seed=0))
    elif a.source == "wikitext103":
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=a.hf_cache)
        tr, va, te = (split_wikitext(ds[s]["text"]) for s in ("train", "validation", "test"))
    else:
        def load(p):
            if p.endswith(".jsonl"):
                return [t for t, _ in read_jsonl(p)]
            return split_wikitext(open(p).readlines()) if "wiki" in p else open(p).read().split("\n\n")
        tr, va, te = load(a.train), load(a.valid), load(a.test)

    half = len(va) // 2
    for name, docs in dict(lm_train=tr, dev_sel=va[:half], val=va[half:], test=te).items():
        n = write_jsonl(os.path.join(a.out, f"{name}.jsonl"), ((d, "en") for d in docs))
        print(f"{name}: {n} docs")
    if a.tok_source and a.tok_source.startswith("hf:"):
        from datasets import load_dataset
        _, name, cfg, n = a.tok_source.split(":")
        ds = load_dataset(name, cfg, split="train", streaming=True)
        docs = (r["text"] for _, r in zip(range(int(n)), ds))
        print("tok_train:", write_jsonl(os.path.join(a.out, "tok_train.jsonl"), ((d, "en") for d in docs)), "docs (HF)")
    elif a.tok_source:
        shutil.copy(a.tok_source, os.path.join(a.out, "tok_train.jsonl"))
    else:
        shutil.copy(os.path.join(a.out, "lm_train.jsonl"), os.path.join(a.out, "tok_train.jsonl"))


if __name__ == "__main__":
    main()
