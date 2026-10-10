#!/usr/bin/env python
"""Encode text splits with a fixed visible vocabulary and an alternative inference method (E2 streams).

--method canonical|longest|dp   one segmentation for every split
--method mix                     train: per-document uniform choice among canonical/longest/dp (fixed seed);
                                 eval splits: canonical
Every output dir also gets cross-evaluation streams test_<m>.bin / val_<m>.bin for m in canonical/longest/dp,
so each model can be scored on all three segmentations of the same text (same ids, same bytes).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scaffold_margin.adapter import load_team_tokenizer  # noqa: E402
from scaffold_margin.bpe import Tokenizer  # noqa: E402
from scaffold_margin.data import read_jsonl, write_stream  # noqa: E402
from scaffold_margin.inference import METHODS, VisibleSegmenter  # noqa: E402

_SEG = None


def load_tok(path):
    d = json.load(open(path))
    return load_team_tokenizer(path) if d.get("format") == "hierarchical-scaffold-bpe-v1" else Tokenizer.load(path)


def _init(path):
    global _SEG
    tok = load_tok(path)
    _SEG = {m: VisibleSegmenter(tok, m) for m in METHODS}


def _enc(job):
    method, text = job
    return _SEG[method].encode(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--text_dir", required=True)
    ap.add_argument("--method", required=True, choices=list(METHODS) + ["mix"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["lm_train:train", "val:val", "test:test"])
    ap.add_argument("--cross", nargs="+", default=["val", "test"])
    ap.add_argument("--mix_seed", type=int, default=20261011)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = load_tok(a.tok)
    np.save(os.path.join(a.out, "token_bytes.npy"), tok.model_token_bytes)
    meta = dict(tokenizer=os.path.abspath(a.tok), n_normal=tok.n_normal, eos_id=tok.eos_id, mode=tok.mode,
                inference=a.method, splits={})
    rng = np.random.default_rng(a.mix_seed)
    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(a.tok,)) as pool:
        def run(stem, name, method_of_doc):
            docs = list(read_jsonl(os.path.join(a.text_dir, f"{stem}.jsonl")))
            methods = method_of_doc(len(docs))
            ids = pool.map(_enc, list(zip(methods, [d for d, _ in docs])), chunksize=32)
            n = write_stream(a.out, name, ids, tok.eos_id, [l for _, l in docs])
            nbytes = sum(len(d.encode()) for d, _ in docs)
            for (d, _), i in list(zip(docs, ids))[:200]:
                assert tok.decode(i).decode("utf-8") == d, "decode mismatch"
            meta["splits"][name] = dict(tokens=n, docs=len(docs), bytes=nbytes,
                                        bytes_per_token=nbytes / max(1, n - len(docs) - 1),
                                        methods={m: methods.count(m) for m in set(methods)})
            print(f"{name}: {n:,} tokens bytes/token={meta['splits'][name]['bytes_per_token']:.4f}", flush=True)

        for spec in a.splits:
            stem, name = spec.split(":")
            if a.method == "mix" and name == "train":
                run(stem, name, lambda n: [METHODS[i] for i in rng.integers(0, 3, n)])
            else:
                m = "canonical" if a.method == "mix" else a.method
                run(stem, name, lambda n, m=m: [m] * n)
        for stem in a.cross:
            for m in METHODS:
                run(stem, f"{stem}_{m}", lambda n, m=m: [m] * n)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
