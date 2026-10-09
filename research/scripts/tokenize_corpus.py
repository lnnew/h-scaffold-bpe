#!/usr/bin/env python
"""Encode text splits with one tokenizer into uint32 streams + BPB bookkeeping files."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scaffold_margin.bpe import Tokenizer
from scaffold_margin.data import encode_docs, read_jsonl, write_stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--text_dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["lm_train:train", "val:val", "test:test", "dev_sel:dev_sel"],
                    help="<jsonl stem>:<output name>; multilingual runs may add e.g. test_de:test_de")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--no_cache", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = Tokenizer.load(a.tok)
    np.save(os.path.join(a.out, "token_bytes.npy"), tok.model_token_bytes)
    meta = dict(tokenizer=os.path.abspath(a.tok), n_normal=tok.n_normal, eos_id=tok.eos_id, mode=tok.mode, splits={})
    for spec in a.splits:
        stem, name = spec.split(":")
        path = os.path.join(a.text_dir, f"{stem}.jsonl")
        if not os.path.exists(path):
            print("skip (missing):", path); continue
        docs = list(read_jsonl(path))
        ids = encode_docs(a.tok, [d for d, _ in docs], a.workers)
        n = write_stream(a.out, name, ids, tok.eos_id, [l for _, l in docs])
        nbytes = sum(len(d.encode()) for d, _ in docs)
        meta["splits"][name] = dict(tokens=n, docs=len(docs), bytes=nbytes, bytes_per_token=nbytes / max(1, n - len(docs) - 1))
        # lossless round-trip check on a sample
        for d, i in list(zip([d for d, _ in docs], ids))[:50]:
            assert tok.decode(i).decode("utf-8") == d, "decode mismatch"
        print(f"{name}: {n:,} tokens, {nbytes:,} bytes, bytes/token={meta['splits'][name]['bytes_per_token']:.3f}")
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
