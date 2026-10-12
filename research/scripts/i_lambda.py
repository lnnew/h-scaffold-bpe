#!/usr/bin/env python
"""Idea 1: compression vs compatibility frontier between canonical Scaffold and min-token DP.

encode:  for each lam, segment val/test with  #tokens + lam * #boundary changes vs canonical  (same visible ids),
         plus the H-Scaffold segmentation mapped into the Scaffold id space.  Writes <out>/seg_<name>/{val,test}.bin
eval:    score frozen checkpoints on every segmentation (no training).
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scaffold_margin.adapter import load_team_tokenizer  # noqa: E402
from scaffold_margin.data import read_jsonl, write_stream  # noqa: E402
from scaffold_margin.inference import VisibleSegmenter  # noqa: E402

_SEG = None


def _init(tok, hs, name, lam):
    global _SEG
    sc = load_team_tokenizer(tok)
    if name == "hscaffold":
        h = load_team_tokenizer(hs, second_pass="full")
        vis = VisibleSegmenter(sc, "canonical").vocab
        _SEG = lambda text: [vis[h.token_bytes[h.from_model[i]]] for i in h.encode(text)]  # noqa: E731
    else:
        s = VisibleSegmenter(sc, "constrained", lam)
        _SEG = s.encode


def _enc(text):
    return _SEG(text)


def encode(a):
    names = [("hscaffold", None)] + [(f"lam{l:g}", l) for l in a.lams]
    sc = load_team_tokenizer(a.tok)
    for name, lam in names:
        out = os.path.join(a.out, f"seg_{name}")
        if os.path.exists(os.path.join(out, "meta.json")):
            continue
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, "token_bytes.npy"), sc.model_token_bytes)
        meta = dict(tokenizer=os.path.abspath(a.tok), n_normal=sc.n_normal, eos_id=sc.eos_id, seg=name, lam=lam, splits={})
        with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(a.tok, a.hs, name, lam)) as pool:
            for split in a.splits:
                docs = [d for d, _ in read_jsonl(os.path.join(a.text_dir, f"{split}.jsonl"))]
                ids = pool.map(_enc, docs, chunksize=1)
                for d, i in zip(docs[:20], ids):
                    assert sc.decode(i).decode("utf-8") == d
                n = write_stream(out, split, ids, sc.eos_id)
                meta["splits"][split] = dict(tokens=n, docs=len(docs))
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
        print(name, {k: v["tokens"] for k, v in meta["splits"].items()}, flush=True)


def evaluate(a):
    import torch

    from scaffold_margin.data import bpb_from_token_nll, doc_ids_for_targets, open_stream
    from scaffold_margin.lm import load_checkpoint
    from scaffold_margin.train_lm import eval_stream

    res = json.load(open(a.results)) if os.path.exists(a.results) else {}
    device = "cuda:0"
    for ck in sorted(glob.glob(a.checkpoints)):
        run = os.path.relpath(os.path.dirname(ck), a.lm_root)
        model, cfg = load_checkpoint(ck, device)
        model.eval()
        amp = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[
            json.load(open(os.path.join(os.path.dirname(ck), "results.json")))["amp"]]
        for seg in sorted(glob.glob(os.path.join(a.out, "seg_*"))):
            for split in a.splits:
                key = f"{run}|{os.path.basename(seg)}|{split}"
                if key in res:
                    continue
                s = open_stream(os.path.join(seg, f"{split}.bin"))
                tb = np.load(os.path.join(seg, "token_bytes.npy"))
                offs = np.load(os.path.join(seg, f"{split}_docs.npy"))
                nll, tg = eval_stream(model, s, tb, cfg.max_seq, device, amp)
                bpb, (dn, db) = bpb_from_token_nll(nll, tg, tb, doc_ids_for_targets(offs, len(s))[1:], len(offs))
                res[key] = dict(bpb=bpb, tokens=int(len(s)), doc_nll_bits=dn.tolist(), doc_bytes=db.tolist())
                print(f"{key} BPB={bpb:.5f}", flush=True)
                json.dump(res, open(a.results, "w"))
        del model
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["encode", "eval"])
    ap.add_argument("--tok")
    ap.add_argument("--hs")
    ap.add_argument("--text_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lams", type=float, nargs="+", default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0])
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--checkpoints", help="glob of model.pt")
    ap.add_argument("--lm_root")
    ap.add_argument("--results")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    encode(a) if a.stage == "encode" else evaluate(a)


if __name__ == "__main__":
    main()
