#!/usr/bin/env python
"""B + C evaluation of frozen checkpoints (no training).

B  loss by training exposure of the target token: test bits and bytes per exposure bucket
   (exposure = how often the token is emitted when the tokenizer encodes the training word counts).
C  robustness: BPB on deterministically perturbed test text (character typos in a fraction of words),
   encoded with the same segmentation the model was trained with.

Manifest (JSON list): {"run": name, "ckpt": path, "tok": tokenizer json, "method": canonical|longest|dp}
"""
import argparse
import json
import os
import pickle
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scaffold_margin.adapter import load_team_tokenizer  # noqa: E402
from scaffold_margin.bpe import Tokenizer  # noqa: E402
from scaffold_margin.data import read_jsonl  # noqa: E402
from scaffold_margin.inference import VisibleSegmenter  # noqa: E402
from scaffold_margin.lm import load_checkpoint  # noqa: E402
from scaffold_margin.train_lm import eval_stream  # noqa: E402

BUCKETS = [0, 100, 1_000, 10_000, 100_000, 1_000_000, 10**18]


def load_tok(path):
    d = json.load(open(path))
    return load_team_tokenizer(path) if d.get("format") == "hierarchical-scaffold-bpe-v1" else Tokenizer.load(path)


def perturb(text, rate, seed):
    """Typo in `rate` of alphabetic words: swap / delete / insert / substitute one inner letter."""
    rng = random.Random(seed)
    out = []
    for w in text.split(" "):
        if len(w) > 3 and w.isalpha() and rng.random() < rate:
            i = rng.randrange(1, len(w) - 1)
            op = rng.randrange(4)
            ch = rng.choice("abcdefghijklmnopqrstuvwxyz")
            w = (w[:i] + w[i + 1] + w[i] + w[i + 2:] if op == 0 else w[:i] + w[i + 1:] if op == 1
                 else w[:i] + ch + w[i:] if op == 2 else w[:i] + ch + w[i + 1:])
        out.append(w)
    return " ".join(out)


def stream_of(seg, docs, eos):
    s = [eos]
    for d in docs:
        s += seg.encode(d) + [eos]
    return np.array(s, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--text_dir", required=True)
    ap.add_argument("--word_counts", required=True)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.05, 0.15])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    docs = [d for d, _ in read_jsonl(os.path.join(a.text_dir, "test.jsonl"))]
    variants = {"clean": docs, **{f"typo{r:g}": [perturb(d, r, 1000 + i) for i, d in enumerate(docs)] for r in a.rates}}
    wc = pickle.load(open(a.word_counts, "rb"))
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    segs, expo = {}, {}
    for job in json.load(open(a.manifest)):
        if job["run"] in res:
            continue
        key = (job["tok"], job["method"])
        if key not in segs:
            tok = load_tok(job["tok"])
            seg = VisibleSegmenter(tok, job["method"])
            e = np.zeros(tok.n_normal + 1, dtype=np.int64)
            for w, n in wc.items():
                for t in seg.encode_word(w):
                    e[t] += n
            segs[key], expo[key] = (tok, seg), e
        tok, seg = segs[key]
        model, cfg = load_checkpoint(job["ckpt"], "cuda:0")
        model.eval()
        tb = tok.model_token_bytes
        amp = torch.bfloat16 if json.load(open(os.path.join(os.path.dirname(job["ckpt"]), "results.json")))["amp"] == "bf16" else torch.float16
        out = {}
        for name, ds in variants.items():
            s = stream_of(seg, ds, tok.eos_id)
            nll, tg = eval_stream(model, s, tb, cfg.max_seq, "cuda:0", amp)
            bits = nll / np.log(2)
            by = tb[tg]
            out[name] = dict(bpb=float(bits.sum() / by.sum()), tokens=int(len(s)), bytes=int(by.sum()))
            if name == "clean":
                e = expo[key][tg]
                rows = []
                for lo, hi in zip(BUCKETS, BUCKETS[1:]):
                    m = (e >= lo) & (e < hi) & (tg != tok.eos_id)
                    rows.append(dict(lo=lo, hi=hi, positions=int(m.sum()), bytes=int(by[m].sum()), bits=float(bits[m].sum()),
                                     bits_per_byte=float(bits[m].sum() / max(1, by[m].sum()))))
                out["by_exposure"] = rows
        res[job["run"]] = out
        print(job["run"], {k: round(v["bpb"], 5) for k, v in out.items() if "bpb" in v}, flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
