#!/usr/bin/env python
"""Build causal scaffold-feature streams (<split>_aux.bin) next to an existing canonical Scaffold data dir.

The token streams are symlinked unchanged, so batches/order are identical to the no-feature baseline.
--shuffle_seed gives the identity-destroying control (same firing positions, permuted scaffold ids).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scaffold_margin.adapter import load_team_tokenizer  # noqa: E402
from scaffold_margin.data import open_stream  # noqa: E402
from scaffold_margin.inference import CausalScaffoldFeatures  # noqa: E402

_F = None


def _init(tok_path, seed):
    global _F
    _F = CausalScaffoldFeatures(load_team_tokenizer(tok_path), seed)


def _feat(args):
    path, a, b = args
    s = open_stream(path)
    return a, np.asarray(_F.features(s[a:b].tolist()), dtype=np.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="canonical Scaffold data dir (train/val/test .bin)")
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--shuffle_seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.src, "meta.json")))
    feats = CausalScaffoldFeatures(load_team_tokenizer(a.tok), a.shuffle_seed)
    for f in os.listdir(a.src):
        if f.endswith((".bin", ".npy", ".json")) and f != "meta.json":
            dst = os.path.join(a.out, f)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(os.path.join(a.src, f)), dst)
    meta.update(aux_size=feats.size, aux_shuffle_seed=a.shuffle_seed, aux_table=len(feats.table), aux_maxk=feats.maxk,
                aux_rule="longest scaffold whose demolition equals the last k>=2 tokens (causal)", aux_coverage={})
    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(a.tok, a.shuffle_seed)) as pool:
        for split in a.splits:
            path = os.path.join(a.src, f"{split}.bin")
            n = len(open_stream(path))
            offs = list(np.load(os.path.join(a.src, f"{split}_docs.npy"))) + [n]
            jobs = [(path, int(offs[i]), int(offs[i + 1])) for i in range(len(offs) - 1)]   # per document (+EOS)
            aux = np.zeros(n, dtype=np.uint32)
            for start, f in pool.imap_unordered(_feat, jobs, chunksize=16):
                aux[start:start + len(f)] = f
            aux.tofile(os.path.join(a.out, f"{split}_aux.bin"))
            meta["aux_coverage"][split] = float((aux > 0).mean())
            print(f"{split}: {n:,} positions, feature fires at {100 * (aux > 0).mean():.3f}%", flush=True)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
