#!/usr/bin/env python
"""Back-of-envelope compute budget with EXPLICIT assumptions (nothing here is measured).

FLOPs/token ~= 6 * (non-embedding params + vocab*d_model) + 12 * layers * d_model * seq   (PaLM-style)
Wall-clock = total FLOPs / (n_gpu * peak_TFLOPs * 1e12 * MFU).   --peak and --mfu are inputs: run
scripts/autotune.py on the target machine to replace the assumed MFU by a measured tokens/s.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scaffold_margin.lm import PRESETS, make_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, required=True, help="training tokens per run (train tokens x epochs)")
    ap.add_argument("--vocab", type=int, nargs="+", default=[32768, 65536])
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--peak", type=float, required=True, help="assumed peak dense fp16 TFLOPs per GPU (look it up for the actual card)")
    ap.add_argument("--mfu", type=float, required=True, help="assumed model-FLOPs utilisation, e.g. 0.25")
    ap.add_argument("--runs", type=int, default=1, help="number of runs to multiply by (methods x seeds x vocab sizes)")
    a = ap.parse_args()
    print(f"ASSUMPTIONS: peak={a.peak} TFLOPs/GPU x {a.gpus} GPUs, MFU={a.mfu}, {a.tokens/1e6:.0f}M tokens/run, seq={a.seq}")
    print(f"{'preset':8s} {'vocab':>6s} {'params(M)':>10s} {'non-emb(M)':>11s} {'GFLOP/tok':>10s} {'hours/run':>10s} {'hours x runs':>13s}")
    for p in PRESETS:
        for V in a.vocab:
            c = make_config(p, V + 1, a.seq)
            f = 6 * (c.n_params(False) + V * c.d_model) + 12 * c.n_layers * c.d_model * a.seq
            h = f * a.tokens / (a.gpus * a.peak * 1e12 * a.mfu) / 3600
            print(f"{p:8s} {V:6d} {c.n_params()/1e6:10.1f} {c.n_params(False)/1e6:11.1f} {f/1e9:10.2f} {h:10.1f} {h*a.runs:13.1f}")


if __name__ == "__main__":
    main()
