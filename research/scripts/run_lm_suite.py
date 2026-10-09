#!/usr/bin/env python
"""Resumable driver: for every (setting, method, seed) tokenise (if needed) and train, skipping runs whose
results.json already exists.  Safe to re-launch after a crash / preemption.

Example (Direction C, 32K):
  python scripts/run_lm_suite.py --tok_dir tok --text_dir data/text --K 32768 \
     --methods scaffold hscaffold hsel_randomT0_f50 hsel_utilpred_f50 --seeds 0 1 2 \
     --preset small --epochs 5 --gpus 2 --micro_batch 8 --runs runs --setting K32768
Tokenisers with identical L0 sets (scaffold, hscaffold, hsel_*) are tokenised separately because their
streams differ; only the model-id mapping is shared.
"""
import argparse, os, subprocess, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok_dir", required=True); ap.add_argument("--text_dir", required=True)
    ap.add_argument("--K", type=int, required=True); ap.add_argument("--setting", required=True)
    ap.add_argument("--methods", nargs="+", required=True, help="file stems: <method>_<K>.json in tok_dir")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--runs", default="runs"); ap.add_argument("--data_root", default="tok_data")
    ap.add_argument("--gpus", type=int, default=2); ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--splits", nargs="+", default=["lm_train:train", "val:val", "test:test"])
    ap.add_argument("--eval_sets", default="val,test")
    ap.add_argument("--tok_pattern", default="{method}_{K}.json")
    ap.add_argument("--port", type=int, default=29500)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--lm_args", nargs=argparse.REMAINDER, default=[], help="everything after --lm_args is forwarded to train_lm")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, PYTHONPATH=os.path.join(here, "..", "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for m in a.methods:
        tok = os.path.join(a.tok_dir, a.tok_pattern.format(method=m, K=a.K))
        data = os.path.join(a.data_root, f"{a.setting}_{m}")
        if not os.path.exists(os.path.join(data, "meta.json")):
            cmd = [sys.executable, f"{here}/tokenize_corpus.py", "--tok", tok, "--text_dir", a.text_dir, "--out", data,
                   "--workers", str(a.workers), "--splits", *a.splits]
            print(" ".join(cmd)); a.dry or subprocess.run(cmd, check=True, env=env)
        for s in a.seeds:
            out = os.path.join(a.runs, a.setting, m, f"seed{s}")
            if os.path.exists(os.path.join(out, "results.json")):
                print("skip (done):", out); continue
            base = ["torchrun", f"--nproc_per_node={a.gpus}", f"--master_port={a.port}"] if a.gpus > 1 else [sys.executable]
            cmd = base + (["-m", "scaffold_margin.train_lm"] if a.gpus > 1 else ["-m", "scaffold_margin.train_lm"]) + \
                  ["--data", data, "--out", out, "--seed", str(s), "--data_seed", "0", "--eval_sets", a.eval_sets] + a.lm_args
            print(" ".join(cmd)); a.dry or subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
