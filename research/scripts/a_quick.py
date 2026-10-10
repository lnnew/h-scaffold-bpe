#!/usr/bin/env python
"""Direction A pilot: neural utility vs frequency and matched random swaps, on two owned GPUs."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from threading import Event

from run_lm_suite import file_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="existing data/text and tok/apool_pool_* root")
    ap.add_argument("--teacher", type=Path, required=True, help="completed checkpoint trained on the pool stream")
    ap.add_argument("--out", type=Path, required=True, help="new, isolated pilot root")
    ap.add_argument("--K", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.steps <= 0 or a.K <= 256 or a.workers <= 0:
        ap.error("steps/workers must be positive and K must exceed the byte vocabulary")
    scripts = Path(__file__).resolve().parent
    source = scripts.parent / "src/scaffold_margin"
    a.root, a.out, a.teacher = a.root.resolve(), a.out.resolve(), a.teacher.resolve()
    pool = a.root / f"tok/apool_pool_{round(a.K * 1.25)}.json"
    teacher_result = json.loads(a.teacher.with_name("results.json").read_text())
    if file_hash(teacher_result["meta"]["tokenizer"]) != file_hash(pool):
        raise ValueError("teacher must use exactly the pool tokenizer and its token-ID mapping")
    methods = ["apool_freq", "apool_lm", "apool_randomswap0"]
    contract = dict(K=a.K, steps=a.steps, batch_tokens=8192, preset="small", seq=256,
                    seeds=[0, 1, 2], methods=methods, eval_split="val", initialization="shared pool-token embedding bank",
                    teacher=file_hash(a.teacher),
                    pool=file_hash(pool), code={str(p.relative_to(scripts.parent)): file_hash(p)
                                               for p in sorted([*source.glob("*.py"), *scripts.glob("*.py")])},
                    claim_scope="development pilot; test untouched; not final efficacy evidence")
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = a.out / "contract.json"
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError("pilot contract changed; choose a new output root")
    manifest.write_text(json.dumps(contract, indent=2))
    tok = a.out / "tok"
    tok.mkdir(exist_ok=True)
    shutil.copyfile(pool, tok / pool.name)
    env = dict(os.environ, PYTHONPATH=str(source.parent) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               OMP_NUM_THREADS="8")

    def execute(cmd, log_name, gpu):
        with open(a.out / log_name, "a") as log:
            subprocess.run([sys.executable, *map(str, cmd)], cwd=scripts.parent,
                           env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT, check=True)

    teacher = "torch:" + str(a.teacher)
    (a.out / "status.json").write_text(json.dumps({"status": "RUNNING", "stage": "utility scoring"}))
    execute([scripts / "a_marginal_utility.py", "--stage", "select", "--K", a.K, "--tok_dir", tok,
             "--text_dir", a.root / "data/text", "--train_stream", a.root / f"tok_data/K{round(a.K * 1.25)}_apool/train.bin",
             "--scorers", "bigram", teacher, "--select_scorer", teacher, "--score_batch", 128,
             "--out", a.out / "selection.json"], "selection.log", 0)
    selection = json.loads((a.out / "selection.json").read_text())
    if selection["teachers"][teacher]["n_swapped_slots"] == 0:
        (a.out / "status.json").write_text(json.dumps({"status": "NO_INTERVENTION", "reason": "utility selected frequency vocabulary"}))
        return
    sys.path.insert(0, str(source.parent))
    from scaffold_margin.bpe import Tokenizer
    for method in methods:
        tokenizer = Tokenizer.load(str(tok / f"{method}_{a.K}.json"))
        if tokenizer.n_normal != a.K or tokenizer.eos_id != a.K:
            raise ValueError("all arms must have K normal tokens plus the same EOS/padded model size")
    suite = [scripts / "run_lm_suite.py", "--tok_dir", tok, "--text_dir", a.root / "data/text", "--K", a.K,
             "--setting", f"A_K{a.K}_s{a.steps}", "--data_root", a.out / "streams", "--runs", a.out / "runs",
             "--gpus", 1, "--workers", a.workers, "--splits", "lm_train:train", "val:val", "--eval_sets", "val"]
    (a.out / "status.json").write_text(json.dumps({"status": "RUNNING", "stage": "tokenization"}))
    execute([*suite, "--methods", *methods, "--prepare_only"], "prepare.log", 0)
    jobs = Queue()
    for seed in contract["seeds"]:
        for method in methods:
            jobs.put((method, seed))
    stop = Event()

    def worker(gpu):
        while not stop.is_set():
            try:
                method, seed = jobs.get_nowait()
            except Empty:
                return
            try:
                execute([*suite, "--methods", method, "--seeds", seed, "--lm_args", "--preset", "small", "--seq", 256,
                         "--batch_tokens", 8192, "--micro_batch", 16, "--max_steps", a.steps, "--warmup_steps", 100,
                         "--lr", 0.0003, "--init_pool", tok / pool.name, "--save_ckpt"], f"{method}_seed{seed}.log", gpu)
            except Exception:
                stop.set()
                raise

    try:
        (a.out / "status.json").write_text(json.dumps({"status": "RUNNING", "stage": "matched retraining"}))
        with ThreadPoolExecutor(max_workers=2) as workers:
            list(workers.map(worker, [0, 1]))
        execute([scripts / "stats.py", "--runs", a.out / "runs", "--ref", "apool_freq", "--split", "val",
                 "--out", a.out / "pilot_stats.json"], "stats.log", 0)
        execute([scripts / "stats.py", "--runs", a.out / "runs", "--ref", "apool_randomswap0", "--split", "val",
                 "--out", a.out / "pilot_stats_vs_random.json"], "stats_vs_random.log", 0)
        (a.out / "status.json").write_text(json.dumps({"status": "COMPLETE", "claim_scope": contract["claim_scope"]}))
    except Exception as e:
        (a.out / "status.json").write_text(json.dumps({"status": "FAILED", "error": str(e)}))
        raise


if __name__ == "__main__":
    main()
