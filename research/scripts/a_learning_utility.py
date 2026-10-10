#!/usr/bin/env python
"""Offline fixed-K slot exchanges: select on adaptation, confirm on separate documents."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from threading import Event

import numpy as np

from run_lm_suite import file_hash

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scaffold_margin.bpe import Tokenizer
from scaffold_margin.data import read_jsonl
from scaffold_margin.direction_a import pool_candidates, with_selected
from scaffold_margin.learning import prepare_aligned, retokenize, run_probe


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


def propose_pairs(pool, base, support, n, seed=0):
    rng = np.random.default_rng(seed)
    residual = np.asarray(pool.meta["resid"])
    adds = [int(t) for t in pool_candidates(pool) if t not in base and support[t] >= 2]
    removes = {int(t) for t in base if support[t] >= 2}
    rng.shuffle(adds)
    pairs = []
    for c in adds:
        eligible = [t for t in removes if len(pool.token_bytes[t]) == len(pool.token_bytes[c])]
        if not eligible:
            continue
        t = min(eligible, key=lambda t: (abs(np.log1p(residual[t]) - np.log1p(residual[c])), t))
        pairs.append([t, c]); removes.remove(t)
        if len(pairs) == n:
            rng.shuffle(pairs)
            return pairs
    raise ValueError(f"only {len(pairs)} supported equal-length exchanges; need {n}")


def swap(base, pairs):
    removed, added = {t for t, _ in pairs}, {c for _, c in pairs}
    if len(removed) != len(pairs) or len(added) != len(pairs) or not removed <= base or added & base:
        raise ValueError("slot exchanges must be unique and preserve the fixed vocabulary size")
    return (base - removed) | added


def random_matched(pool, base, selected, support, seed):
    candidates = pool_candidates(pool)
    residual = np.asarray(pool.meta["resid"])[candidates]
    bins = np.searchsorted(np.quantile(np.log1p(residual), np.arange(.1, 1, .1)), np.log1p(residual))
    key = {int(t): (int(b), len(pool.token_bytes[t])) for t, b in zip(candidates, bins)}
    rng, result = np.random.default_rng(seed), set(base)
    for actual, eligible, removing in [(base - selected, base, True),
                                      (selected - base, set(candidates) - base, False)]:
        strata = sorted({key[t] for t in actual})
        for stratum in strata:
            count = sum(key[t] == stratum for t in actual)
            options = sorted(t for t in eligible if support[t] >= 2 and key[t] == stratum)
            chosen = set(int(t) for t in rng.choice(options, count, replace=False))
            result = result - chosen if removing else result | chosen
    return result


def prepare(a):
    a.out.mkdir(parents=True, exist_ok=False)
    pool_path = a.root / f"tok/apool_pool_{round(a.K * 1.25)}.json"
    text = a.root / "data/text"
    pool = Tokenizer.load(str(pool_path))
    freq = Tokenizer.load(str(a.root / f"tok/apool_freq_{a.K}.json"))
    base = set(int(t) for t in freq.from_model if t >= 256)
    if freq.n_normal != a.K or not base <= set(pool_candidates(pool)):
        raise ValueError("frequency baseline must contain K pool-normal tokens including bytes")
    teacher_meta = json.loads(a.teacher.with_name("results.json").read_text())["meta"]
    if file_hash(teacher_meta["tokenizer"]) != file_hash(pool_path):
        raise ValueError("teacher tokenizer identity differs from the candidate pool")
    docs = [d for d, _ in read_jsonl(str(text / "dev_sel.jsonl"))]
    if len(docs) < 10:
        raise ValueError("at least ten held-out development documents required")
    # These are development splits used by earlier research, not a new sealed test.
    order = np.random.default_rng(20261010).permutation(len(docs))
    select = [docs[i] for i in order[:len(docs) // 2]]
    confirm = [docs[i] for i in order[len(docs) // 2:]]
    if set(select) & set(confirm):
        raise ValueError("selection and confirmation contain duplicate raw documents")
    corpus = {"select": prepare_aligned(pool, select, a.seq - 1),
              "confirm": prepare_aligned(pool, confirm, a.seq - 1)}
    train_docs = [d for d, _ in read_jsonl(str(text / "lm_train.jsonl"))]
    rng = np.random.default_rng(20261010)
    corpus["train"] = prepare_aligned(pool, (train_docs[i] for i in rng.permutation(len(train_docs))),
                                      a.seq - 1, a.max_train_chunks)
    del train_docs
    (a.out / "data").mkdir(); (a.out / "tok").mkdir()
    for name, values in corpus.items():
        np.savez_compressed(a.out / f"data/{name}.npz", **values)
    pool.save(str(a.out / "tok/pool.json"))
    freq = with_selected(pool, base, "frequency")
    freq.save(str(a.out / "tok/frequency.json"))
    support = np.bincount(pool.from_model[corpus["select"]["ids"]], minlength=len(pool.levels))
    np.save(a.out / "data/support.npy", support)
    pairs = propose_pairs(pool, base, support, a.groups * a.swaps)
    groups = [pairs[i * a.swaps:(i + 1) * a.swaps] for i in range(a.groups)]
    for i, group in enumerate(groups):
        with_selected(pool, swap(base, group), f"group{i}").save(str(a.out / f"tok/group{i}.json"))
    scripts = Path(__file__).resolve().parent
    source = scripts.parent / "src/scaffold_margin"
    contract = dict(K=a.K, groups=groups, steps=a.steps, select_groups=a.select_groups,
                    data_seeds=[0, 1], seq=a.seq, batch_chunks=32, micro_batch=8, lr=.0003,
                    teacher=str(a.teacher), teacher_hash=file_hash(a.teacher),
                    input_hashes={p.name: file_hash(p) for p in [pool_path, text / "dev_sel.jsonl", text / "lm_train.jsonl"]},
                    corpus={n: dict(chunks=len(c["bytes"]), bytes=int(c["bytes"].sum()), raw_hash=str(c["raw_hash"]))
                            for n, c in corpus.items()},
                    code={str(p.relative_to(scripts.parent)): file_hash(p)
                          for p in [Path(__file__), *sorted(source.glob("*.py"))]},
                    split_docs={"select": order[:len(docs) // 2].tolist(), "confirm": order[len(docs) // 2:].tolist()},
                    scope="development calibration; one pool teacher; two data orders, not model initialization seeds",
                    optimizer="AdamW reset in every arm; constant learning rate; byte-normalized loss",
                    gate="negative document CI upper bound vs frequency, static U0 and mean of 3 matched random selectors; both data orders improve; matching and exposure checks pass")
    write(a.out / "contract.json", contract)
    write(a.out / "status.json", dict(status="PREPARED", stage="paired probes"))


def load_corpus(out, names):
    return {n: dict(np.load(out / f"data/{n}.npz", allow_pickle=False)) for n in names}


def verify_contract(out):
    contract = json.loads((out / "contract.json").read_text())
    research = Path(__file__).resolve().parents[1]
    for relative, digest in contract["code"].items():
        if file_hash(research / relative) != digest:
            raise ValueError("code changed after preparation; use a new output root")
    if file_hash(contract["teacher"]) != contract["teacher_hash"]:
        raise ValueError("teacher checkpoint changed after preparation")
    return contract


def one_job(a):
    c = verify_contract(a.out)
    folder = a.out / f"{a.phase}/{a.method}/seed{a.data_seed}"
    if folder.exists():
        raise ValueError("job output already exists; preserve it and use a fresh root")
    identity = job_identity(a.out, a.phase, a.method, a.data_seed)
    run_probe(Tokenizer.load(str(a.out / "tok/pool.json")),
              Tokenizer.load(str(a.out / f"tok/{a.method}.json")),
              load_corpus(a.out, ["train", "select"] + (["confirm"] if a.phase == "joint" else [])),
              folder, teacher=c["teacher"], steps=c["steps"], data_seed=a.data_seed,
              seq=c["seq"], batch_chunks=c["batch_chunks"], micro_batch=c["micro_batch"], lr=c["lr"])
    if identity != job_identity(a.out, a.phase, a.method, a.data_seed):
        raise ValueError("job inputs changed during training")
    write(folder / "identity.json", identity)


def job_identity(out, phase, method, seed):
    return dict(contract=file_hash(out / "contract.json"), tokenizer=file_hash(out / f"tok/{method}.json"),
                data={n: file_hash(out / f"data/{n}.npz") for n in ["train", "select"] + (["confirm"] if phase == "joint" else [])},
                phase=phase, method=method, data_seed=seed)


def run_jobs(a, phase, methods):
    jobs = Queue()
    stop = Event()
    for method in methods:
        for seed in [0, 1]:
            jobs.put((method, seed))
    def worker(gpu):
        while not stop.is_set():
            try:
                method, seed = jobs.get_nowait()
            except Empty:
                return
            folder = a.out / f"{phase}/{method}/seed{seed}"
            if (folder / "results.json").exists():
                if json.loads((folder / "identity.json").read_text()) != job_identity(a.out, phase, method, seed):
                    stop.set()
                    raise ValueError("completed job identity changed; use a fresh output root")
                continue
            command = [sys.executable, str(Path(__file__).resolve()), "--stage", "job", "--out", str(a.out),
                       "--phase", phase, "--method", method, "--data_seed", str(seed)]
            try:
                with open(a.out / f"{phase}_{method}_seed{seed}.log", "a") as log:
                    subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4"),
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            except Exception:
                stop.set()
                raise
            write(a.out / f"gpu{gpu}_progress.json", dict(phase=phase, method=method, seed=seed, status="COMPLETE"))
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(worker, [0, 1]))


def summarize_probes(a, c):
    h = str(c["steps"])
    rows = []
    for i in range(len(c["groups"])):
        u0, uh = [], []
        for seed in c["data_seeds"]:
            root = a.out / "probes"
            base = json.loads((root / f"frequency/seed{seed}/results.json").read_text())
            alt = json.loads((root / f"group{i}/seed{seed}/results.json").read_text())
            assert_exposure(base, alt)
            u0.append(base["curves"]["0"]["select"] - alt["curves"]["0"]["select"])
            uh.append(base["curves"][h]["select"] - alt["curves"][h]["select"])
        rows.append(dict(group=i, U0=float(np.mean(u0)), Uh=float(np.mean(uh)),
                         learning_gain=float(np.mean(uh) - np.mean(u0)), Uh_by_data_seed=uh))
    learned = [r["group"] for r in sorted(rows, key=lambda r: -r["Uh"])
               if r["Uh"] > 0][:c["select_groups"]]
    static = [r["group"] for r in sorted(rows, key=lambda r: -r["U0"])][:len(learned)]
    report = dict(rows=rows, learned_groups=learned, static_groups=static)
    write(a.out / "selection.json", report)
    if not learned:
        return []
    pool = Tokenizer.load(str(a.out / "tok/pool.json"))
    base = set(int(t) for t in Tokenizer.load(str(a.out / "tok/frequency.json")).from_model if t >= 256)
    variants = {"learning": swap(base, [p for i in learned for p in c["groups"][i]]),
                "static": swap(base, [p for i in static for p in c["groups"][i]])}
    support = np.load(a.out / "data/support.npy")
    selection_corpus = load_corpus(a.out, ["select"])["select"]
    def token_count(keep):
        return len(retokenize(selection_corpus, pool, with_selected(pool, keep, "matching"))["ids"])
    base_tokens = token_count(base)
    desired = token_count(variants["learning"]) - base_tokens
    matching = []
    for seed in range(3):
        trials = []
        for attempt in range(20):
            draw = seed * 1000 + attempt
            keep = random_matched(pool, base, variants["learning"], support, draw)
            delta = token_count(keep) - base_tokens
            trials.append((abs(delta - desired), draw, delta, keep))
        error, draw, delta, keep = min(trials, key=lambda row: (row[0], row[1]))
        variants[f"random{seed}"] = keep
        matching.append(dict(selector_seed=seed, draw=draw, token_delta=delta,
                             target_delta=desired, error=error, compression_matched=error <= max(1, abs(desired) * .1)))
    for name, keep in variants.items():
        with_selected(pool, keep, name).save(str(a.out / f"tok/{name}.json"))
    report.update(changed_slots=len(base - variants["learning"]), matching=matching,
                  static_same_as_learning=variants["static"] == variants["learning"])
    write(a.out / "selection.json", report)
    return ["frequency", *variants]


def assert_exposure(base, alt):
    keys = ["raw_train_hash", "train_order_hash", "train_exposure_bytes", "steps"]
    if any(base[k] != alt[k] for k in keys) or base["skipped_updates"] or alt["skipped_updates"]:
        raise ValueError("paired runs differ in raw exposure or successful optimizer updates")


def comparison(a, c, reference):
    h = c["steps"]
    differences, seed_delta = [], []
    sizes = None
    for seed in c["data_seeds"]:
        root = a.out / "joint"
        result = json.loads((root / f"learning/seed{seed}/results.json").read_text())
        target = np.load(root / f"learning/seed{seed}/eval_confirm_h{h}.npz")
        refs = []
        for name in reference:
            base = json.loads((root / f"{name}/seed{seed}/results.json").read_text())
            assert_exposure(base, result)
            other = np.load(root / f"{name}/seed{seed}/eval_confirm_h{h}.npz")
            if not np.array_equal(target["doc_bytes"], other["doc_bytes"]):
                raise ValueError("confirmation document bytes differ between arms")
            refs.append(other["doc_nll_bits"])
        if sizes is not None and not np.array_equal(sizes, target["doc_bytes"]):
            raise ValueError("data-order repetitions have different evaluation documents")
        sizes = target["doc_bytes"]
        difference = target["doc_nll_bits"] - np.mean(refs, axis=0)
        differences.append(difference)
        seed_delta.append(float(difference.sum() / sizes.sum()))
    delta = np.mean(differences, axis=0)
    rng = np.random.default_rng(0)
    indices = rng.integers(len(sizes), size=(4000, len(sizes)))
    boot = delta[indices].sum(1) / sizes[indices].sum(1)
    return dict(delta_bpb=float(delta.sum() / sizes.sum()),
                delta_by_data_seed=seed_delta, doc_ci95=np.quantile(boot, [.025, .975]).tolist(),
                data_order_signs_pass=all(x < 0 for x in seed_delta))


def finish(a, c):
    selection = json.loads((a.out / "selection.json").read_text())
    refs = {"frequency": ["frequency"], "random_mean": [f"random{i}" for i in range(3)], "static": ["static"]}
    comparisons = {n: comparison(a, c, names) for n, names in refs.items()}
    gate = all(comparisons[n]["doc_ci95"][1] < 0 and comparisons[n]["data_order_signs_pass"]
               for n in ["frequency", "random_mean", "static"])
    gate = gate and all(m["compression_matched"] for m in selection["matching"])
    means = {}
    for name in ["frequency", "learning", "static", "random0", "random1", "random2"]:
        means[name] = float(np.mean([json.loads((a.out / f"joint/{name}/seed{s}/results.json").read_text())
                                   ["curves"][str(c["steps"])]["confirm"] for s in c["data_seeds"]]))
    report = dict(status="GO_TO_FROM_SCRATCH_CONFIRMATION" if gate else "NO_GO_WITH_CURRENT_SHORT_ADAPTATION",
                  comparisons=comparisons, confirm_bpb=means, scope=c["scope"],
                  changed_slots=selection["changed_slots"], static_same_as_learning=selection["static_same_as_learning"],
                  interpretation="conditional adaptation evidence only; no from-scratch efficacy or novelty claim")
    write(a.out / "decision.json", report)
    write(a.out / "status.json", dict(status="COMPLETE", decision=report["status"]))
    print(json.dumps(report), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prepare", "run", "job"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--teacher", type=Path)
    ap.add_argument("--K", type=int, default=32768)
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--swaps", type=int, default=16)
    ap.add_argument("--select_groups", type=int, default=4)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--max_train_chunks", type=int, default=32768)
    ap.add_argument("--phase", choices=["probes", "joint"], default="probes")
    ap.add_argument("--method")
    ap.add_argument("--data_seed", type=int, default=0)
    a = ap.parse_args()
    a.out = a.out.resolve()
    if min(a.groups, a.swaps, a.steps, a.max_train_chunks) <= 0 or a.seq < 5 or not 1 <= a.select_groups <= a.groups:
        ap.error("invalid positive budgets or group count")
    if a.stage == "prepare":
        if not a.root or not a.teacher:
            ap.error("prepare requires root and teacher")
        a.root, a.teacher = a.root.resolve(), a.teacher.resolve()
        prepare(a)
    elif a.stage == "job":
        if not a.method or a.data_seed not in [0, 1]:
            ap.error("job requires a method and data_seed 0 or 1")
        one_job(a)
    else:
        c = verify_contract(a.out)
        try:
            write(a.out / "status.json", dict(status="RUNNING", stage="paired probes"))
            run_jobs(a, "probes", ["frequency", *[f"group{i}" for i in range(len(c["groups"]))]])
            methods = summarize_probes(a, c)
            if not methods:
                write(a.out / "decision.json", dict(status="NO_GO_NO_POSITIVE_GROUP", scope=c["scope"]))
                write(a.out / "status.json", dict(status="COMPLETE", decision="NO_GO_NO_POSITIVE_GROUP"))
                return
            write(a.out / "status.json", dict(status="RUNNING", stage="joint confirmation"))
            run_jobs(a, "joint", methods)
            finish(a, c)
        except Exception as error:
            write(a.out / "status.json", dict(status="FAILED", error=str(error)))
            raise


if __name__ == "__main__":
    main()
