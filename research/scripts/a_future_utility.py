#!/usr/bin/env python
"""Fixed-budget word/future utility experiment; only owned wine GPUs 0 and 1."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from threading import Event

import numpy as np
from scipy.stats import spearmanr, ttest_1samp

from run_lm_suite import file_hash

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scaffold_margin.aligned_lm import prepare as aligned_prepare, train
from scaffold_margin.bpe import Tokenizer
from scaffold_margin.data import encode_docs, read_jsonl, write_jsonl
from scaffold_margin.direction_a import pool_candidates, with_selected
from scaffold_margin.future import collect, score, summaries
from scaffold_margin.learning import training_batches
from scaffold_margin.scorer import TorchScorer
from scaffold_margin.utility import doc_word_lists


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = [*sorted((root / "src/scaffold_margin").glob("*.py")), Path(__file__).resolve(),
             root / "scripts/run_lm_suite.py"]
    return {str(p.relative_to(root)): file_hash(p) for p in files}


def pg19(out, count=64, max_bytes=16384):
    listing_url = "https://huggingface.co/datasets/deepmind/pg19/raw/main/data/test_files.txt"
    listing = urllib.request.urlopen(listing_url, timeout=30).read()
    names = sorted(listing.decode().splitlines(), key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:count]
    cache = out / "pg19_sources"
    cache.mkdir()
    def fetch(name):
        url = "https://storage.googleapis.com/deepmind-gutenberg/" + name
        raw = urllib.request.urlopen(url, timeout=60).read()
        (cache / Path(name).name).write_bytes(raw)
        text = raw.decode("utf-8")
        start = text.find("\n", len(text) // 4) + 1
        if not start:
            start = len(text) // 4
        excerpt = text[start:].encode()[:max_bytes].decode("utf-8", errors="ignore")
        last_line = excerpt.rfind("\n")
        if last_line > len(excerpt) // 2:
            excerpt = excerpt[:last_line + 1]
        if not excerpt.strip():
            raise ValueError("empty PG-19 excerpt")
        meta = dict(book=name, url=url, source_sha256=hashlib.sha256(raw).hexdigest(),
                    start_character=start, excerpt_sha256=hashlib.sha256(excerpt.encode()).hexdigest(),
                    bytes=len(excerpt.encode()))
        return excerpt, meta
    with ThreadPoolExecutor(max_workers=8) as executor:
        fetched = list(executor.map(fetch, names))
    docs = [text for text, _ in fetched]
    write(out / "pg19_manifest.json", dict(listing_url=listing_url,
          listing_sha256=hashlib.sha256(listing).hexdigest(), books=[m for _, m in fetched],
          selection="first64 SHA256-sorted official test books; quarter-book start; <=16KiB each; no loss filtering"))
    write_jsonl(str(out / "pg19.jsonl"), ((text, "en") for text in docs))
    return docs


def pairs_and_quotas(pool, base, stats, budget):
    candidates = pool_candidates(pool)
    residual = np.asarray(pool.meta["resid"], dtype=float)
    bins = np.searchsorted(np.quantile(np.log1p(residual[candidates]), np.arange(.1, 1, .1)),
                           np.log1p(residual))
    removes, adds = defaultdict(set), []
    for t, values in stats.items():
        if values["n"] >= 2 and residual[t] > 0:
            if t in base:
                removes[len(pool.token_bytes[t])].add(t)
            else:
                adds.append(t)
    grouped = defaultdict(list)
    for c in sorted(adds, key=lambda t: (-residual[t], t)):
        length = len(pool.token_bytes[c])
        available = removes[length]
        if not available:
            continue
        t = min(available, key=lambda t: (abs(np.log1p(residual[t]) - np.log1p(residual[c])), t))
        available.remove(t)
        grouped[(int(bins[t]), int(bins[c]), length)].append([t, c])
    grouped = {key: pairs for key, pairs in grouped.items() if len(pairs) >= 2}
    # Leave alternatives inside every stratum so word/future/random can make different decisions.
    cap = sum(len(pairs) // 2 for pairs in grouped.values())
    budget = min(budget, cap)
    if budget < 4:
        raise ValueError(f"only {budget} matched exchanges with alternative choices")
    keys = sorted(grouped)
    np.random.default_rng(20261011).shuffle(keys)
    quotas = {key: 0 for key in keys}
    while sum(quotas.values()) < budget:
        for key in keys:
            if quotas[key] < len(grouped[key]) // 2:
                quotas[key] += 1
                if sum(quotas.values()) == budget:
                    break
    return grouped, quotas, residual, budget


def apply_pairs(base, pairs):
    removes, adds = {t for t, _ in pairs}, {c for _, c in pairs}
    if len(removes) != len(pairs) or len(adds) != len(pairs) or not removes <= base or adds & base:
        raise ValueError("non-unique or invalid fixed-K exchange")
    return (base - removes) | adds


def token_count(pool, keep, counts):
    """Exact recursive-demolition lengths, computed once in merge dependency order."""
    lengths = np.ones(len(pool.levels), dtype=np.int64)
    for token, (left, right) in enumerate(pool.merges, 256):
        if token not in keep:
            lengths[token] = lengths[left] + lengths[right]
    return int(counts @ lengths[pool.from_model])


def prepare(a):
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "tok").mkdir(); (a.out / "data").mkdir()
    text_root = a.root / "data/text"
    pool_path = a.root / f"tok/apool_pool_{round(a.K * 1.25)}.json"
    pool = Tokenizer.load(str(pool_path))
    freq = Tokenizer.load(str(a.root / f"tok/apool_freq_{a.K}.json"))
    base = set(map(int, freq.from_model[256:]))
    if freq.n_normal != a.K or not base <= set(map(int, pool_candidates(pool))):
        raise ValueError("frequency baseline must be a fixed-K pool subset")
    teacher_meta = json.loads(a.teacher.with_name("results.json").read_text())["meta"]
    if file_hash(teacher_meta["tokenizer"]) != file_hash(pool_path):
        raise ValueError("teacher and candidate pool identities differ")
    files = [text_root / f"{name}.jsonl" for name in ["lm_train", "dev_sel", "val", "test"]]
    contract = dict(K=a.K, source_sha=a.source_sha, source_hashes=source_hashes(), steps=a.steps,
                    seq=a.seq, context_tokens=64, batch_windows=32, micro_batch=8, lr=.0003,
                    seeds=[0, 1, 2], data_seed=0, max_samples=32, scoring_context_pretokens=12,
                    scoring_min_occurrences=2, requested_swaps=32, train_raw_budget_bytes=64 * 1024**2,
                    pair_matching="equal byte length; closest frequency; quotas by removal/addition deciles and byte length",
                    random_draw_limit=1000,
                    min_scheduled_pool_occurrences=8,
                    teacher=str(a.teacher), teacher_hash=file_hash(a.teacher), pool_hash=file_hash(pool_path),
                    input_hashes={p.name: file_hash(p) for p in files},
                    scoring="FP32 common pool+EOS output head; partition in bits; rank residual*shrunk span BPB",
                    primary="PG19 new book excerpts", secondary="WikiText reused test; exploratory",
                    gate="Future beats frequency, word, and one matched-random vocabulary on PG19; all3 seed signs and doc CI upper<0",
                    random_scope="one fixed random vocabulary; three model initialization seeds",
                    scope="fixed-pool fixed-K slot-selection pilot; not full Scaffold vocabulary-construction optimization")
    write(a.out / "contract_draft.json", contract)
    selection_docs = [text for text, _ in read_jsonl(str(text_root / "dev_sel.jsonl"))]
    words = doc_word_lists(selection_docs, pool.pattern)
    edits, seen, rejected = collect(pool, words, max_samples=32, context=12, seq=a.seq)
    print(json.dumps(dict(stage="score", candidates=len(edits), windows=sum(map(len, edits.values())))), flush=True)
    if a.score_cache:
        prior = json.loads((a.score_cache / "contract_draft.json").read_text())
        if any(prior[key] != contract[key] for key in ["input_hashes", "teacher_hash", "pool_hash", "max_samples", "scoring"]):
            raise ValueError("score-cache input or scoring identity mismatch")
        if any(value != contract["source_hashes"].get(name) for name, value in prior["source_hashes"].items()
               if name.startswith("src/")):
            raise ValueError("score-cache implementation changed")
        cached = np.load(a.score_cache / "score_samples.npz")
        rows = {int(t): cached[t] for t in cached.files}
        if set(rows) != set(edits) or any(len(rows[t]) != len(edits[t]) for t in rows):
            raise ValueError("cached windows no longer match deterministic collected edits")
        checks = dict(max_prefix_delta_bits=max(float(np.abs(v[:, 2] - v[:, 0] - v[:, 1]).max()) for v in rows.values()),
                      max_partition_error_bits=0., windows=sum(map(len, rows.values())))
        contract["score_cache"] = dict(root=str(a.score_cache), source_sha=prior["source_sha"],
                                      samples_hash=file_hash(a.score_cache / "score_samples.npz"))
    else:
        scorer = TorchScorer(str(a.teacher), device="cuda:0", batch_size=32, amp=False)
        rows, checks = score(pool, scorer, edits)
        del scorer
    del edits
    import torch
    torch.cuda.empty_cache()
    if checks["max_prefix_delta_bits"] > .001 or checks["max_partition_error_bits"] > 1e-8:
        raise ValueError("prefix invariance or loss partition check failed")
    stats = summaries(rows)
    np.savez_compressed(a.out / "score_samples.npz", **{str(t): v for t, v in rows.items()})
    if not a.training_cache:
        raise ValueError("actual raw training-window cache required for eligibility")
    prior_train = json.loads((a.training_cache / "contract_draft.json").read_text())
    for key in ["input_hashes", "pool_hash", "steps", "seq", "data_seed", "train_raw_budget_bytes"]:
        if prior_train[key] != contract[key]:
            raise ValueError("training support cache contract mismatch")
    cached_train_path = a.training_cache / "data/frequency_train.npz"
    cached_train = dict(np.load(cached_train_path))
    cached_tok = Tokenizer.load(str(a.training_cache / "tok/frequency.json"))
    if set(map(int, cached_tok.from_model[256:])) != base:
        raise ValueError("training cache baseline vocabulary mismatch")
    indices = np.concatenate(list(training_batches(len(cached_train["bytes"]), a.steps, 32, 0)))
    raw_targets = [cached_tok.decode(cached_train["ids"][cached_train["offsets"][i] + cached_train["starts"][i]:
                    cached_train["offsets"][i + 1]]).decode() for i in indices]
    if sum(len(text.encode()) for text in raw_targets) != int(cached_train["bytes"][indices].sum()):
        raise ValueError("cached raw training targets changed bytes")
    pool.save(str(a.out / "tok/pool.json"))
    print(json.dumps(dict(stage="training_eligibility", windows=len(raw_targets))), flush=True)
    encoded_targets = encode_docs(str(a.out / "tok/pool.json"), raw_targets, workers=8)
    support = np.zeros(len(pool.levels), dtype=np.int64)
    support[pool.from_model] = np.bincount(np.concatenate(encoded_targets), minlength=pool.n_normal)
    np.save(a.out / "eligibility_support.npy", support)
    del raw_targets, encoded_targets, cached_train
    eligible_stats = {t: values for t, values in stats.items() if support[t] >= contract["min_scheduled_pool_occurrences"]}
    contract["eligibility"] = dict(cache=str(a.training_cache), cache_hash=file_hash(cached_train_path),
                                  support_hash=file_hash(a.out / "eligibility_support.npy"),
                                  eligible_scored_candidates=len(eligible_stats))
    grouped, quotas, residual, budget = pairs_and_quotas(pool, base, eligible_stats, 32)
    selected = {}
    for method in ["word", "future"]:
        selected[method] = [p for key, quota in quotas.items() for p in sorted(grouped[key],
              key=lambda p: (-(residual[p[1]] * stats[p[1]][method] - residual[p[0]] * stats[p[0]][method]), p))[:quota]]
    variants = {"frequency": base, **{name: apply_pairs(base, pairs) for name, pairs in selected.items()}}
    if variants["future"] == variants["word"]:
        raise ValueError("word and future selected identical vocabularies; comparison has no intervention")
    selection_ids = [pool.encode(text) for text in selection_docs]
    counts = np.bincount([t for doc in selection_ids for t in doc], minlength=pool.n_normal)
    def count_tokens(keep):
        return token_count(pool, keep, counts)
    baseline_count = count_tokens(base)
    future_delta = count_tokens(variants["future"]) - baseline_count
    best = None
    for draw in range(contract["random_draw_limit"]):
        rng = np.random.default_rng(draw)
        pairs = [grouped[key][i] for key, quota in quotas.items()
                 for i in rng.choice(len(grouped[key]), quota, replace=False)]
        keep = apply_pairs(base, pairs)
        if keep in [variants["word"], variants["future"]]:
            continue
        delta = count_tokens(keep) - baseline_count
        trial = (abs(delta - future_delta), draw, delta, pairs, keep)
        if best is None or trial[:2] < best[:2]:
            best = trial
        if trial[0] == 0:
            break
    if best is None:
        raise ValueError("no distinct matched random vocabulary")
    error, draw, delta, selected["random"], variants["random"] = best
    if error > max(1, abs(future_delta) * .1):
        raise ValueError("random selection could not match the future token-count change")
    all_tokens = sorted(stats)
    rank = float(spearmanr([stats[t]["word"] for t in all_tokens], [stats[t]["future"] for t in all_tokens]).statistic)
    weighted_rank = float(spearmanr([residual[t] * stats[t]["word"] for t in all_tokens],
                                    [residual[t] * stats[t]["future"] for t in all_tokens]).statistic)
    half_stats = [summaries({t: v[(v[:, 4] < len(selection_docs) // 2) == (half == 0)]
                            for t, v in rows.items() if ((v[:, 4] < len(selection_docs) // 2) == (half == 0)).sum() >= 2})
                  for half in range(2)]
    common = sorted(set(half_stats[0]) & set(half_stats[1]))
    stability = {method: float(spearmanr([half_stats[0][t][method] for t in common],
                                        [half_stats[1][t][method] for t in common]).statistic)
                 for method in ["word", "future"]} if len(common) >= 3 else {}
    write(a.out / "selection.json", dict(swaps=budget, selected=selected, quotas={str(k): v for k, v in quotas.items()},
          checks=checks, rejected=rejected, scored_candidates=len(stats), seen=seen, word_future_rho=rank,
          weighted_word_future_rho=weighted_rank,
          split_rank_rho=stability, split_common_candidates=len(common),
          token_delta={name: count_tokens(keep) - baseline_count for name, keep in variants.items()},
          random=dict(draw=draw, target_delta=future_delta, delta=delta, error=error, compression_matched=True),
          differing_word_future_slots=len(variants["future"] - variants["word"])))
    for name, keep in variants.items():
        with_selected(pool, keep, name).save(str(a.out / f"tok/{name}.json"))
    pool.save(str(a.out / "tok/pool.json"))
    tokenizers = {name: Tokenizer.load(str(a.out / f"tok/{name}.json")) for name in variants}
    documents = [text for text, _ in read_jsonl(str(text_root / "lm_train.jsonl"))]
    order = np.random.default_rng(20261011).permutation(len(documents))
    train_docs, size = [], 0
    for i in order:
        train_docs.append(documents[i]); size += len(documents[i].encode())
        if size >= contract["train_raw_budget_bytes"]:
            break
    pg_docs = [text for text, _ in read_jsonl(str(a.training_cache / "pg19.jsonl"))]
    shutil.copyfile(a.training_cache / "pg19.jsonl", a.out / "pg19.jsonl")
    shutil.copyfile(a.training_cache / "pg19_manifest.json", a.out / "pg19_manifest.json")
    shutil.copytree(a.training_cache / "pg19_sources", a.out / "pg19_sources")
    digest = lambda text: hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
    pg_hashes = set(map(digest, pg_docs))
    if len(pg_hashes) != len(pg_docs) or pg_hashes & set(map(digest, documents + selection_docs)):
        raise ValueError("new holdout overlaps training or selection documents")
    del documents, rows
    datasets = dict(train=train_docs, val=[text for text, _ in read_jsonl(str(text_root / "val.jsonl"))],
                    wiki=[text for text, _ in read_jsonl(str(text_root / "test.jsonl"))], pg19=pg_docs)
    contract["data"] = {}
    for split, docs in datasets.items():
        print(json.dumps(dict(stage="align", split=split, documents=len(docs))), flush=True)
        prepared = aligned_prepare(pool, tokenizers, docs, seq=a.seq, context=64)
        contract["data"][split] = dict(raw_hash=str(prepared["frequency"]["raw_hash"]),
              bytes=int(prepared["frequency"]["bytes"].sum()), windows=len(prepared["frequency"]["bytes"]), documents=len(docs),
              target_tokens={name: int(np.diff(corpus["offsets"]).sum() - corpus["starts"].sum())
                             for name, corpus in prepared.items()})
        for name, corpus in prepared.items():
            if split == "train" and name != "frequency":
                indices = np.concatenate(list(training_batches(len(corpus["bytes"]), a.steps, 32, 0)))
                targets = np.concatenate([corpus["ids"][corpus["offsets"][i] + corpus["starts"][i]:corpus["offsets"][i + 1]]
                                          for i in indices])
                support = np.bincount(targets, minlength=a.K)
                added = [int(tokenizers[name].to_model[c]) for _, c in selected[name]]
                if support[added].min() < 2:
                    raise ValueError("selected token has insufficient actual training exposure")
                contract.setdefault("training_support", {})[name] = dict(min_added_occurrences=int(support[added].min()))
            np.savez_compressed(a.out / f"data/{name}_{split}.npz", **corpus)
    contract.update(methods=list(variants), swaps=budget, pg19_manifest_hash=file_hash(a.out / "pg19_manifest.json"),
                    selection_hash=file_hash(a.out / "selection.json"),
                    corpus_hashes={p.name: file_hash(p) for p in sorted((a.out / "data").glob("*.npz"))},
                    tokenizer_hashes={p.name: file_hash(p) for p in sorted((a.out / "tok").glob("*.json"))})
    write(a.out / "contract.json", contract)
    write(a.out / "status.json", dict(status="PREPARED", swaps=budget))
    print(json.dumps(dict(stage="prepared", swaps=budget, word_future_rho=rank, matching_error=error)), flush=True)


def verify(out):
    c = json.loads((out / "contract.json").read_text())
    if c["source_hashes"] != source_hashes():
        raise ValueError("source changed after contract freeze")
    for sub, key in [("data", "corpus_hashes"), ("tok", "tokenizer_hashes")]:
        for name, expected in c[key].items():
            if file_hash(out / sub / name) != expected:
                raise ValueError(f"frozen artifact changed: {sub}/{name}")
    if file_hash(out / "selection.json") != c["selection_hash"]:
        raise ValueError("selection changed after contract freeze")
    return c


def job(a):
    c = verify(a.out)
    if a.method not in c["methods"] or a.seed not in c["seeds"]:
        raise ValueError("job is outside the approved contract")
    out = a.out / f"runs/{a.method}/seed{a.seed}"
    identity = dict(contract_hash=file_hash(a.out / "contract.json"), method=a.method, seed=a.seed)
    if out.exists():
        if not (out / "identity.json").exists() or json.loads((out / "identity.json").read_text()) != identity:
            raise ValueError("existing job identity mismatch")
        if (out / "results.json").exists():
            return
        raise ValueError("incomplete job preserved; choose a new run root")
    pool = Tokenizer.load(str(a.out / "tok/pool.json"))
    tokenizer = Tokenizer.load(str(a.out / f"tok/{a.method}.json"))
    if tokenizer.n_normal != c["K"]:
        raise ValueError("fixed vocabulary budget changed")
    corpora = {split: dict(np.load(a.out / f"data/{a.method}_{split}.npz")) for split in c["data"]}
    result = train(pool, tokenizer, corpora, out, steps=c["steps"], seed=a.seed, data_seed=c["data_seed"],
                   seq=c["seq"], batch_windows=c["batch_windows"], micro_batch=c["micro_batch"], lr=c["lr"])
    write(out / "identity.json", identity)
    if result["skipped_updates"]:
        raise ValueError("failed optimization pairing")


def finish(out, c):
    means, comparisons = {}, {}
    for split in ["pg19", "wiki", "val"]:
        means[split] = {name: float(np.mean([json.loads((out / f"runs/{name}/seed{s}/results.json").read_text())
                           ["bpb"][split] for s in c["seeds"]])) for name in c["methods"]}
        comparisons[split] = {}
        for ref in ["frequency", "word", "random"]:
            differences, by_seed, sizes = [], [], None
            for seed in c["seeds"]:
                root = out / "runs"
                target_result = json.loads((root / f"future/seed{seed}/results.json").read_text())
                ref_result = json.loads((root / f"{ref}/seed{seed}/results.json").read_text())
                keys = ["raw_train_hash", "train_order_hash", "train_exposure_bytes", "steps"]
                if any(target_result[k] != ref_result[k] for k in keys):
                    raise ValueError("raw exposure mismatch between training arms")
                target = np.load(root / f"future/seed{seed}/eval_{split}.npz")
                other = np.load(root / f"{ref}/seed{seed}/eval_{split}.npz")
                if not np.array_equal(target["doc_bytes"], other["doc_bytes"]):
                    raise ValueError("evaluation raw bytes differ")
                sizes = target["doc_bytes"]
                d = target["doc_nll_bits"] - other["doc_nll_bits"]
                differences.append(d); by_seed.append(float(d.sum() / sizes.sum()))
            d = np.mean(differences, axis=0)
            idx = np.random.default_rng(0).integers(len(sizes), size=(4000, len(sizes)))
            boot = d[idx].sum(1) / sizes[idx].sum(1)
            ci = np.quantile(boot, [.025, .975]).tolist()
            comparisons[split][ref] = dict(delta_bpb=float(d.sum() / sizes.sum()),
                    delta_by_seed=by_seed, doc_ci95=ci, all_seeds_improve=all(v < 0 for v in by_seed),
                    paired_t_p=float(ttest_1samp(by_seed, 0).pvalue))
    gate = all(row["doc_ci95"][1] < 0 and row["all_seeds_improve"] for row in comparisons["pg19"].values())
    decision = dict(status="GO_TO_INDEPENDENT_REPLICATION" if gate else "NO_GO_FOR_THIS_FUTURE_PROXY",
                    means=means, comparisons=comparisons, primary=c["primary"], scope=c["scope"],
                    caveats=["one teacher and one random vocabulary", "PG19 is book-domain generalization",
                             "WikiText test was used by earlier research", "local score is a proxy, not true retraining utility"])
    write(out / "decision.json", decision)
    write(out / "status.json", dict(status="COMPLETE", decision=decision["status"], fits=12))
    print(json.dumps(dict(status=decision["status"], means=means)), flush=True)


def run(a):
    c = verify(a.out)
    queue, stop = Queue(), Event()
    for seed in c["seeds"]:
        for method in c["methods"]:
            queue.put((method, seed))
    script = Path(__file__).resolve()
    def worker(gpu):
        while not stop.is_set():
            try:
                method, seed = queue.get_nowait()
            except Empty:
                return
            write(a.out / f"gpu{gpu}_progress.json", dict(method=method, seed=seed, status="RUNNING"))
            command = [sys.executable, str(script), "--stage", "job", "--out", str(a.out),
                       "--method", method, "--seed", str(seed)]
            try:
                with (a.out / f"{method}_seed{seed}.log").open("w") as log:
                    subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4"),
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            except Exception:
                stop.set()
                raise
            write(a.out / f"gpu{gpu}_progress.json", dict(method=method, seed=seed, status="COMPLETE"))
            print(json.dumps(dict(method=method, seed=seed, gpu=gpu, status="COMPLETE")), flush=True)
    write(a.out / "status.json", dict(status="RUNNING", stage="12 from-scratch fits"))
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(worker, [0, 1]))
    finish(a.out, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prepare", "run", "job"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--teacher", type=Path)
    ap.add_argument("--source_sha")
    ap.add_argument("--score_cache", type=Path)
    ap.add_argument("--training_cache", type=Path)
    ap.add_argument("--K", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--method")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out = a.out.resolve()
    try:
        if a.stage == "prepare":
            if not a.root or not a.teacher or not a.source_sha:
                ap.error("prepare requires root, teacher, and frozen source SHA")
            prepare(a)
        elif a.stage == "job":
            job(a)
        else:
            run(a)
    except Exception as error:
        if a.out.exists():
            write(a.out / "status.json", dict(status="FAILED", stage=a.stage, error=str(error)))
        raise


if __name__ == "__main__":
    main()
