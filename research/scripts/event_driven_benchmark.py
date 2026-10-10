#!/usr/bin/env python
"""Fixed-input, rotated-order benchmark; every repetition checks exact IDs."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from hscaffold_bpe.model import TokenizerModel
from hscaffold_bpe.tokenizer import HierarchicalTokenizer
from scaffold_margin.bpe import Tokenizer
from scaffold_margin.data import read_jsonl


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def output_hash(outputs):
    digest = hashlib.sha256()
    for ids in outputs:
        digest.update(len(ids).to_bytes(8, "little"))
        digest.update(np.asarray(ids, dtype="<u4").tobytes())
    return digest.hexdigest()


def baseline_class(inputs, backend):
    filename, package, classname = (("bpe.py", "scaffold_margin", "Tokenizer") if backend == "margin"
                                    else ("tokenizer.py", "hscaffold_bpe", "HierarchicalTokenizer"))
    path = inputs.parent / "baseline" / filename
    spec = importlib.util.spec_from_file_location(f"{package}._event_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, classname)


def tokenizers(inputs, size, backend):
    paths = {name: inputs / f"{name}-{size}.json" for name in ("scaffold", "hierarchical")}
    original = baseline_class(inputs, backend)
    if backend == "margin":
        sc, h = (Tokenizer.load(paths[name]) for name in paths)
        def old(tok, mode):
            return original(tok.merges, tok.levels, mode=mode, pattern=tok.pattern,
                            second_pass="full", cache=False)
        return {"scaffold": old(sc, "scaffold"), "full": old(h, "hscaffold"),
                "skip": h.variant(cache=False, second_pass="event"),
                "local": h.variant(cache=False, second_pass="local")}
    sc, h = (TokenizerModel.load(paths[name]) for name in paths)
    return {"scaffold": original(sc, backend=backend), "full": original(h, backend=backend),
            **{mode: HierarchicalTokenizer(h, backend=backend, second_pass=mode)
               for mode in ("skip", "local")}}


def measure(toks, docs, reps):
    reference = [toks["full"].encode(doc) for doc in docs]
    expected = output_hash(reference)
    methods = {}
    for name, tok in toks.items():
        outputs = [tok.encode(doc) for doc in docs]
        if name != "scaffold":
            assert outputs == reference, f"warmup parity failed: {name}"
        for doc, ids in zip(docs, outputs):
            decoded = tok.decode(ids)
            assert decoded == (doc.encode() if isinstance(decoded, bytes) else doc)
        methods[name] = dict(seconds=[], token_count=sum(map(len, outputs)),
                             output_sha256=output_hash(outputs))
    rng = random.Random(20261010)
    order = []
    for rep in range(reps):
        names = list(toks)
        rng.shuffle(names)
        order.append(names)
        for name in names:
            start = time.perf_counter()
            outputs = [toks[name].encode(doc) for doc in docs]
            elapsed = time.perf_counter() - start
            methods[name]["seconds"].append(elapsed)
            assert output_hash(outputs) == methods[name]["output_sha256"], f"repeat parity: {name}"
            print(f"round={rep + 1} method={name} seconds={elapsed:.4f}", flush=True)
    nbytes = sum(len(doc.encode()) for doc in docs)
    for name, data in methods.items():
        times = np.asarray(data["seconds"])
        speedups = np.asarray(methods["full"]["seconds"]) / times
        data.update(MBps_median=float(nbytes / 1e6 / np.median(times)),
                    speedup_over_full_median=float(np.median(speedups)),
                    speedup_range=[float(min(speedups)), float(max(speedups))])
    return dict(documents=len(docs), raw_bytes=nbytes, methods=methods, round_order=order,
                exact_H_output_sha256=expected, exact_ids_all_rounds=True,
                changed_H_vs_scaffold_documents=sum(a != b for a, b in zip(
                    reference, [toks["scaffold"].encode(doc) for doc in docs])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[32768, 65536])
    ap.add_argument("--backends", nargs="+", default=["margin", "python", "tiktoken"])
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert a.reps >= 2
    files = sorted(a.inputs.glob("*.json*"))
    result = dict(source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  host=platform.uname()._asdict(), python=platform.python_version(),
                  packages={name: importlib.metadata.version(name) for name in ("numpy", "tiktoken", "regex")},
                  input_sha256={p.name: file_hash(p) for p in files}, repetitions=a.reps,
                  baseline_sha="0113d8e", baseline_file_sha256={
                      p.name: file_hash(p) for p in (a.inputs.parent / "baseline").glob("*.py")},
                  policy="single CPU process; no encoding cache; warm demolition cache; median; rotated order",
                  measurements=[])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for size in a.sizes:
        for backend in a.backends:
            toks = tokenizers(a.inputs, size, backend)
            for domain in ("wiki_dev_sel", "pg19_excerpts"):
                docs = [doc for doc, _ in read_jsonl(a.inputs / f"{domain}.jsonl", a.max_docs)]
                print(f"BEGIN K={size} backend={backend} domain={domain}", flush=True)
                data = measure(toks, docs, a.reps)
                result["measurements"].append(dict(K=size, backend=backend, domain=domain, **data))
                a.out.write_text(json.dumps(result, indent=2) + "\n")
                print("END " + " ".join(f"{name}={row['MBps_median']:.3f}MB/s"
                                         for name, row in data["methods"].items()), flush=True)


if __name__ == "__main__":
    main()
