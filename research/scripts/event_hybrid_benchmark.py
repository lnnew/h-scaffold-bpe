#!/usr/bin/env python
"""One fixed Hybrid comparison, including separate short pretoken calls."""
import argparse
import importlib.util
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

from event_driven_benchmark import file_hash, measure, tokenizers
from hscaffold_bpe.model import TokenizerModel
from hscaffold_bpe.pretokenize import iter_pretokens
from hscaffold_bpe.tokenizer import HierarchicalTokenizer
from scaffold_margin.data import read_jsonl


def frozen_local(root):
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
        return value
    kernel = module("hscaffold_bpe._frozen_local_kernel", root / "remerge.py")
    tokenizer = module("hscaffold_bpe._frozen_local_tokenizer", root / "tokenizer.py")
    tokenizer.event_remerge = kernel.event_remerge
    return tokenizer.HierarchicalTokenizer


def short_inputs(inputs, out):
    pool = sorted({word.decode() for doc, _ in read_jsonl(inputs / "wiki_dev_sel.jsonl")
                   for word in iter_pretokens(doc) if len(word) <= 32})
    assert len(pool) >= 1024
    docs = random.Random(20261011).sample(pool, 1024)
    path = out.parent / "short_pretokens.jsonl"
    with path.open("w") as stream:
        for text in docs:
            stream.write(json.dumps(dict(text=text, lang="en"), ensure_ascii=False) + "\n")
    return docs, dict(sha256=file_hash(path), unique_pool=len(pool), sampled=len(docs),
                      seed=20261011, max_UTF8_bytes=32, policy="one pretoken per encode call")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=7)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    short, short_meta = short_inputs(a.inputs, a.out)
    old = frozen_local(a.out.parent / "frozen_local")
    data = dict(source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                baseline_sha="0113d8e", local_sha="8c873596e687c557ae7593e55587f7345e66bd4e",
                host=platform.uname()._asdict(), repetitions=a.reps, short_input=short_meta,
                input_sha256={p.name: file_hash(p) for p in a.inputs.glob("*.json*")},
                dispatch=dict(min_expanded_length=128, min_distinct_initial_ranks=4, tuned=False),
                measurements=[])
    for size in (32768, 65536):
        model = TokenizerModel.load(a.inputs / f"hierarchical-{size}.json")
        for backend in ("python", "tiktoken"):
            full = tokenizers(a.inputs, size, backend)["full"]
            toks = dict(full=full, local=old(model, backend=backend, second_pass="local"),
                        hybrid=HierarchicalTokenizer(model, backend=backend, second_pass="hybrid"))
            for domain in ("wiki_dev_sel", "pg19_excerpts", "short_pretokens"):
                docs = short if domain == "short_pretokens" else [doc for doc, _ in read_jsonl(
                    a.inputs / f"{domain}.jsonl")]
                print(f"BEGIN K={size} backend={backend} domain={domain}", flush=True)
                row = measure(toks, docs, a.reps)
                data["measurements"].append(dict(K=size, backend=backend, domain=domain, **row))
                a.out.write_text(json.dumps(data, indent=2) + "\n")
                print("END " + " ".join(f"{name}={v['MBps_median']:.3f}MB/s"
                                         for name, v in row["methods"].items()), flush=True)


if __name__ == "__main__":
    main()
