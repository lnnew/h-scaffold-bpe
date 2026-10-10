"""Amortize timer noise; same frozen algorithm, thresholds and short samples."""
import argparse
import json
from pathlib import Path
import subprocess

from event_driven_benchmark import file_hash, measure, tokenizers
from event_hybrid_benchmark import frozen_local
from hscaffold_bpe.model import TokenizerModel
from hscaffold_bpe.tokenizer import HierarchicalTokenizer
from scaffold_margin.data import read_jsonl

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT.parent / "event_driven_v1" / "inputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "short_amortized.json")
    output = ap.parse_args().out
    output.parent.mkdir(parents=True, exist_ok=True)
    repo = ROOT.parents[2]
    source_sha = "baa49a507cfc247e7c8c0fc2a3842e536e828080"
    for name in ("src/hscaffold_bpe/remerge.py", "src/hscaffold_bpe/tokenizer.py",
                 "src/hscaffold_bpe/model.py", "src/hscaffold_bpe/pretokenize.py",
                 "research/scripts/event_driven_benchmark.py", "research/scripts/event_hybrid_benchmark.py"):
        assert (repo / name).read_bytes() == subprocess.check_output(
            ["git", "show", source_sha + ":" + name], cwd=repo), name
    docs = [doc for doc, _ in read_jsonl(ROOT / "short_pretokens.jsonl")]
    assert len(docs) == 1024
    old = frozen_local(ROOT / "frozen_local")
    result = dict(source_sha=source_sha, repetitions=7,
                  reason="original short rounds were only 2–3 ms with excessive timer noise",
                  unique_samples=1024, repeats_per_sample=128, encode_calls_per_sweep=131072,
                  input_sha256=file_hash(ROOT / "short_pretokens.jsonl"), measurements=[])
    for size in (32768, 65536):
        model = TokenizerModel.load(INPUTS / f"hierarchical-{size}.json")
        for backend in ("python", "tiktoken"):
            print(f"BEGIN amortized-short K={size} backend={backend}", flush=True)
            toks = dict(full=tokenizers(INPUTS, size, backend)["full"],
                        local=old(model, backend=backend, second_pass="local"),
                        hybrid=HierarchicalTokenizer(model, backend=backend, second_pass="hybrid"))
            row = measure(toks, docs * 128, 7)
            result["measurements"].append(dict(K=size, backend=backend, domain="short_pretokens", **row))
            output.write_text(json.dumps(result, indent=2) + "\n")
            print("END amortized-short", size, backend, flush=True)


if __name__ == "__main__":
    main()
