#!/usr/bin/env python
"""Real-vocabulary differential audit and independently counted sparse events."""
import argparse
from collections import Counter
import json
import random
from pathlib import Path

from event_driven_benchmark import file_hash, output_hash, tokenizers
from hscaffold_bpe.model import TokenizerModel
from hscaffold_bpe.tokenizer import HierarchicalTokenizer
from scaffold_margin.bpe import Tokenizer, pretokenize
from scaffold_margin.data import read_jsonl
from scaffold_margin.remerge import event_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hybrid", action="store_true")
    a = ap.parse_args()
    rng = random.Random(20261010)
    texts = ["".join(rng.choices(" abcdeArizona\n\t123.!é한中🙂", k=rng.randrange(1, 200)))
             for _ in range(1000)]
    texts += ["", "\na", "aaa" * 1000, "  \n\n\t", "café<|endoftext|>한글"]
    result = dict(random_seed=20261010, audit=[], events=[])
    for size in (32768, 65536):
        path = a.inputs / f"hierarchical-{size}.json"
        h = Tokenizer.load(path).variant(cache=False, second_pass="full")
        selected = [h.token_bytes[t].decode("utf-8", "replace")
                    for t, level in enumerate(h.levels) if level == 2]
        # Surround L2 surfaces to expose both boundaries; joins exercise many islands.
        stress = ["x " + token + " z" for token in selected]
        docs = texts + stress + [" ".join(stress)]
        model = TokenizerModel.load(path)
        sc = TokenizerModel.load(a.inputs / f"scaffold-{size}.json")
        assert model.visible_ids == sc.visible_ids
        assert all(model.tokens[t] == sc.tokens[t] for t in model.visible_ids)
        for backend in ("margin", "python", "tiktoken"):
            toks = tokenizers(a.inputs, size, backend)
            if a.hybrid and backend != "margin":
                toks["hybrid"] = HierarchicalTokenizer(model, backend=backend, second_pass="hybrid")
            reference = [toks["full"].encode(doc) for doc in docs]
            for name in [key for key in toks if key not in ("full", "scaffold")]:
                outputs = [toks[name].encode(doc) for doc in docs]
                assert outputs == reference, (size, backend, name)
                for doc, ids in zip(docs, outputs):
                    decoded = toks[name].decode(ids)
                    assert decoded == (doc.encode() if isinstance(decoded, bytes) else doc)
            if backend != "margin":
                current_full = HierarchicalTokenizer(model, backend=backend)
                assert [current_full.encode(doc) for doc in docs] == reference
                assert current_full.encode("a", add_eos=True)[-1] == current_full.eos_token_id
            result["audit"].append(dict(K=size, backend=backend, documents=len(docs),
                                        compared_methods=[key for key in toks if key not in ("full", "scaffold")],
                                        stress_L2_surfaces=len(stress), exact=True,
                                        output_sha256=output_hash(reference)))
            print(f"parity K={size} backend={backend} docs={len(docs)} PASS", flush=True)
        for domain in ("wiki_dev_sel", "pg19_excerpts"):
            counts = Counter(word for doc, _ in read_jsonl(a.inputs / f"{domain}.jsonl")
                             for word in pretokenize(doc, h.pattern))
            events = event_stats(h, counts)
            result["events"].append(dict(K=size, domain=domain, **events))
            print(f"events K={size} {domain} pretokens={events['pretokens']} "
                  f"merges={events['merge_events']}", flush=True)
    result["input_sha256"] = {p.name: file_hash(p) for p in a.inputs.glob("*.json*")}
    a.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
