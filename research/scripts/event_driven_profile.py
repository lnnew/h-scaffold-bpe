#!/usr/bin/env python
"""Isolated, warm stage costs of the frozen original tiktoken path."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from event_driven_benchmark import tokenizers
from scaffold_margin.data import read_jsonl


def profile(tok, docs):
    assert not any(special in doc for doc in docs for special in tok.model.special_tokens)
    fast = [tok._fast_encoding.encode_ordinary(doc) for doc in docs]
    internal = [[tok._fast_to_internal[t] for t in ids] for ids in fast]
    early = [[part for t in ids for part in tok._demolish_token(t, 1)] for ids in internal]
    second = [tok._apply_bpe(ids, 1) for ids in early]
    final = [[part for t in ids for part in tok._demolish_token(t, 0)] for ids in second]
    output = [[tok._internal_to_model[t] for t in ids] for ids in final]
    assert output == [tok.encode(doc) for doc in docs]
    eligible = {pair for pair, (_, out) in tok._rank.items() if tok.model.levels[out] <= 1}
    stats = dict(documents=len(docs), expanded_tokens=sum(map(len, internal)),
                 documents_with_L2=sum(any(tok.model.levels[t] == 2 for t in ids) for ids in internal),
                 residual_L2_tokens=sum(tok.model.levels[t] == 2 for ids in internal for t in ids),
                 eligible_untouched_edges=sum((a, b) in eligible and tok.model.levels[a] != 2
                                             and tok.model.levels[b] != 2
                                             for ids in internal for a, b in zip(ids, ids[1:])),
                 initial_eligible_after_L2=sum((a, b) in eligible for ids in early
                                             for a, b in zip(ids, ids[1:])),
                 actual_second_pass_joins=sum(len(a) - len(b) for a, b in zip(early, second)))
    stages = {
        "native_first_pass": lambda: [tok._fast_encoding.encode_ordinary(doc) for doc in docs],
        "native_ID_mapping": lambda: [[tok._fast_to_internal[t] for t in ids] for ids in fast],
        "L2_demolition": lambda: [[part for t in ids for part in tok._demolish_token(t, 1)]
                                  for ids in internal],
        "full_second_merge": lambda: [tok._apply_bpe(ids, 1) for ids in early],
        "L1_demolition": lambda: [[part for t in ids for part in tok._demolish_token(t, 0)]
                                  for ids in second],
        "model_ID_mapping": lambda: [[tok._internal_to_model[t] for t in ids] for ids in final],
    }
    times = {name: [] for name in stages}
    rng = random.Random(37)
    for _ in range(7):
        names = list(stages)
        rng.shuffle(names)
        for name in names:
            start = time.perf_counter()
            value = stages[name]()
            elapsed = time.perf_counter() - start
            times[name].append(elapsed)
            del value
    medians = {name: float(np.median(values)) for name, values in times.items()}
    total = sum(medians.values())
    return dict(**stats, isolated_seconds=times, median_seconds=medians,
                isolated_stage_share={name: value / total for name, value in medians.items()},
                caveat="isolated warmed stages; shares are descriptive, not end-to-end causal attribution")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    result = []
    for size in (32768, 65536):
        tok = tokenizers(a.inputs, size, "tiktoken")["full"]
        for domain in ("wiki_dev_sel", "pg19_excerpts"):
            docs = [doc for doc, _ in read_jsonl(a.inputs / f"{domain}.jsonl")]
            result.append(dict(K=size, domain=domain, **profile(tok, docs)))
            print(size, domain, "profile PASS", flush=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
