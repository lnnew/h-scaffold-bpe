#!/usr/bin/env python
"""Instrument copies of actual kernels; never use these timings as throughput."""
import argparse
from collections import defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import time

import numpy as np

from event_driven_benchmark import tokenizers
from hscaffold_bpe.remerge import event_remerge
from hscaffold_bpe.tokenizer import HierarchicalTokenizer
from scaffold_margin.data import read_jsonl


def instrument(fn, mode, stats, root):
    source = textwrap.dedent(inspect.getsource(fn))
    original_hash = hashlib.sha256(source.encode()).hexdigest()
    if mode == "full":
        changes = [
            ("        best_pair: tuple[int, int] | None = None", "        _scan = _clock()\n        best_pair: tuple[int, int] | None = None"),
            ("        if best_pair is None:", "        _stats['full_scan'] += _clock() - _scan\n        if best_pair is None:"),
            ("        left, right = best_pair", "        _apply = _clock()\n        left, right = best_pair"),
            ("        sequence = result", "        sequence = result\n        _stats['merge_application'] += _clock() - _apply"),
        ]
    else:
        changes = [
            ("    if candidates is None:", "    _gen = _clock()\n    if candidates is None:"),
            ("    if not heap:\n        return list(sequence)", "    _stats['candidate_generation'] += _clock() - _gen\n    if not heap:\n        _out = _clock()\n        result = list(sequence)\n        _stats['output_reconstruction'] += _clock() - _out\n        return result"),
            ("    heapify(heap)", "    _setup = _clock()\n    heapify(heap)"),
            ("    def push(left):", "    _stats['state_setup'] += _clock() - _setup\n\n    def push(left):"),
            ("        rank = heap[0][0]", "        _queue = _clock()\n        _application = 0.0\n        rank = heap[0][0]"),
            ("            values[left] = rule[1]", "            _apply = _clock()\n            values[left] = rule[1]"),
            ("            batch_merge = (*pair, rule[1])", "            batch_merge = (*pair, rule[1])\n            _application += _clock() - _apply"),
            ("        if changed:", "        _stats['merge_application'] += _application\n        _stats['queue_pop_validation'] += _clock() - _queue - _application\n        if changed:"),
            ("            for left in changed:", "            _push = _clock()\n            for left in changed:"),
            ("                    push(left)", "                    push(left)\n            _stats['neighbor_candidate_push'] += _clock() - _push"),
            ("    result = []", "    _out = _clock()\n    result = []"),
            ("\n    return result\n", "\n    _stats['output_reconstruction'] += _clock() - _out\n    return result\n"),
        ]
    for anchor, replacement in changes:
        assert source.count(anchor) == 1, (mode, anchor)
        source = source.replace(anchor, replacement)
    path = root / f"instrumented_{mode}.py"
    path.write_text(source)
    namespace = dict(fn.__globals__, _stats=stats, _clock=time.perf_counter)
    exec(compile(source, str(path), "exec"), namespace)
    return namespace[fn.__name__], original_hash


def collector(tok):
    source = textwrap.dedent(inspect.getsource(HierarchicalTokenizer._hybrid_merge))
    assert source.count("    if not candidates:") == 1
    source = source.split("    if not candidates:")[0] + "    return early, candidates, ranks\n"
    namespace = dict(HierarchicalTokenizer._hybrid_merge.__globals__)
    exec(compile(source, "fused_collector", "exec"), namespace)
    return lambda ids: namespace["_hybrid_merge"](tok, ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    result = dict(caveat="instrumented copies, warm inputs, timers add overhead; not end-to-end throughput",
                  repetitions=7, measurements=[])
    for size in (32768, 65536):
        toks = tokenizers(a.inputs, size, "tiktoken")
        full, tok = toks["full"], toks["local"]
        collect = collector(tok)
        for domain in ("wiki_dev_sel", "pg19_excerpts"):
            docs = [doc for doc, _ in read_jsonl(a.inputs / f"{domain}.jsonl")]
            internal = [[full._fast_to_internal[t] for t in full._fast_encoding.encode_ordinary(doc)] for doc in docs]
            def plain(ids):
                early = []
                for token in ids:
                    early.extend(full._demolish_token(token, 1))
                return early
            early = [plain(ids) for ids in internal]
            expected = [full._apply_bpe(ids, 1) for ids in early]
            collected = [collect(ids) for ids in internal]
            for ids, (emitted, candidates, ranks) in zip(early, collected):
                assert ids == emitted
                all_edges = [(rule[0], i, i + 1) for i, (left, right) in enumerate(zip(ids, ids[1:]))
                             if (rule := tok._remerge_rules.get((left, right))) is not None]
                assert candidates == all_edges
                assert ranks == {r for r, _, _ in candidates}
            stats_full, stats_local = defaultdict(float), defaultdict(float)
            probe_full, hash_full = instrument(type(full)._apply_bpe, "full", stats_full, a.out.parent)
            probe_local, hash_local = instrument(event_remerge, "local", stats_local, a.out.parent)
            rounds = {"full": [], "local": [], "plain_demolition": [], "fused_demolition_and_candidates": []}
            for _ in range(7):
                for mode, stats in (("full", stats_full), ("local", stats_local)):
                    stats.clear()
                    outputs = ([probe_full(full, ids, 1) for ids in early] if mode == "full" else
                               [probe_local(ids, tok._remerge_rules) for ids in early])
                    assert outputs == expected
                    rounds[mode].append(dict(stats))
                for name, fn in (("plain_demolition", plain), ("fused_demolition_and_candidates", collect)):
                    start = time.perf_counter()
                    values = [fn(ids) for ids in internal]
                    rounds[name].append(time.perf_counter() - start)
                    del values
            medians = {mode: {phase: float(np.median([r.get(phase, 0) for r in rows]))
                              for phase in sorted(set().union(*(r.keys() for r in rows)))}
                       for mode, rows in rounds.items() if mode in ("full", "local")}
            dispatch = {"short_full": 0, "empty_skip": 0, "seeded_full": 0, "local": 0}
            for ids, (_, candidates, ranks) in zip(internal, collected):
                key = "short_full" if len(ids) < 128 else ("empty_skip" if not candidates else
                       "seeded_full" if len(ranks) < 4 else "local")
                dispatch[key] += 1
            result["measurements"].append(dict(K=size, domain=domain, rounds=rounds,
                median_seconds=medians, dispatch_counts=dispatch,
                source_function_sha256=dict(full=hash_full, local=hash_local), exact_replays=True))
            print(size, domain, "kernel profiles + fused completeness PASS", flush=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
