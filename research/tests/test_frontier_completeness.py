"""Check complete eligible-edge coverage at actual production batch boundaries."""
import inspect
import random
import sys

import pytest

from hscaffold_bpe.remerge import event_remerge
from hscaffold_bpe.tokenizer import HierarchicalTokenizer
from hscaffold_bpe.model import Merge, TokenizerModel
from scaffold_margin.bpe import Tokenizer


def checked_remerge(sequence, rules, **kwargs):
    lines, start = inspect.getsourcelines(event_remerge)
    anchors = {line.strip(): start + i for i, line in enumerate(lines)}
    checkpoints = {anchors["if not heap:"]: "initial", anchors["rank = heap[0][0]"]: "batch",
                   anchors["batch_merge = (*pair, rule[1])"]: "pending",
                   anchors["if changed:"]: "pending", anchors["result = []"]: "final"}
    count = 0

    def trace(frame, event, arg):
        nonlocal count
        if frame.f_code != event_remerge.__code__ or event != "line" or frame.f_lineno not in checkpoints:
            return trace
        v = frame.f_locals
        phase = checkpoints[frame.f_lineno]
        if phase == "initial":
            values = sequence
            following = list(range(1, len(values))) + [-1] if values else []
        else:
            values, following = v["values"], v["following"]
        edges, left = set(), 0 if values else -1
        while left >= 0:
            right = following[left]
            if right >= 0 and (values[left], values[right]) in rules:
                edges.add((left, right))
            left = right
        valid = {(left, right) for rank, left, right in v["heap"]
                 if following[left] == right and following[right] != -2
                 and (values[left], values[right]) in rules
                 and rules[values[left], values[right]][0] == rank}
        if phase == "pending":
            # Newly exposed edges are deliberately deferred until this batch ends.
            for left in v["changed"]:
                if following[left] != -2:
                    previous = v["previous"][left]
                    if previous >= 0:
                        valid.add((previous, left))
                    if following[left] >= 0:
                        valid.add((left, following[left]))
        assert edges <= valid, f"missing frontier {phase}: {edges - valid}"
        count += 1
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        result = event_remerge(sequence, rules, **kwargs)
    finally:
        sys.settrace(previous)
    assert count > 0
    return result


def test_completeness_initial_batch_pending_and_terminal_states():
    rng = random.Random(23)
    for _ in range(200):
        pairs = [(a, b) for a in range(7) for b in range(7)]
        rng.shuffle(pairs)
        rules = {pair: (rank, rng.randrange(7)) for rank, pair in enumerate(pairs[:20])}
        seq = [rng.randrange(7) for _ in range(rng.randrange(30))]
        assert checked_remerge(seq, rules) == event_remerge(seq, rules)
    with pytest.raises(AssertionError, match="missing frontier initial"):
        checked_remerge([0, 1], {(0, 1): (0, 2)}, starts=set())


def test_saturated_demolition_frontier_is_complete():
    tok = Tokenizer([(97, 98), (99, 97), (100, 257), (258, 98)],
                    [0] * 256 + [2, 0, 0, 0], mode="hscaffold", second_pass="local")
    for word in (b"dcab", b"dcab dcab", b"aaa", b"x"):
        full = tok._merge_loop(list(word))
        early, starts = tok._l2_stage(full)
        assert checked_remerge(early, tok._remerge_rules, starts=starts) == tok._merge_loop(
            early, ok=tok._remerge_ok)


def test_hybrid_fused_seeds_and_first_pair_fallback(monkeypatch):
    tokens = {i: bytes([i]) for i in range(256)}
    pairs = [(97, 98), (99, 100), (101, 102), (103, 104)]
    for i, (a, b) in enumerate(pairs, 256):
        tokens[i] = tokens[a] + tokens[b]
    model = TokenizerModel("hierarchical", 260, tokens,
                           [Merge(a, b, 256 + i, 1) for i, (a, b) in enumerate(pairs)],
                           {t: 0 for t in tokens}, {256 + i: [pair] for i, pair in enumerate(pairs)}, {})
    tok = HierarchicalTokenizer(model, backend="python", second_pass="hybrid")
    seen = []

    def checked(sequence, rules, **kw):
        seen.append(True)
        assert "candidates" in kw
        return checked_remerge(sequence, rules, **kw)

    monkeypatch.setattr("hscaffold_bpe.tokenizer.event_remerge", checked)
    seq = list(b"ab cd ef gh " * 30)
    assert tok._second_merge(seq, False) == tok._apply_bpe(seq, 1)
    assert seen
    seen.clear()
    for seq in (list(b"ab " * 60), list(b"xy " * 60), list(b"ab cd")):
        assert tok._second_merge(seq, False) == tok._apply_bpe(seq, 1)
    assert not seen
