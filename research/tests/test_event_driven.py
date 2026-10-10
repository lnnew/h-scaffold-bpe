from collections import Counter
import random

import pytest

from hscaffold_bpe.model import Merge, TokenizerModel
from hscaffold_bpe.pretokenize import iter_pretokens
from hscaffold_bpe.remerge import event_remerge
from hscaffold_bpe.tokenizer import HierarchicalTokenizer
from hscaffold_bpe.trainer import train_from_counts
from scaffold_margin.bpe import Tokenizer


def reference(sequence, rules, log):
    sequence = list(sequence)
    while True:
        candidates = [rules[a, b] for a, b in zip(sequence, sequence[1:]) if (a, b) in rules]
        if not candidates:
            return sequence
        rank, output = min(candidates)
        pair = next(pair for pair, rule in rules.items() if rule == (rank, output))
        result, i = [], 0
        while i < len(sequence):
            if tuple(sequence[i:i + 2]) == pair:
                result.append(output)
                i += 2
            else:
                result.append(sequence[i])
                i += 1
        log.append((*pair, output))
        sequence = result


def test_heap_batches_before_a_new_lower_rank():
    rules = {(3, 0): (0, 4), (0, 1): (1, 3)}
    log = []
    assert event_remerge([0, 1, 0, 1, 2], rules, log=log) == [3, 3, 2]
    assert log == [(0, 1, 3)]


def test_heap_random_sequences_overlap_stale_entries_and_logs():
    rng = random.Random(20261010)
    for _ in range(2500):
        pairs = [(a, b) for a in range(8) for b in range(8)]
        rng.shuffle(pairs)
        rules = {pair: (rank, rng.randrange(8)) for rank, pair in enumerate(pairs[:24])}
        sequence = [rng.randrange(8) for _ in range(rng.randrange(60))]
        expected_log, actual_log = [], []
        expected = reference(sequence, rules, expected_log)
        assert event_remerge(sequence, rules, log=actual_log) == expected
        assert actual_log == expected_log


def test_frontier_propagates_both_directions_and_allowlist(tmp_path):
    merges = [(97, 98), (99, 97), (100, 257), (258, 98)]
    levels = [0] * 256 + [2, 0, 0, 0]
    full = Tokenizer(merges, levels, mode="hscaffold", second_pass="full", cache=False)
    local = full.variant(second_pass="local")
    for word in (b"dcab", b"cab", b"x", b"", b"dcabdcab"):
        assert local.trace(word) == full.trace(word)
    assert local.trace(b"dcab")["remerges"] == [(99, 97, 257), (100, 257, 258), (258, 98, 259)]
    full.set_allowed_remerge({(99, 97)})
    local.set_allowed_remerge({(99, 97)})
    assert local.trace(b"dcab") == full.trace(b"dcab")
    path = tmp_path / "local.json"
    local.save(path)
    assert Tokenizer.load(path).trace(b"dcab") == full.trace(b"dcab")


def boundary_model():
    # First-pass pretokens split newline and 'a', leaving an eligible edge
    # in the legacy fast backend even though this model has no L2 tokens.
    tokens = {i: bytes([i]) for i in range(256)}
    tokens[256] = b"\na"
    return TokenizerModel(
        variant="hierarchical", target_vocab_size=257, tokens=tokens,
        merges=[Merge(10, 97, 256, 1)], levels={t: 0 for t in tokens},
        parents={256: [(10, 97)]}, special_tokens={},
    )


def test_legacy_fast_no_l2_skip_preserves_boundary_merges():
    model = boundary_model()
    full = HierarchicalTokenizer(model, backend="tiktoken")
    python = HierarchicalTokenizer(model, backend="python")
    assert full.encode("\na") != python.encode("\na")
    for mode in ("skip", "local"):
        tok = HierarchicalTokenizer(model, backend="tiktoken", second_pass=mode)
        assert tok.encode("\na") == full.encode("\na")


@pytest.mark.parametrize("backend", ["python", "tiktoken"])
def test_team_backend_parity_random_specials_and_eos(backend):
    text = ("Arizona zona Arizona data database mate mating. " * 40
            + "The quick brown fox jumps over the lazy dog. " * 20)
    model = train_from_counts(Counter(iter_pretokens(text)), 275, "hierarchical", show_progress=False)
    toks = [HierarchicalTokenizer(model, backend=backend, second_pass=mode)
            for mode in ("full", "skip", "local")]
    rng = random.Random(7)
    docs = ["", text, "café 한글\n\n aaabbb", "Arizona<|endoftext|>zona"]
    docs += ["".join(rng.choices("Arizona mate\n\t 123.-", k=80)) for _ in range(300)]
    for doc in docs:
        expected = toks[0].encode(doc, add_eos=True)
        for tok in toks[1:]:
            assert tok.encode(doc, add_eos=True) == expected
            assert tok.decode(expected[:-1]) == doc


def test_second_pass_validation():
    with pytest.raises(ValueError, match="second_pass"):
        HierarchicalTokenizer(boundary_model(), second_pass="typo")
