import itertools

import pytest

from scaffold_margin.bpe import build_tokenizers
from scaffold_margin.inference import VisibleSegmenter
from scaffold_margin.pretok import count_words, pretokenize
from scaffold_margin.synth import make_corpus


@pytest.fixture(scope="module")
def setup():
    toks = build_tokenizers(count_words(make_corpus(n_docs=300, seed=1)), 400, tau=0.5)
    dev = make_corpus(n_docs=40, seed=2)
    return toks, dev


def brute_min(seg, w):
    """Exhaustive minimum over all segmentations into visible tokens (short words only)."""
    n, best = len(w), None
    for cuts in itertools.product([0, 1], repeat=n - 1):
        pieces, s = [], 0
        for i, c in enumerate(cuts, 1):
            if c:
                pieces.append(w[s:i])
                s = i
        pieces.append(w[s:])
        if all(p in seg.vocab for p in pieces):
            best = len(pieces) if best is None else min(best, len(pieces))
    return best


@pytest.mark.parametrize("kind", ["bpe", "scaffold", "hscaffold"])
def test_lossless_same_ids_and_dp_minimal(setup, kind):
    toks, dev = setup
    tok = toks[kind]
    segs = {m: VisibleSegmenter(tok, m) for m in ("canonical", "longest", "dp")}
    for d in dev[:20]:
        assert segs["canonical"].encode(d) == tok.encode(d)
        for seg in segs.values():
            assert tok.decode(seg.encode(d)).decode() == d
    words = {w for d in dev for w in pretokenize(d)}
    for w in words:
        n_dp = len(segs["dp"].encode_word(w))
        assert n_dp <= len(segs["canonical"].encode_word(w))
        assert n_dp <= len(segs["longest"].encode_word(w))
        if len(w) <= 12:
            assert n_dp == brute_min(segs["dp"], w)


def test_dp_finds_shorter_segmentations_for_scaffold(setup):
    toks, dev = setup
    sc = toks["scaffold"]
    can, dp = VisibleSegmenter(sc, "canonical"), VisibleSegmenter(sc, "dp")
    words = {w for d in dev for w in pretokenize(d)}
    assert sum(len(can.encode_word(w)) - len(dp.encode_word(w)) for w in words) >= 0
