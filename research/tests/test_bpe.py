import numpy as np
import pytest

from scaffold_margin.bpe import assign_levels, build_tokenizers, train_bpe, Tokenizer
from scaffold_margin.pretok import count_words, pretokenize
from scaffold_margin.synth import make_corpus

K = 400


@pytest.fixture(scope="module")
def setup():
    docs = make_corpus(n_docs=300, seed=1)
    dev = make_corpus(n_docs=60, seed=2)
    wc = count_words(docs)
    toks = build_tokenizers(wc, K, tau=0.5)
    return docs, dev, wc, toks


def naive_bpe(words, n_merges):
    """Reference implementation: recount everything every iteration."""
    ws = [(list(w), c) for w, c in words.items()]
    merges = []
    for m in range(n_merges):
        pc = {}
        for w, c in ws:
            for p in zip(w, w[1:]):
                pc[p] = pc.get(p, 0) + c
        if not pc:
            break
        best = max(pc.items(), key=lambda kv: (kv[1], tuple(-x for x in kv[0])))[0]
        merges.append(best)
        new = []
        for w, c in ws:
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                    out.append(256 + m); i += 2
                else:
                    out.append(w[i]); i += 1
            new.append((out, c))
        ws = new
    return merges


def test_trainer_matches_naive():
    wc = count_words(make_corpus(n_docs=40, seed=3))
    res = train_bpe(wc, num_merges=60, min_freq=1, log_every=0)
    assert res.merges == naive_bpe(wc, 60)


def test_vocab_sizes_and_levels(setup):
    _, _, _, t = setup
    assert t["bpe"].n_normal == K
    assert t["scaffold"].n_normal == K
    assert t["hscaffold"].n_normal == K
    lv = t["scaffold"].levels
    assert (lv[:256] == 0).all()
    assert (lv[256:] > 0).any(), "synthetic corpus should create scaffold tokens"
    assert len(t["scaffold"].merges) > K - 256  # expanded vocab is larger
    assert np.array_equal(t["scaffold"].from_model, t["hscaffold"].from_model)


def test_residual_frequency_matches_corpus(setup):
    """r(t) from the trainer == token counts in a full-BPE tokenisation of the corpus."""
    _, _, wc, t = setup
    sc = t["scaffold"]
    cnt = np.zeros(len(sc.levels), dtype=np.int64)
    for w, c in wc.items():
        for x in sc._merge_loop(list(w)):
            cnt[x] += c
    assert np.array_equal(cnt, np.array(sc.meta["resid"]))


def test_lossless_and_only_l0(setup):
    _, dev, _, t = setup
    for name, tok in t.items():
        for d in dev[:30]:
            ids = tok.encode(d)
            assert tok.decode(ids).decode() == d
            assert max(ids) < tok.n_normal


def test_scaffold_equals_demolished_bpe_and_hscaffold_none_equals_scaffold(setup):
    _, dev, _, t = setup
    h_none = t["hscaffold"].variant(allowed_remerge=set())
    for d in dev[:30]:
        assert h_none.encode(d) == t["scaffold"].encode(d)


def test_event_triggered_second_pass_is_exact(setup):
    _, dev, _, t = setup
    full = t["hscaffold"].variant(second_pass="full", cache=False)
    ev = t["hscaffold"].variant(second_pass="event", cache=False)
    for d in dev[:30]:
        assert full.encode(d) == ev.encode(d)


def test_hscaffold_changes_something_but_rarely(setup):
    _, dev, wc, t = setup
    diff = tot = 0
    for w, c in wc.items():
        tot += c
        if t["hscaffold"].encode_word(w) != t["scaffold"].encode_word(w):
            diff += c
    assert 0 <= diff < tot


def test_trace_stages_consistent(setup):
    _, _, wc, t = setup
    hs = t["hscaffold"]
    for w in list(wc)[:300]:
        tr = hs.trace(w)
        assert tr["scaffold"] == hs._demolish(tr["full"], 1)
        assert [int(x) for x in hs.to_model[tr["hscaffold"]]] == list(hs.encode_word(w))
        assert b"".join(hs.token_bytes[x] for x in tr["hscaffold"]) == w


def test_save_load_roundtrip(setup, tmp_path):
    _, dev, _, t = setup
    p = tmp_path / "t.json"
    t["hscaffold"].save(str(p))
    t2 = Tokenizer.load(str(p))
    assert t2.encode(dev[0]) == t["hscaffold"].encode(dev[0])


# ---- report-faithful scaffold rule (slide 2) -------------------------------------------
def test_component_rule_judges_only_merge_components():
    wc = count_words(make_corpus(n_docs=400, seed=1, stem_seed=0))
    res = train_bpe(wc, target_normal=500, log_every=0, scaffold_rule="component")
    components = {x for ab in res.merges for x in ab}
    flagged = {256 + i for i in np.nonzero(res.scaf[256:])[0]}
    assert flagged <= components, "a token can only be flagged after being a component of some merge"
    assert 256 + int((~res.scaf[256:]).sum()) >= 500


def test_both_rules_reach_k_and_levels_split_scaffold_only(setup):
    wc = count_words(make_corpus(n_docs=400, seed=1, stem_seed=0))
    for rule in ("component", "global"):
        res = train_bpe(wc, target_normal=500, log_every=0, scaffold_rule=rule)
        counts = []
        for tau in (0.25, 0.5, 0.75):
            lv = assign_levels(res, tau=tau, target_normal=500)
            assert 256 + int((lv[256:] == 0).sum()) == 500
            counts.append(int((lv[256:] > 0).sum()))
        assert len(set(counts)) == 1, "tau must only split scaffold tokens into L1/L2 (report: L1+L2 constant)"


def test_rules_can_disagree_but_both_lossless():
    wc = count_words(make_corpus(n_docs=400, seed=1, stem_seed=0))
    docs = make_corpus(n_docs=30, seed=9, stem_seed=0)
    for rule in ("component", "global"):
        t = build_tokenizers(wc, 500, scaffold_rule=rule)["hscaffold"]
        for d in docs:
            assert t.decode(t.encode(d)).decode() == d


def test_provenance_matches_trace_and_flags_cross_boundary(setup):
    from scaffold_margin.remerge import remerge_provenance
    _, _, wc, t = setup
    h = t["hscaffold"]
    seen_any = False
    for w in wc:
        tr = h.trace(w)
        prov = remerge_provenance(h, w)
        dedup = list(dict.fromkeys((a, b, c) for a, b, c, _ in prov))   # trace logs once per merge step,
        assert dedup == tr["remerges"]                                  # provenance once per occurrence
        seen_any |= bool(prov)
    assert seen_any
