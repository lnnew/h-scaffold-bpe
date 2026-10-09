import numpy as np
import pytest

from scaffold_margin.bpe import build_tokenizers
from scaffold_margin.pretok import count_words
from scaffold_margin.remerge import build_rule_alts, event_stats, extract_rules, level_counts
from scaffold_margin.scorer import BigramScorer
from scaffold_margin.synth import make_corpus
from scaffold_margin.utility import (CandidateAlts, collect_edits, doc_word_lists, pick_matched,
                                     pick_random, predictability, score_edits, summarize)


@pytest.fixture(scope="module")
def world():
    train = make_corpus(n_docs=800, seed=1, stem_seed=0)
    dev = make_corpus(n_docs=1500, seed=2, stem_seed=0)
    wc = count_words(train)
    t = build_tokenizers(wc, 500)
    return train, dev, wc, t


def test_event_stats_ordering(world):
    _, _, wc, t = world
    s = event_stats(t["hscaffold"], wc)
    assert s["changed_rate"] <= s["accepted_rate"] <= s["attempted_rate"] <= 1
    assert sum(level_counts(t["hscaffold"]).values()) == len(t["hscaffold"].merges)


def test_rule_extraction_and_edits(world):
    train, dev, wc, t = world
    h, sc = t["hscaffold"], t["scaffold"]
    rules = extract_rules(h, wc)
    assert len(rules) > 0
    for r in rules.values():
        assert r["n_occ"] > 0 and r["c_level"] <= 1
    dev_words = doc_word_lists(dev)
    alts = build_rule_alts(h, list(rules), [w for ws in dev_words for w in ws])
    edits = collect_edits(dev_words, sc.encode_word, alts, ctx_words=6, max_per_key=8)
    assert len(edits) > 0
    sc_fit = BigramScorer(sc.n_normal + 1).fit([sc.encode(d) for d in make_corpus(300, seed=5, stem_seed=0)])
    deltas = score_edits(sc_fit, edits, sc.eos_id)
    summ = summarize(deltas)
    assert all(np.isfinite(v["mean"]) for v in summ.values())
    keys = [k for k in summ]
    feats = np.array([[rules[k]["log_n_occ"], rules[k]["log_r_c"], rules[k]["c_len"]] for k in keys])
    labels = np.array([summ[k]["shrunk"] for k in keys])
    res = predictability(feats, ["a", "b", "c"], labels)
    assert res["n"] == len(keys)
    # matched selection helpers
    assert len(pick_random(keys, 3, 0)) == min(3, len(keys))
    assert pick_matched(keys, labels, 2)[0] == keys[int(np.argmax(labels))]


def test_selective_variants_between_scaffold_and_h(world):
    _, dev, wc, t = world
    h, sc = t["hscaffold"], t["scaffold"]
    rules = sorted(set(extract_rules(h, wc)) | set(extract_rules(h, count_words(dev))))
    half = h.variant(allowed_remerge=set(rules[: len(rules) // 2]))
    all_ = h.variant(allowed_remerge=set(rules))
    none = h.variant(allowed_remerge=set())
    for d in dev[:80]:
        assert none.encode(d) == sc.encode(d)
        assert all_.encode(d) == h.encode(d)
        assert half.decode(half.encode(d)).decode() == d
