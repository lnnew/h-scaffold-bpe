import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scaffold_margin import scorer
from scaffold_margin.bpe import build_tokenizers
from scaffold_margin.direction_a import matched_random_swaps, pool_candidates, select_supported
from scaffold_margin.pretok import count_words
from scaffold_margin.synth import make_corpus


def test_unobserved_tokens_keep_frequency_decision():
    assert set(select_supported([10, 11, 12, 13], [99, -4, 3, 99], [10, 11], [0, 1, 1, 0])) == {10, 12}
    with pytest.raises(ValueError, match="finite"):
        select_supported([10], [np.nan], [10], [1])


def test_random_swaps_match_changed_slots_and_frequency_bins():
    candidates = np.arange(100)
    residual = np.arange(100) + 1
    frequency = set(range(50, 100))
    utility = (frequency - {51, 54, 61, 74}) | {11, 22, 23, 35}
    control = set(matched_random_swaps(candidates, residual, frequency, utility, 7))
    assert len(control) == len(frequency)
    assert len(control - frequency) == len(utility - frequency)
    bins = np.searchsorted(np.quantile(np.log1p(residual), np.arange(0.1, 1, 0.1)), np.log1p(residual))
    for changed in (lambda s: s - frequency, lambda s: frequency - s):
        assert sorted(bins[list(changed(control))]) == sorted(bins[list(changed(utility))])
    assert control == set(matched_random_swaps(candidates, residual, frequency, utility, 7))
    supported = np.isin(candidates, list(utility | {52, 62, 72, 12, 24, 36}))
    control = set(matched_random_swaps(candidates, residual, frequency, utility, 7, supported))
    assert (control ^ frequency).issubset(set(candidates[supported]))


def test_common_tokens_keep_initial_vectors_after_ids_shift():
    import torch
    from scaffold_margin.direction_a import with_selected
    from scaffold_margin.lm import LM, init_pool_embeddings, make_config
    pool = build_tokenizers(count_words(make_corpus(250, seed=8)), 400)["scaffold"]
    ids = list(pool_candidates(pool))
    a = with_selected(pool, ids[:64], "freq")
    b = with_selected(pool, ids[1:65], "utility")
    torch.manual_seed(7); ma = LM(make_config("tiny", 384))
    torch.manual_seed(7); mb = LM(make_config("tiny", 384))
    init_pool_embeddings(ma, a, pool, 7)
    init_pool_embeddings(mb, b, pool, 7)
    common = sorted(set(a.from_model) & set(b.from_model))
    assert torch.equal(ma.emb.weight[a.to_model[common]], mb.emb.weight[b.to_model[common]])
    assert torch.equal(ma.emb.weight[a.eos_id], mb.emb.weight[b.eos_id])
    assert torch.equal(ma.blocks[0].q.weight, mb.blocks[0].q.weight)


def test_neural_teacher_controls_saved_vocabulary(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/a_marginal_utility.py"
    spec = importlib.util.spec_from_file_location("direction_a_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    pool = build_tokenizers(count_words(make_corpus(250, seed=8)), 400)["scaffold"]
    pool.save(str(tmp_path / "apool_pool_400.json"))
    (tmp_path / "dev_sel.jsonl").write_text('\n'.join(json.dumps({"text": "sample document"}) for _ in range(4)))
    (tmp_path / "train.bin").write_bytes(b'\0' * 8)

    class Bigram:
        sign = 1
        def __init__(self, *args): pass
        def fit(self, *args): return self

    class Neural:
        sign = -1
        cfg = SimpleNamespace(vocab_size=pool.n_normal + 1)
        def __init__(self, *args, **kwargs): pass

    def labels(tok, words, teacher, *args):
        ids = list(pool_candidates(tok))
        return {int(t): {"shrunk": teacher.sign * (i + 1)} for i, t in enumerate(ids)}, {int(t): 1 for t in ids}

    monkeypatch.setattr(script, "BigramScorer", Bigram)
    monkeypatch.setattr(scorer, "TorchScorer", Neural)
    monkeypatch.setattr(script, "token_value_labels", labels)
    monkeypatch.setattr(sys, "argv", [str(path), "--stage", "select", "--K", "320", "--tok_dir", str(tmp_path),
                                    "--text_dir", str(tmp_path), "--train_stream", str(tmp_path / "train.bin"),
                                    "--scorers", "bigram", "torch:fake", "--out", str(tmp_path / "report.json")])
    script.main()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["selected_scorer"] == "torch:fake"
    assert (tmp_path / "apool_lm_320.json").read_bytes() != (tmp_path / "apool_lm_teacher0_320.json").read_bytes()
    lm = json.loads((tmp_path / "apool_lm_320.json").read_text())
    neural = json.loads((tmp_path / "apool_lm_teacher1_320.json").read_text())
    assert lm["levels"] == neural["levels"]
    assert lm["levels"] != json.loads((tmp_path / "apool_lm_teacher0_320.json").read_text())["levels"]
