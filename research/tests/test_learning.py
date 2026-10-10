import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from scaffold_margin.bpe import Tokenizer
from scaffold_margin.direction_a import with_selected
from scaffold_margin.learning import (batch, evaluate, mapped_model, prepare_aligned,
                                     raw_chunks, retokenize, run_probe)
from scaffold_margin.lm import LM, make_config, save_checkpoint


@pytest.fixture
def pool():
    return Tokenizer([(97, 98), (98, 99), (256, 99), (120, 121)], [0] * 260,
                     mode="scaffold", meta={"resid": [10] * 256 + [9, 8, 7, 6]})


def test_utf8_chunks_and_recursive_demolition_match_real_encoding(pool):
    text = "abc xy 한글🙂\nabc xyz " * 5
    chunks = list(raw_chunks(text, 15))
    assert "".join(chunks) == text
    assert all(0 < len(c.encode()) <= 15 for c in chunks)
    corpus = prepare_aligned(pool, [text, "abc abc xy"], 15)
    for keep in ([256, 259], [257, 258, 259]):
        tokenizer = with_selected(pool, keep, "trial")
        converted = retokenize(corpus, pool, tokenizer)
        expected = [tokenizer.encode(c) for doc in [text, "abc abc xy"] for c in raw_chunks(doc, 15)]
        assert converted["ids"].tolist() == [t for row in expected for t in row]
        x, y = batch(converted, np.arange(len(expected)), tokenizer.eos_id, 16, "cpu")
        assert (y != -100).sum().item() == sum(len(row) + 1 for row in expected)
        assert np.array_equal(converted["bytes"], corpus["bytes"])
        assert (x[:, 0] == tokenizer.eos_id).all()
    with pytest.raises(ValueError, match="context"):
        batch(corpus, [0], pool.eos_id, 2, "cpu")


def test_teacher_remapping_preserves_common_tokens_body_and_eos(pool, tmp_path):
    parent = LM(make_config("tiny", 320, 16))
    path = tmp_path / "teacher.pt"
    save_checkpoint(path, parent, parent.c)
    left = with_selected(pool, [256, 257, 259], "left")
    right = with_selected(pool, [257, 258, 259], "right")
    ml = mapped_model(pool, left, str(path), "cpu", seq=16)
    mr = mapped_model(pool, right, str(path), "cpu", seq=16)
    assert left.to_model[257] != right.to_model[257]
    for token in [97, 257, 259]:
        assert torch.equal(ml.emb.weight[left.to_model[token]], mr.emb.weight[right.to_model[token]])
        assert torch.equal(mr.emb.weight[right.to_model[token]], parent.emb.weight[pool.to_model[token]])
    assert torch.equal(ml.emb.weight[left.eos_id], mr.emb.weight[right.eos_id])
    assert torch.equal(ml.blocks[0].down.weight, parent.blocks[0].down.weight)


def test_uniform_eval_includes_all_bytes_eos_and_ignores_padding(pool):
    class Uniform(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((*x.shape, 320))
    corpus = prepare_aligned(pool, ["abc xy", "🙂abc"], 15)
    score, nll, sizes = evaluate(Uniform(), corpus, pool, 16, "cpu", None, bs=2)
    targets = len(corpus["ids"]) + len(corpus["bytes"])
    assert sizes.sum() == len("abc xy🙂abc".encode())
    assert nll.sum() == pytest.approx(targets * np.log2(pool.n_normal + 1), rel=1e-6)
    assert score == pytest.approx(nll.sum() / sizes.sum())


def test_paired_training_has_identical_raw_exposure_but_different_token_exposure(pool, tmp_path):
    torch.set_num_threads(1)
    corpus = prepare_aligned(pool, ["abc xy " * 80], 15)
    results = []
    for name, keep in [("left", [256, 259]), ("right", [258, 259])]:
        tokenizer = with_selected(pool, keep, name)
        results.append(run_probe(pool, tokenizer, {"train": corpus, "select": corpus}, tmp_path / name,
                                 steps=2, seq=16, batch_chunks=4, micro_batch=2, preset="tiny"))
    for key in ["train_order_hash", "raw_train_hash", "train_exposure_bytes", "skipped_updates"]:
        assert results[0][key] == results[1][key]
    assert results[0]["train_exposure_tokens"] != results[1]["train_exposure_tokens"]
    assert set(results[0]["curves"]) == {"0", "2"}


@pytest.fixture
def script(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("learning_script", scripts / "a_learning_utility.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_random_control_preserves_slot_count_frequency_and_length_strata(pool, script):
    base, selected = {256, 257}, {258, 259}
    control = script.random_matched(pool, base, selected, np.ones(260) * 2, 1)
    assert len(control) == len(base)
    bins = np.searchsorted(np.quantile(np.log1p([9, 8, 7, 6]), np.arange(.1, 1, .1)), np.log1p([9, 8, 7, 6]))
    strata = lambda tokens: Counter((int(bins[t - 256]), len(pool.token_bytes[t])) for t in tokens)
    assert strata(base - control) == strata(base - selected)
    assert strata(control - base) == strata(selected - base)
    with pytest.raises(ValueError, match="unique"):
        script.swap(base, [[256, 258], [256, 259]])


def test_exposure_mismatch_and_skipped_update_invalidate_comparison(script):
    good = dict(raw_train_hash="text", train_order_hash="order", train_exposure_bytes=100,
                steps=128, skipped_updates=0)
    for bad in [dict(good, train_order_hash="other"), dict(good, skipped_updates=1)]:
        with pytest.raises(ValueError, match="exposure"):
            script.assert_exposure(good, bad)


def test_document_comparison_uses_paired_bytes_and_rejects_mismatch(script, tmp_path):
    good = dict(raw_train_hash="text", train_order_hash="order", train_exposure_bytes=100,
                steps=128, skipped_updates=0)
    for seed in [0, 1]:
        for name, bits in [("frequency", [30., 50.]), ("learning", [20., 40.])]:
            folder = tmp_path / f"joint/{name}/seed{seed}"
            folder.mkdir(parents=True)
            (folder / "results.json").write_text(json.dumps(good))
            np.savez(folder / "eval_confirm_h128.npz", doc_nll_bits=bits, doc_bytes=[10, 20])
    from types import SimpleNamespace
    result = script.comparison(SimpleNamespace(out=tmp_path), dict(steps=128, data_seeds=[0, 1]), ["frequency"])
    assert result["delta_bpb"] == pytest.approx(-20 / 30)
    assert result["doc_ci95"][1] < 0
    path = tmp_path / "joint/learning/seed1/eval_confirm_h128.npz"
    np.savez(path, doc_nll_bits=[20., 40.], doc_bytes=[10, 21])
    with pytest.raises(ValueError, match="bytes differ"):
        script.comparison(SimpleNamespace(out=tmp_path), dict(steps=128, data_seeds=[0, 1]), ["frequency"])
