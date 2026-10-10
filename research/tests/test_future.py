import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

from scaffold_margin.aligned_lm import batch, prepare, train
from scaffold_margin.bpe import Tokenizer
from scaffold_margin.direction_a import with_selected
from scaffold_margin.future import Edit, collect, score
from scaffold_margin.lm import LM, make_config, save_checkpoint
from scaffold_margin.pretok import pretokenize
from scaffold_margin.scorer import TorchScorer


@pytest.fixture
def pool():
    return Tokenizer([(97, 98), (256, 99), (120, 121)], [0] * 259,
                     mode="scaffold", meta={"resid": [10] * 256 + [9, 8, 7]})


def test_partition_and_suffix_alignment(pool):
    e = Edit((1, 2, 3), (1, 4, 5, 3), 1, 2, 3, 1, 0)
    class Bits:
        def token_nll(self, *args, **kwargs):
            return [np.array([1., 2., 3.]) * np.log(2), np.array([1., 2., 2., 6.]) * np.log(2)]
    rows, checks = score(pool, Bits(), {257: [e]})
    assert rows[257][0, :3] == pytest.approx([2., 3., 5.])
    assert checks["max_partition_error_bits"] < 1e-12
    with pytest.raises(ValueError, match="identical"):
        Edit(e.before, (1, 4, 5, 9), 1, 2, 3, 1, 0).validate()


def test_collected_edits_preserve_raw_text_and_are_repeatable(pool):
    words = [pretokenize("abc xy abc xy abc xy abc xy")]
    first, seen, rejected = collect(pool, words, max_samples=2, context=2, seq=16)
    second, _, _ = collect(pool, words, max_samples=2, context=2, seq=16)
    assert first == second and seen and not rejected
    for edits in first.values():
        for e in edits:
            e.validate()
            assert pool.decode(e.before) == pool.decode(e.after)
            assert len(e.before) <= 16 and len(e.after) <= 16


def test_neural_token_losses_match_manual_targets_and_prefix(pool, tmp_path):
    torch.set_num_threads(1)
    path = tmp_path / "teacher.pt"
    model = LM(make_config("tiny", 320, 16)).eval()
    save_checkpoint(path, model, model.c)
    scorer = TorchScorer(str(path), device="cpu", batch_size=2, amp=False)
    seqs = [(1, 2, 3), (1, 2, 4, 5)]
    losses = scorer.token_nll(seqs, pool.eos_id, vocab_size=pool.n_normal + 1)
    assert losses[0][:2] == pytest.approx(losses[1][:2], abs=1e-6)
    with torch.no_grad():
        x = torch.tensor([[pool.eos_id, 1, 2]])
        logits = model(x)[..., :pool.n_normal + 1]
        expected = torch.nn.functional.cross_entropy(logits.flatten(0, 1), torch.tensor([1, 2, 3]), reduction="none")
    assert losses[0] == pytest.approx(expected.numpy(), abs=1e-6)
    assert scorer.score(seqs, pool.eos_id) == pytest.approx([a.sum() for a in scorer.token_nll(seqs, pool.eos_id)])


def test_raw_windows_cover_each_byte_once_without_artificial_eos(pool):
    toks = {"base": pool, "split": with_selected(pool, [256, 258], "split")}
    docs = ["abc xy " * 30 + "한글🙂", "abc abc xy " * 20]
    data = prepare(pool, toks, docs, seq=16, context=4)
    assert str(data["base"]["raw_hash"]) == str(data["split"]["raw_hash"])
    assert np.array_equal(data["base"]["bytes"], data["split"]["bytes"])
    assert data["base"]["bytes"].sum() == sum(len(d.encode()) for d in docs)
    for name, corpus in data.items():
        rebuilt = [b"", b""]
        for i, doc in enumerate(corpus["docs"]):
            left = corpus["offsets"][i] + corpus["starts"][i]
            ids = corpus["ids"][left:corpus["offsets"][i + 1]]
            rebuilt[doc] += toks[name].decode(ids)
        assert rebuilt == [d.encode() for d in docs]
        x, y = batch(corpus, np.arange(len(corpus["bytes"])), toks[name].eos_id, "cpu")
        assert x.shape[1] <= 16
        assert not (y == toks[name].eos_id).any()
        assert toks[name].model_token_bytes[y[y >= 0].numpy()].sum() == corpus["bytes"].sum()


def test_oversized_utf8_pretoken_is_lossless(pool):
    docs = ["한글" * 50 + " abc"]
    data = prepare(pool, {"pool": pool}, docs, seq=16, context=4)["pool"]
    assert int(data["split_units"]) > 0
    assert data["bytes"].sum() == len(docs[0].encode())


def test_from_scratch_smoke_saves_weights_and_evaluates_all_bytes(pool, tmp_path):
    torch.set_num_threads(1)
    docs = ["abc xy " * 50, "abc " * 50]
    data = prepare(pool, {"pool": pool}, docs, seq=16, context=4)["pool"]
    out = tmp_path / "run"
    result = train(pool, pool, {"train": data, "test": data}, out, steps=2,
                   preset="tiny", seq=16, batch_windows=4, micro_batch=2)
    assert result["skipped_updates"] == 0 and result["train_exposure_bytes"] > 0
    assert (out / "model.pt").exists() and np.isfinite(result["bpb"]["test"])
    assert np.load(out / "eval_test.npz")["doc_bytes"].sum() == sum(len(d.encode()) for d in docs)


def controller():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("future_controller", scripts / "a_future_utility.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stratified_exchange_budget_preserves_bytes_and_slot_count():
    m = controller()
    pool = Tokenizer([(97, b) for b in range(64)], [0] * 320, mode="scaffold",
                     meta={"resid": [100] * 320})
    base = set(range(256, 288))
    stats = {t: dict(n=2) for t in range(256, 320)}
    groups, quotas, _, budget = m.pairs_and_quotas(pool, base, stats, 8)
    assert budget == 8 and sum(quotas.values()) == 8
    pairs = [p for key, quota in quotas.items() for p in groups[key][:quota]]
    changed = m.apply_pairs(base, pairs)
    assert len(changed) == len(base) and len(changed - base) == 8
    assert all(len(pool.token_bytes[t]) == len(pool.token_bytes[c]) for t, c in pairs)
    with pytest.raises(ValueError, match="invalid"):
        m.apply_pairs(base, pairs + pairs[:1])


def test_fast_token_count_matches_recursive_demolition(pool):
    m = controller()
    counts = np.arange(1, pool.n_normal + 1, dtype=np.int64)
    for keep in [set(), {256}, {257}, {256, 258}]:
        tok = with_selected(pool, keep, "check")
        expected = sum(int(counts[i]) * len(tok._demolish([int(t)], 1)) for i, t in enumerate(pool.from_model))
        assert m.token_count(pool, keep, counts) == expected


def test_finish_checks_training_pairing_and_doc_bytes(tmp_path):
    m = controller()
    c = dict(seeds=[0, 1, 2], methods=["frequency", "word", "future", "random"], primary="PG19", scope="test")
    for name in c["methods"]:
        for seed in c["seeds"]:
            root = tmp_path / f"runs/{name}/seed{seed}"
            root.mkdir(parents=True)
            nll = np.array([10., 20.]) + (0 if name == "future" else 2 + seed * .1)
            result = dict(raw_train_hash="raw", train_order_hash="order", train_exposure_bytes=100,
                          steps=2, bpb={split: float(nll.sum() / 30) for split in ["pg19", "wiki", "val"]})
            (root / "results.json").write_text(json.dumps(result))
            for split in ["pg19", "wiki", "val"]:
                np.savez(root / f"eval_{split}.npz", doc_nll_bits=nll, doc_bytes=np.array([10, 20]))
    m.finish(tmp_path, c)
    assert json.loads((tmp_path / "decision.json").read_text())["status"] == "GO_TO_INDEPENDENT_REPLICATION"
    p = tmp_path / "runs/random/seed1/results.json"
    result = json.loads(p.read_text()); result["train_exposure_bytes"] += 1
    p.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="exposure"):
        m.finish(tmp_path, c)
