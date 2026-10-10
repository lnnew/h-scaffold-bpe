import json, os, subprocess, sys
import numpy as np
import torch

from scaffold_margin.bpe import build_tokenizers
from scaffold_margin.data import bpb_from_token_nll, doc_ids_for_targets, encode_docs, write_stream, open_stream
from scaffold_margin.pretok import count_words
from scaffold_margin.synth import make_corpus
from scaffold_margin.train_lm import eval_stream


class Uniform(torch.nn.Module):
    def __init__(self, V):
        super().__init__(); self.V = V
    def forward(self, x):
        return torch.zeros(*x.shape, self.V)


def test_bpb_accounting_exact(tmp_path):
    docs = make_corpus(60, seed=7)
    tok = build_tokenizers(count_words(make_corpus(300, seed=1)), 400)["hscaffold"]
    tp = str(tmp_path / "t.json"); tok.save(tp)
    ids = encode_docs(tp, docs, workers=1)
    write_stream(str(tmp_path), "test", ids, tok.eos_id)
    s = open_stream(str(tmp_path / "test.bin"))
    # byte lengths of targets sum to raw bytes exactly
    tgt = np.asarray(s[1:], dtype=np.int64)
    assert tok.model_token_bytes[tgt].sum() == sum(len(d.encode()) for d in docs)
    # uniform model: BPB must equal log2(V) * n_targets / bytes, for several window sizes
    V = tok.n_normal + 1
    exp = np.log2(V) * len(tgt) / tok.model_token_bytes[tgt].sum()
    for seq in (16, 64, 1000):
        nll, tg = eval_stream(Uniform(V), s, tok.model_token_bytes, seq, "cpu", None)
        assert abs(bpb_from_token_nll(nll, tg, tok.model_token_bytes)[0] - exp) < 1e-4
    # per-document aggregation is consistent with the total
    offs = np.load(str(tmp_path / "test_docs.npy"))
    d_of = doc_ids_for_targets(offs, len(s))[1:]
    tot, (nd, bd) = bpb_from_token_nll(nll, tg, tok.model_token_bytes, d_of, len(offs))
    assert abs(nd.sum() / bd.sum() - tot) < 1e-6


def test_fixed_budget_training_records_actual_text_exposure(tmp_path):
    tok = build_tokenizers(count_words(make_corpus(120, seed=8)), 400)["scaffold"]
    ids = [tok.encode(d) for d in make_corpus(30, seed=9)]
    write_stream(str(tmp_path), "train", ids, tok.eos_id)
    write_stream(str(tmp_path), "val", ids[:2], tok.eos_id)
    np.save(tmp_path / "token_bytes.npy", tok.model_token_bytes)
    tok_path = tmp_path / "tok.json"
    tok.save(str(tok_path))
    (tmp_path / "meta.json").write_text(json.dumps(dict(n_normal=tok.n_normal, tokenizer=str(tok_path))))
    out = tmp_path / "run"
    env = dict(os.environ, OMP_NUM_THREADS="2")
    proc = subprocess.run([sys.executable, "-m", "scaffold_margin.train_lm", "--data", str(tmp_path), "--out", str(out),
                           "--preset", "tiny", "--seq", "32", "--batch_tokens", "256", "--micro_batch", "4",
                           "--max_steps", "2", "--amp", "fp32", "--eval_sets", "val", "--init_pool", str(tok_path), "--save_ckpt"],
                          env=env, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr[-1500:]
    result = json.loads((out / "results.json").read_text())
    stream = open_stream(str(tmp_path / "train.bin"))
    order = np.random.default_rng([0, 0]).permutation((len(stream) - 1) // 32)
    expected_bytes = sum(int(tok.model_token_bytes[stream[i * 32 + 1 : i * 32 + 33]].sum()) for i in order[:16])
    assert result["train_exposure_tokens"] == 512
    assert result["train_exposure_bytes"] == expected_bytes
    assert np.isfinite(result["bpb"]["val"])
    from pathlib import Path
    evaluator = Path(__file__).parents[1] / "scripts/eval_checkpoint.py"
    frozen_out = tmp_path / "frozen_eval"
    proc = subprocess.run([sys.executable, str(evaluator), "--checkpoint", str(out / "model.pt"), "--data", str(tmp_path),
                           "--split", "val", "--out", str(frozen_out)], env=env, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr[-1500:]
    frozen = json.loads((frozen_out / "results.json").read_text())
    assert abs(frozen["bpb"]["val"] - result["bpb"]["val"]) < 1e-6
    assert json.loads((out / "results.json").read_text()) == result
