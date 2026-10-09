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
