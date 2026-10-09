"""Segmentation scorers: total NLL (nats) of a token-id sequence under some LM.

All scorers take sequences of *model ids* and implicitly prepend EOS as the
start-of-text symbol.  The NLL of the prepended EOS itself is not counted.  Two
segmentations of the same raw text are comparable because the sums cover the same
bytes (this is exactly the quantity bits-per-byte is built from).
"""
from __future__ import annotations

import numpy as np


class BigramScorer:
    """CPU absolute-discounting bigram LM.  A deliberately weak, dependency-free
    teacher used for smoke tests and as a cheap 'is the neural teacher necessary?'
    control; it cannot see context beyond one token."""

    def __init__(self, vocab_size: int, discount: float = 0.75):
        self.V = vocab_size
        self.d = discount

    def fit(self, streams):
        V = self.V
        uni = np.zeros(V, dtype=np.float64)
        keys = []
        for s in streams:
            s = np.asarray(s, dtype=np.int64)
            uni += np.bincount(s, minlength=V)
            keys.append(s[:-1] * V + s[1:])
        keys = np.concatenate(keys) if keys else np.zeros(0, dtype=np.int64)
        self.keys, self.cnt = np.unique(keys, return_counts=True)
        self.ctx = np.bincount(self.keys // V, weights=self.cnt, minlength=V)
        self.n1 = np.bincount(self.keys // V, minlength=V).astype(np.float64)
        self.p_uni = (uni + 1) / (uni.sum() + V)
        return self

    def _logp(self, a, b):
        V = self.V
        k = a * V + b
        idx = np.searchsorted(self.keys, k)
        idx = np.minimum(idx, len(self.keys) - 1)
        hit = self.keys[idx] == k
        c = np.where(hit, self.cnt[idx], 0)
        ca = self.ctx[a]
        lam = np.where(ca > 0, self.d * self.n1[a] / np.maximum(ca, 1), 1.0)
        p = np.where(ca > 0, np.maximum(c - self.d, 0) / np.maximum(ca, 1), 0.0) + lam * self.p_uni[b]
        return np.log(p)

    def score(self, seqs, eos_id: int) -> np.ndarray:
        out = np.zeros(len(seqs))
        for i, s in enumerate(seqs):
            s = np.asarray([eos_id] + list(s), dtype=np.int64)
            out[i] = -self._logp(s[:-1], s[1:]).sum()
        return out


class TorchScorer:
    """Neural teacher: wraps a model from lm.py (checkpoint produced by train_lm.py)."""

    def __init__(self, ckpt_path: str, device: str = "cuda", batch_size: int = 256, amp: bool = True):
        import torch
        from .lm import load_checkpoint

        self.torch = torch
        self.model, self.cfg = load_checkpoint(ckpt_path, device)
        self.model.eval()
        self.device = device
        self.bs = batch_size
        self.amp = amp and device.startswith("cuda")

    def score(self, seqs, eos_id: int) -> np.ndarray:
        torch = self.torch
        out = np.zeros(len(seqs))
        order = np.argsort([len(s) for s in seqs])  # length-bucket to limit padding
        with torch.no_grad():
            for i in range(0, len(order), self.bs):
                idx = order[i : i + self.bs]
                L = max(len(seqs[j]) for j in idx) + 1
                x = torch.full((len(idx), L), eos_id, dtype=torch.long)
                mask = torch.zeros((len(idx), L - 1), dtype=torch.bool)
                for r, j in enumerate(idx):
                    s = seqs[j]
                    x[r, 1 : 1 + len(s)] = torch.tensor(s, dtype=torch.long)
                    mask[r, : len(s)] = True
                x, mask = x.to(self.device), mask.to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
                    logits = self.model(x[:, :-1])
                lp = torch.log_softmax(logits.float(), dim=-1)
                nll = -lp.gather(-1, x[:, 1:, None]).squeeze(-1)
                nll = (nll * mask).sum(1)
                out[idx] = nll.cpu().numpy()
        return out
