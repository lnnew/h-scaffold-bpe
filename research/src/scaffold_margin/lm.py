"""Minimal LLaMA-style decoder (RMSNorm, RoPE, GQA, SwiGLU, tied embeddings).

Written for the target hardware (2x Quadro RTX 8000 = Turing, sm_75):
  * attention goes through F.scaled_dot_product_attention (mem-efficient kernel works on
    sm_75; FlashAttention-2 does not),
  * the training script uses fp16 autocast + GradScaler (no usable bf16 on Turing).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LMConfig:
    vocab_size: int = 32768
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 8
    ffn_dim: int = 1408
    max_seq: int = 1024
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    aux_size: int = 0          # >0: extra input-only embedding (index 0 = none, contributes zero)

    def n_params(self, include_embedding=True) -> int:
        hd = self.d_model // self.n_heads
        attn = self.d_model * (self.n_heads * hd) * 2 + self.d_model * (self.n_kv_heads * hd) * 2
        mlp = 3 * self.d_model * self.ffn_dim
        body = self.n_layers * (attn + mlp + 2 * self.d_model) + self.d_model
        emb = self.vocab_size * self.d_model * (1 if self.tie_embeddings else 2) + self.aux_size * self.d_model
        return body + (emb if include_embedding else 0)


# d_model, layers, heads, kv_heads, ffn.  'llama1b' mirrors Llama-3.2-1B's shape.
PRESETS = {
    "tiny": dict(d_model=128, n_layers=4, n_heads=4, n_kv_heads=4, ffn_dim=384),
    "small": dict(d_model=512, n_layers=8, n_heads=8, n_kv_heads=8, ffn_dim=1408),
    "base": dict(d_model=768, n_layers=12, n_heads=12, n_kv_heads=4, ffn_dim=2048),
    "large": dict(d_model=1024, n_layers=16, n_heads=16, n_kv_heads=4, ffn_dim=2816),
    "llama1b": dict(d_model=2048, n_layers=16, n_heads=32, n_kv_heads=8, ffn_dim=8192),
}


def make_config(preset: str, vocab_size: int, max_seq: int = 1024) -> LMConfig:
    return LMConfig(vocab_size=vocab_size, max_seq=max_seq, **PRESETS[preset])


def init_pool_embeddings(model, tokenizer, pool, seed):
    """Keep common tokens' initial vectors equal even when vocabulary pruning shifts their IDs."""
    if tokenizer.merges != pool.merges or tokenizer.pattern != pool.pattern:
        raise ValueError("initialization pool must use the same merge table and pattern")
    ids = pool.to_model[tokenizer.from_model].tolist() + [pool.eos_id]
    if min(ids) < 0:
        raise ValueError("student vocabulary is not a subset of the initialization pool")
    bank = torch.empty(pool.n_normal + 1, model.c.d_model).normal_(
        0, 0.02, generator=torch.Generator().manual_seed(seed))
    vectors = bank[ids].to(model.emb.weight.device)
    with torch.no_grad():
        model.emb.weight[:len(ids)].copy_(vectors)
        if model.head is not None:
            model.head.weight[:len(ids)].copy_(vectors)


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.w


def rope_cache(seq, hd, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(seq, device=device).float()
    f = torch.outer(t, inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):  # x: B,H,T,hd
    x1, x2 = x[..., ::2].float(), x[..., 1::2].float()
    c, s = cos[None, None, : x.shape[2]], sin[None, None, : x.shape[2]]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2).type_as(x)


class Block(nn.Module):
    def __init__(self, c: LMConfig):
        super().__init__()
        self.c = c
        hd = c.d_model // c.n_heads
        self.n1, self.n2 = RMSNorm(c.d_model, c.norm_eps), RMSNorm(c.d_model, c.norm_eps)
        self.q = nn.Linear(c.d_model, c.n_heads * hd, bias=False)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * hd, bias=False)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * hd, bias=False)
        self.o = nn.Linear(c.n_heads * hd, c.d_model, bias=False)
        self.gate = nn.Linear(c.d_model, c.ffn_dim, bias=False)
        self.up = nn.Linear(c.d_model, c.ffn_dim, bias=False)
        self.down = nn.Linear(c.ffn_dim, c.d_model, bias=False)

    def forward(self, x, cos, sin):
        c = self.c
        B, T, _ = x.shape
        hd = c.d_model // c.n_heads
        h = self.n1(x)
        q = self.q(h).view(B, T, c.n_heads, hd).transpose(1, 2)
        k = self.k(h).view(B, T, c.n_kv_heads, hd).transpose(1, 2)
        v = self.v(h).view(B, T, c.n_kv_heads, hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if c.n_kv_heads != c.n_heads:
            r = c.n_heads // c.n_kv_heads
            k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, -1))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class LM(nn.Module):
    def __init__(self, c: LMConfig):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.d_model, c.norm_eps)
        self.head = None if c.tie_embeddings else nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.aux_emb = nn.Embedding(c.aux_size, c.d_model, padding_idx=0) if c.aux_size else None
        self.apply(self._init)
        if self.aux_emb is not None:
            with torch.no_grad():
                self.aux_emb.weight[0].zero_()
        for n, p in self.named_parameters():  # GPT-2 style scaled residual init
            if n.endswith("o.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layers))
        self._rope = None

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, aux=None):
        c = self.c
        if self._rope is None or self._rope[0].device != idx.device or self._rope[0].shape[0] < idx.shape[1]:
            self._rope = rope_cache(max(c.max_seq, idx.shape[1]), c.d_model // c.n_heads, c.rope_theta, idx.device)
        cos, sin = self._rope
        x = self.emb(idx)
        if aux is not None:
            x = x + self.aux_emb(aux)
        for b in self.blocks:
            x = b(x, cos, sin)
        x = self.norm(x)
        return x @ self.emb.weight.T if self.head is None else self.head(x)


def save_checkpoint(path, model, cfg: LMConfig, extra=None):
    m = model.module if hasattr(model, "module") else model
    torch.save(dict(cfg=asdict(cfg), state=m.state_dict(), extra=extra or {}), path)


def load_checkpoint(path, device="cpu"):
    d = torch.load(path, map_location=device)
    cfg = LMConfig(**d["cfg"])
    m = LM(cfg).to(device)
    m.load_state_dict(d["state"])
    return m, cfg
