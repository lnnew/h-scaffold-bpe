"""Paired vocabulary interventions on identical UTF-8 chunks, with short adaptation."""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .lm import LM, init_pool_embeddings, load_checkpoint, make_config
from .train_lm import pad_vocab


def raw_chunks(text, max_bytes):
    if max_bytes < 4:
        raise ValueError("max_bytes must accommodate a UTF-8 character")
    start, size = 0, 0
    for i, char in enumerate(text):
        width = len(char.encode("utf-8"))
        if size + width > max_bytes:
            yield text[start:i]
            start, size = i, 0
        size += width
    if start < len(text):
        yield text[start:]


def pack(rows, byte_counts, doc_ids):
    lengths = np.asarray([len(row) for row in rows], dtype=np.int64)
    if not len(rows) or (lengths == 0).any():
        raise ValueError("aligned corpus needs nonempty chunks")
    return dict(ids=np.concatenate(rows).astype(np.int32),
                offsets=np.r_[0, np.cumsum(lengths)],
                bytes=np.asarray(byte_counts, dtype=np.int64),
                docs=np.asarray(doc_ids, dtype=np.int64))


def prepare_aligned(pool, docs, max_bytes=255, max_chunks=None):
    rows, sizes, doc_ids = [], [], []
    digest = hashlib.sha256()
    for doc, text in enumerate(docs):
        for chunk in raw_chunks(text, max_bytes):
            raw = chunk.encode("utf-8")
            ids = pool.encode(chunk)
            if pool.decode(ids) != raw:
                raise ValueError("tokenizer does not losslessly represent a raw chunk")
            digest.update(len(raw).to_bytes(4, "little") + raw)
            rows.append(ids); sizes.append(len(raw)); doc_ids.append(doc)
            if max_chunks and len(rows) >= max_chunks:
                result = pack(rows, sizes, doc_ids)
                result["raw_hash"] = np.array(digest.hexdigest())
                return result
    result = pack(rows, sizes, doc_ids)
    result["raw_hash"] = np.array(digest.hexdigest())
    return result


def retokenize(corpus, pool, tokenizer):
    if tokenizer.merges != pool.merges or tokenizer.pattern != pool.pattern:
        raise ValueError("interventions must share the pool merge table and pretokenizer")
    if not set(tokenizer.from_model).issubset(set(pool.from_model)):
        raise ValueError("only pool-normal candidates are supported")
    lookup = [tokenizer.to_model[tokenizer._demolish([int(t)], 1)].tolist()
              for t in pool.from_model]
    rows = []
    for left, right in zip(corpus["offsets"][:-1], corpus["offsets"][1:]):
        rows.append([x for t in corpus["ids"][left:right] for x in lookup[t]])
    result = pack(rows, corpus["bytes"], corpus["docs"])
    for row, count in zip(rows, result["bytes"]):
        if min(row) < 0 or tokenizer.model_token_bytes[row].sum() != count:
            raise ValueError("intervention changed raw byte coverage")
    result["raw_hash"] = corpus["raw_hash"]
    return result


def batch(corpus, indices, eos, seq, device):
    x = np.full((len(indices), seq), eos, dtype=np.int64)
    y = np.full_like(x, -100)
    for row, i in enumerate(indices):
        ids = corpus["ids"][corpus["offsets"][i]:corpus["offsets"][i + 1]]
        if len(ids) + 1 > seq:
            raise ValueError("raw chunk exceeds the common model context")
        x[row, :len(ids) + 1] = np.r_[eos, ids]
        y[row, :len(ids) + 1] = np.r_[ids, eos]
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def training_batches(n_chunks, steps, batch_chunks, data_seed):
    if n_chunks < batch_chunks or min(steps, batch_chunks) <= 0:
        raise ValueError("positive budget and sufficient raw chunks required")
    per_epoch = n_chunks // batch_chunks
    order, previous = None, -1
    for step in range(steps):
        epoch, pos = divmod(step, per_epoch)
        if epoch != previous:
            order = np.random.default_rng([data_seed, epoch]).permutation(n_chunks)
            previous = epoch
        yield order[pos * batch_chunks:(pos + 1) * batch_chunks]


def mapped_model(pool, tokenizer, teacher, device, seed=0, preset="small", seq=256):
    torch.manual_seed(seed)
    if teacher:
        parent, cfg = load_checkpoint(teacher, "cpu")
        if cfg.vocab_size < pool.n_normal + 1:
            raise ValueError("teacher cannot represent pool IDs")
        cfg = replace(cfg, vocab_size=pad_vocab(tokenizer.n_normal + 1), max_seq=seq)
        model = LM(cfg)
        state = parent.state_dict()
        for name, value in model.state_dict().items():
            if name not in ("emb.weight", "head.weight"):
                value.copy_(state[name])
        ids = pool.to_model[tokenizer.from_model].tolist() + [pool.eos_id]
        with torch.no_grad():
            model.emb.weight[:len(ids)].copy_(state["emb.weight"][ids])
            if model.head is not None:
                model.head.weight[:len(ids)].copy_(state["head.weight"][ids])
        del parent
    else:
        cfg = make_config(preset, pad_vocab(tokenizer.n_normal + 1), seq)
        model = LM(cfg)
        init_pool_embeddings(model, tokenizer, pool, seed)
    return model.to(device)


@torch.no_grad()
def evaluate(model, corpus, tokenizer, seq, device, amp_dtype, bs=8):
    model.eval()
    losses = np.zeros(len(corpus["bytes"]), dtype=np.float64)
    for start in range(0, len(losses), bs):
        idx = np.arange(start, min(start + bs, len(losses)))
        x, y = batch(corpus, idx, tokenizer.eos_id, seq, device)
        length = int((y != -100).sum(1).max())
        x, y = x[:, :length], y[:, :length]
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.startswith("cuda")):
            logits = model(x)[..., :tokenizer.n_normal + 1]
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="none").view_as(y)
        losses[idx] = loss.sum(1).double().cpu().numpy() / math.log(2)
    model.train()
    docs = corpus["docs"]
    nll = np.bincount(docs, weights=losses)
    sizes = np.bincount(docs, weights=corpus["bytes"])
    return float(losses.sum() / sizes.sum()), nll, sizes


def run_probe(pool, tokenizer, corpora, out, teacher=None, steps=128, data_seed=0,
              seed=0, preset="small", seq=256, batch_chunks=32, micro_batch=8, lr=3e-4):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if steps <= 0 or batch_chunks <= 0 or micro_batch <= 0 or batch_chunks % micro_batch:
        raise ValueError("positive steps/chunks required; micro_batch must divide batch_chunks")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    amp = torch.float16 if device.startswith("cuda") else None
    data = {name: retokenize(c, pool, tokenizer) for name, c in corpora.items()}
    train = data.pop("train")
    if len(train["bytes"]) < batch_chunks:
        raise ValueError("not enough raw chunks for an update")
    model = mapped_model(pool, tokenizer, teacher, device, seed, preset, seq)
    opt = torch.optim.AdamW([dict(params=[p for p in model.parameters() if p.ndim >= 2], weight_decay=.1),
                            dict(params=[p for p in model.parameters() if p.ndim < 2], weight_decay=0)],
                           lr=lr, betas=(.9, .95), fused=device.startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"), init_scale=1024,
                                  growth_interval=2000)
    result = dict(steps=steps, data_seed=data_seed, init_seed=seed, teacher=teacher,
                  cfg=model.c.__dict__, n_params=model.c.n_params(), curves={},
                  raw_train_hash=str(train["raw_hash"]), train_exposure_bytes=0,
                  train_exposure_tokens=0, skipped_updates=0,
                  protocol="same raw chunks; byte-normalized objective; EOS targets included; padding logits excluded")
    order_digest = hashlib.sha256()
    t0 = time.monotonic()
    def assess(step):
        metrics = {}
        for name, corpus in data.items():
            value, nll, sizes = evaluate(model, corpus, tokenizer, seq, device, amp)
            metrics[name] = value
            np.savez_compressed(out / f"eval_{name}_h{step}.npz", doc_nll_bits=nll, doc_bytes=sizes)
        result["curves"][str(step)] = metrics
        print(json.dumps(dict(step=step, bpb=metrics)), flush=True)
    assess(0)
    for step, idx in enumerate(training_batches(len(train["bytes"]), steps, batch_chunks, data_seed)):
        order_digest.update(idx.astype("<i8").tobytes())
        count = int(train["bytes"][idx].sum())
        for start in range(0, batch_chunks, micro_batch):
            x, y = batch(train, idx[start:start + micro_batch], tokenizer.eos_id, seq, device)
            with torch.autocast("cuda", dtype=amp, enabled=device.startswith("cuda")):
                logits = model(x)[..., :tokenizer.n_normal + 1]
            loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="sum") / count
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite probe loss")
            scaler.scale(loss).backward()
            result["train_exposure_tokens"] += int((y != -100).sum())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        old_scale = scaler.get_scale()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        result["skipped_updates"] += int(scaler.get_scale() < old_scale)
        result["train_exposure_bytes"] += count
        if step + 1 in {min(32, steps), steps}:
            assess(step + 1)
    result.update(train_order_hash=order_digest.hexdigest(), elapsed_s=time.monotonic() - t0)
    (out / "results.json").write_text(json.dumps(result, indent=2))
    return result
