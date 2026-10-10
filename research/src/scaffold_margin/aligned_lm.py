"""From-scratch LM comparisons with common raw targets and overlapping raw context."""
from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .learning import mapped_model, pack, raw_chunks, training_batches
from .lm import save_checkpoint
from .pretok import pretokenize


def prepare(pool, tokenizers, docs, seq=256, context=64, limit=None):
    if not 0 <= context < seq - 4:
        raise ValueError("context must leave room for targets")
    lookups = {}
    for name, tok in tokenizers.items():
        if tok.merges != pool.merges or tok.pattern != pool.pattern:
            raise ValueError("all tokenizers must share the expanded merge table")
        lookups[name] = [tok.to_model[tok._demolish([int(t)], 1)].tolist() for t in pool.from_model]
    rows = {name: [] for name in tokenizers}
    starts = {name: [] for name in tokenizers}
    sizes, doc_ids, digest, split_units = [], [], hashlib.sha256(), 0
    for doc, text in enumerate(docs):
        units = []
        for word in pretokenize(text, pool.pattern):
            ids = pool.encode_word(word)
            if max(sum(len(lookup[t]) for t in ids) for lookup in lookups.values()) > seq - context:
                # shortcut: split only oversized pretokens; report their count in every corpus.
                pieces = [s.encode() for s in raw_chunks(word.decode(), seq - context)]
                split_units += 1
                units.extend(pieces)
            else:
                units.append(word)
        encoded_pool = [pool.encode_word(word) for word in units]
        encoded = {name: [[v for t in row for v in lookup[t]] for row in encoded_pool]
                   for name, lookup in lookups.items()}
        cumulative = [np.r_[0, np.cumsum([len(row) for row in values])] for values in encoded.values()]
        i = 0
        while i < len(units):
            lo = i
            while lo > 0 and max(int(c[i] - c[lo - 1]) for c in cumulative) <= context:
                lo -= 1
            j = i
            while j < len(units) and max(int(c[j + 1] - c[lo]) for c in cumulative) <= seq:
                j += 1
            if j == i:
                raise ValueError("an oversized pretoken could not fit the common context")
            raw = b"".join(units[i:j])
            digest.update(doc.to_bytes(8, "little") + len(raw).to_bytes(4, "little") + raw)
            for name, values in encoded.items():
                prefix = [v for row in values[lo:i] for v in row]
                target = [v for row in values[i:j] for v in row]
                if tokenizers[name].decode(target) != raw:
                    raise ValueError("aligned targets changed raw bytes")
                rows[name].append(prefix + target)
                starts[name].append(len(prefix))
            sizes.append(len(raw)); doc_ids.append(doc)
            i = j
            if limit and len(sizes) >= limit:
                break
        if limit and len(sizes) >= limit:
            break
    result = {}
    for name in tokenizers:
        result[name] = pack(rows[name], sizes, doc_ids)
        result[name].update(starts=np.asarray(starts[name], dtype=np.int64),
                            raw_hash=np.array(digest.hexdigest()), split_units=np.array(split_units))
    return result


def batch(corpus, indices, eos, device):
    length = max(int(corpus["offsets"][i + 1] - corpus["offsets"][i]) for i in indices)
    x = np.full((len(indices), length), eos, dtype=np.int64)
    y = np.full_like(x, -100)
    for row, i in enumerate(indices):
        ids = corpus["ids"][corpus["offsets"][i]:corpus["offsets"][i + 1]]
        x[row, :len(ids)] = np.r_[eos, ids[:-1]]
        start = int(corpus["starts"][i])
        y[row, start:len(ids)] = ids[start:]
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def evaluate(model, corpus, tokenizer, device, amp, bs=8):
    model.eval()
    losses = np.zeros(len(corpus["bytes"]), dtype=np.float64)
    for start in range(0, len(losses), bs):
        idx = np.arange(start, min(start + bs, len(losses)))
        x, y = batch(corpus, idx, tokenizer.eos_id, device)
        with torch.autocast("cuda", dtype=amp, enabled=device.startswith("cuda")):
            logits = model(x)[..., :tokenizer.n_normal + 1]
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="none").view_as(y)
        losses[idx] = loss.sum(1).double().cpu().numpy() / math.log(2)
    model.train()
    nll = np.bincount(corpus["docs"], weights=losses)
    sizes = np.bincount(corpus["docs"], weights=corpus["bytes"])
    return float(nll.sum() / sizes.sum()), nll, sizes


def train(pool, tokenizer, corpora, out, steps=1500, seed=0, data_seed=0, preset="small",
          seq=256, batch_windows=32, micro_batch=8, lr=3e-4):
    out.mkdir(parents=True, exist_ok=False)
    if steps <= 0 or batch_windows % micro_batch:
        raise ValueError("positive steps and divisible micro batches required")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    amp = torch.float16 if device.startswith("cuda") else None
    model = mapped_model(pool, tokenizer, None, device, seed, preset, seq)
    opt = torch.optim.AdamW([dict(params=[p for p in model.parameters() if p.ndim >= 2], weight_decay=.1),
                            dict(params=[p for p in model.parameters() if p.ndim < 2], weight_decay=0)],
                           lr=lr, betas=(.9, .95), fused=device.startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"), init_scale=1024,
                                  growth_interval=2000)
    train_data = corpora["train"]
    result = dict(steps=steps, init_seed=seed, data_seed=data_seed, cfg=model.c.__dict__,
                  n_params=model.c.n_params(), raw_train_hash=str(train_data["raw_hash"]),
                  train_exposure_bytes=0, train_exposure_tokens=0, skipped_updates=0, bpb={})
    digest, t0, warmup = hashlib.sha256(), time.monotonic(), min(200, max(1, steps // 10))
    model.train()
    with (out / "log.jsonl").open("w") as log:
        for step, idx in enumerate(training_batches(len(train_data["bytes"]), steps, batch_windows, data_seed)):
            rate = lr * ((step + 1) / warmup if step < warmup else
                         .1 + .9 * .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
            for group in opt.param_groups:
                group["lr"] = rate
            digest.update(idx.astype("<i8").tobytes())
            count, loss_sum = int(train_data["bytes"][idx].sum()), 0.
            for pos in range(0, batch_windows, micro_batch):
                x, y = batch(train_data, idx[pos:pos + micro_batch], tokenizer.eos_id, device)
                with torch.autocast("cuda", dtype=amp, enabled=device.startswith("cuda")):
                    logits = model(x)[..., :tokenizer.n_normal + 1]
                loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="sum") / count
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                scaler.scale(loss).backward()
                loss_sum += float(loss.detach())
                result["train_exposure_tokens"] += int((y != -100).sum())
            scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            old_scale = scaler.get_scale()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            result["skipped_updates"] += int(scaler.get_scale() < old_scale)
            result["train_exposure_bytes"] += count
            if step % 50 == 0 or step + 1 == steps:
                rec = dict(step=step + 1, loss_nats_per_byte=loss_sum, lr=rate,
                           gnorm=float(norm), elapsed_s=time.monotonic() - t0)
                line = json.dumps(rec)
                print(line, flush=True); log.write(line + "\n"); log.flush()
    result.update(train_order_hash=digest.hexdigest(), train_time_s=time.monotonic() - t0)
    if result["skipped_updates"]:
        raise FloatingPointError("skipped optimizer updates invalidate paired exposure")
    # Save final weights before any holdout is evaluated; no checkpoint selection on test.
    save_checkpoint(out / "model.pt", model, model.c, dict(seed=seed, steps=steps))
    for name, corpus in corpora.items():
        if name == "train":
            continue
        value, nll, sizes = evaluate(model, corpus, tokenizer, device, amp)
        result["bpb"][name] = value
        np.savez_compressed(out / f"eval_{name}.npz", doc_nll_bits=nll, doc_bytes=sizes)
        print(json.dumps(dict(split=name, bpb=value)), flush=True)
    (out / "results.json").write_text(json.dumps(result, indent=2))
    return result
