"""Train a small LLaMA-style LM from scratch on a pre-tokenised stream; report BPB.

Launch (2 GPUs):  torchrun --nproc_per_node=2 -m scaffold_margin.train_lm --data DIR --out DIR ...
Single process / CPU also works (used for smoke tests).

Design choices that matter for the research question
----------------------------------------------------
* init seed (--seed) and data-order seed (--data_seed) are separate, so seed variance can
  be attributed (the brief stresses that all earlier results are single-seed).
* the same number of *epochs over the same raw text* is used for every tokenizer by
  default (--epochs); --max_steps gives a compute-matched alternative.
* evaluation dumps per-token NLL so that BPB can be re-aggregated per document (paired
  bootstrap) and per language without re-running the model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .data import bpb_from_token_nll, doc_ids_for_targets, open_stream
from .lm import LM, make_config, save_checkpoint


def pad_vocab(n, mult=64):
    return (n + mult - 1) // mult * mult


@torch.no_grad()
def eval_stream(model, stream, token_bytes, seq, device, amp_dtype, bs=16, aux_stream=None):
    """Contiguous windows: x = s[iL:iL+L], y = s[iL+1:iL+L+1]; each token after the first is
    predicted exactly once (context restarts at window starts, same as in training)."""
    model.eval()
    s = np.asarray(stream, dtype=np.int64)
    sa = None if aux_stream is None else np.asarray(aux_stream, dtype=np.int64)
    n_t = len(s) - 1
    nll = np.zeros(n_t, dtype=np.float32)
    starts = list(range(0, n_t, seq))
    for i in range(0, len(starts), bs):
        group = starts[i : i + bs]
        # group windows of equal length together; the last (short) one runs alone
        full = [st for st in group if st + seq <= n_t]
        short = [st for st in group if st + seq > n_t]
        for sts, L in ((full, seq), *(([st], n_t - st) for st in short)):
            if not sts:
                continue
            x = torch.tensor(np.stack([s[st : st + L] for st in sts]), device=device)
            y = torch.tensor(np.stack([s[st + 1 : st + L + 1] for st in sts]), device=device)
            xa = None if sa is None else torch.tensor(np.stack([sa[st : st + L] for st in sts]), device=device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.startswith("cuda") and amp_dtype is not None):
                logits = model(x) if xa is None else model(x, xa)
            l = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.reshape(-1), reduction="none").view(len(sts), L)
            for r, st in enumerate(sts):
                nll[st : st + L] = l[r].cpu().numpy()
    model.train()
    return nll, s[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with train.bin, <eval>.bin, token_bytes.npy, meta.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch_tokens", type=int, default=262144, help="global tokens per optimizer step")
    ap.add_argument("--micro_batch", type=int, default=8, help="sequences per GPU per forward")
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--max_steps", type=int, default=0, help="overrides --epochs when > 0")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0, help="model init seed")
    ap.add_argument("--data_seed", type=int, default=0, help="data order seed")
    ap.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--eval_sets", default="val,test", help="comma list of <name>.bin files to evaluate at the end")
    ap.add_argument("--eval_every", type=int, default=0, help="steps between val-BPB evals (0=end only)")
    ap.add_argument("--save_ckpt", action="store_true")
    ap.add_argument("--init_pool", help="pool tokenizer for common-token-matched embedding initialization")
    ap.add_argument("--max_train_tokens", type=int, default=0, help="truncate train stream (smoke tests)")
    ap.add_argument("--aux", action="store_true", help="use <split>_aux.bin input-only features (meta['aux_size'])")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()
    device = f"cuda:{local}" if use_cuda else "cpu"
    if world > 1:
        dist.init_process_group("nccl" if use_cuda else "gloo")
    if use_cuda:
        torch.cuda.set_device(local)
    master = rank == 0
    os.makedirs(args.out, exist_ok=True)

    # precision: Turing (sm_75) has no usable bf16 -> fp16 + loss scaling
    if args.amp == "auto":
        if use_cuda and torch.cuda.is_bf16_supported(including_emulation=False) and torch.cuda.get_device_capability()[0] >= 8:
            amp = "bf16"
        else:
            amp = "fp16" if use_cuda else "fp32"
    else:
        amp = args.amp
    amp_dtype = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=None)[amp]
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16"))

    meta = json.load(open(os.path.join(args.data, "meta.json")))
    token_bytes = np.load(os.path.join(args.data, "token_bytes.npy"))
    vocab = pad_vocab(meta["n_normal"] + 1)
    cfg = make_config(args.preset, vocab, args.seq)
    if args.aux:
        cfg.aux_size = int(meta["aux_size"])
    aux_of = lambda name: open_stream(os.path.join(args.data, f"{name}_aux.bin")) if args.aux else None  # noqa: E731

    torch.manual_seed(args.seed)
    model = LM(cfg).to(device)
    if args.init_pool:
        from .bpe import Tokenizer
        from .lm import init_pool_embeddings
        init_pool_embeddings(model, Tokenizer.load(meta["tokenizer"]), Tokenizer.load(args.init_pool), args.seed)
    if args.compile:
        model = torch.compile(model)
    raw_model = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local] if use_cuda else None)
    decay = [p for n, p in raw_model.named_parameters() if p.ndim >= 2]
    nodecay = [p for n, p in raw_model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=args.wd), dict(params=nodecay, weight_decay=0.0)],
                            lr=args.lr, betas=(0.9, 0.95), fused=use_cuda)

    train = open_stream(os.path.join(args.data, "train.bin"))
    train_aux = aux_of("train")
    if args.max_train_tokens:
        train = train[: args.max_train_tokens]
        train_aux = None if train_aux is None else train_aux[: args.max_train_tokens]
    assert train_aux is None or len(train_aux) == len(train), "aux stream misaligned" 
    L = args.seq
    n_blocks = (len(train) - 1) // L
    per_step_seqs = args.batch_tokens // L
    assert per_step_seqs % (world * args.micro_batch) == 0, \
        f"batch_tokens/seq ({per_step_seqs}) must be divisible by world*micro_batch ({world * args.micro_batch})"
    accum = per_step_seqs // (world * args.micro_batch)
    steps_per_epoch = n_blocks // per_step_seqs
    if steps_per_epoch == 0:
        raise ValueError("train stream is too short for one batch_tokens update")
    total_steps = args.max_steps or max(1, int(args.epochs * steps_per_epoch))
    if master:
        print(f"[train] amp={amp} vocab={vocab} params={cfg.n_params()/1e6:.1f}M (non-emb {cfg.n_params(False)/1e6:.1f}M) "
              f"train_tokens={len(train)/1e6:.1f}M steps/epoch={steps_per_epoch} total_steps={total_steps} accum={accum}", flush=True)

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        p = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))

    def block_order(epoch):
        return np.random.default_rng([args.data_seed, epoch]).permutation(n_blocks)

    def get_batch(order, step_in_epoch, micro):
        base = step_in_epoch * per_step_seqs + (micro * world + rank) * args.micro_batch
        idx = order[base : base + args.micro_batch]
        x = np.stack([np.asarray(train[i * L : i * L + L], dtype=np.int64) for i in idx])
        y = np.stack([np.asarray(train[i * L + 1 : i * L + L + 1], dtype=np.int64) for i in idx])
        xa = None if train_aux is None else torch.from_numpy(
            np.stack([np.asarray(train_aux[i * L : i * L + L], dtype=np.int64) for i in idx])).to(device)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device), int(token_bytes[y].sum()), xa

    def val_bpb():
        vs = open_stream(os.path.join(args.data, "val.bin"))
        nll, tg = eval_stream(raw_model, vs, token_bytes, args.seq, device, amp_dtype, aux_stream=aux_of("val"))
        return bpb_from_token_nll(nll, tg, token_bytes)[0]

    log = open(os.path.join(args.out, "log.jsonl"), "a") if master else None
    t0 = time.time()
    tokens_seen = 0
    bytes_seen = 0
    model.train()
    for step in range(total_steps):
        epoch, sie = divmod(step, max(1, steps_per_epoch))
        order = block_order(epoch)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss_acc = 0.0
        for micro in range(accum):
            x, y, batch_bytes, xa = get_batch(order, sie, micro)
            bytes_seen += batch_bytes
            sync = (micro == accum - 1) or world == 1
            ctx = model.no_sync() if (world > 1 and not sync) else torch.enable_grad()
            with ctx:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda and amp_dtype is not None):
                    logits = model(x) if xa is None else model(x, xa)
                loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1)) / accum
                scaler.scale(loss).backward()
            loss_acc += loss.item()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        tokens_seen += args.batch_tokens
        if master and (step % 20 == 0 or step == total_steps - 1):
            rec = dict(step=step, loss=loss_acc, lr=lr_at(step), gnorm=float(gn), tok_s=tokens_seen / (time.time() - t0),
                       elapsed=time.time() - t0)
            if args.eval_every and step > 0 and step % args.eval_every == 0:
                rec["val_bpb"] = val_bpb()
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
    train_time = time.time() - t0
    exposure = torch.tensor(bytes_seen, dtype=torch.int64, device=device)
    if world > 1:
        dist.all_reduce(exposure)

    if master:
        res = dict(args=vars(args), cfg=cfg.__dict__, amp=amp, train_time_s=train_time, total_steps=total_steps,
                   n_params=cfg.n_params(), train_tokens=int(len(train)),
                   train_exposure_tokens=tokens_seen, train_exposure_bytes=int(exposure.item()), meta=meta, bpb={})
        for name in args.eval_sets.split(","):
            p = os.path.join(args.data, f"{name}.bin")
            if not os.path.exists(p):
                continue
            s = open_stream(p)
            nll, tg = eval_stream(raw_model, s, token_bytes, args.seq, device, amp_dtype, aux_stream=aux_of(name))
            doc_off = np.load(os.path.join(args.data, f"{name}_docs.npy"))
            d_of = doc_ids_for_targets(doc_off, len(s))[1:]
            tot, (nll_d, by_d) = bpb_from_token_nll(nll, tg, token_bytes, d_of, len(doc_off))
            res["bpb"][name] = tot
            np.savez_compressed(os.path.join(args.out, f"eval_{name}.npz"), doc_nll_bits=nll_d, doc_bytes=by_d)
            print(f"[eval] {name}: BPB={tot:.4f}", flush=True)
        json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)
        if args.save_ckpt:
            save_checkpoint(os.path.join(args.out, "model.pt"), raw_model, cfg)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
