#!/usr/bin/env python
"""Measure, don't guess: for each (preset, vocab) find the largest micro-batch that fits and the tokens/s
it achieves on ONE GPU, in the precision train_lm.py will use.  Multiply by n_gpu x ~0.9 for DDP.
Results guide --micro_batch / --batch_tokens and whether --compile helps.  Uses random tokens."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, torch.nn.functional as F
from scaffold_margin.lm import LM, PRESETS, make_config


def trial(preset, V, seq, mb, amp, compile_):
    dev = "cuda:0"
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = LM(make_config(preset, V + 64, seq)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == torch.float16))
    f = torch.compile(m) if compile_ else m
    x = torch.randint(0, V, (mb, seq), device=dev)
    def step():
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            lg = f(x)
        loss = F.cross_entropy(lg.float().view(-1, lg.shape[-1]), x.view(-1))
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    for _ in range(2): step()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(4): step()
    torch.cuda.synchronize()
    r = dict(micro_batch=mb, tok_s=4 * mb * seq / (time.time() - t), peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9)
    del m, opt, f
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", nargs="+", default=["small", "base", "large"])
    ap.add_argument("--vocab", type=int, nargs="+", default=[32768, 65536])
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", default="autotune.json")
    a = ap.parse_args()
    cap = torch.cuda.get_device_capability()[0]
    amp = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}.get(a.amp, torch.bfloat16 if cap >= 8 else torch.float16)
    res = []
    for p in a.presets:
        for V in a.vocab:
            best = None
            for mb in (1, 2, 4, 8, 16, 32, 64):
                try:
                    r = trial(p, V, a.seq, mb, amp, a.compile)
                except torch.OutOfMemoryError:
                    break
                except Exception as e:
                    print("error", p, V, mb, repr(e)[:200]); break
                best = r
                print(p, V, r, flush=True)
            res.append(dict(preset=p, vocab=V, best=best))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
