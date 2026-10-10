#!/usr/bin/env python
"""Evaluate a frozen checkpoint without retraining or overwriting its development results."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_lm_suite import check_identity, file_hash
from scaffold_margin.data import bpb_from_token_nll, doc_ids_for_targets, open_stream
from scaffold_margin.lm import load_checkpoint
from scaffold_margin.train_lm import eval_stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="separately encoded evaluation streams")
    ap.add_argument("--out", type=Path, required=True, help="new evaluation output; training results remain unchanged")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    train = json.loads(a.checkpoint.with_name("results.json").read_text())
    meta = json.loads((a.data / "meta.json").read_text())
    tokenizer_hash = file_hash(meta["tokenizer"])
    run_identity = a.checkpoint.with_name("identity.json")
    trained_tokenizer_hash = (json.loads(run_identity.read_text())["data"]["tokenizer"] if run_identity.exists()
                              else file_hash(train["meta"]["tokenizer"]))
    if tokenizer_hash != trained_tokenizer_hash or meta["n_normal"] != train["meta"]["n_normal"]:
        raise ValueError("evaluation tokenizer must have exactly the checkpoint's token-ID mapping")
    source = Path(__file__).resolve().parents[1] / "src/scaffold_margin"
    identity = dict(checkpoint=file_hash(a.checkpoint), tokenizer=tokenizer_hash,
                    inputs={name: file_hash(a.data / name) for name in
                            [a.split + ".bin", a.split + "_docs.npy", "token_bytes.npy"]},
                    code={name: file_hash(source / name) for name in ["lm.py", "train_lm.py", "data.py"]},
                    evaluator=file_hash(__file__), split=a.split)
    check_identity(a.out, identity, a.out / "results.json", False)
    if (a.out / "results.json").exists():
        print("skip verified evaluation:", a.out)
        return
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, cfg = load_checkpoint(str(a.checkpoint), device)
    model.eval()
    amp = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[train["amp"]] if device.startswith("cuda") else None
    stream = open_stream(str(a.data / (a.split + ".bin")))
    token_bytes = np.load(a.data / "token_bytes.npy")
    offsets = np.load(a.data / (a.split + "_docs.npy"))
    nll, targets = eval_stream(model, stream, token_bytes, cfg.max_seq, device, amp)
    doc_ids = doc_ids_for_targets(offsets, len(stream))[1:]
    bpb, (doc_nll, doc_bytes) = bpb_from_token_nll(nll, targets, token_bytes, doc_ids, len(offsets))
    if not np.isfinite(bpb):
        raise ValueError("non-finite evaluation BPB")
    np.savez_compressed(a.out / ("eval_" + a.split + ".npz"), doc_nll_bits=doc_nll, doc_bytes=doc_bytes)
    result = dict(checkpoint=str(a.checkpoint.resolve()), training_args=train["args"], cfg=vars(cfg),
                  data_meta=meta, split=a.split, bpb={a.split: bpb}, identity=identity,
                  note="frozen checkpoint evaluation; does not establish performance at a larger model/budget")
    (a.out / "results.json").write_text(json.dumps(result, indent=2))
    print(f"{a.split} BPB={bpb:.6f}")


if __name__ == "__main__":
    main()
