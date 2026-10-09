#!/usr/bin/env python
"""Direction B data: per-language train / dev / val / test jsonl files.

  <out>/train_<l>.jsonl   first `--train_bytes` of language l after the held-out docs
  <out>/dev_<l>, val_<l>, test_<l>.jsonl   held-out docs taken from the START of the stream
Held-out docs are taken first so that capping train bytes never changes them.
Natural language sizes (bytes in train_<l>) define the 'natural' mixture used for allocation.
Sources: synthetic (offline smoke test) | hf (wikimedia/wikipedia, config 20231101.<lang>)
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scaffold_margin.data import write_jsonl


def hf_docs(lang):
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    for r in ds:
        yield r["text"]


def synthetic_docs(lang, sizes={"l0": 1500, "l1": 600, "l2": 200, "l3": 60}):
    from scaffold_margin.synth import make_corpus, _SYL
    i = int(lang[1:])
    rot = _SYL[i * 5:] + _SYL[: i * 5]
    return iter(make_corpus(sizes[lang] + 60, seed=10 + i, alphabet=[s + ("" if i % 2 == 0 else "aa") for s in rot]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["synthetic", "hf"])
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train_bytes", type=float, default=200e6)
    ap.add_argument("--mix_alpha", type=float, default=0.5,
                    help="exponent for the FIXED LM-training mixture: share_l ~ n_l**mix_alpha (same for every tokenizer)")
    ap.add_argument("--eval_docs", type=int, default=300, help="docs per held-out split (dev/val/test) per language")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sizes, train_docs, val_docs = {}, {}, []
    for lang in a.langs:
        it = synthetic_docs(lang) if a.source == "synthetic" else hf_docs(lang)
        docs = [[] for _ in range(3)]
        for k in range(3):
            n_ev = a.eval_docs if a.source == "hf" else 20
            for _ in range(n_ev):
                docs[k].append(next(it))
        tr, size = [], 0
        for d in it:
            if size >= a.train_bytes:
                break
            tr.append(d); size += len(d.encode())
        for name, ds in zip(("dev", "val", "test"), docs):
            write_jsonl(f"{a.out}/{name}_{lang}.jsonl", ((d, lang) for d in ds))
        write_jsonl(f"{a.out}/train_{lang}.jsonl", ((d, lang) for d in tr))
        sizes[lang], train_docs[lang] = size, tr
        val_docs += docs[1]
        print(f"{lang}: train {size/1e6:.1f} MB ({len(tr)} docs); held-out {a.eval_docs} docs x3")
    # fixed LM training mixture + pooled val (the tokenizer allocation sweep never touches these)
    import numpy as np
    rng = np.random.default_rng(0)
    tot = sum(sizes.values())
    z = sum(v ** a.mix_alpha for v in sizes.values())
    mix = []
    for lang in a.langs:
        target = tot * (sizes[lang] ** a.mix_alpha / z)       # bytes wanted from this language
        docs_l = train_docs[lang]
        reps = target / max(1, sizes[lang])
        take = list(docs_l) * int(reps)
        extra = int(round((reps - int(reps)) * len(docs_l)))
        take += [docs_l[i] for i in rng.choice(len(docs_l), size=extra, replace=False)] if extra else []
        mix += [(d, lang) for d in take]
        print(f"mix {lang}: {target/1e6:.1f} MB target ({reps:.2f}x of its {sizes[lang]/1e6:.1f} MB)")
    rng.shuffle(mix)
    write_jsonl(f"{a.out}/lm_train.jsonl", mix)
    write_jsonl(f"{a.out}/val.jsonl", [(d, "mix") for d in val_docs])


if __name__ == "__main__":
    main()

