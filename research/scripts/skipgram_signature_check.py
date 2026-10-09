#!/usr/bin/env python
"""Does the label double shift produce the reported *signature* (next-token NLL worse than unigram)?

Count-model stand-in (one-token context) on real English text with a real tokenizer:
  * model trained for x_t -> x_{t+1}                      (what training should optimise)
  * model trained for x_t -> x_{t+2}                      (what `model(labels=x[:, 1:])` optimises)
  * both scored on the aligned next-token task that hscaffold_bpe.lm.evaluate measures, vs a unigram.
A real LM has longer context so absolute numbers differ; the point is the ordering and the size of the gap.
Default text: English documentation that exists on most machines (no downloads).  Pass --text_files for WikiText.
"""
import argparse, glob, os, sys, random, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scaffold_margin.bpe import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", required=True, help="e.g. <team repo>/artifacts/tokenizers/scaffold-32768.json")
ap.add_argument("--text_files", nargs="*", default=None, help="plain-text files (one document each); default: docs found on this machine")
a = ap.parse_args()

# --- gather English prose that happens to be on this machine (docs, licenses, READMEs, pydoc topics)
files = list(a.text_files or [])
for root in () if a.text_files else ("/usr/local/lib/python3.13/dist-packages", "/usr/lib/python3", "/usr/share/doc", "/usr/share/common-licenses"):
    for ext in ("*.md", "*.rst", "*.txt", "METADATA"):
        files += glob.glob(f"{root}/**/{ext}", recursive=True)
files = [f for f in files if os.path.isfile(f) and 2000 < os.path.getsize(f) < 400000 and "test" not in f.lower()]
docs = []
if not a.text_files:
    import pydoc_data.topics as T
    docs += list(T.topics.values())
for f in sorted(set(files)):
    try: docs.append(open(f, encoding="utf-8").read())
    except Exception: pass
docs = [d for d in docs if sum(c.isascii() for c in d) > 0.97 * len(d)]
random.Random(0).shuffle(docs)
cut = int(0.8 * len(docs))
tr, te = docs[:cut], docs[cut:]
print(f"{len(docs)} English docs, {sum(map(len,docs))/1e6:.1f} MB; train {len(tr)} / test {len(te)}")

tok = Tokenizer.load(a.tokenizer)
def enc(ds):
    out = []
    for d in ds:
        out.append(np.array([tok.eos_id] + tok.encode(d), dtype=np.int64))
    return out
Tr, Te = enc(tr), enc(te)
V = tok.n_normal + 1
print("train tokens", sum(map(len, Tr)), "test tokens", sum(map(len, Te)), "vocab", V)

def pairs(streams, gap):
    a = np.concatenate([s[:-gap] for s in streams if len(s) > gap]); b = np.concatenate([s[gap:] for s in streams if len(s) > gap])
    return a, b
uni = np.bincount(np.concatenate(Tr), minlength=V).astype(float)
p_uni = (uni + 1) / (uni.sum() + V)

class Cond:                                     # absolute-discounting conditional model p(b | a) with unigram back-off
    def __init__(self, a, b, d=0.75):
        k = a * V + b; self.keys, self.cnt = np.unique(k, return_counts=True); self.d = d
        self.ctx = np.bincount(self.keys // V, weights=self.cnt, minlength=V); self.n1 = np.bincount(self.keys // V, minlength=V).astype(float)
    def logp(self, a, b):
        k = a * V + b; i = np.minimum(np.searchsorted(self.keys, k), len(self.keys) - 1)
        c = np.where(self.keys[i] == k, self.cnt[i], 0); ca = self.ctx[a]
        lam = np.where(ca > 0, self.d * self.n1[a] / np.maximum(ca, 1), 1.0)
        return np.log(np.where(ca > 0, np.maximum(c - self.d, 0) / np.maximum(ca, 1), 0) + lam * p_uni[b])

# held-out *next-token* evaluation pairs (what hscaffold_bpe.lm.evaluate measures)
ea, eb = pairs(Te, 1)
nll = lambda lp: -lp.mean()
res = {
  "unigram": nll(np.log(p_uni[eb])),
  "model trained for next token  (x_t -> x_t+1)": nll(Cond(*pairs(Tr, 1)).logp(ea, eb)),
  "model trained two-ahead      (x_t -> x_t+2)": nll(Cond(*pairs(Tr, 2)).logp(ea, eb)),   # what the buggy loss optimises, scored as next-token
}
print(f"\nNext-token test NLL (nats/token) on real English with tokenizer {os.path.basename(a.tokenizer)}:")
for k, v in res.items(): print(f"  {k:46s} {v:6.3f}   PPL {math.exp(v):9.1f}")
