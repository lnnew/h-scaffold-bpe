#!/usr/bin/env python
"""E3: inference gap and scaffold-token composition per language/script (tokenizer-only, CPU).

For each multilingual tokenizer pair (BPE / Scaffold-BPE, same alpha):
  * bytes/token per language for BPE canonical, Scaffold canonical / longest / dp, BPE dp
  * parity (CV, Gini, max/min of bytes/token across languages) per inference.  The test sets are NOT parallel
    text, so this is the same within-language compression parity used by b_build_tokenizers, not a
    parallel-corpus premium.
  * which scripts the scaffold (hidden) tokens belong to, versus the visible vocabulary; incomplete UTF-8
    fragments are counted separately.
"""
import argparse
import json
import os
import sys
import unicodedata
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scaffold_margin.bpe import Tokenizer  # noqa: E402
from scaffold_margin.data import read_jsonl  # noqa: E402
from scaffold_margin.inference import VisibleSegmenter  # noqa: E402
from scaffold_margin.pretok import pretokenize  # noqa: E402


def script_of(b: bytes) -> str:
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return "partial-utf8"
    kinds = Counter()
    for ch in s:
        if ch.isspace() or not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        kinds[name.split(" ")[0] if name else "other"] += 1
    if not kinds:
        return "non-letter"
    k = kinds.most_common(1)[0][0]
    return {"LATIN": "latin", "CYRILLIC": "cyrillic", "DEVANAGARI": "devanagari"}.get(k, "other")


def parity(v):
    x = np.array(sorted(v))
    n = len(x)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
    return dict(cv=float(x.std() / x.mean()), gini=gini, max_over_min=float(x.max() / x.min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="label:bpe.json:scaffold.json")
    ap.add_argument("--text_dir", required=True)
    ap.add_argument("--langs", nargs="+", default=["en", "de", "ru", "hi"])
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    for spec in a.pairs:
        label, bp, sp = spec.split(":")
        bpe, sc = Tokenizer.load(bp), Tokenizer.load(sp)
        segs = {"bpe": VisibleSegmenter(bpe, "canonical"), "bpe_dp": VisibleSegmenter(bpe, "dp"),
                "scaffold": VisibleSegmenter(sc, "canonical"), "scaffold_longest": VisibleSegmenter(sc, "longest"),
                "scaffold_dp": VisibleSegmenter(sc, "dp")}
        res = dict(per_lang={}, parity={}, composition={})
        for split in a.splits:
            bpt = {m: {} for m in segs}
            for lang in a.langs:
                wc = Counter()
                nbytes = 0
                for t, _ in read_jsonl(os.path.join(a.text_dir, f"{split}_{lang}.jsonl")):
                    wc.update(pretokenize(t, sc.pattern))
                    nbytes += len(t.encode())
                row = {}
                for m, seg in segs.items():
                    n_tok = sum(n * len(seg.encode_word(w)) for w, n in wc.items())
                    row[m] = dict(tokens=n_tok, bytes_per_token=nbytes / n_tok)
                    bpt[m][lang] = nbytes / n_tok
                g = row["scaffold"]["tokens"] - row["scaffold_dp"]["tokens"]
                row["scaffold_dp_gap_rel"] = -g / row["scaffold"]["tokens"]
                row["bpe_to_scaffold_rel"] = row["scaffold"]["tokens"] / row["bpe"]["tokens"] - 1
                res["per_lang"][f"{split}_{lang}"] = row
                print(f"[E3 {label} {split} {lang}] bpe->scaffold {100*row['bpe_to_scaffold_rel']:+.3f}%  "
                      f"scaffold->dp {100*row['scaffold_dp_gap_rel']:+.3f}%  "
                      f"bpe->bpe_dp {100*(row['bpe_dp']['tokens']/row['bpe']['tokens']-1):+.3f}%", flush=True)
            res["parity"][split] = {m: parity(list(v.values())) for m, v in bpt.items()}
        hidden = [t for t in range(256, len(sc.levels)) if sc.levels[t] >= 1]
        visible = [t for t in range(256, len(sc.levels)) if sc.levels[t] == 0]
        res["composition"] = dict(
            n_hidden=len(hidden), n_visible=len(visible),
            hidden=dict(Counter(script_of(sc.token_bytes[t]) for t in hidden)),
            visible=dict(Counter(script_of(sc.token_bytes[t]) for t in visible)),
            bpe_vocab=dict(Counter(script_of(bpe.token_bytes[t]) for t in range(256, len(bpe.levels)))))
        print(f"[E3 {label}] composition {json.dumps(res['composition'])}", flush=True)
        out[label] = res
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
