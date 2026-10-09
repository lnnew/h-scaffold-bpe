"""Load a tokenizer trained by the team's `hscaffold_bpe` package into this repo's `Tokenizer`.

The team's format (`hierarchical-scaffold-bpe-v1`): tokens keyed by id (0-255 bytes, 256 = <|endoftext|>,
257+ = merge outputs), `merges` = [left, right, output, freq], `levels` = {id: 0/1/2}, `parents`.
Only tokenizers in which every token has exactly one parent pair can be represented here (checked).

Why this exists: it lets every analysis in this repo (audit, rule extraction, utility labels, Direction A
pool selection, statistics) run on the *actual* 32K/64K tokenizers behind the report, and it lets us check
that the two implementations segment text identically (tests/test_adapter.py).
"""
from __future__ import annotations

import base64
import json

import numpy as np

from .bpe import Tokenizer

_MODES = {"bpe": "bpe", "scaffold": "scaffold", "hierarchical": "hscaffold"}


def load_team_tokenizer(path: str, mode: str | None = None, **kw) -> Tokenizer:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("format") != "hierarchical-scaffold-bpe-v1":
        raise ValueError("not a hscaffold_bpe tokenizer file")
    tokens = {int(i): base64.b64decode(v) for i, v in d["tokens"].items()}
    levels = {int(i): int(v) for i, v in d["levels"].items()}
    multi = [i for i, ps in d["parents"].items() if len(ps) > 1]
    if multi:
        raise ValueError(f"{len(multi)} tokens have several parent pairs; not representable")
    special = set(d["special_tokens"].values())
    for b in range(256):
        assert tokens[b] == bytes([b])
    my_id = {b: b for b in range(256)}
    merges, lv = [], [0] * 256
    for left, right, out, _freq in d["merges"]:
        my_id[out] = 256 + len(merges)
        merges.append((my_id[left], my_id[right]))
        lv.append(levels[out])
    meta = dict(source="hscaffold_bpe", variant=d["variant"], team_ids={str(v): k for k, v in my_id.items() if v >= 256},
                level2_threshold=d.get("metadata", {}).get("level2_threshold"), merge_freq=[m[3] for m in d["merges"]])
    mode = mode or _MODES[d["variant"]]
    kw.setdefault("pattern", "gpt2")      # the pretokenizer the team's tokenizers were trained with
    return Tokenizer(merges, np.array(lv, dtype=np.int8), mode=mode, meta=meta, **kw)


def find_tokenizer(tok_dir: str, name: str, K: int) -> str:
    """Path of '<name>_<K>.json' (this repo) or '<name>-<K>.json' (team's naming; 'hscaffold' -> 'hierarchical')."""
    import os

    alias = {"hscaffold": "hierarchical"}
    for cand in (f"{name}_{K}.json", f"{name}-{K}.json", f"{alias.get(name, name)}-{K}.json"):
        p = os.path.join(tok_dir, cand)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no tokenizer for {name}/{K} in {tok_dir}")


def attach_residuals(tok: Tokenizer, word_counts: dict) -> Tokenizer:
    """The team's files do not store r(t) (residual frequency) or the judgement ratios.  r(t) is recomputable
    from a pretoken count table (full-BPE token counts); the judgement ratio is not, so it stays unavailable
    (features that need it become NaN and are ignored by the selectors)."""
    resid = np.zeros(len(tok.levels), dtype=np.int64)
    for w, c in word_counts.items():
        for t in tok._merge_loop(list(w)):
            resid[t] += c
    tok.meta["resid"] = resid.tolist()
    return tok


def team_pool_tokenizer(word_counts: dict, k_pool: int, team_repo: str) -> Tokenizer:
    """Direction-A pool built with the TEAM's trainer: Scaffold-BPE to `k_pool` normal tokens (their K counts
    <|endoftext|>, hence k_pool + 1).  Levels are collapsed to normal / scaffold (2) as in direction_a.pool_tokenizer."""
    import os
    import sys
    import tempfile

    sys.path.insert(0, os.path.join(team_repo, "src"))
    from hscaffold_bpe.trainer import train_from_counts

    model = train_from_counts(word_counts, k_pool + 1, "scaffold", show_progress=False)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pool.json")
        model.save(path)
        t = load_team_tokenizer(path, mode="scaffold", cache=False)
    lv = np.where(t.levels > 0, 2, 0).astype(np.int8)
    pool = Tokenizer(t.merges, lv, mode="scaffold", pattern=t.pattern,
                     meta=dict(t.meta, kind="apool", k_pool=k_pool))
    attach_residuals(pool, word_counts)
    return pool
