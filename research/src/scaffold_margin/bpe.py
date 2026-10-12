"""Byte-level BPE / Scaffold-BPE / Hierarchical Scaffold-BPE.

Terminology (matches the research brief)
----------------------------------------
* A *token* has an expanded id: 0..255 are raw bytes, 256+i is the result of merge i.
* r(t)  = residual frequency of t: how often t occurs in the training corpus after
          *all* merges were applied (the expanded-vocabulary tokenisation).
* Judgement (report slide 2 + the team's trainer): after a merge (a,b)->t, tokens whose frequency is below
  q = f(Qhead) of the NEW queue head are scaffold (leave V); a scaffold token is revived when the head falls to its
  frequency (the team keeps it in the queue).  `scaffold_rule="global"` implements this exactly (see train_bpe).
* Level: L0 = not scaffold; scaffold tokens are L2 if the lowest r/q since they were flagged is < tau (tau=0.5;
  sticky until revival), else L1.  Bytes are always L0.  `levels` is stored explicitly in the tokenizer so that other
  selection rules (Direction A) can assign L0/L2 arbitrarily.

Encoding modes
--------------
bpe         plain greedy-rank BPE on a vocabulary whose levels are all 0.
scaffold    full BPE, then demolish every token with level >= 1 into its children
            until only L0 tokens remain (Scaffold-BPE, Lian et al. 2025).
hscaffold   full BPE -> demolish L2 -> second merge pass restricted to merges whose
            result is not L2 (and, optionally, to an allow-list of re-merge rules)
            -> demolish remaining L1.   (H-Scaffold-BPE, our extension)

Everything returned to callers is a *model id*: the rank of the token among L0
tokens (ascending expanded id).  Two tokenizers whose L0 sets coincide therefore
share model ids, which is what lets one LM score both segmentations.
"""
from __future__ import annotations

import heapq
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .pretok import pretokenize

_BIG = 1 << 60


# ----------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------
@dataclass
class TrainResult:
    merges: list          # [(a, b)], token id of merge i is 256 + i
    merge_freq: list      # pair frequency at the moment merge i was performed
    resid: np.ndarray     # residual frequency r(t), len = 256 + len(merges)
    q: int                # head-of-queue pair frequency when training stopped
    scaffold_mode: bool
    scaf: Optional[np.ndarray] = None    # bool, token currently flagged scaffold (len 256+merges)
    ratio: Optional[np.ndarray] = None   # r(t)/f(Qhead) at the token's LAST judgement (flagged tokens)
    rule: str = "component"


def train_bpe(
    word_counts: dict,
    num_merges: Optional[int] = None,
    target_normal: Optional[int] = None,
    max_merges: Optional[int] = None,
    min_freq: int = 2,
    log_every: int = 2000,
    scaffold_rule: str = "global",
    scaffold_factor: float = 1.0,
) -> TrainResult:
    """Train BPE.

    * `num_merges` given            -> plain BPE with exactly that many merges.
    * `target_normal` given         -> Scaffold-BPE: keep merging until the number of
      non-scaffold tokens (|V|) reaches `target_normal`.

    `scaffold_rule` (only matters for Scaffold-BPE):
      "global" (default): after every merge compare EVERY merged token's frequency with f(Qhead) of the
          new queue head; f(x) < f(Qhead) -> scaffold (leaves V), otherwise it is (re)activated.  Only the
          merge's components can newly fall below the head (all other tokens are unchanged and f(Qhead) only
          falls), so flagging equals the team's rule; reactivation of ANY scaffold token when the head falls to
          its frequency equals the team's queue-based revival.  Verified against `hscaffold_bpe.trainer`:
          identical merges, scaffold sets and L1/L2 levels on synthetic counts (tests/test_adapter.py).
      "component": literal reading of report slide 2 (judge only a and b; revive only when judged again).
          It does NOT match the team's trainer - kept only to show the difference.

    `scaffold_factor` c (global rule only): a token is scaffold iff f(x) < c * f(Qhead).  c = 1 is Scaffold-BPE;
    larger c hides more intermediate tokens (their slots go to further merges), c -> 0 approaches plain BPE.
    """
    assert scaffold_rule in ("component", "global")
    assert scaffold_factor == 1.0 or scaffold_rule == "global"
    assert (num_merges is None) != (target_normal is None)
    scaffold = target_normal is not None
    if max_merges is None:
        max_merges = num_merges if not scaffold else 3 * target_normal

    words, cnts = [], []
    for w, c in word_counts.items():
        words.append(list(w))
        cnts.append(c)

    freq = np.zeros(256 + max_merges, dtype=np.int64)
    pair_counts: dict = defaultdict(int)
    pair_words: dict = defaultdict(set)
    for wi, (w, c) in enumerate(zip(words, cnts)):
        for t in w:
            freq[t] += c
        for p in zip(w, w[1:]):
            pair_counts[p] += c
            pair_words[p].add(wi)
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    def head():
        while heap:
            negc, p = heap[0]
            cur = pair_counts.get(p, 0)
            if cur == -negc and cur > 0:
                return cur, p
            heapq.heappop(heap)
            if cur > 0:
                heapq.heappush(heap, (-cur, p))
        return 0, None

    scaf = np.zeros(256 + max_merges, dtype=bool)
    ratio = np.zeros(256 + max_merges, dtype=np.float64)
    n_normal = 256
    merges, merge_freq = [], []
    q = 0
    while len(merges) < max_merges:
        cnt, pair = head()
        if pair is None or cnt < min_freq:
            q = cnt if pair is not None else 0
            break
        a, b = pair
        c_id = 256 + len(merges)
        merges.append(pair)
        merge_freq.append(cnt)
        changed = set()
        total_k = 0
        for wi in pair_words.pop(pair, ()):
            w = words[wi]
            out, i, n, k = [], 0, len(w), 0
            while i < n:
                if i < n - 1 and w[i] == a and w[i + 1] == b:
                    out.append(c_id)
                    i += 2
                    k += 1
                else:
                    out.append(w[i])
                    i += 1
            if k == 0:
                continue
            cw = cnts[wi]
            total_k += k * cw
            for p in zip(w, w[1:]):
                pair_counts[p] -= cw
                changed.add(p)
            for p in zip(out, out[1:]):
                pair_counts[p] += cw
                pair_words[p].add(wi)
                changed.add(p)
            words[wi] = out
        freq[a] -= total_k
        freq[b] -= total_k
        freq[c_id] += total_k
        for p in changed:
            v = pair_counts[p]
            if v > 0:
                heapq.heappush(heap, (-v, p))
            else:
                pair_counts.pop(p, None)
        q, _ = head()
        if log_every and len(merges) % log_every == 0:
            print(f"  [bpe] merge {len(merges)}  head_freq={q}", flush=True)
        if scaffold:
            n_normal += 1                      # the new token t enters V (f(t) = pair count >= f(Qhead))
            if q > 0:
                if scaffold_rule == "component":
                    for x in {a, b}:
                        if x < 256:
                            continue           # raw bytes are never scaffold (needed for coverage)
                        if freq[x] < q:
                            cur = freq[x] / q
                            if not scaf[x]:
                                scaf[x] = True
                                n_normal -= 1
                                ratio[x] = cur
                            else:
                                ratio[x] = min(ratio[x], cur)   # L2 is sticky: keep the lowest ratio since flagging
                        elif scaf[x]:          # "may revive later"
                            scaf[x] = False
                            n_normal += 1
                else:
                    m = len(merges)
                    fr = freq[256 : 256 + m]
                    new = fr < scaffold_factor * q
                    old = scaf[256 : 256 + m]
                    n_normal += int(old.sum()) - int(new.sum())
                    cur = fr / q
                    # ratio = lowest r/f(Qhead) since the token was (re-)flagged: r only falls at merges and
                    # f(Qhead) only falls over time, so this equals the team's "L2 is sticky until revival"
                    ratio[256 : 256 + m] = np.where(new & ~old, cur, np.where(new, np.minimum(ratio[256 : 256 + m], cur), ratio[256 : 256 + m]))
                    scaf[256 : 256 + m] = new
            if n_normal >= target_normal:
                break
    n = 256 + len(merges)
    return TrainResult(merges, merge_freq, freq[:n].copy(), int(q), scaffold,
                       scaf[:n].copy(), ratio[:n].copy(), scaffold_rule)


def assign_levels(res: TrainResult, tau: float = 0.5, target_normal: Optional[int] = None) -> np.ndarray:
    """L0/L1/L2 per the report: L0 = not scaffold; scaffold tokens are L2 if r/f(Qhead) < tau at
    their last judgement, else L1 (tau <= ratio < 1).  `ratio` is stored, so tau can be changed
    without retraining (this is how a tokenizer-only tau ablation works)."""
    n = len(res.resid)
    lv = np.zeros(n, dtype=np.int8)
    if not res.scaffold_mode:
        return lv
    sc = res.scaf[256:]
    lv[256:] = np.where(~sc, 0, np.where(res.ratio[256:] < tau, 2, 1))
    r = res.resid[256:]
    if target_normal is not None:
        excess = 256 + int((lv[256:] == 0).sum()) - target_normal
        if excess > 0:  # revivals can overshoot K by a few; demote the weakest normal tokens
            l0 = np.nonzero(lv[256:] == 0)[0]
            weakest = l0[np.argsort(r[l0], kind="stable")[:excess]]
            lv[256 + weakest] = 1
    return lv
    r = res.resid[256:]
    q = res.q
    lv[256:] = np.where(r >= q, 0, np.where(r >= tau * q, 1, 2))
    if target_normal is not None:
        excess = 256 + int((lv[256:] == 0).sum()) - target_normal
        if excess > 0:  # dynamic re-classification can overshoot; demote the weakest
            l0 = np.nonzero(lv[256:] == 0)[0]
            weakest = l0[np.argsort(r[l0], kind="stable")[:excess]]
            lv[256 + weakest] = 1
    return lv


class _RankSet:
    """Indexable 'mask' over merge ranks backed by a set (sparse allow-list)."""

    def __init__(self, ranks):
        self.r = set(ranks)

    def __getitem__(self, i):
        return i in self.r


# ----------------------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------------------
class Tokenizer:
    def __init__(self, merges, levels, mode="scaffold", allowed_remerge=None,
                 second_pass="event", cache=True, meta=None, pattern="margin"):
        assert mode in ("bpe", "scaffold", "hscaffold")
        assert second_pass in ("event", "full", "local")
        self.merges = [tuple(m) for m in merges]
        self.levels = np.asarray(levels, dtype=np.int8)
        assert len(self.levels) == 256 + len(self.merges)
        self.mode = mode
        self.second_pass = second_pass
        self.pattern = pattern
        self.meta = dict(meta or {})
        self.rank = {m: i for i, m in enumerate(self.merges)}
        tb = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            tb.append(tb[a] + tb[b])
        self.token_bytes = tb
        # model ids = rank among L0 tokens
        l0 = np.nonzero(self.levels == 0)[0]
        self.n_normal = len(l0)
        self.to_model = np.full(len(self.levels), -1, dtype=np.int64)
        self.to_model[l0] = np.arange(len(l0))
        self.from_model = l0
        self.eos_id = self.n_normal
        self.model_token_bytes = np.array([len(tb[t]) for t in l0] + [0], dtype=np.int64)
        self.set_allowed_remerge(allowed_remerge)
        self._cache: Optional[dict] = {} if cache else None

    # -- re-merge allow-list ---------------------------------------------------------
    def set_allowed_remerge(self, allowed):
        """allowed: None (all eligible re-merges) or a set of (a, b) rules."""
        self.allowed_remerge = None if allowed is None else {tuple(r) for r in allowed}
        ok = np.zeros(len(self.merges), dtype=bool)
        for i, (a, b) in enumerate(self.merges):
            if self.levels[256 + i] <= 1 and (self.allowed_remerge is None or (a, b) in self.allowed_remerge):
                ok[i] = True
        self._remerge_ok = ok
        self._remerge_rules = {pair: (r, 256 + r) for pair, r in self.rank.items() if ok[r]}
        if getattr(self, "_cache", None) is not None:
            self._cache.clear()

    # -- core operations -------------------------------------------------------------
    def _merge_loop(self, ids, ok=None, log=None):
        rank, merges = self.rank, self.merges
        ids = list(ids)
        while len(ids) > 1:
            best_r = _BIG
            for i in range(len(ids) - 1):
                r = rank.get((ids[i], ids[i + 1]))
                if r is not None and r < best_r and (ok is None or ok[r]):
                    best_r = r
            if best_r == _BIG:
                break
            a, b = merges[best_r]
            c = 256 + best_r
            out, i, n = [], 0, len(ids)
            while i < n:
                if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(c)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            if log is not None:
                log.append((a, b, c))
            ids = out
        return ids

    def _demolish(self, ids, min_level):
        lv, merges = self.levels, self.merges
        out = []
        for t in ids:
            if lv[t] < min_level:
                out.append(t)
                continue
            stack = [t]
            while stack:
                x = stack.pop()
                if lv[x] >= min_level:
                    a, b = merges[x - 256]
                    stack.append(b)
                    stack.append(a)
                else:
                    out.append(x)
        return out

    def _l2_stage(self, full):
        if self.second_pass != "local":
            return self._demolish(full, 2), None
        early, starts = [], set()
        for token in full:
            if self.levels[token] < 2:
                early.append(token)
            else:
                start = len(early)
                early.extend(self._demolish([token], 2))
                starts.update(range(max(0, start - 1), len(early)))
        return early, starts

    def _second_merge(self, early, starts, changed, log=None):
        if self.second_pass == "local":
            from hscaffold_bpe.remerge import event_remerge
            return event_remerge(early, self._remerge_rules, starts, log)
        if changed or self.second_pass == "full":
            return self._merge_loop(early, ok=self._remerge_ok, log=log)
        return early

    def trace(self, word: bytes) -> dict:
        """Every intermediate stage of encoding one pretoken (expanded ids)."""
        full = self._merge_loop(list(word))
        scaf = self._demolish(full, 1)
        after_l2, starts = self._l2_stage(full)
        log: list = []
        has_l2 = len(after_l2) != len(full) or after_l2 != full
        pass2 = self._second_merge(after_l2, starts, has_l2, log)
        h = self._demolish(pass2, 1)
        return dict(full=full, scaffold=scaf, after_l2=after_l2, remerges=log,
                    pass2=pass2, hscaffold=h, has_l2=has_l2)

    def encode_ablated(self, word: bytes, ablate) -> list:
        """Scaffold-mode encoding in which the tokens in `ablate` are additionally treated as
        scaffold (i.e. removed from the normal vocabulary).  Expanded ids; no cache."""
        ablate = set(ablate)
        lv, merges = self.levels, self.merges
        out = []
        for t in self._merge_loop(list(word)):
            stack = [t]
            while stack:
                x = stack.pop()
                if lv[x] >= 1 or x in ablate:
                    a, b = merges[x - 256]
                    stack.append(b)
                    stack.append(a)
                else:
                    out.append(x)
        return out

    def encode_with_rules(self, word: bytes, rules) -> list:
        """H-Scaffold encoding of one pretoken with an ad-hoc re-merge allow-list
        (expanded ids).  Used for per-rule marginal analysis; does not touch the cache."""
        full = self._merge_loop(list(word))
        after_l2 = self._demolish(full, 2)
        if after_l2 != full:
            ok = _RankSet(self.rank[r] for r in rules if r in self.rank and self.levels[256 + self.rank[r]] <= 1)
            after_l2 = self._merge_loop(after_l2, ok=ok)
        return self._demolish(after_l2, 1)

    def encode_word_expanded(self, word: bytes) -> list:
        full = self._merge_loop(list(word))
        if self.mode == "bpe":
            return full
        if self.mode == "scaffold":
            return self._demolish(full, 1)
        after_l2, starts = self._l2_stage(full)
        after_l2 = self._second_merge(after_l2, starts, after_l2 != full)
        return self._demolish(after_l2, 1)

    def encode_word(self, word: bytes) -> tuple:
        c = self._cache
        if c is not None:
            hit = c.get(word)
            if hit is not None:
                return hit
        ids = self.encode_word_expanded(word)
        out = tuple(int(x) for x in self.to_model[ids])
        assert min(out) >= 0, "non-L0 token leaked into output"
        if c is not None:
            c[word] = out
        return out

    def encode(self, text: str) -> list:
        out: list = []
        for w in pretokenize(text, self.pattern):
            out.extend(self.encode_word(w))
        return out

    def decode(self, model_ids: Iterable[int]) -> bytes:
        return b"".join(
            b"" if i == self.eos_id else self.token_bytes[self.from_model[i]] for i in model_ids
        )

    # -- io --------------------------------------------------------------------------
    def to_json(self) -> dict:
        return dict(merges=self.merges, levels=self.levels.tolist(), mode=self.mode,
                    allowed_remerge=None if self.allowed_remerge is None else sorted(self.allowed_remerge),
                    second_pass=self.second_pass, meta=self.meta, pattern=self.pattern)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_json(), f)

    @classmethod
    def load(cls, path: str, **overrides) -> "Tokenizer":
        with open(path) as f:
            d = json.load(f)
        if d.get("format") == "hierarchical-scaffold-bpe-v1":     # a tokenizer trained by the team's package
            from .adapter import load_team_tokenizer

            return load_team_tokenizer(path, **overrides)
        d.update(overrides)
        return cls(d["merges"], d["levels"], mode=d["mode"], allowed_remerge=d.get("allowed_remerge"),
                   second_pass=d.get("second_pass", "event"), meta=d.get("meta"), pattern=d.get("pattern", "margin"))

    def variant(self, **kw) -> "Tokenizer":
        """Copy with a different mode / allow-list / levels (shares merges)."""
        args = dict(merges=self.merges, levels=self.levels, mode=self.mode,
                    allowed_remerge=self.allowed_remerge, second_pass=self.second_pass,
                    cache=self._cache is not None, meta=self.meta, pattern=self.pattern)
        args.update(kw)
        return Tokenizer(**args)


def build_tokenizers(word_counts: dict, vocab_size: int, tau: float = 0.5, scaffold_rule: str = "global",
                     pattern: str = "margin") -> dict:
    """Train BPE and Scaffold-BPE at one normal-vocabulary size.

    Returns {'bpe', 'scaffold', 'hscaffold'}; the last two share merges and levels, so
    their L0 vocabularies and model ids are identical by construction.
    """
    bpe_res = train_bpe(word_counts, num_merges=vocab_size - 256)
    bpe = Tokenizer(bpe_res.merges, assign_levels(bpe_res), mode="bpe", pattern=pattern,
                    meta=dict(vocab_size=vocab_size, kind="bpe"))
    sc_res = train_bpe(word_counts, target_normal=vocab_size, scaffold_rule=scaffold_rule)
    lv = assign_levels(sc_res, tau=tau, target_normal=vocab_size)
    meta = dict(vocab_size=vocab_size, tau=tau, q=sc_res.q, merge_freq=sc_res.merge_freq,
                resid=sc_res.resid.tolist(), ratio=sc_res.ratio.tolist(), scaffold_rule=scaffold_rule)
    sc = Tokenizer(sc_res.merges, lv, mode="scaffold", pattern=pattern, meta=dict(meta, kind="scaffold"))
    hs = sc.variant(mode="hscaffold", meta=dict(meta, kind="hscaffold"))
    return dict(bpe=bpe, scaffold=sc, hscaffold=hs)
