"""Separate a vocabulary ablation's word and identical-continuation prediction effects."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .direction_a import pool_candidates


@dataclass(frozen=True)
class Edit:
    before: tuple
    after: tuple
    start: int
    before_end: int
    after_end: int
    suffix_bytes: int
    doc: int
    word_bytes: int = 1

    def validate(self):
        if not (0 <= self.start < self.before_end < len(self.before)
                and self.start < self.after_end < len(self.after) and self.suffix_bytes > 0):
            raise ValueError("word and nonempty continuation spans required")
        if (self.before[:self.start] != self.after[:self.start]
                or self.before[self.before_end:] != self.after[self.after_end:]):
            raise ValueError("prefix and suffix target IDs must be identical")


def collect(pool, docs_words, max_samples=32, context=12, seq=256, seed=0):
    rng = np.random.default_rng(seed)
    candidates = set(map(int, pool_candidates(pool)))
    edits, seen, memo = defaultdict(list), Counter(), {}
    rejected = Counter()
    for doc, words in enumerate(docs_words):
        encoded = [pool.encode_word(w) for w in words]
        for i, word in enumerate(words):
            if i + 1 == len(words):
                continue
            for model_id in set(encoded[i]):
                token = int(pool.from_model[model_id])
                if token not in candidates:
                    continue
                key = (word, token)
                if key not in memo:
                    memo[key] = tuple(map(int, pool.to_model[pool.encode_ablated(word, {token})]))
                alternative = memo[key]
                if alternative == encoded[i]:
                    continue
                lo, hi = max(0, i - context), min(len(words), i + context + 1)
                def parts():
                    prefix = tuple(t for row in encoded[lo:i] for t in row)
                    suffix = tuple(t for row in encoded[i + 1:hi] for t in row)
                    return prefix, suffix
                prefix, suffix = parts()
                while max(len(prefix) + len(encoded[i]), len(prefix) + len(alternative)) + len(suffix) > seq:
                    if lo < i:
                        lo += 1
                    elif hi > i + 2:
                        hi -= 1
                    else:
                        break
                    prefix, suffix = parts()
                if not suffix or max(len(prefix) + len(encoded[i]), len(prefix) + len(alternative)) + len(suffix) > seq:
                    rejected["context_overflow"] += 1
                    continue
                if pool.decode(alternative) != word:
                    raise ValueError("ablation changed the raw target pretoken")
                edit = Edit(prefix + encoded[i] + suffix, prefix + alternative + suffix,
                            len(prefix), len(prefix) + len(encoded[i]), len(prefix) + len(alternative),
                            sum(map(len, words[i + 1:hi])), doc, len(word))
                edit.validate()
                seen[token] += 1
                n = seen[token]
                slot = len(edits[token]) if len(edits[token]) < max_samples else int(rng.integers(n))
                if slot < max_samples:
                    if slot == len(edits[token]):
                        edits[token].append(edit)
                    else:
                        edits[token][slot] = edit
    return dict(edits), dict(seen), dict(rejected)


def score(pool, scorer, edits, chunk=512):
    flat = [(token, edit) for token, rows in edits.items() for edit in rows]
    rows, max_prefix_error, max_partition_error = defaultdict(list), 0., 0.
    for start in range(0, len(flat), chunk):
        group = flat[start:start + chunk]
        losses = scorer.token_nll([s for _, e in group for s in (e.before, e.after)],
                                 pool.eos_id, vocab_size=pool.n_normal + 1)
        for (token, e), before, after in zip(group, losses[::2], losses[1::2]):
            e.validate()
            if len(before) != len(e.before) or len(after) != len(e.after):
                raise ValueError("scorer must return exactly one NLL per target")
            word = (after[e.start:e.after_end].sum() - before[e.start:e.before_end].sum()) / np.log(2)
            future = (after[e.after_end:].sum() - before[e.before_end:].sum()) / np.log(2)
            prefix = (after[:e.start].sum() - before[:e.start].sum()) / np.log(2)
            total = (after.sum() - before.sum()) / np.log(2)
            max_prefix_error = max(max_prefix_error, abs(prefix))
            max_partition_error = max(max_partition_error, abs(total - word - future - prefix))
            rows[token].append([word, future, total, e.suffix_bytes, e.doc, e.word_bytes])
    return {t: np.asarray(v) for t, v in rows.items()}, dict(max_prefix_delta_bits=max_prefix_error,
            max_partition_error_bits=max_partition_error, windows=len(flat))


def summaries(rows):
    result = {}
    for token, values in rows.items():
        n = len(values)
        # Shrink towards zero: an unobserved token provides no evidence for an exchange.
        result[token] = dict(n=n, word=float(values[:, 0].sum() / values[:, 5].sum() * n / (n + 4)),
                             future=float(values[:, 1].sum() / values[:, 3].sum() * n / (n + 4)),
                             future_bpb=float(values[:, 1].sum() / values[:, 3].sum()),
                             suffix_bytes=int(values[:, 3].sum()))
    return result
