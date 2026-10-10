from __future__ import annotations

import re as stdlib_re
from functools import cache
from pathlib import Path

from .model import TokenizerModel
from .pretokenize import PATTERN, iter_pretokens
from .remerge import event_remerge


class HierarchicalTokenizer:
    def __init__(self, model: TokenizerModel, backend: str = "auto", second_pass: str = "full"):
        model.validate()
        if second_pass not in {"full", "skip", "local", "hybrid"}:
            raise ValueError("second_pass must be full, skip, local, or hybrid")
        self.second_pass = second_pass
        self.model = model
        self._rank: dict[tuple[int, int], tuple[int, int]] = {}
        for rank, merge in enumerate(model.merges):
            self._rank.setdefault((merge.left, merge.right), (rank, merge.output))
        self._remerge_rules = {
            pair: rule for pair, rule in self._rank.items() if model.levels[rule[1]] <= 1
        }
        self._visible = model.visible_ids
        self._internal_to_model = {token_id: index for index, token_id in enumerate(self._visible)}
        self._model_to_internal = dict(enumerate(self._visible))
        self._special_pattern = None
        if model.special_tokens:
            choices = sorted(model.special_tokens, key=len, reverse=True)
            self._special_pattern = stdlib_re.compile(
                "(" + "|".join(map(stdlib_re.escape, choices)) + ")"
            )
        self._fast_encoding = None
        self._fast_to_internal: dict[int, int] = {}
        if backend not in {"auto", "python", "tiktoken"}:
            raise ValueError("backend must be auto, python, or tiktoken")
        if backend != "python":
            try:
                import tiktoken

                mergeable_ids = [
                    token_id
                    for token_id in sorted(model.tokens)
                    if token_id not in model.special_tokens.values()
                ]
                mergeable_ranks = {
                    model.tokens[token_id]: rank for rank, token_id in enumerate(mergeable_ids)
                }
                self._fast_to_internal = dict(enumerate(mergeable_ids))
                self._fast_encoding = tiktoken.Encoding(
                    name=f"{model.variant}-{model.target_vocab_size}",
                    pat_str=PATTERN.pattern,
                    mergeable_ranks=mergeable_ranks,
                    special_tokens={},
                )
            except ImportError:
                if backend == "tiktoken":
                    raise

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> HierarchicalTokenizer:
        return cls(TokenizerModel.load(path), **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self._visible)

    @property
    def eos_token_id(self) -> int:
        internal = self.model.special_tokens["<|endoftext|>"]
        return self._internal_to_model[internal]

    def _apply_bpe(self, sequence: list[int], max_output_level: int = 2,
                   first_pair: tuple[int, int] | None = None) -> list[int]:
        while len(sequence) > 1:
            best_pair: tuple[int, int] | None = None
            best_rank = len(self.model.merges) + 1
            best_output = -1
            if first_pair is not None:
                best_pair = first_pair
                best_rank, best_output = self._rank[first_pair]
                first_pair = None
            else:
                for left, right in zip(sequence, sequence[1:]):
                    candidate = self._rank.get((left, right))
                    if candidate is None:
                        continue
                    rank, output = candidate
                    if self.model.levels[output] <= max_output_level and rank < best_rank:
                        best_pair = (left, right)
                        best_rank = rank
                        best_output = output
            if best_pair is None:
                break
            left, right = best_pair
            result: list[int] = []
            index = 0
            while index < len(sequence):
                if (
                    index + 1 < len(sequence)
                    and sequence[index] == left
                    and sequence[index + 1] == right
                ):
                    result.append(best_output)
                    index += 2
                else:
                    result.append(sequence[index])
                    index += 1
            sequence = result
        return sequence

    @cache
    def _demolish_token(self, token_id: int, max_level: int) -> tuple[int, ...]:
        if self.model.levels[token_id] <= max_level:
            return (token_id,)
        alternatives = self.model.parents.get(token_id, [])
        if not alternatives:
            raise ValueError(f"Scaffold token {token_id} has no decomposition")
        candidates = [
            self._demolish_token(left, max_level) + self._demolish_token(right, max_level)
            for left, right in alternatives
        ]
        return min(candidates, key=lambda value: (len(value), value))

    def _second_merge(self, sequence: list[int], saturated: bool) -> list[int]:
        if self.second_pass == "hybrid" and len(sequence) >= 128:
            return self._hybrid_merge(sequence)
        if self.second_pass == "skip" or (self.second_pass == "local" and saturated):
            if not any(self.model.levels[t] == 2 for t in sequence):
                # The fast backend concatenates pretokens: unchanged boundary
                # edges can still be eligible, so a blind no-L2 skip is unsafe.
                if saturated or not any(
                    (a, b) in self._remerge_rules for a, b in zip(sequence, sequence[1:])
                ):
                    return sequence
        early: list[int] = []
        starts = set() if saturated and self.second_pass == "local" else None
        for token_id in sequence:
            start = len(early)
            early.extend(self._demolish_token(token_id, max_level=1))
            if starts is not None and self.model.levels[token_id] == 2:
                starts.update(range(max(0, start - 1), len(early)))
        if self.second_pass == "local":
            return event_remerge(early, self._remerge_rules, starts)
        return self._apply_bpe(early, max_output_level=1)

    def _hybrid_merge(self, sequence: list[int]) -> list[int]:
        early, candidates, ranks = [], [], set()
        for token_id in sequence:
            for part in self._demolish_token(token_id, max_level=1):
                left = len(early) - 1
                if left >= 0:
                    rule = self._remerge_rules.get((early[left], part))
                    if rule is not None:
                        candidates.append((rule[0], left, left + 1))
                        ranks.add(rule[0])
                early.append(part)
        if not candidates:
            return early
        # Fixed before measurement; no corpus-tuned threshold search.
        if len(ranks) < 4:
            _, left, right = min(candidates)
            return self._apply_bpe(early, 1, first_pair=(early[left], early[right]))
        return event_remerge(early, self._remerge_rules, candidates=candidates)

    def _encode_pretoken(self, token: bytes) -> list[int]:
        sequence = self._apply_bpe(list(token), max_output_level=2)
        return self._demolish_sequence(sequence, saturated=True)

    def _demolish_sequence(self, sequence: list[int], saturated: bool = False) -> list[int]:
        if self.model.variant == "hierarchical":
            sequence = self._second_merge(sequence, saturated)
        final: list[int] = []
        for token_id in sequence:
            final.extend(self._demolish_token(token_id, max_level=0))
        return final

    def _encode_regular_text(self, text: str) -> list[int]:
        if self._fast_encoding is not None:
            fast_ids = self._fast_encoding.encode_ordinary(text)
            internal = [self._fast_to_internal[token_id] for token_id in fast_ids]
            return self._demolish_sequence(internal)
        internal: list[int] = []
        for pretoken in iter_pretokens(text):
            internal.extend(self._encode_pretoken(pretoken))
        return internal

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        internal_ids: list[int] = []
        parts = self._special_pattern.split(text) if self._special_pattern else [text]
        for part in parts:
            if not part:
                continue
            special_id = self.model.special_tokens.get(part)
            if special_id is not None:
                internal_ids.append(special_id)
            else:
                internal_ids.extend(self._encode_regular_text(part))
        if add_eos:
            internal_ids.append(self.model.special_tokens["<|endoftext|>"])
        return [self._internal_to_model[token_id] for token_id in internal_ids]

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.model.tokens[self._model_to_internal[token_id]] for token_id in ids)
        return raw.decode("utf-8", errors="replace")
