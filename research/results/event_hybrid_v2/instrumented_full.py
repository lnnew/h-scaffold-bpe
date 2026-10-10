def _apply_bpe(self, sequence: list[int], max_output_level: int = 2) -> list[int]:
    while len(sequence) > 1:
        _scan = _clock()
        best_pair: tuple[int, int] | None = None
        best_rank = len(self.model.merges) + 1
        best_output = -1
        for left, right in zip(sequence, sequence[1:]):
            candidate = self._rank.get((left, right))
            if candidate is None:
                continue
            rank, output = candidate
            if self.model.levels[output] <= max_output_level and rank < best_rank:
                best_pair = (left, right)
                best_rank = rank
                best_output = output
        _stats['full_scan'] += _clock() - _scan
        if best_pair is None:
            break
        _apply = _clock()
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
        _stats['merge_application'] += _clock() - _apply
    return sequence
