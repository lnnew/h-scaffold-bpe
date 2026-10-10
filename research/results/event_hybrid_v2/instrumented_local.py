def event_remerge(sequence, rules, starts=None, log=None, *, candidates=None):
    """`rules` maps pairs to (rank, output). None seeds every initial edge.

    A restricted frontier is safe only when all omitted edges were already
    saturated by the first pass. Original positions break ties within a batch.
    """
    n = len(sequence)
    _gen = _clock()
    if candidates is None:
        heap = []
        for left in range(n - 1) if starts is None else sorted(starts):
            if 0 <= left < n - 1:
                rule = rules.get((sequence[left], sequence[left + 1]))
                if rule is not None:
                    heap.append((rule[0], left, left + 1))
    else:
        # Hybrid collects eligible edges while emitting demolished tokens.
        heap = list(candidates)
    _stats['candidate_generation'] += _clock() - _gen
    if not heap:
        _out = _clock()
        result = list(sequence)
        _stats['output_reconstruction'] += _clock() - _out
        return result
    _setup = _clock()
    heapify(heap)
    values = list(sequence)
    previous = list(range(-1, n - 1))
    following = list(range(1, n)) + [-1]

    _stats['state_setup'] += _clock() - _setup

    def push(left):
        if left < 0 or following[left] < 0:
            return
        right = following[left]
        rule = rules.get((values[left], values[right]))
        if rule is not None:
            heappush(heap, (rule[0], left, right))

    while heap:
        _queue = _clock()
        _application = 0.0
        rank = heap[0][0]
        changed = []
        batch_merge = None
        while heap and heap[0][0] == rank:
            _, left, right = heappop(heap)
            if following[left] != right or following[right] == -2:
                continue
            pair = (values[left], values[right])
            rule = rules.get(pair)
            if rule is None or rule[0] != rank:
                continue
            _apply = _clock()
            values[left] = rule[1]
            following[left] = following[right]
            if following[right] >= 0:
                previous[following[right]] = left
            following[right] = -2
            changed.append(left)
            batch_merge = (*pair, rule[1])
            _application += _clock() - _apply
        _stats['merge_application'] += _application
        _stats['queue_pop_validation'] += _clock() - _queue - _application
        if changed:
            if log is not None:
                log.append(batch_merge)
            # Delay new candidates until the entire reference batch is done;
            # reused output tokens can expose a rank lower than this batch.
            _push = _clock()
            for left in changed:
                if following[left] != -2:
                    push(previous[left])
                    push(left)
            _stats['neighbor_candidate_push'] += _clock() - _push
    _out = _clock()
    result = []
    left = 0
    while left >= 0:
        result.append(values[left])
        left = following[left]
    _stats['output_reconstruction'] += _clock() - _out
    return result
