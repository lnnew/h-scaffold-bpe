"""Text I/O, corpus tokenisation to .bin streams, and BPB bookkeeping.

Text splits live as jsonl files ({"text": ..., "lang": ...} per line) produced by
scripts/prepare_text.py, so everything downstream is independent of where data came from.

Token streams on disk (uint32):  [EOS] doc1 [EOS] doc2 [EOS] ...   with one extra
`<split>_docs.npy` array holding the stream offset of each document's first token.
Because every real token is a prediction target and EOS counts as 0 bytes, summing
target-token byte lengths gives *exactly* the raw UTF-8 size of the documents: BPB is
comparable across tokenizers (same numerator semantics, same denominator).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from typing import Iterator

import numpy as np


def read_jsonl(path: str, limit: int | None = None) -> Iterator[tuple]:
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            d = json.loads(line)
            yield d["text"], d.get("lang", "en")


def write_jsonl(path: str, docs) -> int:
    n = 0
    with open(path, "w") as f:
        for text, lang in docs:
            f.write(json.dumps(dict(text=text, lang=lang), ensure_ascii=False) + "\n")
            n += 1
    return n


# ---- parallel encoding -----------------------------------------------------------------
_TOK = None


def _init(tok_path, overrides):
    global _TOK
    from .bpe import Tokenizer

    _TOK = Tokenizer.load(tok_path, **overrides)


def _enc(text):
    return _TOK.encode(text)


def encode_docs(tok_path: str, texts: list, workers: int = 4, overrides: dict | None = None) -> list:
    overrides = overrides or {}
    if workers <= 1:
        _init(tok_path, overrides)
        return [_enc(t) for t in texts]
    with mp.get_context("fork").Pool(workers, initializer=_init, initargs=(tok_path, overrides)) as p:
        return p.map(_enc, texts, chunksize=64)


def write_stream(out_dir: str, name: str, docs_ids: list, eos_id: int, langs: list | None = None):
    parts, offs, pos = [np.array([eos_id], dtype=np.uint32)], [], 1
    for ids in docs_ids:
        offs.append(pos)
        a = np.asarray(ids, dtype=np.uint32)
        parts += [a, np.array([eos_id], dtype=np.uint32)]
        pos += len(a) + 1
    stream = np.concatenate(parts)
    stream.tofile(os.path.join(out_dir, f"{name}.bin"))
    np.save(os.path.join(out_dir, f"{name}_docs.npy"), np.array(offs, dtype=np.int64))
    if langs is not None:
        with open(os.path.join(out_dir, f"{name}_langs.json"), "w") as f:
            json.dump(langs, f)
    return len(stream)


def open_stream(path: str) -> np.memmap:
    return np.memmap(path, dtype=np.uint32, mode="r")


# ---- BPB ---------------------------------------------------------------------------------
def doc_ids_for_targets(offsets: np.ndarray, n_stream: int) -> np.ndarray:
    """Document index of every stream position (the leading EOS gets doc 0)."""
    d = np.zeros(n_stream, dtype=np.int64)
    d[offsets] = 1
    return np.maximum(np.cumsum(d) - 1, 0)


def bpb_from_token_nll(token_nll: np.ndarray, targets: np.ndarray, token_bytes: np.ndarray,
                       doc_of_target: np.ndarray | None = None, n_docs: int | None = None):
    """token_nll[i] (nats) is the loss of predicting targets[i]. Returns total BPB and,
    if doc_of_target is given, per-document (nll_bits, bytes) arrays for bootstrap/by-language."""
    b = token_bytes[targets]
    tot = float(token_nll.sum() / np.log(2) / b.sum())
    if doc_of_target is None:
        return tot, None
    nll_d = np.bincount(doc_of_target, weights=token_nll, minlength=n_docs) / np.log(2)
    by_d = np.bincount(doc_of_target, weights=b, minlength=n_docs)
    return tot, (nll_d, by_d)


# ---- parallel word counting ------------------------------------------------------------
def _count_chunk(args):
    from .pretok import count_words

    texts, pattern = args
    return count_words(texts, pattern)


def count_words_parallel(docs, workers: int = 4, chunk_docs: int = 2000, pattern: str = "margin"):
    """docs: iterable of str. Returns Counter of pretokens (bytes).  pattern='gpt2' = the team's pretokenizer."""
    from collections import Counter

    total: Counter = Counter()

    def chunks():
        buf = []
        for d in docs:
            buf.append(d)
            if len(buf) >= chunk_docs:
                yield buf, pattern
                buf = []
        if buf:
            yield buf, pattern

    if workers <= 1:
        for c in chunks():
            total.update(_count_chunk(c))
        return total
    with mp.get_context("fork").Pool(workers) as p:
        for cnt in p.imap_unordered(_count_chunk, chunks()):
            total.update(cnt)
    return total
