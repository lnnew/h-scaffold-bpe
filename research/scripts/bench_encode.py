#!/usr/bin/env python
"""Phase 4: encoding throughput, separating three questions (H3 in the brief).

  scaffold          baseline (demolition only)
  hscaffold-full    original H-Scaffold: second merge pass scans EVERY pretoken
  hscaffold-event   second pass only when an L2 token was actually demolished (exactly equal output)
  hsel-<name>-event event-triggered + selective allow-list (rejecting rules saves little by itself
                    because the trigger is still 'an L2 token was demolished')

Cache is OFF by default: a per-pretoken cache hides the cost of the algorithm on Zipfian text and is
not what the earlier report measured.  --cache reports the cached (deployment-like) numbers too.
Single process, pure Python: compare ratios, not absolute MB/s, with the earlier Rust/C-level numbers.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scaffold_margin.adapter import find_tokenizer
from scaffold_margin.bpe import Tokenizer
from scaffold_margin.data import read_jsonl


def run(tok, docs, nbytes, reps):
    best = 0
    out = None
    for _ in range(reps):
        if tok._cache is not None:
            tok._cache.clear()
        t = time.perf_counter()
        out = [tok.encode(d) for d in docs]
        best = max(best, nbytes / 1e6 / (time.perf_counter() - t))
    return best, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok_dir", required=True); ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--text", required=True, help="jsonl (use dev_sel, not test)")
    ap.add_argument("--variants", nargs="*", default=[], help="names of hsel_<name>_<K>.json to include")
    ap.add_argument("--max_docs", type=int, default=2000); ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--variants_dir", default=None, help="where hsel_<name>_<K>.json live (default: --tok_dir)")
    ap.add_argument("--cache", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    docs = [d for d, _ in read_jsonl(a.text, a.max_docs)]
    nbytes = sum(len(d.encode()) for d in docs)
    K, cache = a.K, a.cache
    sc = Tokenizer.load(find_tokenizer(a.tok_dir, "scaffold", K))
    base = Tokenizer.load(find_tokenizer(a.tok_dir, "hscaffold", K))
    cfgs = {"scaffold": sc.variant(cache=cache),
            "hscaffold-full": base.variant(second_pass="full", cache=cache),
            "hscaffold-event": base.variant(second_pass="event", cache=cache)}
    for v in a.variants:
        t = Tokenizer.load(f"{a.variants_dir or a.tok_dir}/hsel_{v}_{K}.json")
        cfgs[f"hsel-{v}-full"] = t.variant(second_pass="full", cache=cache)
        cfgs[f"hsel-{v}-event"] = t.variant(second_pass="event", cache=cache)
    res, ref_out = {}, {}
    for name, tok in cfgs.items():
        mbs, out = run(tok, docs, nbytes, a.reps)
        res[name] = dict(MBps=mbs)
        ref_out[name] = out
        print(f"{name:36s} {mbs:8.3f} MB/s")
    for v in ["hscaffold"] + [f"hsel-{v}" for v in a.variants]:
        full, ev = ref_out[f"{v}-full"] if v != "hscaffold" else ref_out["hscaffold-full"], ref_out[f"{v}-event"] if v != "hscaffold" else ref_out["hscaffold-event"]
        assert full == ev, f"event-triggered output differs from full second pass for {v}"
    res["_event_equals_full"] = True
    res["_relative_to_scaffold"] = {k: v["MBps"] / res["scaffold"]["MBps"] for k, v in res.items() if isinstance(v, dict) and "MBps" in v}
    print(json.dumps(res["_relative_to_scaffold"], indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
