# One bounded Hybrid trial — 2026-10-11

Latest user contract: verify frontier completeness, measure kernel phases, and
try one Hybrid only if candidate metadata can be collected without another
sequence scan. Stop local optimization if the original-source comparison fails.
No new LM training or utility scoring. Preserve v1 (`babe2904`) and all A/B work.

## Fixed implementation before timing

The native first pass is opaque, but L2 demolition already emits an early sequence.
Collect every eligible adjacent edge during that emission, including untouched
fast-backend boundary edges. This adds per-edge lookup/metadata cost, not another
whole-sequence traversal; the cost is explicitly included in end-to-end timing.

- First-pass expanded length <128: existing full re-merge; no collector/heap.
- Length >=128: emit early tokens and candidate entries together.
- No candidates: skip second merge.
- Fewer than 4 distinct initial eligible ranks: full re-merge, using the already
  collected minimum pair for its first batch, avoiding a repeated initial scan.
- Otherwise: use the existing local heap with those precomputed candidates.
- Thresholds 128 and 4 are fixed once, not selected from new benchmark outcomes.
  Length and eligible rank diversity are proxies for the region/cost affected;
  they are not an oracle for runtime or a guarantee of speed.

## Completeness contract

At initialization and between completed reference batches, every eligible edge
must have a valid heap entry: E(S) ⊆ valid(Q). Stale extra entries are permitted.
During a rank-wide batch, new edges are deliberately deferred to match the
reference's simultaneous replacement. The intra-batch invariant is
E(S) ⊆ valid(Q) ∪ eligible_edges_adjacent_to_pending_changed_nodes.
After propagation that pending set is empty, recovering the boundary invariant.

Tests inspect the actual production function at initialization, batch boundaries,
before propagation, and termination; a deliberately missing seed must fail.
Rank batch logs/outputs are also compared with an independent reference. This
is executable invariant checking plus an induction argument, not a mechanized
proof covering all possible inputs. Fused seeds must include untouched edges.

## Measurement and stopping rule

- Use frozen original `0113d8e` code and identical 32K/64K tokenizers.
- Repeat Wiki dev and PG19 timing 7 times with rotated method order.
- Explicit short-input workload: one document per pretoken from the fixed Wiki
  corpus. Preserve these input boundaries; this is not a whole-text throughput
  claim. Use 1024 fixed short samples, chosen before timing, per vocabulary.
- Backends: team Python and tiktoken; canonical margin was already evaluated in
  v1 and is not changed in this trial.
- Compare frozen full / previous local / Hybrid. Verify exact IDs, lossless decode,
  valid output IDs, EOS and output hashes on every timing round.
- Real-vocabulary random/L2 stress audit includes Hybrid and new full fallback.
- Report original full scan, initial candidate generation, queue/validation,
  propagation, merge application and state setup/output costs. Diagnostic copies
  of actual functions are instrumented; timer overhead makes tiny queue timings
  descriptive, not uninstrumented throughput or causal proof.
- Go only for reproducible positive paired throughput in both sizes/domains,
  with short-input non-regression. Require at least 5% median improvement and
  min paired ratio >1 on both long domains at both sizes; short median >=1 and
  short min >=0.98. These practical guards avoid declaring noise a success.
- If this gate fails, stop local tuning and record the negative result. Do not
  change thresholds, add another dispatch rule, retrain an LM, or design utility.
