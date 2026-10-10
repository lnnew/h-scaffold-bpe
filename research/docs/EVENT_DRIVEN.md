# Event-Driven H-Scaffold-BPE protocol (2026-10-10)

User authorized a direction change from A (vocabulary-slot utility) to C (exact
encoding acceleration). Future A v4 and its negative result remain immutable;
B stays separate. This study uses CPU only, with no LM retraining or GPU takeover.

## Question and contract

Can sparse second-pass candidate propagation preserve the existing H-Scaffold
token IDs while reducing encoding time? Preserve vocabulary, L0/L1/L2 levels,
demolition choices, rank-wide left-to-right batches, special tokens and EOS.
Keep the default team API on `full`; research defaults remain `event` (existing
no-L2 skip). Add explicit `local` for experiments, with `skip` as an ablation.

The canonical per-pretoken first pass is saturated: no unchanged edge is a
merge candidate. Seed only edges inside or adjacent to an L2 decomposition;
after a merge, add the two changed neighboring edges. All candidates of the
current minimum rank are processed left-to-right before exposing new ranks.
An induction on these batches establishes equivalence: omitted initial edges
are ineligible; only merges can change their eligibility; all changed edges
are then inserted. Stale heap entries are checked against current adjacency.

The team's tiktoken path concatenates pretokens before second-pass merging.
Unchanged cross-pretoken edges can be eligible even without L2. Preserve this
legacy behavior: `skip` checks unchanged candidates before skipping; `local`
seeds **all** eligible initial edges, then propagates locally. The dirty-frontier
proof applies only to saturated input, not to this whole-document input.

## Fixed comparison

- Original source: commit `0113d8e`, frozen copies in the result `baseline/`.
  Do not time a refactored `full` as the only control.
- Original team tokenizers: Scaffold and Hierarchical at 32768 and 65536.
- Domains: all 30 WikiText development-selection documents, and 64 pre-existing
  PG19 excerpts. This is a runtime benchmark, not tuning or LM evaluation.
- Backends: canonical research Python, team Python, team tiktoken. Compare
  modes **within** each backend; cross-backend equality is not assumed.
- Seven rounds; deterministic rotated method order; one process; no encoding
  cache; identical warm demolition caches; exclude construction and input I/O.
- Report median decimal MB/s, paired speedup median and range. Rounds measure
  timing stability on this host, not generalization across machines.
- Assert exact H token IDs and lossless decoding on every document. Hash output
  IDs per document with explicit lengths and little-endian uint32 values; every
  timing round must reproduce its method's hash. Record input/model/source hashes.
- Differential audit: 2500 arbitrary heap cases in tests; 1000 fixed random texts
  per model/backend plus L2-token stress, Unicode, special tokens and long input.
- Count fresh L2-hit pretokens, actual merge occurrences and changed outputs;
  do not confuse these denominators or reuse the report's sample counts.

## Gates declared before the full benchmark

1. Correctness: zero ID mismatches, valid L0 outputs, lossless decoding. Any
   mismatch blocks an acceleration claim until fixed and rechecked.
2. Practical speed: a local version is promising if the 64K tiktoken end-to-end
   median gain is at least 10% on both domains, with no regression at 32K. Smaller
   or domain-specific gains remain reportable but do not justify a strong claim.
3. Local vs skip: compare them directly. If skip accounts for the gain and local
   adds overhead, recommend skip and report the local optimization as negative.
4. 5 MB/s at 64K is an aspirational target on the **new recorded host**, not an
   expected result or a comparison with the old 4-B200 report's absolute rates.
5. Novelty: an event heap alone is not new. Assess the scaffold-demolition frontier
   and exact batch/boundary contract against prior fast BPE implementations.

## Quality claim boundary

Identical IDs imply identical deterministic loss/BPB for a fixed valid model and
evaluation protocol, with no retraining required. This does not establish that
H-Scaffold improves LM quality. The historical 2.513/2.530 BPB cannot be reused:
the existing audit identified a label-double-shift bug and unverified run lineage.
No valid original H checkpoint has been identified for a numerical BPB check.

## Closest verified work

[Faster Superword Tokenization](https://arxiv.org/html/2604.05192v2), section 5,
also accelerates inference by restricting rule search to applicable pretokens
and using a whole-pretoken vocabulary shortcut. Its main two-phase training
construction preserves BoundlessBPE/SuperBPE models; standard-format superword
exports are approximate. Our claim is narrower: preserving a scaffold-demolition
second pass, including rank batches and the existing backend boundary behavior.
The prior work already supports the general principle of avoiding global scans;
we do not claim to introduce that principle or prove publication-level novelty.
