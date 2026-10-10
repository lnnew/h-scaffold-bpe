# Word versus future utility: fixed-K slot selection

User authorized execution on 2026-10-10: A first; B remains independent. Use only owned
wine GPUs0/1, preserve earlier evidence, and compare methods by retraining from scratch.

Question: does a token's effect on predicting unchanged subsequent text identify a more
useful vocabulary slot than its effect on predicting the changed pretoken itself?

## Frozen scoring and vocabulary selection

- Existing Scaffold pool40960 and frequency baseline32768; retain all256 bytes. EOS is extra.
- Existing pool teacher, one checkpoint; FP32 forward scoring with pool40960+EOS softmax fixed.
- Score on existing `dev_sel`, never on final test. Twelve preceding/following pretokens,
  shortened jointly when either alternative would exceed256 tokens. Nonempty suffix required.
- Before/after contain the exact same prefix IDs, suffix IDs, and raw text. Only one pretoken
  is ablated. Prefix loss invariance and total=prefix+word+future partition are checked.
- Positive delta is bits(split)-bits(keep). Store per-occurrence word/future/total bits,
  each span's raw bytes, and source document. No synthetic EOS target is added.
- Rank tokens by residual training count times span BPB improvement, with n/(n+4) shrinkage
  towards zero. Word and future have separate byte denominators. This is a local proxy;
  overlapping continuation windows mean it is not an exact corpus-wide marginal utility.
- At most32 exchanges. Qualify tokens with at least2 scored occurrences; pair an added
  and removed token with identical byte length and closest residual frequency.
  Stratify pairs by (removal decile, addition decile, byte length); match each direction's
  distribution across selectors. A frequency-top-K boundary need not permit same-decile
  addition/removal pairs. Preserve the initial failed prepare and reuse unchanged scoring.
  Quotas and disjoint exchange pairs depend only on coverage/frequency/length. Keep at
  least two possible pairs per quota. Effective budget=min(32, sum floor(pair_count/2)).
- Word/future rank exchanges inside identical quotas. Random uses the same pair universe
  and quotas. Among at most100 draws, choose by selection token-count proximity to Future,
  never by LM loss. Require <=max(1token,10% absolute delta) error. Preserve the selected draw.
- One fixed random vocabulary, model initialization seeds0/1/2. Results are conditional on
  that random draw and teacher. If word and future select identical vocabularies, stop.
- Report word/future rank correlation and score stability on separate selection-document halves.

## From-scratch training

- Frequency / Word / Future / Matched Random, all K32768+EOS, small42.51M architecture.
- Common transformer initialization and token-string-matched random input/output embeddings;
  no teacher weights are copied into the student models.
- Deterministically shuffled complete WikiText training documents, at least64MiB raw text.
- Build shared raw target windows at pretoken boundaries using all four tokenizers. Up to64
  tokens of common raw preceding context, total context<=256 under every tokenizer. Each
  target byte is scored exactly once. Only oversized pretokens are split; record their count.
- Padding/context targets ignored. EOS provides starting context only, never an artificial
  window-end target. Softmax excludes unused padded vocabulary rows.
-1500 updates,32 raw windows/update, microbatch8; data order seed0 common across methods.
  Byte-normalized objective; AdamW(.9,.95), weight decay.1, clip1, LR3e-4,
 150-step warmup then cosine to.1x. FP16 GradScaler initial1024. Any skipped update invalidates pairing.
- Record raw corpus/order hashes, target bytes/tokens and final checkpoints. Save fixed final
  checkpoint before evaluation. No test-based checkpoint selection or budget extension.

## Evaluation and decision

- Primary new confirmation:64 official PG-19 test books, selected by filename SHA256 order.
  One <=16KiB excerpt per book begins at the first line after the book's quarter position.
  No performance filtering. Preserve source URLs, source hashes, offsets, excerpt hashes.
  Exact normalized document duplicate checks against training and selection. This supports
  book-domain generalization, not a fresh in-domain Wikipedia test or contamination-free claim.
- Secondary WikiText test: previously consumed by this research, explicitly exploratory.
- Existing validation evaluated only at the fixed final step, without checkpoint selection.
- Each method uses the same raw target spans and raw context. Record document NLL and bytes.
- Primary comparisons: Future-frequency, Future-word, Future-random. Paired initialization
  seed deltas and4000 document-bootstrap replicates on seed-averaged NLL; report both.
  Three seeds do not establish selector/teacher robustness or a precise population variance.
- Proceed only when all three PG19 comparisons have CI upper<0 and all3 seed deltas<0.
  These are pilot gates, not multiplicity-adjusted publication-level significance claims.
  Otherwise stop this proxy; do not automatically add another utility or launch64K/1B.

Novelty remains unconfirmed. The contribution candidate is causal-LM loss decomposition
and fixed-budget retraining validation, not the first use of context or LM likelihood in tokenization.
