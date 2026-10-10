# Learning-aware slot exchange: development protocol

This is researcher direction A: normal-vocabulary slot allocation. Direction B remains separate;
this does not change H-Scaffold encoding-time re-merging.

For a fixed normal vocabulary V of K tokens, replace t by c without changing K. Compare
BPB after h adaptation updates of each vocabulary from a common pool checkpoint:

    U_h(t -> c | V) = BPB_dev(theta_h^V, V) - BPB_dev(theta_h^V', V')
    U_h = U_0 + (learning_progress(V') - learning_progress(V))

U_0 measures the checkpoint's immediate preference; U_h measures conditional short-adaptation
value, not from-scratch marginal utility. Merely using contextual loss is not novel: SaGe
(https://aclanthology.org/2023.eacl-main.45/) overshoots a vocabulary and prunes using SkipGram
ablation loss, periodically relearning embeddings. Our novelty candidate is prospective,
fixed-capacity deletion/replacement interventions and calibration against subsequent learning.
Novelty is unconfirmed.

## Fixed first-stage contract

- Same existing frequency-built pool, K=32768 normal tokens (including 256 bytes), plus EOS.
- Pool-normal candidates only; previously excluded intermediate scaffolds are not optimized.
- Development documents are partitioned by a frozen seed into selection and confirmation halves.
  They were used in earlier research: neither half is a newly sealed final test. Test is not loaded.
- Candidate tokens must occur at least twice in selection text. Propose 256 disjoint equal-byte-length
  exchanges, grouped randomly into sixteen groups of sixteen. These estimate group utility, not
  identifiable individual-token effects.
- UTF-8 chunks <=255 bytes are fixed before tokenization. Every arm gets the same raw chunks,
  raw training order, and byte exposures. Each sequence is [EOS] + content -> content + [EOS].
  Padding targets are ignored; output logits beyond K+EOS are excluded.
- Train 128 updates, 32 raw chunks/update, microbatch8, common seq256. Optimize summed token NLL
  divided by raw batch bytes, including EOS NLL. Constant AdamW lr0.0003, reset moments, fp16
  on wine's Turing GPUs; two data-order seeds0/1 with one common teacher (not two model seeds).
- Remap all common token vectors and EOS by pool token identities; copy the common transformer.
  Evaluate h=0,32,128. Any skipped optimizer update or unequal raw exposure invalidates pairing.
- Rank group U_128 on selection only. Accept at most four positive groups. Rank U_0 separately
  as the frozen-score ablation with the same group count. No confirmation loss enters selection.
- Evaluate the joint learned and static selections, frequency, and three random selectors.
  Random selectors exactly match changed-slot count, observed support, residual-frequency deciles
  and token byte lengths, separately for removals/additions. For each selector choose among twenty
  draws by selection-text token count only; token-count change must be within max(1 token,10%)
  of the learned selection's change. Report imperfect matching and fail the promotion gate.
- Combining groups can have interactions: joint performance is measured again, not inferred
  from the sum of group scores. Confirmation metrics are consumed only after selection is frozen.
- Gate: learning-minus-frequency, learning-minus-static U0 and learning-minus-mean(random0..2) document bootstrap upper
  95% bounds <0; both data orders improve; compression matching and pairing checks pass.
  Bootstrap uncertainty is conditional on the teacher/data orders, not teacher initialization.

34 individual probe fits plus12 joint fits =5888 updates. Short raw contexts change the evaluation
setting relative to previous flattened-stream BPB; do not compare the absolute numbers directly.
Short chunks can split words and raise EOS frequency. If promoted, verify longer-context stability
and train final fixed vocabularies from scratch across independent initialization seeds, using a
new final holdout. A failed gate stops 64K/1B expansion.

## Implementation and reproduction

`scaffold_margin.learning` provides raw chunk alignment, recursive pool demolition, remapped
warm starts and byte-normalized training. Existing tokenizer construction and LM remain unchanged.
`scripts/a_learning_utility.py` freezes input/source/checkpoint hashes, runs only owned GPU0/1,
and keeps group selection independent of confirmation. Results and document NLL are immutable
per-job outputs. Existing completed outputs are reused only if their input identity matches.

```
python research/scripts/a_learning_utility.py --stage prepare \
  --root /root/home/ubuntu/research \
  --teacher /root/home/ubuntu/research/runs/purpose_a_v1/teacher_pool40960/model.pt \
  --out /root/home/ubuntu/research/runs/learning_a_v1
python research/scripts/a_learning_utility.py --stage run \
  --out /root/home/ubuntu/research/runs/learning_a_v1
```

The operator checks GPU ownership before running. Never resume an incomplete job in place or
alter a running source snapshot. Use a fresh root on code/input changes. A go result authorizes
the next small from-scratch confirmation, not a claimed successful tokenizer or automatic1B launch.
