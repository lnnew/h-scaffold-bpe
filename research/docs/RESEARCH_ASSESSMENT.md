# Research assessment: Scaffold-BPE extensions A / B / C

**Status of this document.** Updated after reading the team's report (`hscaffold_bpe_report_v2.pdf`; numbers transcribed in
`docs/PRIOR_REPORT_NUMBERS.md`) and after auditing the team's code (`yerinNam/h-scaffold-bpe`; findings in §1.4 —
**the reported LM results were produced by a training bug**). No new LM experiment has been run. Everything below is (i) an audit of the
numbers given in the research brief, (ii) a literature check, (iii) a design, and (iv) a protocol with explicit
go/no-go gates. Where I give a judgement it is a *prior to be updated by the gates*, not a finding.

Tags used throughout
* **[OBS]** established observation — taken from the brief's H-Scaffold report; single seed, not reproduced here.
* **[HYP]** researcher hypothesis (A and B are the researcher's own; C is the assistant's).
* **[NEW]** newly proposed mechanism in this repo.
* **[CHK]** a check or calculation done while preparing this repo (assumptions stated).

---------------------------------------------------------------------------------------------------

## 0. Bottom line

| | A: marginal-utility vocabulary | B: multilingual allocation | C: selective re-merging (optional) |
|---|---|---|---|
| Core question | Is frequency a good proxy for slot value? | Is per-language parity optimal for LM quality? | Are all re-merges equally useful? |
| Where the idea acts | vocabulary construction (which tokens get slots) | corpus mixture / slot allocation across languages | encoding-time segmentation |
| Expected effect size | likely small (see §3) — *measured by a cheap gate before any LM* | large and easy to measure in tokenizer metrics; LM effect unknown | **tiny by construction** (§1.3) |
| Closest prior work | VOLT, Unigram-LM pruning, PickyBPE, Zouhar et al. | Petrov et al., Ali et al., XLM-V, VoCAP, Arnett et al. | PickyBPE (training-time re-use of freed pieces), Unigram segmentation |
| Novelty | moderate | low-to-moderate (crowded) | low as a method; the audit is the useful part |
| Cost to falsify | hours of CPU/1 GPU (tokenizer-only gate) | days (needs a multilingual LM at ≥2 allocations × seeds) | hours (audit) — LM-level test is probably unmeasurable |
| Explainable in 5 min | yes | yes | yes |

**Recommendation (prior, pending the gates).** Of the researcher's two questions, **A is the better first
project for this hardware and for a course-scale deliverable**: it is the direct extension of Scaffold-BPE
(a vocabulary-construction change), it has a decisive and cheap *tokenizer-only* falsification gate
(`scripts/a_marginal_utility.py`), and it cannot be confounded by corpus composition. **B has the larger and more
detectable effects and the more interesting science, but** it is the most crowded area, Scaffold-BPE is incidental
to it (it becomes a corpus-mixture study), and the interesting half — *indirect* cross-lingual transfer — is hard
to isolate at the model size 2× RTX 8000 can train. The honest plan is: run A's gate and B's tokenizer-only
sweep first (both cheap), then spend LM compute on whichever clears its gate. Neither is guaranteed to.
**C should stay a separate, optional comparison**; I do not recommend it as the main contribution, and I
deliberately did not let its code reuse influence this ranking (§1.3 explains why the data argue against it).

---------------------------------------------------------------------------------------------------

## 1. Problem, background, and what the existing evidence shows

### 1.1 BPE → Scaffold-BPE → H-Scaffold-BPE
* BPE keeps every intermediate merge even when its residual frequency collapses (`Ari`+`zona`→`Arizona`: `zona`
  is rarely seen alone afterwards). [Lian et al. 2025, as summarised in the brief.]
* Scaffold-BPE marks a token as scaffold when its residual frequency r(t) falls below the frequency q of the head of
  the merge queue; scaffolds stay in the merge structure (expanded vocabulary) but are demolished into normal tokens
  at encoding, freeing slots that later merges fill. Implemented in `bpe.py`.
* H-Scaffold-BPE (team's extension) adds levels L0/L1/L2 (L2: r < τq, τ=0.5), demolishes L2 first, runs a second
  merge pass, then demolishes L1. Same final normal vocabulary; only encoding differs. Implemented in `bpe.py`
  (`mode="hscaffold"`), with an exactness-tested event-triggered variant.
* **Fidelity [CHK] — now verified against the team's code.** Official Scaffold-BPE code was never released; the team's trainer
  (`hscaffold_bpe.trainer`) is an independent implementation (report slide 16). It keeps scaffold tokens in the merge queue
  and reactivates one when the queue head falls to its frequency, identifies tokens by their byte string, and makes L2 sticky.
  This repo's default `scaffold_rule="global"` (re-judge all tokens after each merge) reproduces **identical merges, L0 sets and L1/L2
  levels** on synthetic counts for K ∈ {800, 1500, 2500} × τ ∈ {0.25, 0.5, 0.75} (`tests/test_adapter.py`; real-corpus equivalence not
  tested — no corpus here). The literal slide-2 reading (`"component"`: judge only a, b; revive only when judged again) does **not**
  match the team's trainer and is kept only for contrast — an earlier version of this repo wrongly made it the default.
  On the team's six committed tokenizers this repo's *encoder* gives the same segmentation as the team's python backend (§1.4).
  Still my assumption: the stop rule when revivals overshoot K (weakest demoted), irrelevant for the team's artifacts.

### 1.2 What the reported experiments do and do not show
[OBS] (brief §4 and report slides 8–14; single seed; llama-3.2-1b shape; BF16, 8-bit AdamW, 4×B200; WikiText-103 evaluation):

| | 32K | 64K |
|---|---|---|
| Scaffold-BPE | 2.635 | 2.530 |
| H-Scaffold-BPE | 2.659 (+0.92%) | 2.513 (−0.69%) |

Fertility differences < 0.04%; second-merge events 51 (32K) / 24 (64K) at τ=0.5, measured on the WikiText-103 test set (slides 14–15);
encoding 35–46% slower. **Test PPL is 3.7–4.1K for all six models** (slide 10), vs 3.78K / 3.93K / 4.08K for BPE / Scaffold / H at 32K.

### 1.3 Audit of the motivation  **[CHK] — this is the most important section**
**(a) The re-merge events are too rare to produce the reported BPB gap.** Take the brief's numbers at face value:
event rate ρ = 51/247K ≈ 2.1×10⁻⁴ (32K) and 24/247K ≈ 1.0×10⁻⁴ (64K) per pretoken; ΔBPB = +0.024 (32K), −0.017 (64K).
With ≈4.5 bytes per pretoken (assumption; WikiText-103 raw is ≈5 bytes per whitespace word), the BPB shift
corresponds to ≈0.11 bits (32K) and ≈0.077 bits (64K) per pretoken *averaged over all pretokens*, i.e.
**≈500 bits (32K) and ≈800 bits (64K) per re-merge event**. A single event changes the segmentation of one
pretoken (a handful of tokens); its likelihood effect is plausibly of order ≤10 bits. The reported ΔBPB is therefore
~50–100× larger than a direct effect of the re-merges can explain. This holds whether the 247K counts types or
occurrences (re-merges fire on rare words, so the occurrence-weighted rate is, if anything, lower). Note the "247K" denominator is ambiguous in
the slides (64K·0.5 also lists "L2 pretokens = 247"); even if the true sample were 100× smaller the required effect would still be ~5× too large.
Unless the event rate is mis-measured by ≥1–2 orders of magnitude, **the 32K/64K reversal is far more likely
seed/optimisation noise or an unmatched pipeline difference than a causal effect of re-merging.**
`scripts/phase0_audit.py` measures the true attempted/accepted/changed rates, and `scripts/c_label_rules.py`
prints a *predicted* ΔBPB(H − Scaffold) under a teacher for direct comparison with the reported one.

**(b) The LMs in the report did not learn next-token prediction — root cause found (code audit, §1.4).** Slide 10 gives test PPL ≈ 3.7–4.1K,
consistent with the BPB column (ln 3,784.75/ln 2 = 11.9 bits/token ÷ 4.499 bytes/token = 2.64 ≈ 2.648), so the BPB arithmetic is fine. The
cause is in `lm.py`: it calls `model(input_ids=x[:, :-1], labels=x[:, 1:])`, and HuggingFace shifts `labels` internally, so the models were
trained to predict the token **two** steps ahead while `evaluate()` scored aligned next-token loss. Evidence: (i) transformers 5.19 — the HF
loss equals the two-ahead CE to 6 digits; (ii) a controlled Markov demo with known entropies: buggy training gives NLL 5.62 nats, *worse than the
unigram entropy 4.68*, fixed training 2.17 vs the true 2.07 (`scripts/demo_label_shift.py` in the team repo); (iii) the team's val/test NLL
(8.17–8.31 nats, all six models) shows the same "worse than unigram" signature. Not verified: the team's training logs (not in the repo), the
the transformers version actually used for the runs (the project allows `>=4.56,<5`; the shift was verified with 4.56.2 and 5.19), and a GPU rerun. **Consequence: the six LM results — including the
32K/64K reversal — carry no information about tokenizer quality.** They must be redone with the fixed loss.

**(c) Seed variance is unmeasured.** In a 4-run toy suite (synthetic data, tiny model) the across-seed std of BPB
(0.024) was ~2× the Scaffold-vs-H gap (0.012), while the document-bootstrap CI excluded zero. This is exactly why
`scripts/stats.py` demands both seed-level and document-level evidence. The brief's 0.7–0.9% gaps need ≥3 seeds.

**(d) Protocol details — now confirmed from the team's README/scripts.** Tokenizer corpus: Wikipedia EN, 1B whitespace words. LM text:
WikiText-103 train, 5 epochs, micro-batch 16×1024, grad-accum 1 (≈16K tokens/step), lr 3e-4, 8-bit AdamW (`bitsandbytes`), BF16,
flash-attn kernel, 4×B200; ≈34–36K steps per run. Events were counted on the WikiText-103 **test** set. A Wikipedia-EN tokenizer corpus very
likely contains the WikiText-103 test articles (affects merges only, identically for all methods — state it); never use test text to *choose*
anything. Still unknown: whether init/data-order seeds were identical across methods (the code's default `--seed 42` suggests yes), and whether
"2nd merge" counts occurrences or unique pretokens.

**(f) The report's own limitations agree with this audit** (slide 16): single seed (min. 3 needed), independent Scaffold implementation, only τ=0.5 evaluated
(ablation tokenizer-only), PPL biased across tokenizers. Slide 12–13 narratives ("H-Scaffold best at 64K", "more effective at larger vocabularies") rest on gaps of
0.009–0.017 BPB from one run each; the vocabulary-sensitivity deltas (−0.126 / −0.105 / −0.146) differ by ≤0.04 — also within plausible seed noise.

**(e) Consequence.** (1) There is currently **no valid LM evidence either way** on H-Scaffold vs Scaffold: the only data are from a mis-trained model.
(2) The tokenizer-side observations (fertility, bytes/token, speed, L2 ablation counts, L1+L2 constant across τ) are unaffected and were reproduced
from the team's artifacts. (3) The event-rate argument (a) still says a direct BPB effect of re-merging should be tiny, so after the fix the *expected*
outcome is "no detectable difference", which is itself worth establishing with ≥3 seeds. (4) C's premise ("not all re-merges are beneficial") remains
untested; A and B never relied on the H-Scaffold results (instruction 2). The first experiment of the project is therefore **the corrected rerun
of Scaffold vs H-Scaffold at 32K/64K with seeds**, which doubles as Gate G0.

### 1.4 Audit of the team's repository (`yerinNam/h-scaffold-bpe`) **[CHK]**
Read-only audit; full notes and the fix are in that repo's `docs/AUDIT.md` (a patch is in `patches/` here because this session could not push there).
* **Critical — label double shift in `lm.py`** (see 1.3b). Fix: explicit aligned loss `next_token_loss()` used by training and evaluation + regression tests.
* **Verified identical:** Scaffold and Hierarchical artifacts have the same token table, merges and visible set; they differ only in levels
  (|S| = 2,601 / 4,889 = report). The ablation artifacts (t025/t075) reproduce the report's L2/L1 counts exactly.
* **Verified:** this repo's encoder == the team's python backend on all six committed tokenizers (BPE/Scaffold/H × 32K/64K) for prose, multilingual text and 300
  random strings; this repo's default trainer == the team's trainer on synthetic counts (above).
* **Low severity — fast path ≠ reference for H-Scaffold:** the tiktoken path runs the second merge pass on the whole text's token sequence, so a re-merge can
  straddle a pretoken boundary (e.g. `' '`+`'\n'`); 10/300 and 5/300 random strings (32K/64K) differed, all on whitespace/newline boundaries; 0 mismatches on natural
  prose. Effect on WikiText-style text is expected to be negligible but **not measured** (`scripts/compare_team_backend.py` measures it on real text).
* **Not bugs, worth knowing:** BPB uses an *estimated* byte count (`source_bytes × evaluated/total tokens`); data are read sequentially without shuffling;
  the LM stack (bf16, flash-attn kernels, bitsandbytes) targets B200s and does not run as-is on Turing GPUs — use this repo's `train_lm.py` there.

---------------------------------------------------------------------------------------------------

## 2. Related work (verified this session unless marked)

Verified via search results this session (abstract/venue level; full texts were not accessible from the sandbox):
* **Scaffold-BPE** — Lian et al., AAAI 2025, arXiv 2404.17808, doi 10.1609/AAAI.V39I23.34633. Dynamic scaffold removal; expanded vocabulary; scaffolds used as temporary intermediates at encoding.
* **BPE Gets Picky** (PickyBPE) — Chizhov, Arnett, Korotkova, Yamshchikov, EMNLP 2024 main, arXiv 2409.04599. Deletes merges that leave "junk" intermediate tokens *during training*; reports better vocabulary efficiency, fewer under-trained tokens, no compression loss, MT results flat-or-better. (The IoS criterion is from a third-party summary — check the paper.)
* **SuperBPE** — Liu, Hayase, Hofmann, Oh, Smith, Choi, arXiv 2503.13423. Two-stage pretokenization curriculum (subwords, then superwords across whitespace); up to 33% fewer tokens, +4.0 avg points on 30 tasks at 8B.
* **Tokenization Is More Than Compression** — Schmidt et al., EMNLP 2024 main, arXiv 2402.18376. PathPiece (minimal-token segmentation); 64 LMs 350M–2.4B; fewer tokens does not reliably improve downstream results; pre-tokenization and BPE initialisation matter.
* **Tokenizer Choice for LLM Training: Negligible or Crucial?** — Ali et al., Findings NAACL 2024, arXiv 2310.08754. Fertility/parity not always predictive of downstream; multilingual tokenizers needed ≈3× vocabulary of English-only for 5 European languages.
* **VOLT** — Xu et al., ACL 2021, arXiv 2012.15671. "Marginal utility of vocabularization" (negative derivative of corpus entropy w.r.t. size) via optimal transport; NMT.
* **Tokenization and the Noiseless Channel** — Zouhar et al., ACL 2023, arXiv 2306.16842. Rényi efficiency (α=2.5) correlates with BLEU (≈0.78–0.82; arXiv vs ACL pages differ) far better than compressed length.
* **Language Model Tokenizers Introduce Unfairness Between Languages** — Petrov et al., NeurIPS 2023, arXiv 2305.15425 (token "premium").
* **XLM-V** — Liang et al., EMNLP 2023, arXiv 2301.10472 (de-emphasise sharing; per-cluster capacity via ALP); **VoCAP** — Zheng et al., arXiv 2109.07306 (ALP, per-language capacity).
* **Explaining and Mitigating Crosslingual Tokenizer Inequities** — Arnett et al., NeurIPS 2025, arXiv 2510.21909 (per-language vocabulary sizes to equalise compression; ~7,000 tokenizers); **MUTANT**, arXiv 2511.03237 (script-aware allocation; seen only as a search snippet).
* Not re-verified this session (cited from memory): Kudo 2018 (Unigram LM / subword regularization, ACL 2018).

### Required novelty check
| Property | Scaffold-BPE | PickyBPE | Unigram LM | This repo |
|---|---|---|---|---|
| Removes intermediate BPE tokens | yes | yes (training time) | prunes by likelihood | A, C reuse scaffold |
| Reconsiders segmentation after removal | no (demolish only) | pieces return to the corpus and can be re-merged during training* | Viterbi over remaining pieces | C: at **encoding** time, keeping the merge table |
| Model-based utility for the choice | no (frequency) | no (IoS) | unigram likelihood | A/C: neural-teacher marginal bits |
| Final model vocabulary unchanged by the re-merge step | n/a | no | n/a | C yes; A changes *which* tokens, not size |

\*from my understanding of the method; **not verified from the paper's full text**. Honest reading: a PickyBPE-style
"re-merge after removal" exists in spirit at training time, so C's novelty is only the encoding-time, vocabulary-
preserving variant; A's novelty is using a *neural* marginal-likelihood criterion inside a Scaffold-style pool, which
is closest to VOLT/Unigram-style likelihood pruning, not to anything in the BPE-refinement line. I did not find a
paper doing exactly A or C, but my search was not exhaustive; **do not claim novelty without a longer search**.

---------------------------------------------------------------------------------------------------

## 3. Direction A — Marginal-utility Scaffold-BPE (researcher)

**Hypothesis [HYP].** Scaffold-BPE refills freed slots with the next most *frequent* merges. Frequency measures how
often a token occurs, not how many bits it saves. Choosing replacement tokens by estimated marginal LM utility
may give a better vocabulary at equal size.

**Mechanism [NEW] (`direction_a.py`, `scripts/a_marginal_utility.py`).**
1. Run Scaffold-BPE to K_pool = K+P normal tokens (default P = 0.25K); keep the expanded merge table.
2. Choose which K pool tokens stay normal; all others become scaffold (kept for merging, demolished at encoding —
   any subset is valid because demolition bottoms out in bytes).
3. Value of token t: 
   V(t) = Σ_{occurrences o of t} [NLL_teacher(o, t demolished into pieces) − NLL_teacher(o, t kept)], estimated as
   r(t)·mean-bits(t) with empirical-Bayes shrinkage for tokens with few held-out occurrences. NLL is the sum over the
   window, so both compression and prediction difficulty enter; windows cover identical bytes.
4. Keep the K−256 highest-V non-byte tokens. Controls at the same pool: frequency-top-K (r), compression-top-K
   (r·(pieces−1)), random-K. Plus plain Scaffold-BPE at K.

**Why it might help / hurt.** Frequency ≡ V when mean bits per occurrence is constant. It is not: a token that
splits a hard-to-predict continuation (rare proper noun pieces) can save fewer bits than its count suggests, and a
token that merges two highly predictable pieces saves almost nothing. Against: the margin tokens have r≈q, so each
carries ≈q occurrences out of ~10⁹ — swapping many of them changes few corpus tokens; the teacher is trained on one
segmentation and can prefer it; first-order additivity ignores interactions; and Schmidt et al. / Zouhar et al.
warn that simple intrinsic proxies mispredict downstream quality, which cuts against *any* tokenizer-side utility,
including this one.

**Minimal falsifiable experiment.**
* *Gate (tokenizer-only, hours):* with a teacher trained on the pool vocabulary, report Spearman(r, V), Jaccard of
  the freq- vs LM-selected sets, and the *held-out* predicted ΔBPB of LM-selected vs freq-selected (labels from even
  dev documents, evaluation on odd ones). Proceed only if Jaccard < ~0.9 **and** |predicted ΔBPB| ≳ measured MDE/3 (§8).
* *LM test:* {Scaffold@K, pool-freq, pool-comp, pool-LM, pool-random×3} × 3 seeds at one K (then the other).
  Supported only if pool-LM beats pool-freq **and** pool-random under `stats.py`'s two-level criterion.
* *Null/negative meaning:* identical sets → frequency already is the utility at this scale; sets differ but BPB
  does not → teacher-proxy utility does not transfer to retraining (itself a useful result, cf. Schmidt et al.).

**Feasibility.** High. CPU for pool training; teacher scoring is forward-only (≈ windows × 2 × ctx tokens);
LM runs: 7 methods × 3 seeds = 21 per K with the small preset.

---------------------------------------------------------------------------------------------------

## 4. Direction B — Multilingual utility-aware allocation (researcher)

**Hypothesis [HYP].** Parity is not the objective: giving high-resource languages more capacity can cost
low-resource languages tokenization efficiency yet improve *learning* (and hence transfer), so mean multilingual
quality may peak away from parity. Do not assume parity is optimal.

**Mechanism [NEW] (kept minimal).** Allocation exponent α: tokenizer-training share p_l ∝ n_l^α
(α=1 natural/high-resource-heavy, α=0 uniform ≈ closest to parity). Implemented by re-weighting per-language word
counts (`b_build_tokenizers.py`), Scaffold-BPE vs BPE at each α. **The LM training mixture is held fixed**, so
tokenizer allocation is the only varying factor. No new algorithm, per the instruction not to bolt B onto A or C.

**Measured, separately.** *Direct*: per-language bytes/token (parity: CV, max/min, Gini). *Total*: per-language BPB,
equal-weight mean (every language matters) and byte-weighted mean. *Indirect*: residual of
ΔBPB_l = slope·Δlog(bytes/token)_l + residual_l, slope fitted across (α, language) pairs (`b_analyze.py`).
This split is exploratory (few points, linear), and "residual" lumps transfer with capacity and embedding effects —
I do not claim it isolates transfer.

**Minimal falsifiable experiment.** 4–6 languages spanning resource levels and scripts, α ∈ {0, 0.5, 1}, 3 seeds,
one K. "Parity is optimal" is supported iff equal-weight mean BPB is minimal at α=0 and increases monotonically with α
beyond seed noise. If an interior α wins, parity is not optimal *in this setup*.

**Caveats that limit the claim.** At ≤~100M-parameter models, low-resource languages are data-limited, so
transfer conclusions may not extrapolate; at fixed epochs, different α yields different token counts per language
(compute is not matched — report both fixed-epoch and fixed-token views); BPB across languages is comparable only
on identical-meaning text (Wikipedia is not parallel; FLORES-style parallel eval sets are a better choice for the
final comparison and `prepare_multilingual.py` should be pointed at one if available).
Prior work is extensive (Petrov, Ali, XLM-V, VoCAP, Arnett, MUTANT); the contribution would be the controlled
*allocation → LM quality* curve with a decomposition, not the idea of reallocation.

**Feasibility.** Medium. Tokenizer-only sweep is cheap. LM stage: 3 α × 3 seeds = 9 runs per kind, each on a
multilingual mix; data volume and per-language held-out sets are the main practical obstacle.

---------------------------------------------------------------------------------------------------

## 5. Direction C — Selective re-merging (assistant-suggested; optional)

**Status.** Alternative/comparison only. **[HYP, assistant's]** not all re-merges help; utility depends on the
alternative segmentation, not on residual frequency alone.

**Mechanism [NEW] (`remerge.py`, `scripts/c_label_rules.py`).** Keep L0/L1/L2, expanded vocabulary, L2 demolition,
and final L1 demolition. Replace "re-merge every eligible pair" by an **allow-list of re-merge rules** (a,b)→c;
the pass-2 loop only applies allowed rules. Rule-level allow-list (not per-occurrence) keeps the tokenizer a pure
function with a lookup table and no runtime model. Utilities (each a separate ablation):
* U_freq(c) = log(1 + r(c)); * U_comp(c) = α·(−Δtokens per event) + β·log(1 + r(c));
* U_LM(rule) = mean over dev occurrences of (NLL(Scaffold segmentation) − NLL(rule enabled alone)) in bits,
  window-local, teacher-dependent; a **cheap deployable selector** is a ridge model on features
  {log occurrences, log r(c), collapse ratio, merge rank, length, Δtokens, boundary-only fraction, capitalised
  fraction, L2 ratio} trained on U_LM labels with out-of-fold evaluation. Oracle (dev-label top-m) is reported and
  labelled non-deployable.
* Controls: random matched in **#rules** and random matched in **traffic** (sum of rule occurrences) — rule-count
  matching alone is confounded because frequency-selected rules change far more tokens.
* Rules that only fire after another re-merge (`chain_only`) have no isolated effect and are excluded from
  selection; their share is reported.

**Efficiency (H3).** After full BPE no adjacent pair is mergeable, so after L2 demolition only pairs touching the
demolished span can merge: the second pass can be **event-triggered** (run only if an L2 token was demolished) with
*exactly identical output* (tested). `bench_encode.py` separates full pass / event-triggered / selective. Rejecting
rules does not by itself speed anything up; the trigger does.

**Minimal falsifiable experiment.** Phase 0 (reproduction + audit, §8) → Phase 1 labels and predictability →
LM test of {Scaffold, H, random-matched, utility-selected} × 3 seeds × {32K, 64K} only if the gate passes.

**Evidence from the report (slide 15) [OBS].** L2 tokens are mostly rare proper-noun prefixes and odd fragments (▁descend, perors, ▁charg, thor, inness);
re-merges produce tokens such as `uming`, `▁Verd`, `Hus`, `ruct`, and 48/51 cross an original boundary. These are rare words, so each event touches
text that is rare in training *and* test — consistent with a negligible BPB footprint.

**Honest expectation.** Given §1.3, the LM-level effect of any re-merge selection is likely unmeasurable at this
scale (ρ≈10⁻⁴ of pretokens). The defensible contributions from C are (1) the audit, (2) the exact event-triggered
speed-up, (3) a possibly-negative "utility is not predictable / events too rare" result. A carefully verified
negative result is an acceptable outcome per the brief.

---------------------------------------------------------------------------------------------------

## 6. Experimental protocol (shared)

* **Splits** (`prepare_text.py`): `lm_train`; `tok_train` (= `lm_train` by default; a Wikipedia dump contains the
  WikiText-103 test articles — if used, state it; it affects only tokenizer merges, equally for all methods);
  `dev_sel` (selection/labels **only**); `val` (monitoring/hyper-parameters); `test` (final report **only**).
* **Metric.** Test BPB = Σ NLL_nats / ln 2 / Σ target-token bytes; EOS counts 0 bytes; exact byte conservation tested.
  Per-token NLL is aggregated per document (`eval_<split>.npz`) so paired bootstrap and per-language BPB need no re-run.
* **Seeds.** Model-init seed and data-order seed are separate flags; paired design = same seeds across methods.
  ≥3 seeds for every principal comparison; 5 for the Phase-0 noise estimate on the small preset.
* **Statistics** (`stats.py`): paired difference over seeds (mean, std, paired t, Wilcoxon if n≥5), document-level paired
  bootstrap on seed-averaged losses, minimum detectable effect (80% power, α=.05) from the observed paired std, and a
  verdict that requires **both** levels to clear zero with the same sign.
* **Compute match.** Default: same epochs over the same raw text (equal raw-data exposure, unequal token counts).
  Report the fixed-step alternative for B.
* **Baselines/ablations.** C: Scaffold, H (all), random-#rules, random-traffic, freq, comp, utility-predicted, oracle.
  A: Scaffold@K, pool-freq, pool-comp, pool-LM, pool-random. B: α ∈ {0, 0.5, 1} × {BPE, Scaffold}.
* **Teacher reliability.** Re-score labels with ≥2 teachers (bigram floor; neural trained on Scaffold stream; neural trained
  on H stream). Report cross-teacher label Spearman; if signs flip with the teacher, the "utility" is teacher bias.

## 7. Implementation plan
Unchanged from Scaffold-BPE: BPE training loop, scaffold detection (q = queue head), expanded vocabulary, demolition.
Added: levels stored explicitly (`Tokenizer.levels`), allow-list in the second pass (C), `encode_ablated` (A),
event extraction/features (C), pool selection (A), reweighted counts (B), LM trainer and BPB evaluation (shared).
Open engineering items: BPE training in pure Python — measure RAM/time at the real corpus size before committing to a
1B-token tokenizer corpus (use `--max_docs`); encoding is pure Python (compare *ratios* with earlier Rust-level
numbers, not absolute MB/s).

## 8. Feasibility, cost and go/no-go
Costs are relative and assumption-based; do not read them as measurements. `scripts/budget.py` computes
FLOPs/token = 6(N_nonemb + V·d) + 12·L·d·T and wall-clock = FLOPs/(GPUs·peak·MFU) from **user-supplied** peak and MFU;
`scripts/autotune.py` replaces MFU by measured tokens/s on the target machine. Run counts, not hours:
C-main 4 methods×3 seeds×2 K = 24 runs (+ Phase-0 noise runs); A 7×3 = 21 per K; B 3 α × 3 seeds × (1–2 kinds).
The earlier 1B-parameter × 5-epoch protocol is almost certainly infeasible ×24 on 2 RTX 8000: use `small`/`base`
presets and say so in the write-up.

| Gate | Proceed to LM training if… | Stop / change direction if… |
|---|---|---|
| G0 (all) | `phase0_audit` checks all true and |S| in the report's range; `sanity_baselines`: LM BPB clearly below the bigram floor and PPL orders below the report's ≈3.7K; 5-seed noise estimated | any check fails (fix first); LM ≈ unigram floor / PPL ≈ report's (training broken — the report's models were in this state) |
| G-C | predicted \|ΔBPB\| of H vs Scaffold (teacher) and of selected vs random ≥ MDE/3; label-predictability CV Spearman > ~0.2 with CI excluding 0 | events too rare (predicted effect ≪ MDE) → report "unmeasurable at LM level", publish audit + speed-up only |
| G-A | Jaccard(freq, LM) < ~0.9 and held-out predicted \|ΔBPB\| ≥ MDE/3; teacher labels agree across ≥2 teachers (Spearman > 0.5) | sets ≈ identical, or teacher-dependent signs |
| G-B | tokenizer sweep shows meaningful parity change across α (e.g., max/min bytes-per-token changes ≥10%) | negligible change across α |
The MDE/3 rule is a heuristic (a predicted effect much below the detectable one cannot be confirmed at this budget);
revise it once the real seed std is known.

## 9. Five-minute presentation outline
1. *Scaffold-BPE in one slide:* intermediate tokens waste slots; scaffolds keep structure, free slots (`Ari`+`zona`).
2. *What we found (and didn't):* H-Scaffold reversal at 32K/64K is single-seed; re-merge events ~10⁻⁴ of pretokens;
   required per-event effect ~500–800 bits ⇒ the reversal is probably not causal. (Say this plainly.)
3. *Two researcher questions, kept separate:* A — frequency ≠ value of a slot; B — parity ≠ optimum.
4. *A in one picture:* same pool, different selection (frequency vs marginal bits); gate = do the sets differ, and by how much BPB?
5. *B in one picture:* α sweep: parity metrics vs LM BPB per language; falsified if α=0 wins monotonically.
6. *Optional C:* exact event-triggered re-merge (speed) + audit; selective re-merge only if the gate passes.
7. *What would change our mind:* gates in §8; a verified null is a result.
