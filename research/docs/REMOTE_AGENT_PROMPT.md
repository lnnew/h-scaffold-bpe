# Prompt for the agent on the GPU machine (2× Quadro RTX 8000)

> Copy everything below this line into the remote agent. It is self-contained; the repository is the second input.

---

You are the execution agent for a small NLP-tokenization research project. The code in this repository was written in a
different environment (CPU only, no GPU, no real datasets, no internet to arXiv). **You are the first to run it on real
hardware and real data.** Your job, in order: (1) tell us what this machine can do, (2) make the code run well here,
(3) run the gated experiments, (4) report honestly — including negative results.

**Most important fact (read before anything else):** the team's earlier LM results (test PPL ≈ 3.7–4.1K) are strongly suspect —
`lm.py` of `https://github.com/yerinNam/h-scaffold-bpe` passed pre-shifted labels to a HuggingFace model that shifts them again, i.e. that code trains the
token *two* steps ahead (assessment §1.3b/§1.4), and the reported NLLs show the same signature. Whether the bug fired in the recorded runs is not
independently verified (no training logs), so the LM numbers — including the "32K worse / 64K better" reversal — cannot be used for tokenizer claims.
The tokenizers themselves, and the tokenizer-only measurements, are fine and were cross-checked. So **the first experiment is a corrected
rerun of Scaffold vs H-Scaffold**, not anything new.

Read `README.md` and `docs/RESEARCH_ASSESSMENT.md` first (≈15 minutes). The three research directions are **separate**; do not
merge them into one algorithm and do not add architecture beyond what is in the repo:
* **A** marginal-utility replacement-token selection (researcher's idea) — `scripts/a_marginal_utility.py`
* **B** multilingual allocation / is parity optimal? (researcher's idea) — `scripts/b_*.py`
* **C** selective re-merging (assistant's optional suggestion) — `scripts/phase0_audit.py`, `c_label_rules.py`, `bench_encode.py`

Ground rules (these are what make results believable):
1. **Never tune or select on the test split.** Selection/labels use `dev_sel`; monitoring uses `val`; `test` is for the final table only.
2. **Do not change the BPB code path** (`data.bpb_from_token_nll`, `train_lm.eval_stream`) except to fix a demonstrated bug; if you do, extend `tests/test_pipeline.py` and say so in the report.
3. `pytest -q` must pass before and after any code change you make. Keep changes small and list each one in the report.
4. Every comparison needs ≥3 seeds (same init/data seeds across methods) and goes through `scripts/stats.py`. A claim is supported only if `stats.py` prints `SUPPORTED`.
5. Report negative and null results as plainly as positive ones. Do not run extra variants to "find" an effect; the variants are pre-specified in the assessment.
6. Do not commit datasets, token streams or checkpoints (`.gitignore` covers them); do commit JSON reports and logs under `results/`.
7. If something in the brief or the assessment looks wrong against what you measure, say so — the assessment itself flags several suspicious numbers (§1.3).

## Stage 0 — Environment report (do this first, before any long job)

Run `python scripts/env_probe.py env_report.json` and write `results/ENV_REPORT.md` that answers **every** question below.
Where the probe cannot answer, find out (commands, docs, or ask the human operator). We will use your answers to tune settings
and to decide the experiment scale, so exact numbers matter more than prose.

**Hardware / software**
1. GPUs: exact model, count, memory each, compute capability (expected 7.5 / Turing), driver version, NVLink bridge present? (`nvidia-smi topo -m`; `p2p_access_0_1`, measured GPU0→GPU1 GB/s).
2. CPU cores, RAM, **free disk** on the filesystem you will write to; is there a faster scratch disk? Any per-job time limit or GPU sharing with other users?
3. Python version; installed torch + the CUDA version it was built for. **Does that torch wheel match the driver?** (A CUDA-13 wheel needs a recent driver; choose the wheel for *this* driver — do not blindly upgrade.) Can `torch.cuda.is_available()` be True and does a fp16 matmul reach a sane TFLOP/s (probe prints it)?
4. Which SDPA attention backends run (expected: mem-efficient yes, FlashAttention no on sm_75)? Does `--compile` (torch.compile/Triton) work on this GPU, and does it help (`autotune.py --compile` vs not)?
5. Is bf16 emulated/slow (expected)? We therefore train in **fp16 + GradScaler** (already the `--amp auto` behaviour on <sm_80). Confirm no NaNs/overflow in a 200-step run at the target learning rate; if unstable, report what you changed (lr, warmup, clip, loss scale).
6. NCCL: does 2-GPU DDP start and scale? Measure tokens/s on 1 GPU vs 2 GPUs for the same per-GPU micro-batch.

**Data and network**
7. Network: can the machine reach huggingface.co, pypi.org, github.com (probe prints HTTP codes)? Any proxy/offline mode? Where is the HF cache?
8. Which corpora are **already on disk**? Specifically: WikiText-103 (raw), an English Wikipedia dump (how large?), multilingual Wikipedia or another multilingual corpus (which languages, how many MB each), any parallel evaluation set (FLORES-200 or similar).
9. Is the **team repository** `https://github.com/yerinNam/h-scaffold-bpe` reachable (it is public)? Do you have, or can you get: the Wikipedia-EN 1B-word tokenizer corpus (or its `*.counts.pkl` from `hscaffold count`), WikiText-103 (raw), and the **training logs** of the original runs (`train_log.jsonl`, not in the repo)? Were 8B runs also done (their launch script includes llama-3.1-8b)?

**Already answered by the report and the code audit — confirm with the human, do not re-ask from scratch**
10. Tokenizer corpus: Wikipedia EN, 1B whitespace words. LM text: WikiText-103 train, 5 epochs, micro-batch 16×1024, lr 3e-4, 8-bit AdamW, BF16, flash-attn kernel, 4×B200. Tokenizer events were counted on WikiText-103 **test**.
11. "Second-merge events" in the τ ablation: counts of L2-containing pretokens / second merges / cross-boundary merges on ≈247K pretokens (denominator ambiguous in the slides — ask).
12. The previous LMs were essentially unlearned (test PPL ≈ 3.7–4.1K) because of the label double shift in `lm.py` (fix + tests are in `patches/`). Ask the human only whether the original training logs exist (to confirm: the logged train loss would have been a two-ahead loss) and whether init/data seeds were identical across methods (the code defaults to `--seed 42`).
13. Time/compute budget: total GPU-hours available, wall-clock deadline, and whether runs may be preempted (the suite runner is resumable).
14. Which languages for Direction B (suggested: 4–6 spanning scripts and resource levels, e.g. en, de, ru, hi, sw, vi) and is a parallel evaluation set available?
15. Priority if time runs out: which of A, B, C first? (Default order below: C-audit → A gate → B tokenizer sweep → LM stages for whichever pass their gate.)

Post `results/ENV_REPORT.md` (and tell the human) before starting any run longer than ~30 minutes. While you wait, you may do
everything that is CPU-only (tests, tokenizer training on a subset, audits).

## Stage 0b — Get the team's repository and apply the fix

> **If you are on branch `jihyun/research` of `lnnew/h-scaffold-bpe`:** this stage is already done. The team code is the repo root (`TEAM=$(git rev-parse --show-toplevel)`), the label-shift fix is included, `patches/` does not exist here, and the scripts live in `research/scripts` (run them from `research/`).


```bash
git clone https://github.com/yerinNam/h-scaffold-bpe ../h-scaffold-bpe && export TEAM=$PWD/../h-scaffold-bpe
cd $TEAM && git checkout -b fix-lm-label-shift && git am $OLDPWD/patches/0001-*.patch   # (optional: then 0002-*.patch for the audit notes); the core is lm.next_token_loss()
pip install -e ".[dev]" && pytest -q        # expect: 12 passed after 0001; 18 passed + 4 xfailed after 0002 (the xfails document a known whitespace-boundary divergence, AUDIT.md item 2)
```
We cannot push to that repository from the authoring environment; the patch is the hand-off. Do not trust any LM number from `artifacts/lm/`.
**Reuse the team's committed tokenizers** (`$TEAM/artifacts/tokenizers/{bpe,scaffold,hierarchical}-{32768,65536}.json`, plus `artifacts/ablation/tokenizers/t025|t075/`): no retraining is needed to compare Scaffold vs H-Scaffold; all scripts here read that format directly (`hierarchical` ≡ `hscaffold`). Our `train_tokenizers.py` default rule reproduces the team's trainer exactly on synthetic counts, so it is only needed for new vocabularies. The team's LM stack (bf16, flash-attn kernels, bitsandbytes, 1B/8B) targets B200s — on these Turing GPUs use this repo's `train_lm.py` (its loss alignment is unit-tested).

## Stage 1 — Make it run, then tune for this machine

```bash
pip install -r requirements.txt            # keep the torch build that matches the driver; do not downgrade/upgrade it blindly
pytest -q                                  # must pass (30 tests; the 13 adapter/trainer-equivalence tests need the team repo and skip otherwise)
python scripts/autotune.py --presets small base large --vocab 32768 65536 --out results/autotune.json
python scripts/autotune.py --presets small base --vocab 32768 --compile --out results/autotune_compile.json
python scripts/budget.py --tokens <train_tokens*epochs> --peak <fp16 tensor TFLOPs of the card, look it up> --mfu <measured/peak from autotune> --runs <planned runs>
```
What to optimise (in this order) and how to decide:
* **Model size.** The earlier protocol (LLaMA-3.2-1B shape, 5 epochs, 4×B200, bf16) is almost certainly infeasible for ≥24 runs here.
  Pick the largest preset from {`small` ≈43M/59M params at 32K/64K vocab, `base`, `large`} such that the *entire planned run list* fits the budget
  (Stage-0 answer 13). State the choice and the arithmetic. Large vocabularies make the embedding/softmax a big share of a small model —
  keep the *same* preset for all methods within a comparison.
* **Micro-batch / accumulation.** Use `autotune.json`'s largest fitting micro-batch (fp16). `--batch_tokens` must be divisible by
  `seq × n_gpu × micro_batch`; default 262,144 tokens/step. If there is no NVLink/P2P (PCIe only), prefer larger accumulation to amortise all-reduce.
* **Throughput switches to try and measure** (keep only if faster *and* loss curves match): `--compile`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
  `OMP_NUM_THREADS`; data loading (memmap is used; if the GPUs wait on the CPU, say so). Do not change numerics (precision, optimizer) for speed without a matched-loss check.
* **Learning rate.** The default 6e-4 is for the `small` preset; scale down for larger presets. Do a short LR sweep (3 values, 1 seed, `val` BPB only) once, then freeze it for **all** methods.
* Report measured tokens/s per preset/vocab, the chosen settings, and every deviation from defaults.

## Stage 2 — Data and counts

```bash
python scripts/prepare_text.py --source wikitext103 --out data/text      # lm_train / dev_sel / val / test jsonl (valid is split dev_sel/val)
# pretoken counts with the TEAM's pretokenizer (pattern gpt2), needed by the rule/pool scripts. Ideally the tokenizer corpus (Wikipedia-EN 1B) via the team's
# `hscaffold count`; as a stand-in, WikiText-103 train (say so in the report):
python scripts/train_tokenizers.py --text data/text/lm_train.jsonl --out tok --pattern gpt2 --counts_only --workers $(nproc)
```
Only if you need *new* vocabularies: `train_tokenizers.py` (pure Python, cost at your corpus size **unmeasured** — start with `--max_docs`, report wall-clock and peak RSS).
`--scaffold_rule global` (default) reproduces the team's trainer; `component` is a literal slide-2 reading that does not.

## Stage 3 — Gate G0 (corrected rerun) and Direction C gate

**3a. Tokenizer-only audit on the team's tokenizers (CPU).**
```bash
TOK=$TEAM/artifacts/tokenizers; ABL=$TEAM/artifacts/ablation/tokenizers
for K in 32768 65536; do
  python scripts/phase0_audit.py --tok_dir $TOK --K $K --text_dir data/text --out results/audit_$K.json \
     --tau_files 0.25=$ABL/t025/hierarchical-$K.json 0.5=$TOK/hierarchical-$K.json 0.75=$ABL/t075/hierarchical-$K.json
done
python scripts/compare_team_backend.py --team_repo $TEAM --tokenizer $TOK/hierarchical-32768.json --text data/text/test.jsonl --split_lines
```
Expected: all `checks` true; |S| = 2,601 / 4,889; L1/L2 counts equal the report's table exactly (already verified on the committed files); attempted vs accepted vs changed rates and the
`report_target` columns on WikiText-103 **test** (`--repro_split test`, used only for this table). `compare_team_backend` measures, on real text, how often the team's fast tiktoken path
differs from its reference path (known divergence on whitespace boundaries; expected ≈0 on WikiText lines — report the number).

**3b. Tokenise and check the floor.**
```bash
for m in scaffold hierarchical; do python scripts/tokenize_corpus.py --tok $TOK/$m-32768.json --text_dir data/text --out tok_data/K32768_$m --workers $(nproc); done
python scripts/sanity_baselines.py --data tok_data/K32768_scaffold
```
(Our encoder is pure Python with a per-pretoken cache — measure its throughput first; it is pretoken-local like the team's reference backend.)

**3c. Gate G0 = the corrected rerun (the real first experiment).** Our trainer is fp16 (not bf16) with fused AdamW (not 8-bit): a deliberate difference to report.
**Training health first:** a short run must drop far below the unigram floor (PPL in the tens–hundreds, not ≈3.7K); if not, stop and debug (LR, warmup, fp16 scaling, data) before anything else.
Then 5 seeds of Scaffold vs H-Scaffold at K=32768, and the same at 65536:
```bash
python scripts/run_lm_suite.py --tok_dir $TOK --tok_pattern "{method}-{K}.json" --text_dir data/text --K 32768 --setting K32768 \
   --methods scaffold hierarchical --seeds 0 1 2 3 4 --gpus 2 --lm_args --preset <p> --micro_batch <mb> --epochs <e> --lr <lr>
python scripts/stats.py --runs runs --ref scaffold --out results/stats_K32768.json
```
Report across-seed std, MDE, and whether any difference clears **both** seed- and document-level tests. The expected outcome (assessment §1.3a,e) is no detectable difference; that is a valid result.

**3d. Direction C Phase 1 (labels, predictability, variants)** — only after G0 is understood. Variants are written to `--variants_dir` (never into the team's artifacts):
```bash
python scripts/c_label_rules.py --tok_dir $TOK --K 32768 --text_dir data/text --word_counts tok/word_counts.pkl \
   --train_stream tok_data/K32768_scaffold/train.bin --variants_dir tok_variants \
   --scorers bigram torch:runs/K32768/scaffold/seed0/model.pt torch:runs/K32768/hierarchical/seed0/model.pt --out results/c_32768.json
python scripts/bench_encode.py --tok_dir $TOK --variants_dir tok_variants --K 32768 --text data/text/dev_sel.jsonl --variants utilpred_f50 randomT0_f50 --out results/bench_32768.json
```
(`model.pt` exists only if you add `--save_ckpt` to `--lm_args` for those seeds; if the script stops with "no candidate rule fired", the dev text is too small or events are too rare — report that.)
**Go/no-go for C's LM stage** (assessment §8): proceed only if predicted |ΔBPB| and selected-vs-random differences are ≥ MDE/3 **and** label predictability (CV Spearman) is meaningful and teacher-consistent. Otherwise stop C at "unmeasurable at LM level", and report the audit and the `bench_encode` result (event-triggered vs full second pass).
If it passes: methods `hsel_utilpred_f50 hsel_randomT0_f50 hsel_randomT1_f50 hsel_randomT2_f50 hsel_freq_f50 scaffold hierarchical`, 3 seeds, both K (use `--tok_dir tok_variants` for the `hsel_*` ones).

## Stage 4 — Direction A gate

Current priority supersedes the earlier G0/C-first schedule: use `PURPOSE_FIRST.md` and
`scripts/a_quick.py` for the development pilot. Do not restart old G0 jobs on wine GPUs 0/1.
Do not infer a no-go from a Jaccard threshold alone: count changed slots and measure retrained loss.

```bash
python scripts/a_marginal_utility.py --stage pool --tok_dir tok --K 32768 --pool_frac 0.25 --team_repo $TEAM   # pool built with the TEAM's trainer; uses tok/word_counts.pkl
python scripts/tokenize_corpus.py --tok tok/apool_pool_40960.json --text_dir data/text --out tok_data/K32768_apool_pool      # K_pool = 1.25K
python scripts/a_marginal_utility.py --stage select --tok_dir tok --K 32768 --text_dir data/text --train_stream tok_data/K32768_apool_pool/train.bin \
   --scorers bigram [torch:<teacher trained on tok_data/K32768_apool_pool>] --out results/a_32768.json
```
The neural teacher must be trained on the **pool** vocabulary (it must contain every candidate token). Gate per assessment §8. If it passes:
`run_lm_suite.py ... --tok_pattern "{method}_{K}.json" --methods scaffold apool_freq apool_comp apool_lm apool_random0 apool_random1 apool_random2 --seeds 0 1 2`.
(`apool_*` tokenizers have different L0 sets than `scaffold`, hence separate embeddings/streams — that is correct.)

## Stage 5 — Direction B

```bash
python scripts/prepare_multilingual.py --source hf --langs <langs> --out data/ml_text --train_bytes <per-language cap>   # or build the same file layout from local data
python scripts/b_build_tokenizers.py --text_dir data/ml_text --langs <langs> --K 32768 --alphas 0 0.5 1 --out tok_ml
```
Gate: parity metrics must change materially across α (printed). `prepare_multilingual.py` also writes the **fixed** LM mixture (`lm_train.jsonl`, `--mix_alpha`, same for every tokenizer) and pooled `val.jsonl`; per-language `test_<l>.jsonl` are the held-out sets. Train one tokenizer-specific LM per α (alphas are floats, so directories are `alpha0.0`, `alpha0.5`, `alpha1.0`):
```bash
LANGS="en de ru hi"                                   # <- your languages
SPLITS="lm_train:train val:val"; EVALS="val"
for l in $LANGS; do SPLITS="$SPLITS test_$l:test_$l"; EVALS="$EVALS,test_$l"; done
for A in 0.0 0.5 1.0; do
  python scripts/run_lm_suite.py --tok_dir tok_ml --text_dir data/ml_text --K 32768 --setting alpha$A --methods scaffold \
     --tok_pattern "ml_a${A}_{method}_{K}.json" --data_root tok_data_ml --runs runs_ml --seeds 0 1 2 --gpus 2 \
     --splits $SPLITS --eval_sets $EVALS --lm_args --preset <p> --micro_batch <mb> --epochs <e> --lr <lr>
done
python scripts/b_analyze.py --runs runs_ml --tok_report tok_ml/b_tokenizer_report.json --langs $LANGS --text_dir data/ml_text --kind scaffold --ref_alpha 1.0 --out results/b_analysis.json
```
Report equal-weight and byte-weighted means, per-language BPB and the (exploratory) direct/indirect split, plus the fixed-epoch **and** fixed-token views (rerun with `--lm_args ... --max_steps N` for the latter). Prefer a parallel test set (FLORES-like) if available; Wikipedia test text differs in content across languages. `stats.py` compares *methods within one setting*, so it does not directly compare α settings: use `b_analyze` for the α curve and compute the per-language across-seed spread yourself from each `results.json` (≥3 seeds) before calling any α difference real.

## What to send back (in `results/REPORT.md`, plus the JSON files)
1. `ENV_REPORT.md` answers (Stage 0), chosen preset/settings and measured tokens/s.
2. Phase-0 audit tables, noise floor (std, MDE), floor-baseline check, and every discrepancy with the brief's numbers.
3. For each of C, A, B separately: gate numbers → decision (proceed/stop and why) → if run: the `stats.py` table verbatim with its verdict lines.
4. Encoding benchmark ratios (scaffold / H-full / H-event / selective).
5. Code changes you made (file, why), test status, and anything you could not verify.
6. A direct answer to the project's goal: *which of A or B is more compelling and feasible, given your measurements*, with C kept as a separately-evaluated option. If the data cannot discriminate, say that.
