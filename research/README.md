# Research extensions on top of the team code (branch `jihyun/research`)

This folder holds the utility-aware extensions (directions A / B / C, see `docs/RESEARCH_ASSESSMENT.md`). It lives **inside** the team repo
so it can use the team's tokenizers, trainer and data format directly. Everything here is new files under `research/`; no team file is modified
(the label-shift fix from PR #1 is on this branch only because it is a separate commit underneath).

```bash
cd research
pip install -r requirements.txt && PYTHONPATH=src python -m pytest -q tests      # 30 tests; the 13 adapter tests use the team package at the repo root
```
Run scripts from `research/` (they resolve `../src`). The team repo root is the "team repo": `TEAM=$(git rev-parse --show-toplevel)`.

Working rules for this branch
1. Add new files under `research/`; avoid editing team files (keeps merges with upstream conflict-free).
2. When `yerinNam/h-scaffold-bpe` `main` moves, **merge** it in (no rebase), so every experiment stays tied to a commit.
3. One commit per experiment step, results as small JSON under `research/results/`; large files (token streams, checkpoints) stay out of git.
4. Never open an upstream PR from this branch. Fixes for the team go in short branches cut from upstream `main`
   (PR #1 is an example). Tag milestones (`g0-result`, ...).

---

# Scaffold-Margin

Research code for three **separate** follow-ups to Scaffold-BPE (Lian et al., AAAI 2025):

| | Direction | Origin | Question | Entry points |
|---|---|---|---|---|
| **A** | Marginal-utility Scaffold-BPE | researcher | Is token *frequency* a good proxy for the value of a vocabulary slot? Choose replacement tokens by model-based marginal utility. | `scripts/a_marginal_utility.py`, `src/scaffold_margin/direction_a.py` |
| **B** | Multilingual utility-aware allocation | researcher | Is per-language tokenization *parity* actually optimal for multilingual LM quality? | `scripts/prepare_multilingual.py`, `b_build_tokenizers.py`, `b_analyze.py` |
| **C** | Selective re-merging for H-Scaffold-BPE | assistant-suggested, optional | Are all scaffold re-merges equally useful, and can that be predicted cheaply? | `scripts/phase0_audit.py`, `c_label_rules.py`, `bench_encode.py` |

They share infrastructure (tokenizer, LM trainer, BPB evaluation, statistics) but **do not share
algorithms**. Read `docs/RESEARCH_ASSESSMENT.md` for motivation, prior work, novelty, falsification
tests and go/no-go gates. **If you are the agent on the GPU machine, start with
`docs/REMOTE_AGENT_PROMPT.md`.**

## What exists, and how much of it has been verified

The repository started empty: the team's original code was **not available**, so Scaffold-BPE and H-Scaffold-BPE were
re-implemented from the brief and, later, the team's report (`hscaffold_bpe_report_v2.pdf`, slide 2 shows the scaffold
judgement rule; its numbers are in `docs/PRIOR_REPORT_NUMBERS.md`). Numbers in the report are therefore *targets to
reproduce*, not outputs of this code. The report itself notes that the official Scaffold-BPE code was never released.

Verified in the authoring sandbox (CPU only, synthetic data, no GPU, no real corpus):
* 30 pytest tests (13 of them cross-check against the team repo and skip without it): trainer == naive reference BPE; residual frequencies == recount; every output token is L0;
  lossless decode; H-Scaffold with an empty allow-list == Scaffold; event-triggered second pass == full second pass;
  BPB accounting exact (uniform-model check, per-document aggregation); save/load.
* Every script ran end-to-end on a synthetic corpus, including 2-process gloo DDP training, the neural-teacher
  scorer (CPU), and the paired-statistics script.

**Not** verified: anything on GPU (fp16 stability, memory, speed, NCCL), behaviour on real text, runtime of
BPE training on ~1B tokens (pure Python; measured only ~2-4 ms/merge on a 57K-type synthetic corpus), and
fidelity to the official Scaffold-BPE code (arXiv was unreachable from the sandbox).

## Relation to the team's repository (`yerinNam/h-scaffold-bpe`)
This repo is built to run **on top of** it: `Tokenizer.load` reads the team's tokenizer JSON directly (`adapter.py`), scripts accept their
artifacts, and `patches/` holds the fix (0001, to be proposed as a draft PR) and audit notes (0002) for a training bug found there (label double shift in `lm.py` — the reported LM results are suspect and need a corrected rerun;
see `docs/RESEARCH_ASSESSMENT.md` §1.3b/§1.4). Cross-checks: identical segmentation on all six committed tokenizers, identical trainer output on
synthetic counts (`tests/test_adapter.py`, needs the team repo; set `HSCAFFOLD_REPO`).

## Layout
```
src/scaffold_margin/
  pretok.py     regex pretokenizer (byte-level)          bpe.py       trainer + Tokenizer (bpe/scaffold/hscaffold)
  remerge.py    event stats, rule extraction            utility.py   edits, teacher deltas, predictability, selection
  scorer.py     bigram / neural teacher                 direction_a.py  pool + token value
  lm.py         LLaMA-style model                       train_lm.py  DDP trainer + BPB eval
  data.py       jsonl, streams, BPB bookkeeping         synth.py     offline smoke corpus
scripts/        see docs/REMOTE_AGENT_PROMPT.md for the run order
tests/          pytest -q
```

## Quick smoke test (CPU, ~1 minute)
```bash
pip install -r requirements.txt
pytest -q
W=/tmp/smoke
python scripts/prepare_text.py --source synthetic --out $W/text
python scripts/train_tokenizers.py --text $W/text/tok_train.jsonl --out $W/tok --vocab 600 --workers 2
python scripts/phase0_audit.py --tok_dir $W/tok --K 600 --text_dir $W/text --out $W/audit.json --sample_pretokens 20000
```
