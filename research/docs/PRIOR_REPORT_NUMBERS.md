# Numbers transcribed from the team's report (`hscaffold_bpe_report_v2.pdf`, 17 slides)

Transcribed by hand from the slides; **re-check against the PDF before quoting**. "Inferred" marks things the slides do not
state. These are *reproduction targets*, not outputs of this repository's code.

## Setup (slides 1, 5)
Tokenizer corpus: Wikipedia EN ≈1B tokens, byte-level base. Tokenizers: BPE, Scaffold-BPE, H-Scaffold-BPE at 32K and 64K.
LM: llama-3.2-1b architecture, from scratch, 5 epochs, **WikiText-103** evaluation; B200×4, **BF16, 8-bit AdamW**, seq 1024.
Metrics: fertility, bytes/token, domain σ/CV (WikiText-103, WMT17, BLiMP), BPB (primary), PPL.
Official Scaffold-BPE code was **not released** → the team's Scaffold-BPE is an independent implementation too (slide 16).

## Scaffold rule used (slide 2)
After each merge (a,b)→t and the frequency update: `if f(a) < f(Qhead): Scaffold(a)=True` (removed from V; "may revive later"),
where f(Qhead) is the frequency of the **new** queue top. Example: (Ari, zona)→Arizona, f(Qhead)=43,598; f(Ari) 73,020→29,421, f(zona) 48,945→5,346.
H-Scaffold levels: r≥q → L0; 0.5q≤r<q → L1; r<0.5q → L2. Encoding: full merge → demolish L2 → re-merge (no L2 output) → demolish L1.
→ the team's trainer additionally keeps scaffold tokens in the queue and revives them when the head falls to their frequency; this is implemented as `scaffold_rule="global"` (default) in `bpe.py` and reproduces the team's trainer exactly on synthetic counts. The literal slide-2 reading is `"component"` (does not match).

## Tokenizer metrics (slide 8)
| Method | Vocab | Wiki fertility | Bytes/token | BLiMP fertility | Domain σ | Domain CV | MB/s | Train time |
|---|---|---|---|---|---|---|---|---|
| BPE | 32K | 1.166 | 4.499 | 1.295 | 0.0638 | 0.0530 | 6.35 | 651.8 s |
| BPE | 64K | 1.112 | 4.717 | 1.192 | 0.0392 | 0.0345 | 6.31 | 712.0 s |
| Scaffold | 32K | 1.161 | 4.520 | 1.287 | 0.0622 | 0.0519 | 5.99 | 596.3 s |
| Scaffold | 64K | 1.109 | 4.729 | 1.186 | 0.0371 | 0.0328 | 6.30 | 657.1 s |
| H-Scaffold | 32K | 1.161 | 4.520 | 1.287 | 0.0622 | 0.0519 | 3.91 | 663.9 s |
| H-Scaffold | 64K | 1.109 | 4.730 | 1.185 | 0.0371 | 0.0328 | 3.39 | 721.5 s |
Encoding −34.7% (32K) / −46.2% (64K) vs Scaffold. Tokenizer train time +10.6% (slide 8) / +11.35% (slide 11, 32K) — two different figures in the report.

## LM results (slide 10) — single seed
| Method | Vocab | tok/s | Time | Val PPL | Val BPB | Test PPL | Test BPB |
|---|---|---|---|---|---|---|---|
| BPE | 32K | 74,350 | 2.24 h | 3,624.99 | 2.596 | 3,784.75 | 2.648 |
| BPE | 64K | 70,237 | 2.25 h | 3,555.23 | 2.472 | 3,738.33 | 2.522 |
| Scaffold | 32K | 73,898 | 2.25 h | 3,676.07 | 2.589 | 3,778.13 | 2.635 |
| Scaffold | 64K | 70,311 | 2.25 h | 3,738.36 | 2.481 | 3,928.79 | 2.530 |
| H-Scaffold | 32K | 72,894 | 2.27 h | 3,962.57 | 2.612 | 4,079.89 | 2.659 |
| H-Scaffold | 64K | 70,091 | 2.26 h | 3,525.20 | 2.463 | 3,712.39 | 2.513 |
Vocab sensitivity (slide 13), test BPB 32K→64K: BPE −0.126, Scaffold −0.105, H-Scaffold −0.146.

## L2 threshold ablation (slide 14; tokenizer-only; τ=0.5 "reuses existing model")
| Vocab | τ | L2 | L1 | Avg fertility | L2 pretokens | 2nd merge | Cross-boundary |
|---|---|---|---|---|---|---|---|
| 32K | 0.25 | 1,322 | 1,279 | 1.199181 | 180 | 26 | 24 |
| 32K | 0.50 | 2,118 | 483 | 1.198974 | 466 | 51 | 48 |
| 32K | 0.75 | 2,477 | 124 | 1.198921 | 627 | 78 | 72 |
| 64K | 0.25 | 2,595 | 2,294 | 1.133075 | 103 | 13 | 13 |
| 64K | 0.50 | 3,988 | 901 | 1.133045 | 247 | 24 | 23 |
| 64K | 0.75 | 4,677 | 212 | 1.133022 | 339 | 28 | 26 |
Derived: L1+L2 is constant across τ → **|S| = 2,601 (32K), 4,889 (64K)**; this fixes the scaffold *set* and τ only splits it
(consistent with a stored per-token ratio). Fertility differences < 0.022% (32K) / < 0.005% (64K).
Text on slide 14/15: "51 per 247K pretokens at 32K·0.5"; L2 analysis is on the WikiText-103 **test** set (slide 15).

## Derived checks (mine; see assessment §1.3)
* BPB and PPL are mutually consistent: ln(3,784.75)/ln2 = 11.89 bits/token ÷ 4.499 bytes/token = 2.64 ≈ 2.648. So the BPB *accounting* looks right.
* …which pointed at the models, and the code audit found why: `lm.py` trained with HF-shifted labels *and* pre-shifted labels (two-ahead target) while
  evaluation was aligned — see `docs/RESEARCH_ASSESSMENT.md` §1.3b/§1.4 and the team repo's `docs/AUDIT.md`. **All six LM results are suspect (bug signature present); unusable for tokenizer comparison until rerun.**
* LM training text: WikiText-103 train (confirmed by the team's README; 74,350 tok/s × 2.24 h ÷ 5 epochs ≈ 120M tokens/epoch agrees). Batch 16×1024, lr 3e-4,
  8-bit AdamW, 34–36K steps (from `artifacts/lm/*/result.json`).
* Event rates (WikiText-103 test, ~247K pretokens, inferred from slide text): L2-containing 466 (0.19%), 2nd merge 51 (0.021%), cross-boundary 48.
