# Scaffold-BPE 인코딩 격차 연구 (infer_v1)

2026-10-11 · source `5c7d1fd0d3538a7ed653e693a966fae94d595bc9` · wine GPU0/1 · run root `runs/infer_v1`

**결론: Scaffold vocab에서 인코딩만 바꾸면 토큰 수가 0.3~0.6%(영어), 최대 1.9%(독일어·러시아어) 줄어든다.
H-Scaffold 재병합은 이 격차의 1~5%만 메운다. 그러나 토큰 수 감소는 LM 품질(BPB)로 이어지지 않았다.
같은 vocab에서 canonical / longest-match / DP로 학습한 LM의 test BPB 차이는 0.0006 이하이고 모두 통계적으로 0과 구분되지 않는다.**

## 동기 (survey)

Cognetta et al. (2026) *Tokenization: A Survey for Modern NLP* §3.1.3은 Scaffold-BPE의 인코딩이
학습 때의 merge 순서를 그대로 따르지 않아 최적이 아닌 분할을 낼 수 있다고 지적한다(PickyBPE와 대비).
이 연구는 그 격차를 처음으로 측정하고, H-Scaffold가 얼마나 메우는지, 격차를 메우는 것이 LM에 의미가 있는지 확인했다.

## 검증

- 새 인코딩 코드로 만든 BPE/Scaffold의 canonical train/val/test stream이 기존 `tok_data/K32768_*`와 **바이트 단위로 동일**하다.
- DP가 최소 토큰 수인지는 단어 길이 12 이하에서 모든 분할을 전수 탐색해 확인했다. 단위 테스트 4개 PASS(로컬, wine).
- 모든 학습 run: BPE 1,814 step, Scaffold 계열 1,806 step, 학습 바이트 1.0776 GB(차이 0.01% 미만).

## E1 인코딩 격차 (WikiText-103, CPU)

같은 Scaffold vocab에서 각 방식의 토큰 수 변화 (Scaffold canonical 대비)

| | 32K val | 32K test | 64K val | 64K test |
|---|---:|---:|---:|---:|
| BPE (자기 vocab) | +0.37% | +0.47% | +0.26% | +0.26% |
| H-Scaffold | −0.03% | −0.02% | −0.004% | −0.007% |
| Longest-match | −0.35% | −0.30% | −0.16% | −0.17% |
| Min-token DP | −0.54% | −0.57% | −0.28% | −0.33% |
| H-Scaffold가 메운 비율 (Scaffold→DP 격차 중) | 5.3% | 2.9% | 1.4% | 2.2% |

DP가 더 짧게 자르는 pretoken은 전체의 0.3~0.6%다. 예시: `Ker|shaw` ← `K|ers|haw`, `semi|aqu|atic` ← `sem|ia|qu|atic`.

**한계:** BPE에도 같은 종류의 격차가 있다(32K test BPE→BPE-DP −0.50%). Scaffold 철거가 더하는 몫은 약 0.07%p다.

## E0 자리 비우기: leaf pruning vs 숨기기 (BPE, CPU)

토큰 노출 빈도는 WikiText train 전체(104M pretoken)에서 셌다. 표의 값은 val/test 토큰 수 증가다.

| 비운 자리 k | 32K leaf | 32K 숨기기 | 64K leaf | 64K 숨기기 |
|---:|---:|---:|---:|---:|
| 5,000 | +0.23~0.32% | +0.12~0.13% | +0.002% | +0.002% |
| 10,000 | +1.46~1.75% | +1.13~1.35% | +0.08~0.09% | +0.03~0.04% |

1.6MB 텍스트로 돌린 파일럿에서는 두 방법의 차이가 보이지 않았지만, 전체 코퍼스 기준으로는 숨기기가 낫다.
Pruned BPE(arXiv 2608.00837)와 방향이 겹치므로 부록용이다.

## E3 다국어 (en/de/ru/hi 32K, CPU)

| α=1.0 test | BPE→Scaffold (vocab 효과) | Scaffold→DP (인코딩 효과) |
|---|---:|---:|
| en | −0.52% | −0.92% |
| de | −0.74% | −1.92% |
| ru | −0.79% | −1.72% |
| hi | −0.53% | −0.96% |

- 모든 언어에서 인코딩 효과가 vocab 효과보다 크다(α=0도 같다).
- 언어 간 형평성(Gini)은 거의 변하지 않는다: α=1.0 0.1114 / 0.1113 / 0.1109(DP).
- 숨겨진 scaffold 토큰 중 데바나가리 문자 비율이 높다: α=0에서 23.8%(보이는 vocab에서는 14.3%).
- 잘린 UTF-8 조각은 숨겨진 토큰 중 15개뿐이다.
- 이 test 셋은 병렬 데이터가 아니다.

## E2 LM: 같은 vocab에서 인코딩만 다르게 (GPU, 15 run)

설정: 42.5M LLaMA형 모델, seq 1024, 2 epoch, batch 131K tokens, lr 6e-4, fp16 + compile. seed 0/1/2(초기화 seed와 데이터 순서 seed를 같이 바꿈).
Mix는 문서마다 canonical / longest / DP 중 하나를 무작위로 골라 학습했다(survey §3.9 "작은 분할 후보 집합").

평균 test BPB (행: 학습할 때 쓴 분할, 열: 평가할 때 쓴 분할)

| 학습 분할 | 자기 분할 | canonical | longest | DP |
|---|---:|---:|---:|---:|
| BPE | 1.0917 | 1.0917 | 1.2042 | 1.1889 |
| Scaffold canonical | 1.0931 | 1.0931 | 1.1958 | 1.1827 |
| Scaffold longest | 1.0925 | 1.2059 | 1.0925 | 1.1038 |
| Scaffold DP | 1.0932 | 1.1923 | 1.1004 | 1.0932 |
| Scaffold mix | 1.1107 | 1.1107 | 1.0973 | 1.0970 |

Scaffold canonical 대비 차이 (seed별 차이 / 문서 단위 bootstrap 95% CI)

| 비교 | 평균 차이 | seed별 차이 | 문서 CI | 판단 |
|---|---:|---|---|---|
| DP − canonical | +0.0001 | +0.0003, +0.0004, −0.0004 | [−0.0018, +0.0022] | 차이 없음 |
| longest − canonical | −0.0006 | +0.0002, +0.0014, −0.0034 | [−0.0025, +0.0012] | 차이 없음 |
| BPE − Scaffold | −0.0014 | +0.0029, −0.0060, −0.0011 | [−0.0027, +0.0001] | 차이 없음 (BPE 쪽으로 약간) |
| mix − canonical | +0.0176 | +0.017, +0.018, +0.017 | [+0.016, +0.019] | mix가 확실히 나쁨 |

- **토큰 수 0.57% 감소가 BPB 차이로 이어지지 않는다.** 차이는 seed 노이즈보다 작다(BPE의 seed 간 차이는 최대 0.008).
- **학습 때와 다른 분할로 평가하면 BPB가 0.08~0.11 나빠진다.** 이는 인코딩 방식 간 차이의 100배가 넘는다(Sawada & Goyal 2025의 merge 순서 이탈 결과와 같은 방향).
- **Mix는 학습 분할 쪽 성능이 0.004~0.018 나빠지는 대신, 최악의 경우가 1.111에 그친다.** 단일 분할 모델이 다른 분할을 받으면 1.18~1.21이다.
- **Scaffold vs BPE의 LM 이득은 이번 3 seed에서 재현되지 않았다.** 팀의 기존 보고서에는 label-shift 버그가 있었다. 버그를 고친 이번 학습에서는 차이가 없다.

## 결론과 주장 범위

- 측정된 것: Scaffold의 인코딩 격차(survey의 지적), H-Scaffold가 메우는 비율(1~5%), 언어별 격차 크기.
- 측정되지 않은 것: 격차를 메우는 것의 LM 이득. 이 규모(42.5M, 2 epoch, 3 seed)에서는 효과가 감지되지 않았다. 더 큰 모델이나 다른 과제(형태론, 철자)에서의 효과는 시험하지 않았다.
- H-Scaffold의 재병합은 압축 측면에서도 작고, 격차를 전부 메우는 DP조차 LM 이득이 감지되지 않았다. 따라서 H-Scaffold의 L2 재병합에 LM 이득을 기대할 근거는 약하다.

## 재현

```sh
SRC=h-scaffold-bpe-infer-v1; P=h-scaffold-bpe/.venv/bin/python; export PYTHONPATH=$SRC/src:$SRC/research/src
$P $SRC/research/scripts/i_gap.py --tok_dir $SRC/artifacts/tokenizers --text_dir data/text --word_counts tok/word_counts.pkl --out e0_e1.json
$P $SRC/research/scripts/i_multilingual.py --pairs a1.0:...bpe:...scaffold a0:... --text_dir data/ml_text_imb --out e3.json
bash runs/infer_v1/tokenize_all.sh && runs/infer_v1/gpu_queue.sh 0 0 & runs/infer_v1/gpu_queue.sh 1 1
$P research/scripts/i_analyze.py --runs runs/infer_v1 --out analysis.json && python figures.py
```

checkpoint 15개는 wine에 남아 있다(`CHECKPOINTS.sha256`). 이 폴더에는 per-document NLL, 학습 로그, results.json이 보존되어 있다.
