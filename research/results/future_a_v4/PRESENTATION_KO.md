# Does a Good Token Help Predict What Comes Next?

## 0:00–0:45 — 질문

“BPE와 Scaffold-BPE는 주로 빈도를 기준으로 vocabulary를 구성합니다. 그런데 자주 등장하는 토큰이 언어모델 학습에 가장 유용한 토큰일까요? 제 연구 질문은 고정된 vocabulary 예산에서 어떤 토큰에 슬롯을 줄 것인가입니다. 이번에는 토큰 자체의 예측 손실 대신, 그 토큰을 읽은 뒤 다음 내용을 예측하는 능력을 기준으로 삼았습니다.”

## 0:45–1:40 — 방법

“같은 문자열을 하나의 토큰으로 유지한 경우와 작은 토큰으로 분해한 경우를 teacher에게 입력합니다. 앞 문맥과 뒤에서 평가할 token IDs는 같습니다. Word Utility는 바뀐 문자열 구간의 loss 차이이고, Future Utility는 문자열 자체를 제외한 동일한 후속 구간의 loss 차이입니다. 두 방법 모두 학습 빈도로 가중한 점수를 사용해 vocabulary 슬롯을 교환합니다.

Teacher의 선호가 실제 학습 가치인지 확인하려고, 최종 모델은 teacher 가중치를 복사하지 않고 처음부터 다시 학습했습니다. 이것이 proxy 점수만 보고 개선을 주장하는 것과의 차이입니다.”

## 1:40–2:40 — 공정한 비교

“후보 pool 40,960개와 최종 vocabulary 32,768개를 고정했습니다. Frequency, Word, Future, matched random 네 조건을 세 initialization seed로 학습해 총 12개 모델을 비교했습니다. Utility 방법은 각각 32개 슬롯을 교환하며, random도 같은 수와 같은 길이·빈도 구간을 사용합니다. 개발 자료의 Future와 random 토큰 수 차이도 1개로 맞췄습니다.

각 모델이 보는 원문 순서와 target bytes는 정확히 같습니다. 실제 Future는 학습 입력 구간의 약 2.5%를 바꿨습니다. Primary 평가는 새 PG-19 책 발췌 64개이고, 이전에 사용한 WikiText test는 보조 결과로만 제시합니다.”

## 2:40–4:00 — 결과

![결과 그래프](/Users/jihyunyoo/Documents/ChatGPT/MLP/h-scaffold-bpe/research/results/future_a_v4/pg19_comparison.png)

| 방법 | PG-19 평균 BPB ↓ |
|---|---:|
| Frequency | 3.877189 |
| Word Utility | 3.870175 |
| Future Utility | 3.871945 |
| Matched Random | **3.862845** |

“Future는 frequency보다 평균 BPB는 낮았지만, 한 seed에서는 악화됐습니다. Word보다 평균은 높고 차이의 문서 구간은 0을 포함합니다. 가장 중요한 random 대비는 세 seed 모두 Future가 나빴습니다. 평균 차이는 +0.009100 BPB입니다.

오른쪽 그래프의 문서 bootstrap 구간도 양수지만, 이 구간은 관측한 세 seed에 조건부입니다. 세 seed만으로 random이 모든 학습 조건에서 우월하다고 주장하지는 않습니다. 우리가 정한 Future 개선 기준은 통과하지 못했습니다. WikiText에서도 Future 평균이 가장 높았습니다.”

## 4:00–5:00 — 의미와 한계

“Future 점수는 개발 문서를 나눠도 순위 상관이 약 0.71로 안정적이었습니다. 하지만 안정적으로 측정되는 점수라는 것과, 그 점수가 재학습 후 토큰 가치를 예측한다는 것은 다른 문제였습니다.

결론은 모든 utility 기반 vocabulary 선택이 틀렸다는 것이 아닙니다. 이번 고정 teacher의 local future loss는, 작은 고정예산 실험에서 matched random보다 좋은 슬롯을 찾지 못했습니다. 따라서 현재 수식을 큰 모델로 확대할 근거는 없습니다. 연구의 기여 후보는 새로운 개선 알고리즘이라는 주장보다, 압축과 teacher loss를 넘어 실제 재학습과 강한 random 대조군으로 vocabulary utility를 검증한 반증 실험입니다. 선행연구 대비 novelty는 아직 확정하지 않았습니다.”

자세한 계약·통계·코드 검사는 [RESULTS_KO.md](/Users/jihyunyoo/Documents/ChatGPT/MLP/h-scaffold-bpe/research/results/future_a_v4/RESULTS_KO.md)에 있다. 64K/1B와 multilingual 방향 B는 실행하지 않았다.
