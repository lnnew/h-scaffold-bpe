# 연구 목적과 가장 빠른 검증 경로

2026-10-10. A 우선, B는 별도로 준비한다는 사용자의 지시를 반영한다.
기준 revision: `b451363bcc8e0eb23c1765ab5147c755966925c7`.

## 리뷰 결론

- A의 pool vocabulary 선택 코드는 목적과 맞았다. 그러나 여러 scorer를 넣으면 첫 scorer만
  최종 `apool_lm`에 반영했다. 실제 예약 명령은 bigram을 먼저 넣어서 neural 결과를 사용하지
  않는 상태였다. scorer별 vocabulary 저장과 명시적 `--select_scorer`로 수정했다.
- 기존 bigram proxy의 held-out ΔBPB(utility−frequency)는 32K `+0.008085`,
  64K `+0.000754`였다. 양수는 악화다. 이는 기존 proxy의 약한 근거이며, 재학습 후 BPB가 아니다.
- utility 관측이 없는 토큰을 0점으로 재배분하면 데이터 부족을 토큰의 무가치함으로 오해한다.
  이제 관측되지 않은 토큰의 frequency 선택 상태를 고정한다.
- 학습 실행기는 결과 파일의 존재만으로 다른 budget/tokenizer까지 건너뛰었다.
  이제 텍스트·tokenizer·관련 코드·학습 인자를 SHA-256 identity로 검증한다.
  기존 identity 없는 캐시는 새 실행 루트로 분리한다. 과거 결과는 덮어쓰지 않는다.
- 기존 report의 LM 숫자는 A/B의 증거가 아니다. 팀 main에는 label double shift가 있었고
  현재 branch에서는 수정됐다. 원래 숫자가 그 코드로 생성됐다는 실행 provenance는 확보하지
  못했으므로, 해당 숫자의 정확한 원인을 확정하지 않는다.
- H-Scaffold의 Python/tiktoken 경계 처리 차이는 `hello \nworld`에서 재현됐다.
  이번 A 실험은 동일한 research tokenizer 경로만 사용한다. C/backend 수정은 우선 경로에 넣지 않는다.

## A: 변경한 알고리즘

동일한 Scaffold pool, 동일한 merge table을 사용하고 **최종 normal vocabulary를 선택**한다.
encoding 단계의 selective re-merging 연구로 바꾸지 않는다.

1. pool의 normal token은 `K_pool = 1.25 K`개다. bytes 256개는 항상 유지한다.
2. frequency baseline은 residual frequency 순서로 `K−256`개의 후보를 유지한다.
3. teacher는 반드시 pool의 token-ID mapping으로 학습한다. 체크포인트와 tokenization metadata를 대조한다.
4. `dev_sel` 짝수 문서에서 token별 `NLL(demolished)−NLL(kept)`를 측정하고 shrinkage를 적용한다.
   점수는 `V(t)=residual_frequency(t) × shrunk_mean_delta_bits(t)`다.
5. 관측되지 않은 token의 baseline 상태를 고정하고, 관측된 token의 가용 슬롯만 V로 선택한다.
   홀수 문서의 proxy는 진단용이다. 최종 품질은 별도 LM 재학습으로 측정한다.
6. random 대조군은 frequency baseline에서 utility와 동일한 수의 슬롯을 바꾸고,
   제거/추가 방향 각각의 residual-frequency decile 분포를 맞춘다.

이는 fixed teacher와 독립 token ablation의 proxy다. 여러 token 제거의 상호작용과
teacher segmentation 편향이 남는다. 재학습 비교가 이 proxy의 유효성을 검증한다.

## A 파일럿: 실행 계약

| 항목 | 고정 설정 |
|---|---|
| tokenizer | K=32768 normal tokens + EOS 1개; 모든 arm의 padded model vocab 동일 |
| primary arms | `apool_freq`, `apool_lm`, `apool_randomswap0` |
| LM | 기존 `small` preset, seq=256, FP16 on Turing |
| budget | 각 arm/seed 1500 updates × 8192 tokens = 12,288,000 token exposures |
| seeds | model init 0,1,2; data order 0; random-swap seed 0 고정 |
| text | 동일 WikiText raw train corpus, 전체 corpus에서 고정 token budget으로 학습 |
| selection | `dev_sel`만 사용 |
| pilot evaluation | `val`만 사용; test는 tokenization/evaluation 대상에서도 제외 |
| resources | 사용자 승인 wine GPU 0,1에 개별 single-GPU job 배분 |
| evidence | immutable `contract.json`, stream/run `identity.json`, per-doc NLL, 실제 raw byte exposure |

고정 token budget은 compute 관점의 비교다. tokenizer별 실제 읽힌 raw bytes는 다를 수 있으므로
`train_exposure_bytes`도 저장한다. 최종 유효성 확인에는 같은 raw-data epoch 관점의 비교를 추가한다.
동일 seed는 같은 크기의 초기 parameter tensor와 data-order seed를 맞추며, 서로 다른 token 문자열의
embedding을 의미적으로 동일하게 만든다는 뜻은 아니다.

```bash
# 새 코드에서 실행. 기존 pool와 그 pool로 학습 완료된 teacher를 재사용한다.
python research/scripts/a_quick.py \
  --root /root/home/ubuntu/research \
  --teacher /root/home/ubuntu/research/runs/purpose_a_v1/teacher_pool40960/model.pt \
  --out /root/home/ubuntu/research/runs/purpose_a_v1/pilot
```

진행 기준: utility가 frequency와 matched random보다 validation BPB가 낮고,
paired seed 및 document CI가 같은 방향이면 최종 확인 단계로 진행한다.
표본이 작아 불확실하거나 개선이 없으면 64K/1B 실험을 자동으로 늘리지 않는다.
먼저 coverage, proxy의 held-out sign, teacher 품질과 simultaneous-removal 상호작용을 확인한다.
관측 가능한 slot 변화가 0이면 현재 selector에는 개입이 없으므로 재학습을 생략한다.
파일럿 null 결과는 이 proxy/규모의 실패이며, 모든 marginal utility 연구의 반증은 아니다.

최종 확인: 선택 규칙과 hyperparameters를 validation으로 동결한 뒤 test BPB를 측정한다.
별도 budget/setting 루트에서 고정 raw-data epochs 비교를 추가한다. 기존 report의 절대 BPB와
소형 파일럿의 절대 BPB를 직접 연결하지 않는다. 큰 모델 확인은 파일럿 신호 이후에만 진행한다.

## B: 독립 준비와 주장 범위

현재 imbalance를 준 tokenizer sweep은 실제 개입이 있다. validation의 Scaffold bytes/token
max/min은 alpha 0에서 2.113, alpha 1에서 1.653이다. **alpha 0이 parity라는 가정은 틀렸다.**
uniform sampling과 measured parity를 구분한다. 서로 다른 문자 체계의 bytes/token은 UTF-8 영향을
받으므로, 비용의 공평함을 주장할 때는 같은 내용의 parallel text token 수 등도 확인한다.

첫 B 실험은 동일 raw LM mixture/budget/seeds에서 alpha별 language BPB 변화와
실제로 측정한 parity 지표의 관계를 평가한다. 목표함수(언어별 동일 가중치/자료 비중 가중치)는
미리 구분하고 language 내 변화량도 보고한다. BPE는 Scaffold 특유의 효과를 주장할 때 추가한다.

이 실험만으로 “영어 학습 개선이 다른 언어의 지식/추론 능력을 높였다”고 주장할 수 없다.
회귀 residual은 embedding/capacity/optimization 등도 포함한다. 전이를 분리하려면 recipient의
표현·노출을 고정한 개입과, 토큰화 자체로 설명되지 않는 별도 cross-language task가 필요하다.
이 단계는 A와 합치지 않는다. B의 GPU 학습은 현재 우선 큐에 자동 추가하지 않는다.

선행 연구와의 차이는 vocabulary utility의 재학습 검증으로 판단한다.
[Scaffold-BPE](https://arxiv.org/abs/2404.17808)는 빈도 기반 scaffold allocation,
[PickyBPE](https://aclanthology.org/2024.emnlp-main.925/)는 intermediate token 정리라는
가까운 출발점이다. 이름만 다르다는 이유로 novelty를 주장하지 않는다.
