# H-Scaffold의 목적을 다르게 구현하기 (infer_v2): ① λ 연속선, ③ 인과적 scaffold 입력 특징

2026-10-12 · source `5032892` (학습), `1322cbf` (checkpoint 로더 수정, 평가 전용) · wine GPU0/1 + quiche GPU0/5/6/7

**결론**
- **① λ 연속선:** 기존 Scaffold 모델에 넣었을 때 BPB가 좋아지는 분할은 하나도 없다. 분할을 바꾼 만큼 손해가 커진다(H-Scaffold +0.0008, 최소 토큰 분할 +0.012).
  그러나 토큰 수가 똑같이 최소인 분할이라도, 원래 분할에 가장 가까운 것을 고르면 손해가 7.6배 줄어든다(+0.090 → +0.012).
- **③ scaffold 입력 특징:** 효과가 감지되지 않는다. 진짜 id와 셔플 id의 차이는 +0.0006이고 seed마다 방향이 바뀐다. 특징이 켜지는 위치가 0.25%뿐이라 효과가 생길 여지 자체가 작다.

## ① λ 연속선 (학습 없음, 기존 checkpoint 평가)

각 pretoken을 `토큰 수 + λ × (Scaffold 기본 분할과 다른 경계 수)`가 최소가 되도록 자른다(정확한 DP).
λ≥1이면 기본 분할과 같고(단위 테스트로 확인), λ=0이면 토큰 수 최소다(동점이면 기본 분할에서 덜 벗어난 쪽).
H-Scaffold 분할은 같은 id 공간으로 옮겨 함께 평가했다. infer_v1의 checkpoint 12개(분할 4종 × seed 3개)를 9개 분할 × val/test로 평가했다.

| test 분할 | 토큰 변화 | 기본 분할 모델 ΔBPB | DP 모델 ΔBPB | Longest 모델 ΔBPB | Mix 모델 ΔBPB |
|---|---:|---:|---:|---:|---:|
| 기본 (λ=1) | 0 | 0 | 0 | 0 | 0 |
| H-Scaffold | −0.017% | +0.0008 | +0.0001 | +0.0001 | +0.0006 |
| λ=0.75 | −0.030% | +0.0008 | −0.0005 | −0.0006 | +0.0004 |
| λ=0.5 | −0.034% | +0.0009 | −0.0006 | −0.0007 | +0.0003 |
| λ=0.4 | −0.067% | +0.0009 | −0.0024 | −0.0024 | −0.0003 |
| λ=0.3 / 0.2 | −0.552% | +0.0112 | −0.0193 | −0.0190 | −0.0025 |
| λ=0 / 0.1 | −0.570% | +0.0118 | −0.0201 | −0.0198 | −0.0027 |
| (infer_v1) DP, 긴 토큰 먼저 | −0.570% | **+0.0896** | — | — | — |

(ΔBPB = 같은 모델을 기본 분할 test에 넣었을 때와의 차이. seed 3개 평균.)

- **분할을 바꾸면 손해다.** 기본 분할 모델에 대해 어느 λ도 BPB를 낮추지 않는다. 손해는 바꾼 양에 비례해 커진다.
- **H-Scaffold의 호환성:** 거의 무해하지만 0은 아니다(+0.0008).
- **최소 분할 중 어느 것을 고르느냐가 중요하다.** 같은 최소 토큰 수(281,725)에서 기본 분할에 가까운 것을 고르면 +0.012, 긴 토큰을 먼저 고르면 +0.090이다.
  압축만 보고 고른 분할은 호환성을 크게 잃는다. 최대 압축이 필요하다면 "원래 분할에서 최소한만 벗어나는 최소 분할"이 훨씬 안전하다.
- **연속선은 계단 모양이다.** λ=0.4에서 0.3으로 넘어갈 때 대부분의 압축이 일어난다. 이 구간부터 단순히 조각을 합치는 게 아니라 경계를 옮기기 시작한다.
- DP·Longest로 학습한 모델은 λ가 낮을수록 BPB가 좋아진다(자기가 학습한 분할 쪽으로 가까워짐). Mix 모델은 ±0.003 이내로 가장 둔감하다.

## ③ 인과적 scaffold 입력 특징 (GPU, 6 + 4 run)

**정의:** 위치 t에서 "직전 k≥2개 토큰이 정확히 어떤 scaffold 토큰의 철거 결과와 같으면, 그중 가장 긴 scaffold의 id"를 준다. 해당하지 않으면 0이다.
이 값은 t까지의 토큰만으로 계산되므로 미래 정보가 새지 않는다(단위 테스트: 앞부분만 넣어 계산해도 값이 같음).
입력 쪽 임베딩 테이블(2,602×512, 파라미터 +1.33M)에 더해지고, 0번은 0으로 고정된다. 출력 vocab은 바뀌지 않는다.

**대조군:** scaffold id를 무작위로 섞어서, 특징이 켜지는 위치와 파라미터 수는 같고 id만 틀리게 했다.
token stream, 데이터 순서, 하이퍼파라미터는 infer_v1의 Scaffold 기본 분할 run과 동일하다(fp16, compile).
특징이 켜지는 위치는 train 0.249%, test 0.254%다.

| arm (wine, seed 0/1/2) | test BPB 평균 | seed별 | val 평균 |
|---|---:|---|---:|
| 특징 없음 (infer_v1) | 1.09308 | 1.0915 / 1.0921 / 1.0956 | 1.08602 |
| 셔플 id (대조군) | 1.09222 | 1.0954 / 1.0919 / 1.0894 | 1.08455 |
| scaffold id | 1.09285 | 1.0941 / 1.0957 / 1.0888 | 1.08537 |

| 비교 | 평균 차이 | seed별 차이 | 문서 bootstrap 95% CI |
|---|---:|---|---|
| scaffold − 셔플 | +0.0006 | −0.0013, +0.0038, −0.0006 | [+0.0002, +0.0011] |
| scaffold − 특징 없음 | −0.0002 | +0.0026, +0.0036, −0.0069 | [−0.0012, +0.0006] |
| 셔플 − 특징 없음 | −0.0009 | +0.0039, −0.0002, −0.0063 | [−0.0018, −0.0000] |

- **진짜 id에 이득이 없다.** 문서 CI는 0을 살짝 넘지만 방향이 "진짜 id가 더 나쁨"이고, seed 3개 중 2개는 진짜 id가 더 좋다. 결론은 "이득 감지 안 됨"이다.
- **quiche 반복 실험(A6000, 같은 설정):** seed 0은 scaffold 1.0953 / 셔플 1.0972, seed 2는 1.0877 / 1.0864. 역시 방향이 갈린다.
- **run 간 노이즈:** 같은 설정·seed인데 wine과 quiche가 0.001~0.002 다르고, seed 2는 두 특징 arm 모두 특징 없음 run보다 0.007 낮다.
  학습 비결정성까지 포함한 노이즈가 최대 0.007 수준이다. 이번 실험에서 측정한 모든 효과가 이보다 작다.

## 해석

- **목적은 맞지만 정보량이 작다.** "scaffold를 버리면서 잃는 정보를 살리자"는 H-Scaffold의 목적은 이번에 세 가지 방식으로 구현해 봤다.
  분할을 바꾸는 방식(H-Scaffold, λ, DP)과 입력에 정보를 더하는 방식(③) 모두 이 규모에서 LM 이득이 없었다.
  scaffold가 관여하는 위치 자체가 0.25~0.6%로 작은 것이 가장 단순한 설명이다.
- **이득은 없지만 확실한 손해는 있다.** 학습 때와 다른 분할을 쓰면 손해이고, 그 손해는 어떤 최소 분할을 고르느냐에 크게 좌우된다(7.6배).
- **범위:** 42.5M 모델, 2 epoch, 32K, WikiText-103. 더 큰 모델, 64K, 다른 과제에서는 시험하지 않았다.

## 재현

```sh
SRC=h-scaffold-bpe-infer-v2; export PYTHONPATH=$SRC/src:$SRC/research/src
python $SRC/research/scripts/i_lambda.py encode --tok scaffold-32768.json --hs hierarchical-32768.json --text_dir data/text --out lambda
python h-scaffold-bpe-infer-v2b/research/scripts/i_lambda.py eval --out lambda --checkpoints 'runs/infer_v1/lm/scaffold_*/seed*/model.pt' --lm_root runs/infer_v1/lm --results lambda/eval.json
python $SRC/research/scripts/i_aux_data.py --src runs/infer_v1/data/scaffold_canonical --tok scaffold-32768.json --out data/scaffold_aux [--shuffle_seed 20261012]
aux_queue.sh <python> $SRC/research data . <gpu> scaffold_aux:<seed> ...   # train_lm --aux, 설정은 infer_v1과 동일
python i_analyze_v2.py --lambda_dir lambda --aux_runs . --base_runs ../infer_v1 --out analysis_v2.json && python figures.py
```

checkpoint는 wine `runs/infer_v2`와 quiche `~/ssd/jihyun_scaffold/runs`에 있다(`CHECKPOINTS.sha256`).
quiche 학습 도중 서버 전체 load average가 약 5,500까지 올라 SSH가 일시적으로 끊겼다. 원인은 다른 컨테이너의 55GB 프로세스가 메모리 부족으로 강제 종료된 것이었다(커널 로그). 학습 4개는 모두 정상 완료됐다.
