# A 우선 검증 결과: 현재 utility 점수의 특별한 이득은 미입증

이번 작업은 연구자의 **vocabulary slot 선택(A)**을 검증한다. C의 encoding-time
selective re-merging으로 방향을 바꾸지 않았다. B는 별도 실험으로 준비했다.

## 실제 수행한 비교

동일 Scaffold pool에서 32768개의 normal token과 EOS를 사용했다. Frequency, neural utility,
관측 범위·변경 슬롯 수·빈도 구간을 맞춘 random 선택을 비교했다. Utility가 바꾼 슬롯은 63개다.
공통 token/EOS 초기 embedding과 같은 seed의 model body를 맞췄다.

- 모델: LLaMA 형태의 기존 small preset, 약42.51M parameters, seq256, FP16.
- 각 방법은 initialization seed0/1/2로 재학습했다. Data-order seed0과 random-selection seed0은 고정했다.
- 각 실행은1500 updates ×8192 tokens. 같은 raw corpus를 사용하며 실제 byte exposure도 기록했다.
- `dev_sel`로만 token을 선택했다. Validation 후 세 방법/예산을 동결하고 모든 최종 체크포인트를 test 평가했다.
- Test62문서, 총1,287,656 UTF-8 bytes. 9개 평가의 문서별 byte 배열이 완전히 일치함을 확인했다.

| 방법 | Validation BPB 평균 | Test BPB 평균 |
|---|---:|---:|
| Pool frequency | 1.507995 | 1.536514 |
| Neural utility | 1.506837 | 1.534764 |
| Matched random | 1.506308 | 1.534767 |

Utility−frequency의 test 평균 차이는 **−0.001749 BPB(−0.114%)**였다.
3개 model seed 모두 개선됐고 paired t p=0.000412였다. 그러나 문서 bootstrap95%구간은
**[−0.004032, +0.000677]**로0을 포함한다. 특정 test split의 관측 평균과 문서 모집단에 대한
불확실성을 구분해야 한다.

Utility−matched random의 평균 차이는 **−0.00000247 BPB**,
paired t p=0.998, 문서 bootstrap95%구간은 **[−0.003280, +0.004328]**이다.
Utility의 우위가 확인되지 않았으며, 이 결과를 두 방법의 동등성 증명으로 해석하지도 않는다.

## 연구 판단

현재 score를 이용한 64K/1B 확대는 **No-go**다. Frequency 대비 관측 평균이 낮다는 사실만으로
utility criterion의 특별한 기여를 주장하면 안 된다. Matched random도 거의 같은 평균을 보였다.
이는 현재 fixed-teacher proxy/데이터/규모에 대한 결과이며, 모든 marginal utility 접근의 반증은 아니다.

다음 수정 대상은 모델 크기보다 **utility 추정**이다. 개발 데이터 coverage를 넓히고,
fixed-teacher token demolition 점수가 실제 재학습 이득을 예측하는지 먼저 검증해야 한다.
현재8674개 후보만 selection evidence가 있고, 관측하지 못한 후보의 baseline 선택은 고정했다.
또한 후보는 frequency로 만든 pool의 normal tokens에 한정된다. 모든 intermediate scaffold의
유지/삭제 또는 vocabulary construction 전체를 최적화했다고 주장할 수 없다.

## 코드에서 바로잡은 점

1. 첫 scorer만 `apool_lm`에 저장하던 오류 수정. 각 teacher 선택을 별도로 저장하고 scorer를 명시한다.
2. 관측되지 않은 token의 utility를0으로 보고 자의적으로 배분하지 않는다.
3. Random control도 같은 관측 범위 안에서 같은 수/빈도 구간의 슬롯을 바꾼다.
4. 공통 token 초기값을 token ID 이동과 무관하게 맞춘다.
5. Tokenizer/텍스트/관련 코드/학습 인자/initialization pool hash가 달라진 캐시 재사용을 거부한다.
6. 고정 token budget에서 실제 raw byte 노출을 기록한다. Frozen test 결과는 학습 결과와 별도로 저장한다.

Local 전체53tests, wine 핵심12tests 및 checkpoint 재평가2tests가 통과했다.
CPU2-process DDP의 global token/byte exposure도 기대값과 일치했다.
다운로드한 문서별NLL을 로컬에서 재집계한 평균/차이/p-value/bootstrap구간은 서버와1e−12 이내로 일치했다.
GPU에서 학습9개와 test 평가9개가 완료됐고, 현재 GPU0/1은 비어 있다.

## 재현 정보와 범위

- 학습 source SHA: `dae94ffa1048644aa03766e56c4f19f0851c5563`.
- 평가 source SHA: `08e0461c21e7682a503446a92fa4baaea57d4661`.
- SSH: `wine-jihyun-container`.
- 학습/평가 root: `/root/home/ubuntu/research/runs/purpose_a_v2/pilot`.
- Code entry: `research/scripts/a_quick.py`, `research/scripts/eval_checkpoint.py`.
- 이 폴더의 `decision.json`, 두 `stats_vs_*.json`, `runs/`의 checkpoint hash/문서별NLL이 근거다.
- B의 measured parity/전이 분리 설계는 `research/docs/PURPOSE_FIRST.md`에 있다. B LM은 실행하지 않았다.
- 64K/1B, 다른 teacher/selection seed, 고정 raw-epoch 비교 및 novelty 전체 조사는 미완료다.

기존G0 학습과 watchdog은 사용자의 지시로 중단했다. 로그와 큐 원본은
`/root/home/ubuntu/research/runs/preserved_g0/20261010T061152Z`에 보존했다.
중간 체크포인트가 없어 해당 학습 가중치에서 직접 재개할 수는 없다.
