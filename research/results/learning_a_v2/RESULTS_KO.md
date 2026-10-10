# 학습 후 슬롯 교환 utility: 구현 완료, 현재 추정법 확대는 No-go

연구자 방향 A인 vocabulary slot 선택을 구현하고 실제 wine GPU0/1에서 검증했다.
Encoding-time selective re-merging(C)으로 방향을 바꾸지 않았으며 B는 별도로 유지했다.

## 무엇을 측정했는가

고정 K=32768 normal tokens(+EOS)에서 토큰을 삭제하고 대체하는 교환을 비교했다.
공통 pool checkpoint의 transformer와 token/EOS 벡터를 동일하게 옮겼다.
U0는 교환 직후의 BPB 차이, U128은 같은 원문 배치로128회 적응한 뒤의 BPB 차이다.
이것은 짧은 warm-start 적응의 조건부 가치이며 처음부터 학습한 LM의 marginal utility가 아니다.

- 기존 pool teacher 하나, small LM 약42.51M parameters, FP16, seq256.
- 두 반복은 model initialization seed가 아니라 data-order seed0/1이다.
- 각 update32개 원문 chunk, chunk당 최대255 UTF-8 bytes. 모든 방법의 원문·순서·byte 노출을 일치시켰다.
- Seed0 노출1,040,467 bytes, seed1 노출1,036,762 bytes. 같은 seed 안에서30개 실행의 노출·순서 hash가 일치한다.
- 모든 chunk의 content와 종료 EOS를 예측하고 padding targets/logits를 제외했다. Byte-normalized NLL로 AdamW를 수행했다.
- 첫 준비는 개발 문서에 두 번 이상 나온 추가 후보가67개라256개 교환을 만들지 못해 실패했다. GPU는 사용하지 않았다.
- 후보 관측 기준을 실제 두 학습 순서 각각에서2회 이상 등장하는 토큰으로 옮겼다.163개 추가 후보가 자격을 갖췄다.
- 8묶음×16슬롯을 두 학습 순서에서 비교했다. 총18개 개별 실행 후,64개 슬롯을 바꾸는4묶음을 선택했다.
- 선택 문서15개/192,317 bytes, 확인 문서15개/320,724 bytes. 서로 원문이 겹치지 않는다.
  두 분할 모두 기존 연구의 개발 데이터이므로 새 sealed test라고 주장하지 않는다. 기존 test는 읽지 않았다.

## 실제 공동 확인 결과

선택 문서에서 U128로 선택한 묶음은[3,1,6,0], U0 기준은[6,7,4,2]였다.
그룹별 효과를 더해 최종 성능을 가정하지 않고, 조합한 어휘를 다시 학습·평가했다.
Frequency, learning, static U0, random0/1/2를 두 data order에서 비교해12개 실행을 추가했다.
전체30 fits,3840 updates가 완료됐다.

| 방법 | 확인 BPB 평균 |
|---|---:|
| Frequency |1.799151|
| 학습 후 utility |1.798247|
| 교환 직후 U0 기준 |1.799106|
| Matched random 평균 |1.798563|
| Matched random0 |1.799280|
| Matched random1 |1.798661|
| Matched random2 |1.797748|

아래 차이는 학습 후 utility−대조군이며 음수가 개선이다.

| 대조군 | 평균 BPB 차이 | 문서 bootstrap95%구간 | 두 data order에서 모두 개선 |
|---|---:|---|---|
| Frequency |−0.000905|[−0.001724,−0.000051]|예|
| U0 |−0.000859|[−0.002448,+0.000656]|아니오|
| Random3개 평균 |−0.000316|[−0.001574,+0.000760]|아니오|

개발 확인 조건에서 frequency 대비 작은 이득은 관측됐다. 그러나 U0와 random에 대한
고유한 우위가 없다. Random2의 관측 평균은 학습 후 utility보다 낮았다. 이 평균 비교를
방법의 동등성이나 random의 일반적인 우위로 해석하지 않는다.

Random 대조군은 교환 슬롯 수, 실제 학습 관측 자격, 제거/추가 토큰의 residual-frequency
decile 및 token byte length를 각각 맞췄다. 선택 문서 token-count 변화도 학습 선택+9개에
대해 random은+9/+8/+10개로 허용오차1token 안이었다. 압축 비교에 확인 문서의 손실은 사용하지 않았다.

## 중요한 실패 신호와 판단

1. 관측한8묶음의 U128 순위는 data order를 바꾸면 불안정했다. Spearman rho=−0.238이다.
   작은 후보 집합에 대한 기술 통계이며 모집단의 음의 상관을 증명하는 값은 아니다.
2. U0와 평균 U128 순위 상관은−0.214였다. 다른 선택을 만들 수 있다는 것만으로 더 좋은 선택임이 입증되지는 않는다.
3. Frequency의 확인 BPB는 적응 전1.784178에서 적응 후 평균1.799151로 악화됐다.
   Learning도1.784437에서1.798247로 악화됐다. 현재 비교는 개선된 학습보다는 상대적으로
   덜 악화된 적응을 포착할 수 있다. 원인은 아직 분리하지 않았다.
4. 원문 chunk는 단어를 자를 수 있고 EOS 빈도를 높인다. Optimizer reset/constant LR,
   짧은 문맥, 한 teacher의 편향을 별도로 검증하지 않았다. 기존 flattened-stream BPB와 절대값을 비교하면 안 된다.

사전에 정한 gate는 frequency, U0, random 평균 모두에 대해 문서CI 상한<0,
두 data order에서 모두 개선, 압축 matching 및 정확한 raw-exposure pairing 통과였다.
**결정: NO_GO_WITH_CURRENT_SHORT_ADAPTATION.** U0/random 비교가 gate를 통과하지 못했다.
현재 score로 predictor를 학습하거나64K/1B로 확대하지 않는다. 이는 A 전체의 반증이 아니다.
다음 연구는 모델 확대보다 안정적인 적응 기준 확보와 rank 재현성부터 다뤄야 한다.

## 코드 및 검증

- `research/src/scaffold_margin/learning.py`: UTF-8 chunk, 동일 원문 배치, token-ID remap,
  byte-normalized adaptation 및 문서별 BPB.
- `research/scripts/a_learning_utility.py`: fixed-K 교환, U0/U128 분해, 독립 확인,
  matched random, 입력/소스/teacher hash와 재사용 검증, GPU0/1 queue.
- 소스 SHA `ceebfc1006a96670529b503e4c2b9a679f0f3a68`.
- 로컬 기존 root12 + research49 =61 tests 통과. Wine 최신 learning tests8 통과.
-30개 실행 모두 skipped optimizer updates0, 같은 data seed에서 원문 순서와 byte 노출 일치.
- 로컬 문서별NLL 재집계가 서버의3개 비교 평균 및 bootstrap 구간과1e−12 이내로 일치한다.
- 근거: `contract.json`, `selection.json`, `decision.json`, `VALIDATION.json`,
  `probes/` 및 `joint/`의 results/identity/document-NLL 파일.
- Wine run `/root/home/ubuntu/research/runs/learning_a_v2`, 고정 소스
  `/root/home/ubuntu/research/h-scaffold-bpe-learning-a-v2`. 기존 teacher와 실패한v1 준비는 보존했다.
- Probe 최종 weights는 저장하지 않았고, 공통 teacher/소스/입력 hash/배치 순서/곡선/NLL로 재현한다.
- 실행이 정상 종료됐고 wine GPU0/1은1MiB/0%로 비어 있다. 추가 학습은 실행하지 않았다.
- 처음부터 재학습, 새 최종 holdout, teacher/model-init seed 다양화, novelty 확정은 미완료다.
