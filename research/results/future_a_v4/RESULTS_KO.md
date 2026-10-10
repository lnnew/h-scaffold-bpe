# Future Utility: 구현·재학습 완료, 현재 proxy 확대는 No-go

**Vocabulary slot 선택이라는 연구자 방향 A를 구현하고 12개 LM을 처음부터 학습했다. Future Utility의 우위는 확인하지 못했다. 이번 조건에서는 matched random보다 PG-19 BPB가 세 seed 모두 높았다.** B는 별도 준비 문서로 유지했고 C의 encoding-time selective re-merging으로 방향을 바꾸지 않았다.

## 검증한 질문과 구현 범위

좋은 토큰은 그 문자열 자체를 잘 예측하게 하는가, 아니면 읽은 뒤 동일한 후속 내용을 잘 예측하게 하는가?

기존 Scaffold pool 40,960개에서 normal vocabulary 32,768개를 선택한다. Frequency baseline은 pool의 residual frequency로 선택한 고정 어휘다. 전체 Scaffold vocabulary 학습 알고리즘을 최적화한 결과는 아니다. 각 방법은 baseline에서 32개 슬롯을 교환하며 모든 256 byte 토큰을 보존한다.

기존 pool teacher 하나에서 토큰 유지/분해를 비교한다. 같은 prefix, 같은 원문, 같은 suffix token IDs, 같은 pool+EOS output head를 사용한다. 변경 pretoken의 loss를 Word, 동일 suffix의 loss를 Future로 분리했다. 토큰 앞의 loss는 변화하지 않는지 검사한다. 토큰 자체의 loss는 Future 점수에 포함하지 않는다.

실제 점수는 다음과 같다. L은 bits 단위 NLL, n은 관측 수다.

```text
U_word(t)   = n/(n+4) × Σ[L_word(split) − L_word(keep)] / Σbytes_word
U_future(t) = n/(n+4) × Σ[L_suffix(split) − L_suffix(keep)] / Σbytes_suffix
S(t)        = residual_train_frequency(t) × U(t)
exchange_score(remove t, add c) = S(c) − S(t)
```

양수는 teacher 아래에서 유지가 유리하다는 뜻이다. 이는 byte-weighted local proxy이며, 겹치는 suffix와 토큰 간 상호작용 때문에 실제 corpus marginal utility와 같지 않다. suffix는 최대 12개 후속 pretokens이며, 전후 입력이 모두 256 tokens 이내가 되도록 함께 줄인다. prefix 길이·position 효과도 점수에 포함될 수 있다.

## 통제와 실제 실행

- Frequency / Word / Future / Matched random, 각 3개 독립 model initialization seed(0/1/2). 데이터 순서 seed는 모두 0이다. 교사 가중치는 student에 복사하지 않았다. 공통 transformer 및 문자열이 같은 토큰의 초기 random embedding을 맞췄다.
- Student 42,508,800 parameters, FP16, seq 256, update 1500, update당 공통 원문 구간 32개. 동일 raw target와 preceding raw context를 제공하고 byte-normalized NLL로 학습했다. Padding/context는 target에서 제외하며 인위적인 구간 끝 EOS target을 넣지 않았다.
- 공통 준비 corpus: 3,676 WikiText 문서, 67,135,626 bytes, 74,175 target windows. 실제 각 모델은 중복 없는 48,000 windows, **43,453,915 target bytes**를 학습했다. 모든 실행의 raw corpus/order hash와 노출 byte 수가 같다. optimizer update 누락은 0회다.
- 개발 문서에서 54,700개 counterfactual을 FP32로 계산했다. 토큰당 최대 32개 관측, 최소 2개 관측을 요구했다. 실제 예정 학습 원문의 pool occurrence가 8회 이상인 후보만 남기고, 최종 tokenization에서도 추가 토큰이 최소 2회 등장하는지 검사했다. 실제 최소치는 Word 9회, Future/random 5회다.
- 교환 후보의 byte 길이를 같게 하고 가장 가까운 residual frequency로 짝지었다. Removal decile/addition decile/byte length별 교환 quota와 슬롯 수는 세 selector가 같다. 동일 decile의 추가·삭제가 불가능한 frequency 경계는 별도 방향 decile로 처리했다.
- Random은 같은 후보·quota에서 최대 1000번 뽑고, LM loss 없이 개발 자료의 토큰 수 변화가 Future와 가까운 draw를 선택했다. 실제 draw 218: Future −48, random −47 tokens, 허용 오차 내 1개 차이. Word는 −121이다. 이 matching은 개발 자료 기준이며 모든 corpus의 token 수를 동일하게 만드는 통제는 아니다.
- Word와 Future는 32개 교환 중 12개 pair를 공유한다. 최종 vocabulary의 대칭 차이는 80개(한쪽 집합 차이 40개)다. 전체 scored candidate의 Word/Future 순위 상관은 0.5836, 별도 개발 문서 절반에서 Future 순위 안정성은 0.7090(공통 후보 1829개)이다. 안정적인 점수 자체가 retraining utility의 타당성을 입증하지 않는다.
- Expanded token IDs로 독립 대조한 실제 학습 입력 변경률은 Word 2.6979%, Future 2.5208%, random 2.7104%다. Target 변경률은 각각 2.1354%, 1.9875%, 2.0896%다. 변경 window 비율은 변경 byte 비율과 다르다.

## 평가와 결과

Primary는 이번 연구에서 새로 사용한 **PG-19 test 64개 책의 발췌**, 총 1,041,425 bytes다. 공식 파일명의 SHA256 정렬로 고르고, 각 책의 1/4 지점 이후 첫 줄부터 최대 16 KiB를 사용했다. 성능으로 책을 선택하지 않았다. 이는 book-domain generalization 시험이다. WikiText test는 이전 연구에서 이미 사용했으므로 exploratory secondary다. 두 corpus의 BPB 절댓값을 직접 비교하지 않는다.

모든 방법의 원문 평가 구간/byte 수가 같으며, 정해진 마지막 checkpoint를 저장한 뒤 평가했다. Test loss로 vocabulary나 checkpoint를 선택하지 않았다.

| 방법 | PG-19 평균 BPB | WikiText 평균 BPB |
|---|---:|---:|
| Frequency | 3.877189 | 1.560084 |
| Word Utility | 3.870175 | 1.560026 |
| Future Utility | 3.871945 | 1.560240 |
| Matched Random | 3.862845 | 1.560055 |

낮을수록 좋다. 평균은 3개 initialization seed 기준이다.

| PG-19 비교: Future − 기준 | 평균 ΔBPB | seed 0 / 1 / 2 | 문서 bootstrap 95% 구간 |
|---|---:|---|---|
| Frequency | −0.005244 | −0.030771 / −0.003819 / +0.018858 | [−0.007504, −0.003554] |
| Word | +0.001770 | +0.011239 / −0.000974 / −0.004956 | [−0.000180, +0.003528] |
| Matched random | **+0.009100** | **+0.003237 / +0.015552 / +0.008512** | **[+0.008048, +0.010155]** |

문서 bootstrap은 seed별 문서 NLL을 평균한 뒤 64개 책을 4000번 재표집했다. **이 구간은 관측한 세 seed에 조건부이며 seed 모집단의 신뢰구간이 아니다.** 참고로 Future−random의 3개 seed paired t-test p=0.1254여서 seed 모집단에 대한 유의한 열세도 주장하지 않는다.

Frequency보다 평균은 좋아도 seed 2에서는 악화됐다. Word 대비 문서 구간은 0을 포함한다. Random 대비는 관측한 세 seed 모두 악화됐다. 미리 고정한 gate(모든 대조군 대비 모든 seed 개선 및 PG-19 문서 CI 상한<0)를 충족하지 못해 **NO_GO_FOR_THIS_FUTURE_PROXY**다. WikiText에서도 Future의 평균은 네 방법 중 가장 높다.

![PG-19 비교](/Users/jihyunyoo/Documents/ChatGPT/MLP/h-scaffold-bpe/research/results/future_a_v4/pg19_comparison.png)

왼쪽 점은 각 seed, 선은 평균이다. 오른쪽은 위의 조건부 문서 구간이며 음수가 Future에 유리하다.

## 연구적·코드적 판단

연구자 목적 A에 맞는 slot-selection 실험을 실행했지만, **vocabulary 선택으로 LM 성능을 개선했다는 목적은 현재 충족하지 못했다.** 고정 teacher의 미래 loss를 사용한다고 재학습 후 토큰 가치가 더 잘 예측되는 것은 아니라는 결과다. 모든 marginal-utility 접근의 실패나 frequency의 최적성을 뜻하지 않는다.

현재 proxy를 64K/1B로 확대하거나, test 결과를 본 뒤 utility 수식을 바꿔 같은 holdout에서 다시 평가하지 않는 것을 권한다. 5분 발표는 “안정적인 local utility가 실제 학습 가치를 예측하는가?”라는 반증 결과로 정리할 수 있다. 처음으로 LM utility/context를 이용한 tokenizer라는 novelty 주장은 하지 않는다. 정확한 선행연구 대비 독창성 검증은 완료하지 않았다.

코드 검증은 전체 70 tests, 서버 focused 9 tests가 통과했다. 공통 raw window, loss partition/prefix invariance, fixed-K 교환, lossless decoding, recursive demolition과 DP token-count 일치, EOS/padding 처리 및 tiny from-scratch 저장·평가를 검사했다. 원문 노출/order/평가 bytes를 12개 실제 실행에서 대조했다. 로컬 문서 NLL로 평균·delta·bootstrap·gate를 독립 재계산했으며 서버 대비 최대 오차는 **0**이다.

학습 전 matching/coverage gate에서 실패한 `future_a_v1/v2/v3`는 원본 점수·입력·로그와 함께 보존했다. Student test loss를 사용해 수정하지 않았다. 성공 실행의 source SHA는 `89e2da363a16edc00f454975136a99efe10ccc09`다.

## 보존 자료와 남은 범위

- [최종 판단](/Users/jihyunyoo/Documents/ChatGPT/MLP/h-scaffold-bpe/research/results/future_a_v4/decision.json), [고정 실험 계약](/Users/jihyunyoo/Documents/ChatGPT/MLP/h-scaffold-bpe/research/results/future_a_v4/contract.json), [슬롯 선택](/Users/jihyunyoo/Documents/ChatGPT/MLP/h-scaffold-bpe/research/results/future_a_v4/selection.json), [원문 노출 분석](/Users/jihyunyoo/Documents/ChatGPT/MLP/h-scaffold-bpe/research/results/future_a_v4/EXPOSURE.json), [독립 통계 검사](/Users/jihyunyoo/Documents/ChatGPT/MLP/h-scaffold-bpe/research/results/future_a_v4/AUDIT.json).
- [재계산 코드](/Users/jihyunyoo/Documents/ChatGPT/MLP/h-scaffold-bpe/research/results/future_a_v4/analyze.py), [PNG](/Users/jihyunyoo/Documents/ChatGPT/MLP/h-scaffold-bpe/research/results/future_a_v4/pg19_comparison.png), [SVG](/Users/jihyunyoo/Documents/ChatGPT/MLP/h-scaffold-bpe/research/results/future_a_v4/pg19_comparison.svg), [5분 발표 원고](/Users/jihyunyoo/Documents/ChatGPT/MLP/h-scaffold-bpe/research/results/future_a_v4/PRESENTATION_KO.md).
- [Checkpoint 목록·SHA256](/Users/jihyunyoo/Documents/ChatGPT/MLP/h-scaffold-bpe/research/results/future_a_v4/CHECKPOINTS.json): 12개 마지막 가중치는 `/root/home/ubuntu/research/runs/future_a_v4/runs/<method>/seed<seed>/model.pt`에 보존했다. 로컬에는 metadata와 문서별 loss를 복사했으며 큰 가중치/학습 corpus NPZ는 서버에 있다.
- 실제 student training 시간 합은 **0.9165 GPU-hours**다. Teacher/scoring/평가/CPU 준비 시간은 이 수치에 포함하지 않았다. wine GPU0/1은 완료 후 각각 1 MiB/0%였다. 기존 G0 재시작 guard는 유지했다.

한 teacher, 한 random vocabulary, 세 initialization seed, 32개 교환, 짧은 small-LM 학습의 pilot이다. Teacher/selector seed 반복, 충분히 수렴한 장기 학습, position 효과 분리, 64K/1B, encoding throughput과 B의 multilingual LM 실험은 실행하지 않았다. B 준비 계획은 [B 준비 계획](/Users/jihyunyoo/Documents/ChatGPT/MLP/h-scaffold-bpe/research/docs/PURPOSE_FIRST.md)에 별도로 남아 있다. Whole-document exact duplicate 검사는 통과했지만 corpus contamination 부재를 입증한 것은 아니다.
