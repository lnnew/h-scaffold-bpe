# 재병합의 희소성은 빠른 토크나이저를 보장할까?

5분 발표 · 결론: **정확성 검증 성공 / 마지막 Hybrid의 가속 실패 / 최적화 중단**.

## 0:00–0:40 — 질문

“H-Scaffold는 L2 토큰을 분해한 후 다시 병합합니다. 실제 join은 드물지만,
발견하기 위한 탐색 비용까지 작다는 뜻은 아닙니다. 같은 token IDs를 유지하면서
필요한 후보만 처리하면 실제 인코딩을 가속할 수 있는지 실험했습니다.”

그림: `Full merge → L2 demolition → Second merge → L1 demolition`.
원래 A의 vocabulary utility 연구와 구분되는 효율 연구임을 한 문장으로 밝힌다.

## 0:40–1:40 — 정확성

“초기와 batch 경계에서는 모든 eligible edge가 valid queue에 있어야 합니다.
단, 원본은 같은 rank의 pair들을 왼쪽부터 한 batch로 병합합니다.
batch 중에 새 후보를 즉시 처리하면 원본과 달라질 수 있어 pending frontier를 따로 둡니다.
batch가 끝나면 바뀐 양옆을 넣어 completeness를 회복합니다.

기존 fast backend에는 pretoken 경계 후보가 남아 L2 위치만 seed하면 틀립니다.
새 collector는 기존 경계 후보도 포함합니다. 실제 함수 상태의 불변조건 검사와
16,236 Hybrid 차등 조건을 통과했지만, 모든 텍스트를 열거한 증명이라고 말하지 않습니다.”

## 1:40–2:30 — 마지막 Hybrid 한 가지

“짧은 sequence는 full을 씁니다. 길이가 128 이상일 때만 기존 demolition의 출력 생성
중 후보를 수집합니다. 후보가 없으면 skip, rank 종류가 4 미만이면 최소 후보를 재사용한
full fallback, 나머지만 local heap입니다. 임계값은 결과를 보고 다시 바꾸지 않았습니다.
별도 sequence 순회는 없지만 lookup과 Python emission 비용은 포함했습니다.”

실험: 고정 32K/64K, Wiki 30문서·PG19 64발췌문, 두 backend, 7회 rotated-order.
짧은 pretoken 1,024개는 같은 sample을 ×128 반복해 2ms 수준의 timer noise를 줄였다.
한 M4 Pro CPU에서 수행했고 GPU나 LM 학습은 없었다.

## 2:30–3:30 — 실제 결과

화면: [RESULTS.png](RESULTS.png).

“가속은 실패했습니다. 64K fast 경로에서 Wiki는 **9.31 → 7.63 MB/s**, PG19는
**5.60 → 5.41 MB/s**입니다. 같은 회차의 paired speedup은 **0.838× / 0.911×**였습니다.
32K도 **0.903× / 0.957×**로 개선되지 않았습니다.
보정한 짧은 입력도 32K **0.863×**, 64K **0.831×**였습니다.”

MB/s 중앙값과 paired ratio 중앙값은 다른 통계다. plot의 error bar는 min–max이지 CI가 아니다.
다른 실행 시간의 v1 숫자를 비교하지 않고 고정 원본 대비로 판단했다.

## 3:30–4:30 — 병목

“queue pop/검증은 약 0.01–0.08ms로 작았습니다. 큰 비용은 초기 후보 생성,
full scan과 batch 적용, sequence 생성·출력이었습니다.
별도 순회를 없앤 fused collector도 무료가 아닙니다. 예를 들어 64K PG19에서
plain demolition은 약 **19.5ms**, demolition과 후보 수집을 합치면 약 **56.0ms**였습니다.
이는 같은 작업이 아니므로 full scan을 대체하는 절약도 고려해야 하지만,
최종 end-to-end에서 그 비용을 상쇄하지 못했습니다.”

계측 사본에는 timer overhead가 있다. phase 표를 그대로 더해 인과적 총시간이라고 말하지 않는다.

## 4:30–5:00 — 종료와 의미

“출력 동등성은 확인했지만 실제 throughput 개선은 확인하지 못했습니다.
사전에 약속한 대로 추가 local 튜닝은 중단했습니다.
heap 자체가 새 연구 기여는 아닙니다. 얻은 결과는 frontier/batch/boundary 조건과
비용 분석, fixed Hybrid의 음성 결과입니다.
LM 품질이나 새로운 어휘 효용 목표를 달성했다고 말하지 않습니다.”

## 예상 질문

- **추가 scan이 없는데 왜 느린가?** 같은 emission 안에서도 lookup·Python inner loop·metadata 비용이 생긴다.
- **heap이 느린가?** pop 자체는 주요 병목이 아니었다. 초기 발견과 sequence 작업을 구분해야 한다.
- **더 튜닝하면 되지 않나?** 이번 계약은 한 가지 마지막 검증이었다. 실제 개선이 없어 중단했다.
- **BPB는?** 동일 IDs가 같은 유효 모델의 결정론적 loss를 보존하는 것과 품질 개선은 다르다.
  이번에는 LM/BPB 평가를 실행하지 않았다.
- **original A 목적은?** vocabulary-slot utility 연구는 별개다. 이번 효율 실험으로 해결했다고 주장하지 않는다.
