# 재병합이 희소하면 빨라질까? Event-Driven H-Scaffold의 검증

발표 성격: **정확한 가속 알고리즘을 구현했으나 현실 입력에서 가속 가설이 지지되지 않은 실험.**
원래 A의 vocabulary utility 연구와 구분한다. “새 tokenizer가 LM을 개선했다”는 발표가 아니다.

## 0:00–0:45 — 문제

“H-Scaffold는 BPE 후 L2 scaffold를 분해하고 다시 병합한 뒤, 남은 L1을 분해합니다.
실제 재병합은 매우 드뭅니다. 새 측정에서 64K Wiki는 약 9.9만 pretoken 중 9번뿐입니다.
그런데 현재 코드에는 두 번째 탐색과 demolition 비용이 있습니다.
필요한 주변만 처리하면 같은 분할을 더 빨리 얻을 수 있을까요?”

화면: `BPE → L2 demolition → second merge → L1 demolition` 네 상자.
역사적 BPB 수치는 label-shift 문제가 있어 보여주지 않는다.

## 0:45–1:45 — 방법과 정확성 조건

“L2가 없는 pretoken은 skip합니다. L2가 있으면 분해로 드러난 edge를 heap에 넣고,
병합 뒤 바뀐 양옆만 다시 넣습니다. 낮은 merge rank부터 처리하되, 같은 rank는
왼쪽부터 모두 병합하는 기존 batch 규칙을 지킵니다.

단, 기존 fast backend는 pretoken을 합친 뒤 문서 전체에서 재병합합니다.
L2가 없어도 경계 후보가 남을 수 있어 blind skip은 틀립니다.
fast 경로는 초기 후보를 전부 검사해야 같은 token IDs를 보존합니다.”

작은 예시: `"\na"`는 pretoken 경계 때문에 L2 없이도 기존 fast second pass에서 합쳐질 수 있다.
heap 자체와 neighbor update는 기존 tiktoken에도 있어 새 기법이라고 주장하지 않는다.

## 1:45–2:30 — 실험

“원본 32K·64K tokenizer, Wiki 개발 30문서와 PG19 64발췌문을 사용했습니다.
한 Apple M4 Pro에서 세 backend를 각각 비교했고, 수정 전 코드를 고정한 full,
skip, local, Scaffold를 7회씩 순서를 바꿔 측정했습니다.
GPU나 LM 재학습은 필요하지 않았습니다.
무작위·L2 stress 등 24,354 조건과 실제 benchmark의 564 조건에서 token IDs가 일치했습니다.”

조건 수는 문서×모델×backend 수다. 24,354개의 독립 문서가 아니다.
timing 범위는 반복 min–max이며 seed CI가 아니다.

## 2:30–3:30 — 결과

화면: [RESULTS.png](RESULTS.png)의 두 패널.

“정확성은 통과했습니다. 하지만 가속은 실패했습니다.
64K Wiki fast 경로는 full **8.35 MB/s**, local **7.34 MB/s**였고,
PG19는 **5.19 → 5.04 MB/s**였습니다.
회차별 paired speedup 중앙값도 각각 **0.882× / 0.976×**로 감소했습니다.
32K PG19는 약 1.003×였지만 일관된 이득이라고 볼 수 없었습니다.
사전에 정한 64K 두 도메인 1.10× 기준은 통과하지 못했습니다.”

## 3:30–4:30 — 왜 직관과 달랐나

“희소성의 단위가 달랐습니다. pretoken에는 L2가 드물어도 PG19 문서는 64개 모두에
L2가 있어 문서 단위 skip이 작동하지 않았습니다.
또 fast 32K PG19에는 L2와 무관한 초기 eligible edge가 88개 있었습니다.
문서 전체 초기 검사와 linked arrays 생성 비용도 계속 남았습니다.

원본 stage profile에서 second merge는 약 23–36%이고, 첫 merge·demolition·ID mapping이
나머지였습니다. Python 후보 관리가 실제로 절약하는 일보다 비쌀 수 있습니다.
이 profile은 별도 warm 측정의 비중이므로 정확한 인과적 비용 분해라고 말하지 않습니다.”

## 4:30–5:00 — 결론과 기여

“현재 local 구현은 채택하지 않고 기본 경로를 유지합니다.
가속 성공 대신, 동등성을 보장하는 frontier/batch 조건과 blind skip의 반례,
실제 입력에서 가속되지 않는 결과를 얻었습니다.
동일 token IDs는 같은 유효 모델의 BPB를 보존하지만 LM 품질 개선의 증거는 아닙니다.
새 목표나 어휘 효용 연구는 별도이며, Python heap이란 표현만으로 신규성을 주장하지 않습니다.”

## 예상 질문

- **5 MB/s를 넘었으니 성공 아닌가?** 같은 호스트의 원본도 5 MB/s를 넘었다. 기준 대비 가속이 핵심이다.
- **BPB도 같게 측정했나?** 기존 유효 H checkpoint는 식별되지 않았고 이번에는 재평가하지 않았다.
  같은 IDs·같은 유효 모델·결정론적 protocol의 동일 loss는 논리적 귀결이다.
- **왜 더 튜닝하지 않았나?** 사전 기준을 실패했다. 현재 구현의 이득을 주장하기 위해
  test 도메인에 맞춘 변경을 반복하기보다 음성 결과를 보존했다.
- **논문 신규성인가?** heap와 local update는 기존 구현에도 있다. 독립 논문 신규성은 미확인이다.
- **원래 연구 목적 A는?** 이번 실험은 C의 효율 연구다. A의 vocabulary-slot utility 검증을 대신하지 않는다.
