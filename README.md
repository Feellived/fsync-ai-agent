# F-Sync — 의도-반응 괴리도(Gap) 분석 시스템

기획 의도와 실제 소비자 반응의 **괴리를 라벨 단위로 정량화**하는 파이프라인.
기존 소셜 리스닝이 "반응이 어땠는가"만 측정한다면, 여기서는 **"의도한 메시지가 실제로 전달됐는가"** 를 수치로 답한다.

케이스: 엔플라잉(N.Flying) 디지털 싱글 **'환절기'** 의 프로덕션 기획안 vs 유튜브 팬 댓글.

> **팀 원본**: [pasohyun/fsync-ai-agent](https://github.com/pasohyun/fsync-ai-agent) — FNC 미래 일경험 프로젝트 8팀 '심사숙고' (2026-05-26 ~ 07-03).
> 이 저장소는 본인(유주형) 기여분을 포함한 미러이며, **FNC 대외비 기획 문서와 그 원문이 포함된 중간 산출물은 제외**했다. 아래 [제외된 자료](#제외된-자료) 참고.

---

## 핵심 결과

'환절기' 기획안이 의도한 메시지 3개 중 **1개만 전달**됐다.

| 라벨 | 기준선 | 환절기 | lift | 의도 | 판정 |
|---|---:|---:|---:|:---:|---|
| 연주_악기 | 4.48% | 7.86% | **×1.75** | — | 🟡 자생 |
| 장기_팬덤 | 12.66% | 18.81% | ×1.49 | — | 보통 |
| **청량_여름** | 5.88% | 8.06% | **×1.37** | high | ✅ **정렬** |
| 보컬_라이브 | 12.26% | 14.44% | ×1.18 | — | 보통 |
| 신규_유입 | 8.39% | 8.81% | ×1.05 | — | 보통 |
| **밴드_정체성** | 5.83% | 5.75% | **×0.99** | high | 🔴 **누수** |
| 음악성 | 40.07% | 30.52% | ×0.76 | — | 낮음 |
| **위로_공감** | 9.31% | 5.32% | **×0.57** | med | 🔴 **누수** |
| 이별_감성 | 1.12% | 0.44% | ×0.39 | — | 낮음 |

- **'청량·여름'** 은 통상 대비 1.37배로 의도대로 전달됐다.
- **'밴드·정체성'** 은 기획안이 가장 힘줘 밀었으나 통상 수준(×0.99)에 머물렀다 — 메시지가 반응으로 전환되지 않았다.
- **'위로·공감'** 은 오히려 평균 이하(×0.57). 의도와 반응이 정반대로 벌어진 지점.
- 의도하지 않았는데 **'연주·악기'가 1.75배**로 가장 크게 튀었다. 팬들은 기획이 말한 감성보다 **연주 그 자체**에 반응했다.

기준선 = 밴드 아티스트 8팀 유튜브 댓글 107,505건 / 대상 = '환절기' 댓글 4,000건 / 임계값 τ=0.80 / 분석 9라벨.

---

## 방법

```
① 반응 수집·정제  →  ② 라벨 체계 유도  →  ③ 이중 라벨링 + 분류기 학습
                                              ↓
        ⑥ 의도 라벨 추출  →  ⑦ 괴리도 산출  ←  ④⑤ 기준선/대상 반응 분포
```

**핵심은 ④ 기준선(baseline)이다.** 특정 곡의 댓글에서 "음악성 30%"라는 숫자만 보면 높은지 낮은지 알 수 없다. 같은 분류기·같은 임계값으로 **동종 밴드 8팀 전체**의 반응 분포를 먼저 구해 두고, 대상 곡을 그 대비 비율로 읽는다.

라벨별 상대 강도:

```
① hit(c,l) = 1[ σ(z_c,l) ≥ τ ]          τ = 0.80, 멀티라벨이라 한 댓글이 복수 히트 가능
② H(l) = Σ_c hit(c,l),  N = Σ_{l∈L} H(l)
③ p(l) = H(l) / N                        분모는 댓글 수가 아니라 전체 히트 수
④ lift(l) = p_target(l) / p_baseline(l)
```

판정 — `lift ≥ 1.2` 인 의도 라벨은 **정렬**, 미만이면 **누수**. 의도하지 않았는데 `lift ≥ 1.5` 면 **자생**.

> 단일 스칼라 '일치율'은 산출하지 않는다. 누수(의도했으나 미달)와 자생(의도 없었으나 초과)은 방향이 반대라 하나로 합치면 서로 상쇄되어 해석이 무너진다. 판정은 항상 라벨 단위다.

---

## 모델

12라벨 멀티라벨 분류. **Sigmoid 독립 이진 분류**(BCEWithLogitsLoss) — 한 댓글이 여러 반응을 동시에 담을 수 있으므로 softmax가 아니다.

백본 비교 (silver validation, micro F1):

| 모델 | micro F1 | macro F1 |
|---|---:|---:|
| **intfloat/multilingual-e5-large** ★ 채택 | **0.762** | 0.710 |
| klue/roberta-large | 0.739 | 0.678 |
| microsoft/deberta-v3-large | 0.721 | 0.675 |
| answerdotai/ModernBERT-large | 0.697 | 0.643 |

**사람이 직접 라벨링한 골드 홀드아웃 680건 기준 실측: micro F1 0.652 / macro F1 0.611.**
이게 진짜 성능이다. silver validation의 0.762와 11%p 벌어져 있는 게 정상이며, 오토라벨링 데이터로만 잰 숫자는 낙관 편향된다.

라벨별 편차가 크다 — 신뢰하고 읽어도 되는 라벨과 아닌 라벨을 구분해야 한다:

| 신뢰 구간 | 라벨 (골드 기준 F1) |
|---|---|
| 양호 | 연주_악기 .82 · 보컬_라이브 .73 · 청량_여름 .72 · 기타_노이즈 .71 · 음악성 .70 |
| 보통 | 비주얼_멤버매력 .65 · 밴드_정체성 .62 · 신규_유입 .60 · 이별_감성 .53 |
| **주의** | 역주행_기대 .51 · **위로_공감 .42** (P=.28) · **장기_팬덤 .33** (P=.21) |

`위로_공감`과 `장기_팬덤`은 recall만 높고 precision이 무너진 과탐지 상태다. 최종 9라벨에 남아 있으므로, 이 두 라벨의 lift는 다른 라벨보다 넓은 오차를 감안해 읽어야 한다.

### 12라벨

| 축 | 라벨 |
|---|---|
| 음악·컨셉 | `밴드_정체성` `연주_악기` `보컬_라이브` `음악성` `이별_감성` `위로_공감` `청량_여름` |
| 팬 행동 | `신규_유입` `장기_팬덤` `역주행_기대` |
| 기타 | `비주얼_멤버매력` `기타_노이즈` |

최종 Gap 분석은 `기타_노이즈`·`비주얼_멤버매력`·`역주행_기대`를 제외한 **9라벨**로 수행한다.

---

## 학습 데이터

1. **크롤링** — 밴드 아티스트 8팀(엔플라잉·QWER·Day6·엑스디너리히어로즈·드래곤포니·CNBLUE·FTISLAND·LUCY) 유튜브 댓글. 한국어 비율 필터 + 중복 제거 → **107,505건**
2. **오토라벨링(silver)** — Claude Sonnet Batch API, 12라벨 structured output → `01_corpus/corpus_labeled_1.jsonl`
3. **골드 홀드아웃(gold)** — 팀원 4명이 코드북 기준으로 직접 라벨링한 **680건**. 희귀 라벨 stratified 보강. **절대 학습에 쓰지 않고 채점에만 사용** → `01_corpus/gold_680_final.jsonl`

---

## 저장소 구조

```
00_crawling/            수집·정제·라벨링 스크립트
  youtube.py              YouTube Data API v3 댓글 수집
  filter_korean.py        한국어 비율 필터
  labeling/               골드 시트 생성·병합, Claude Batch 오토라벨러
01_corpus/              코퍼스 (10.7만건) + 골드 680건 + 팀원 라벨링 시트
03_doc_probs/           기획 문서의 라벨 확률 (집계값만 — 원문 포함분은 제외)
04_reaction/            추론 결과 + 반응 분포
  전체/OUTPUT/            기준선: 8팀 107,505건
  환절기/output/          대상: 4,000건 (sigmoid/softmax, 분포 그래프)
  아티스트별/             8팀 개별 포지셔닝
05_최종결과/            발표용 산출물 (반응 추이·포지셔닝·워드클라우드)
06_대시보드/            Streamlit 대시보드 + 동봉 데이터
gap_analysis/           Gap 분석 파이프라인 ★
  intent_labels.json      릴리즈별 의도 라벨·강도 (라벨만, 원문 없음)
  step1_build_intent_vectors.py   의도 라벨 추출
  step2_build_reaction_vector.py  반응 분포 산출 (수식 ①②③)
  step3_compute_gap.py            괴리도 산출 (수식 ④)
  run_*_sigmoid.py        댓글 배치 추론 (best_model.pt 필요)
model_training_script/  학습 파이프라인 + train/val jsonl
model_output/           백본별 학습 로그·평가 지표 (가중치는 미포함)
```

---

## 실행

### 대시보드 — 추가 자료 없이 바로 실행

```bash
pip install -r 06_대시보드/requirements.txt
streamlit run 06_대시보드/app.py
```

데이터가 `06_대시보드/data/` 에 동봉돼 있어 압축을 어디에 풀어도 동작한다.

### Gap 분석 — 추가 자료 없이 바로 실행

표준 라이브러리만 쓴다. 이미 산출된 sigmoid JSONL을 입력으로 받는다.

```bash
python gap_analysis/step1_build_intent_vectors.py --release hwanjeolgi
```

```bash
python gap_analysis/step3_compute_gap.py --release hwanjeolgi --baseline "05_최종결과/corpus_reaction_08_9label.json" --target "05_최종결과/hwanjeolgi_reaction_080_9label.json"
```

반응 분포부터 다시 만들려면:

```bash
python gap_analysis/step2_build_reaction_vector.py --sigmoid 04_reaction/전체/OUTPUT/corpus_me5_sigmoid.jsonl --out gap_analysis/out/corpus_reaction_080_9label.json --tau 0.80
```

```bash
python gap_analysis/step2_build_reaction_vector.py --sigmoid "04_reaction/환절기/output/hwanjeolgi_me5_sigmoid.jsonl" --out gap_analysis/out/hwanjeolgi_reaction_080_9label.json --tau 0.80 --by-date
```

### 재추론·재학습 — 모델 가중치 필요

`model_output/me5_large_v2/best_model.pt` (약 2.2GB)가 있어야 한다. `.gitignore` 대상이라 저장소에 없다 — [문의](#제외된-자료)하면 전달한다.

```bash
python gap_analysis/run_corpus_sigmoid.py
```

### 수집·라벨링 — API 키 필요

```bash
export YOUTUBE_API_KEY=...
```

```bash
export ANTHROPIC_API_KEY=...
```

키는 셸 프로필에 두고 코드·깃에 넣지 않는다.

---

## 제외된 자료

공개 저장소라 다음은 의도적으로 제외했고, **커밋 히스토리에서도 제거**했다.

| 제외 대상 | 이유 |
|---|---|
| `02_intent/` 전체 (기획안·발매보고서 PDF 8건, 추출 텍스트, 라인별 의도 추출) | FNC 대외비. `"(p.4) …"` 형태로 원문을 페이지 번호와 함께 인용하고 있었음 |
| `03_doc_probs/*/doc_probs_line*/` | `per_item[].text` 에 기획안 문장이 그대로 포함 |
| 발명 명세서 | 특허 출원 예정 |
| `best_model.pt` | 2.2GB, `.gitignore` |

의도 라벨은 **라벨명과 강도(high/med/low)만** 추려 `gap_analysis/intent_labels.json` 으로 재구성했다. 파이프라인은 이 파일만으로 완결되며, 기획 문서 원문 없이도 step1~3이 끝까지 돈다.

데모·심사 등으로 원자료나 모델 가중치가 필요하면 저장소 소유자에게 연락 바랍니다.

---

## 팀

**심사숙고 (8팀)** — 팀장 유주형 · 강수빈 · 박소현 · 정지민

본인(유주형) 주요 기여: 라벨 체계 설계 및 코드북, 골드 홀드아웃 라벨링 파이프라인(시트 생성·병합·IAA), Claude Batch 오토라벨러, Gap 분석 설계.
