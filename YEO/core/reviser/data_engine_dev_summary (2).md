# 📘 Data Engine 개발 및 입출력 구조 정리

이 문서는 **Module B (Data Engine)** 개발 과정을 요약하고, 코드 설계, 입력/출력 구조를 PM과 다음 GPT 채팅방에서 쉽게 이해할 수 있도록 정리한 것입니다.

---

## 1. 개발 진행 개요

1. **IntentResolver 모듈 (Module A)**

   - 사용자 자연어 질의를 받아 기간(JSON)으로 변환.
   - 예: `"저번주 감정 요약 보여줘"` →
     ```json
     {
       "mode": "weekly",
       "date": null,
       "start": "2025-08-18",
       "end": "2025-08-24",
       "source": "저번주 감정 요약 보여줘",
       "timezone": "Asia/Seoul"
     }
     ```
   - Data Engine의 입력으로 바로 사용.

2. **Data Engine (Module B)**

   - 3단계 구성: **Scanner → Normalizer → Aggregator**
   - 역할:
     - 지정된 기간 안의 `analysis_result_*.json` 파일 수집
     - 감정 분포를 7감정 풀셋으로 정규화
     - 일간 카드와 기간 리포트 집계

3. **Summarizer & Export (Module C)**

   - Data Engine 출력(JSON)을 받아 요약 문구 생성, 파일 저장/DB 매핑.
   - 이 방에서는 아직 본격 구현 전, Data Engine까지 집중.

---

## 2. Data Engine 상세 설계

### (1) Scanner

- **입력 예시**: date\_range = {

      "mode": "weekly",

      "date": None,

      "start": "2025-08-18",

      "end": "2025-08-26",

      "source": "저번주인가 일이 많아서 우울했던거 같은데 어땠더라?"

  }
- **처리**: 기간 안의 파일 스캔, 파일명에서 날짜 추출
- **출력 구조**:
  ```python
  files = {
    "2025-08-23": [Path(".../analysis_result_20250823_050448.json"), ...],
    "2025-08-24": [Path(".../analysis_result_20250824_091200.json")]
  }
  ```

### (2) Normalizer

- **입력**: 개별 `analysis_result_*.json`
- **처리**:
  - 주요 필드 추출: `sentiment_score`, `dominant_overall_emotion`, `emotion_distribution`
  - Top-K 감정만 들어온 분포 → **7감정 풀셋**으로 확장 (`없는 감정=0`, 합=1로 정규화)
- **출력 구조**:
  ```python
  {
    "score": 60,  # sentiment_score
    "top": "기쁨", # dominant_overall_emotion
    "dist": {"기쁨":0.45, "중립":0.30, "슬픔":0.25, "분노":0, "불안":0, "상처":0, "당황":0}
  }
  ```

### (3) Aggregator

#### A. Daily Aggregator

- **입력**: 하루에 해당하는 파일 리스트 (NormalizedRecord[])
- **집계 규칙**:
  - 점수 = **평균(mean)** (중앙값 → 평균으로 변경)
  - 주감정 = **다수결** (동률 → 추후 연속성 고려 가능)
  - 분포 = **합산 후 정규화**
- **출력 구조 (DailyCardJSON)**:
  ```python
  daily_cards = {
    "2025-08-23": {"score": 63, "top": "기쁨", "dist": {...}},
    "2025-08-24": {"score": 58, "top": "불안", "dist": {...}}
  }
  ```

#### B. Period Aggregator

- **입력**: 여러 일간 카드
- **집계 규칙**:
  - 평균 점수 (정수 반올림)
  - 우세 감정 Top-K 빈도 계산
  - trend: 날짜별 score/top 목록
  - notable\_changes:
    - Δscore ≥ 10 → 점수 급변 이벤트
    - dominant 감정 변화 → 감정 전환 이벤트
- **출력 구조 (PeriodReportJSON)**:
  ```python
  period = {
    "average_sentiment_score": 36,
    "dominant_emotions": [
      {"emotion": "상처", "percentage": 0.5},
      {"emotion": "중립", "percentage": 0.25},
      {"emotion": "기쁨", "percentage": 0.25}
    ],
    "trend": {
      "by_day": [
        {"date": "2025-08-18", "score": 20, "top": "상처"},
        {"date": "2025-08-21", "score": 35, "top": "중립"},
        {"date": "2025-08-23", "score": 30, "top": "상처"},
        {"date": "2025-08-24", "score": 60, "top": "기쁨"}
      ],
      "notable_changes": [
        {"date": "2025-08-21", "event": "점수 급변", "delta": 15},
        {"date": "2025-08-21", "event": "주감정 전환: 상처→중립"},
        {"date": "2025-08-23", "event": "주감정 전환: 중립→상처"},
        {"date": "2025-08-24", "event": "점수 급변", "delta": 30},
        {"date": "2025-08-24", "event": "주감정 전환: 상처→기쁨"}
      ]
    }
  }
  ```

---

## 3. 입력 → 출력 전체 흐름

1. **Intent 모듈 출력** → `date_range` JSON
2. **Scanner** → 날짜별 파일 dict
3. **Normalizer** → 파일별 점수/감정 분포 표준화
4. **Aggregator** →
   - DailyCardJSON (날짜별 카드)
   - PeriodReportJSON (기간 리포트)

---

## 4. 요약

- **입력**: IntentResolver의 기간 JSON + analysis\_result\_\*.json 파일들
- **처리**: Scanner → Normalizer → Aggregator
- **출력**:
  - `daily_cards`: 날짜별 점수/주감정/분포
  - `period`: 기간 평균, 우세감정 TopK, trend, notable\_changes

이로써 Data Engine은 **자연어 질의 → 기간 데이터 요약 JSON** 변환의 핵심 루프를 완성했습니다.

