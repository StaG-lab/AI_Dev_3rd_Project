# 🎯 감정 요약 파이프라인 (A–B–C 모듈 통합)

## 1. 최초 입력 (User Query)
- 자연어 한국어 문장 (예: `"지지난주에 우리 뭐했더라?"`, `"7월 한 달 정리해줘"`)
- 입력 형식: `str`

---

## 2. 모듈 흐름

### 🅰️ IntentResolver (intent.py)
- **입력**: 사용자 질의 (자연어 query)  
- **출력**: 의도 JSON  
  ```json
  {
    "mode": "daily|weekly|monthly|range|error",
    "date": "YYYY-MM-DD|null",
    "start": "YYYY-MM-DD|null",
    "end": "YYYY-MM-DD|null",
    "source": "<원문 그대로>",
    "error": "null|ambiguous_period|cannot_parse"
  }
  ```
- **특징**:
  - 프롬프트에 TODAY 반영.
  - 모호 → range / 해석 불가 → error.  
  - resolve_dates 제거 → LLM 출력 그대로 사용.

---

### 🅱️ Data Engine (data_engine.py)
- **입력**: intent JSON (A 모듈 출력)  
- **출력**: 기간 리포트 JSON (PeriodReport)  
  ```json
  {
    "average_sentiment_score": 36,
    "dominant_emotions": [
      {"emotion": "상처", "percentage": 0.5},
      {"emotion": "중립", "percentage": 0.25},
      {"emotion": "기쁨", "percentage": 0.25}
    ],
    "trend": {
      "by_day": [
        {"date": "2025-08-18", "score": 20, "top": "상처"},
        {"date": "2025-08-19", "score": null, "top": null},
        ...
      ],
      "notable_changes": [
        {"date": "2025-08-24", "event": "점수 급변", "delta": 30},
        {"date": "2025-08-24", "event": "주감정 전환: 상처→기쁨"}
      ]
    },
    "meta": {
      "days_requested": ["2025-08-18", "2025-08-26"],
      "days_covered": ["2025-08-18", "2025-08-21", "2025-08-23", "2025-08-24"],
      "status": "ok|intent_error|invalid_range"
    }
  }
  ```
- **특징**:
  - 하루 집계 = dist 합산 → 정규화 → argmax.  
  - 빈 날 = score/top = null.  
  - notable_changes = 급변/주감정 전환.  
  - intent가 error → meta.status=intent_error.

---

### 🅲 Summarizer (summarize.py)
- **입력**: period JSON (B 모듈 출력)  
- **출력**: 요약 결과 JSON  
  ```json
  {
    "summary_message": "8월 18일부터 24일까지 감정 변화가 심했지만...",
    "status": "ok|error|fallback"
  }
  ```
- **특징**:
  - meta.status=error → 안내문 반환 (Gemini 호출 안 함).  
  - 정상 → Gemini 호출, JSON 스키마 강제.  
  - 파싱 실패 → fallback (평균점수+주감정만).  
  - 빈 데이터는 장황하지 않게 "해당 기간엔 감정 기록이 없습니다" 등으로 처리 권장.

---

## 3. 최종 출력 (to User)
- **형식**: `"summary_message"` 문자열  
- **상태**: status 값으로 분기
  - `"ok"`: 정상 요약 → summary_message 출력  
  - `"error"`: 기간 해석 실패 → 챗봇에서 “기간을 다시 말씀해주세요” 안내  
  - `"fallback"`: 간단 보고 + “나중에 다시 시도” 안내  

---

## 4. 주의사항
- 📌 **Intent 해석 모호성**  
  - “지난주/지지지난주” 같이 모호하면 range로 처리 → 범위가 넓어질 수 있음.  
- 📌 **빈 데이터**  
  - Period가 전부 None일 때 Gemini가 장황해질 수 있음 → 코드에서 한 줄 요약으로 대체 권장.  
- 📌 **API Key 관리**  
  - intent/summarizer는 Gemini API 키 필요 → 환경변수로 관리.  
- 📌 **실행 환경**  
  - `reports/analysis/analysis_result_*.json` 파일들에 의존.  
  - 파일 이름 및 경로가 현재 구조에 강하게 묶여있음 → 구조 바뀌면 로직 수정 필요.  
- 📌 **DB/ERD 연계**  
  - 현재는 JSON 파일을 직접 읽는 방식.  
  - ERD/DB 연결은 아직 구현되지 않았음 → 추후 백엔드 연동 시 수정 필요.  

---

## 5. 사용법 (메인 챗봇 예시)

```python
from intent import IntentResolver
from data_engine import run_data_engine
from summarize import Summarizer
from pathlib import Path

def handle_emotion_summary(query: str):
    # A: 의도 파악
    resolver = IntentResolver(api_key="YOUR_KEY", debug=False)
    intent = resolver.run(query)

    # B: 데이터 집계
    period = run_data_engine(intent, Path("./reports/analysis"))

    # C: 요약
    summarizer = Summarizer(api_key="YOUR_KEY", debug=False)
    summary = summarizer.summarize_period(period)

    return summary

# 실행 예시
print(handle_emotion_summary("지지난주에 우리 뭐했더라?"))
```

