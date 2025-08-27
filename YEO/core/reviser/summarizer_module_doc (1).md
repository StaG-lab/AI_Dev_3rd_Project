# 📘 Module C: Summarizer (summarize.py) 정리

---

## 1. 역할
- **Module A (Intent)** → 기간 intent JSON 생성  
- **Module B (Data Engine)** → 지정된 기간의 `analysis_result_*.json`을 스캔·정규화·집계해서 **기간 리포트(PeriodReportJSON)** 생성  
- **Module C (Summarizer)** → 이 기간 리포트를 받아 **사람이 읽을 수 있는 요약 메시지**를 생성  

즉, **데이터 집계 결과 → 대화체 요약 리포트** 변환을 담당하는 모듈입니다.  

---

## 2. 입력
- Data Engine의 **기간 리포트 JSON**
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

## 3. 메커니즘
1. **LLM 프롬프트 빌드**  
   - 입력 JSON을 넣고, 챗봇 톤의 요약을 요청  
   - 예시 프롬프트:  
     ```
     너는 사용자의 감정을 돌아봐주는 챗봇이야.
     아래 기간 리포트를 참고해서, 전체 분위기와 중요한 변화를 대화체로 요약해줘.
     출력은 JSON 하나만 포함해야 해.
     { "chatbot_message": "<대화체 요약문>" }
     ```

2. **Gemini API 호출**  
   - `generate_content` 사용  
   - 실패 시 → 규칙 기반 fallback (“이번 기간 평균은 X점, 주요 감정은 Y입니다…”)

3. **출력 구성**  
   - `return result`  
   - 즉, 챗봇 메시지 JSON만 반환  

---

## 4. 코드 (최소 MVP 버전)

```python
# summarize.py
import os, json
import google.generativeai as genai
from typing import Dict, Any, Optional

class Summarizer:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Gemini API 키가 없습니다.")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model)

    def _call_gemini(self, prompt: str) -> str:
        resp = self.model.generate_content(prompt)
        return getattr(resp, "text", "")

    def summarize_period(self, period: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        너는 사용자의 감정을 돌아봐주는 챗봇이야.
        아래 기간 리포트를 참고해서 대화체 요약을 만들어.
        출력은 반드시 JSON 형식이어야 하고, 안에는 문장만 넣어.

        [OUTPUT SCHEMA]
        {{ "chatbot_message": "<대화체 요약문>" }}

        [DATA]
        {json.dumps(period, ensure_ascii=False)}
        """
        try:
            text = self._call_gemini(prompt)
            result = json.loads(text)
        except Exception:
            result = {"chatbot_message": f"이번 기간 평균은 {period.get('average_sentiment_score')}점, "
                                         f"주요 감정은 {period.get('dominant_emotions',[{}])[0].get('emotion','없음')}입니다."}
        return result
```

---

## 5. 출력 예시
```json
{
  "chatbot_message": "이번 주는 다소 힘든 시기였네요. 초반에는 상처가 많았지만, 21일에는 잠시 중립으로 바뀌었고, 마지막에는 기쁨으로 마무리되었습니다. 기복이 컸지만 긍정적인 마무리라 다행이에요!"
}
```

---

## 6. 사용법
```python
from summarize import Summarizer
from data_engine import aggregate_period, scan, aggregate_daily

summ = Summarizer(api_key="YOUR_KEY")

# Data Engine 결과(period) 입력
period_summary = summ.summarize_period(period)

print(period_summary["chatbot_message"])
# → 챗봇 톤의 대화체 요약이 출력됨
```

---

## 7. 정리
- **입력**: Data Engine이 만든 PeriodReportJSON  
- **처리**: LLM 기반 챗봇 프롬프트 → 대화체 요약 생성  
- **출력**: `{ "chatbot_message": "..." }` JSON  
- **MVP 상태**: `return result`만 반환 → 단순하고 명확  
- **향후 확장**:
  - DB 매핑(report_tbl)까지 연결 가능  
  - 일간 카드(daily) 포함도 가능  
  - 캐릭터별 챗봇 톤(도담이/지혜/모모) 분기 가능  

