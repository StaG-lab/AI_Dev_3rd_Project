📘 IntentResolver 모듈 설명서 (개발자용)

## 1. 개요
`IntentResolver`는 한국어 자연어 질의를 입력받아, **기간 intent(JSON)** 으로 변환하는 모듈이다.

- Google Gemini API(`gemini-1.5-flash`)를 호출하여 질의를 해석한다.
- 반환된 텍스트에서 JSON만 안전하게 추출하고, 필요 시 후처리(`resolve_dates`)를 통해 기간을 **주/월 단위 정규화**한다.
- `debug` 모드 지원: 개발 중에는 Gemini의 원문과 reasoning을 확인하고, 운영 시에는 JSON만 사용 가능하다.

---

## 2. 클래스 초기화
```python
resolver = IntentResolver(
    api_key="YOUR_API_KEY",
    model="gemini-1.5-flash",
    tz="Asia/Seoul",
    logger=None,
    debug=False
)
```

- **api_key**: 필수. Gemini API 키
- **model**: 기본 `"gemini-1.5-flash"`. 다른 Gemini 모델로 교체 가능
- **tz**: 시간대. 기본 `"Asia/Seoul"`
- **logger**: 선택적. 로그 기록용 객체 (`log_info`, `save_intermediate_result` 메서드 지원 시)
- **debug**:
  - `True`: `{ "raw_output": "...", "intent": {...} }` 반환 (원문+JSON)
  - `False`: intent JSON만 반환

---

## 3. 주요 메서드
### `run(query: str) -> dict`
엔드투엔드 실행 (프롬프트 구성 → Gemini 호출 → 파싱 → 후처리)

내부 동작 순서:
1. **프롬프트 구성**: `INTENT_SYSTEM_PROMPT` + 질의 텍스트
2. **Gemini 호출**: `generate_content()`으로 응답 획득
3. **JSON 파싱**: `_parse_json_with_raw()` → `{ "raw_output": 전체응답, "intent": {...} }`
   - JSON만 파싱 실패 시에도 `raw_output`은 항상 남음
4. **디버그 모드 분기**:
   - `debug=True` → raw + intent 그대로 반환
   - `debug=False` → intent JSON만 추출
5. **날짜 보정(resolve_dates)**:
   - weekly → 항상 월~일 7일 범위
   - monthly → 1일~말일
   - range → start/end 순서 보정 및 미래 clamp
   - daily → 단일 날짜

---

## 4. 반환 형식
### 운영 모드 (`debug=False`)
```python
res = resolver.run("지난주 감정 요약 보여줘")
print(res)

# 출력
{
  "mode": "weekly",
  "source": "지난주 감정 요약 보여줘",
  "timezone": "Asia/Seoul",
  "start": "2025-08-18",
  "end": "2025-08-24"
}
```

### 디버그 모드 (`debug=True`)
```python
res = resolver.run("저저번 주인가 저번주인가 일이 많아서 우울했던거 같은데 어땠더라?")
print(res["raw_output"])  # Gemini 전체 응답 (JSON + 설명)
print(res["intent"])      # JSON만
```

출력 예시:
```json
{
  "raw_output": "{\n  \"mode\": \"range\", ... }\n\n저저번주는 2025-08-11 ~ 2025-08-17...",
  "intent": {
    "mode": "range",
    "date": null,
    "start": "2025-08-11",
    "end": "2025-08-24",
    "source": "저저번 주인가..."
  }
}
```

---

## 5. 내부 구조 요약
### 5.1 프롬프트
- `INTENT_SYSTEM_PROMPT`는 오늘 날짜를 포함하여 JSON 스키마와 규칙을 명시.
- 예시(F-string):
```python
from datetime import datetime
import zoneinfo

today = datetime.now(zoneinfo.ZoneInfo("Asia/Seoul")).date().isoformat()
INTENT_SYSTEM_PROMPT = f"""
너는 한국어 질의를 읽고 기간 의도를 JSON으로 변환하는 도구다.
[SCHEMA] {{ "mode": ..., "date": ..., "start": ..., "end": ..., "source": ... }}
[CONTEXT] - TODAY: {today}
"""
```

### 5.2 Gemini 호출
```python
resp = self.model.generate_content(
    prompt,
    generation_config = {
        "temperature": 0.0,      # 결정적 출력
        "top_p": 1.0,
        "max_output_tokens": 512,
        "response_mime_type": "text/plain"
    }
)
```

### 5.3 JSON 파싱
- 정규식으로 첫 번째 `{ ... }` 블록 추출 후 `json.loads()`
- 실패 시 `raw_output`에 전체 응답 저장

### 5.4 후처리(`resolve_dates`)
- **daily**: 단일 날짜 (오늘/어제/지정일)
- **weekly**:
  - "지난주/저번주" → 직전 주 월~일
  - "저저번주" → 2주 전 월~일
  - 기본: 오늘이 속한 주
- **monthly**:
  - "지난달" → 직전 달 1일~말일
  - "이번달" → 1일~오늘 (말일 전이면 clamp)
- **range**: 입력 구간, start>end 시 교환

---

## 6. 권장 사용 패턴
- **개발/디버깅 단계**:
  - `debug=True`, 프롬프트에 "설명도 붙여라" 지시어 추가
  - reasoning과 JSON을 동시에 확인하며 프롬프트/후처리 튜닝
- **운영 단계**:
  - `debug=False`, JSON only 프롬프트
  - 토큰 절약 + 안정적 결과

