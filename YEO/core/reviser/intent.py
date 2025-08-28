# %%
import os, re, json
from datetime import datetime, timedelta, date
import zoneinfo
import google.generativeai as genai
from typing import Optional, Dict, Any

# %%

# 디버그용

# INTENT_SYSTEM_PROMPT_TMPL = f"""
# 너는 한국어 질의를 읽고 기간 의도를 JSON으로 변환하는 도구다.
# 출력은 반드시 JSON 하나만 하고, 그 뒤에 설명을 붙일 수 있다.

# [SCHEMA]
# {{
#   "mode": "daily|weekly|monthly|range",
#   "date": "YYYY-MM-DD|null",
#   "start": "YYYY-MM-DD|null",
#   "end": "YYYY-MM-DD|null",
#   "source": "<원문 그대로>"
# }}

# [CONTEXT]
# - TODAY: {today}

# - 그 외에 기간을 해석할 수 없으면 {{"mode":"error","error":"cannot_parse","source":"..."}} 로 지정하세요.
# """

INTENT_SYSTEM_PROMPT_TMPL = """
너는 한국어 질의를 읽고 기간 의도를 JSON으로 변환하는 도구다. 그 뒤에 설명을 붙일 수 있다.

[SCHEMA]
{{
  "mode": "daily|weekly|monthly|range|error",
  "date": "YYYY-MM-DD|null",
  "start": "YYYY-MM-DD|null",
  "end": "YYYY-MM-DD|null",
  "source": "<원문 그대로>",
  "error": "선택사항|null"
}}

[CONTEXT]
- TODAY: {today}
- 날짜/기간은 반드시 ISO 형식(YYYY-MM-DD)으로 출력한다.
- daily = 특정 하루, weekly = 월요일~일요일, monthly = 1일~말일, range = 임의의 날짜 구간.
- "이번주"는 TODAY가 속한 주(월~일).
- "지난주"는 이번주보다 1주 전(월~일).
- "지지난주"는 이번주보다 2주 전(월~일).
- "이번달"은 TODAY가 속한 달의 1일~말일.
- "지난달"은 이번달보다 1달 전의 1일~말일.
- "올해/작년" 등 연 단위도 같은 방식으로 확장 가능하다.
- "며칠 전", "몇 주 전" 같은 상대 기간도 TODAY 기준으로 계산한다.
[모호성 처리 규칙]
- 질의에 여러 개의 기간 표현이 동시에 등장하고, 그 중 어느 것이 맞는지 단정할 수 없으면:
  → "mode":"range" 로 지정하고,
  → "start"는 가장 이른 기간의 시작일, "end"는 가장 늦은 기간의 종료일로 설정한다.
- 질의가 너무 불명확해서 어떤 기간도 특정할 수 없으면:
  → {{"mode":"error","error":"cannot_parse","source":"..."}} 로 반환한다.
"""


# %%
# -----------------------------------------
# IntentResolver 클래스
# -----------------------------------------
class IntentResolver:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        tz: str = "Asia/Seoul",
        logger: Optional[object] = None,
        debug: bool = False,
        apply_resolve_in_debug: bool = True,  # 추가
    ):
        self.tz = tz
        self.tzinfo = zoneinfo.ZoneInfo(tz)
        self.model_name = model
        self.logger = logger
        self.debug = debug
        self.apply_resolve_in_debug = apply_resolve_in_debug
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Gemini API 키를 찾을 수 없습니다.")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(self.model_name)

    # -----------------------------------------
    # 로깅 유틸
    # -----------------------------------------
    def _log_info(self, msg, data=None):
        if self.logger and hasattr(self.logger, "log_info"):
            self.logger.log_info(f"[IntentResolver] {msg}", data)

    def _save_intermediate(self, name, data):
        if self.logger and hasattr(self.logger, "save_intermediate_result"):
            self.logger.save_intermediate_result(name, data)

    # -----------------------------------------
    # Gemini 호출
    # -----------------------------------------
    def _build_user_prompt(self, query: str) -> str:
        return f'입력 질의: "{query}"'

    def _call_gemini(self, prompt: str) -> str:
        resp = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "top_p": 1.0,
                "max_output_tokens": 512,
                "response_mime_type": "text/plain"
            }
        )
        text = getattr(resp, "text", None)
        if not text and getattr(resp, "candidates", None):
            for c in resp.candidates:
                if getattr(c, "content", None) and getattr(c.content, "parts", None):
                    for p in c.content.parts:
                        if getattr(p, "text", None):
                            text = p.text
                            break
                if text:
                    break
        if not text:
            raise RuntimeError("Gemini 응답에 텍스트가 없습니다.")
        return text

    # -----------------------------------------
    # JSON 파서 (raw + intent 동시 반환)
    # -----------------------------------------
    def _parse_json_with_raw(self, s: str) -> Dict[str, Any]:
        s_clean = s.strip()
        s_clean = re.sub(r"^```json\s*|\s*```$", "", s_clean, flags=re.IGNORECASE|re.MULTILINE)
        m = re.search(r"\{.*\}", s_clean, flags=re.DOTALL)
        if not m:
            return {"raw_output": s_clean, "intent": None}

        try:
            parsed = json.loads(m.group(0), strict=False)
        except Exception as e:
            return {"raw_output": s_clean, "intent": None, "error": str(e)}

        return {"raw_output": s_clean, "intent": parsed}

    # -----------------------------------------
    # Public API
    # -----------------------------------------
    def parse_intent(self, query: str) -> Dict[str, Any]:
        today = datetime.now(zoneinfo.ZoneInfo("Asia/Seoul")).date().isoformat()
        INTENT_SYSTEM_PROMPT = INTENT_SYSTEM_PROMPT_TMPL.format(today=today)
        full_prompt = INTENT_SYSTEM_PROMPT + "\n\n" + self._build_user_prompt(query)


        full_prompt = INTENT_SYSTEM_PROMPT + "\n\n" + self._build_user_prompt(query)
        self._log_info("Gemini 호출", {"query": query})
        raw_text = self._call_gemini(full_prompt)

        result = self._parse_json_with_raw(raw_text)
        self._save_intermediate("intent_raw_from_gemini", result)

        intent = result.get("intent")
        if not intent:
            # 강제 daily 금지: 에러 모드로 명시적으로 전달
            return {
                "mode": "error",
                "error": "JSON parse failed",
                "source": query,
                "timezone": self.tz,
                "raw": raw_text if self.debug else None
            }
        if self.debug:
            intent["raw"] = raw_text
        return intent

    # def resolve_dates(self, intent: Dict[str, Any], now: Optional[datetime] = None) -> Dict[str, Any]:
    #     """
    #     Gemini가 준 intent JSON을 후처리: 주/월 정규화, 미래 clamp 등
    #     (여기는 이전 코드에 있던 보정 로직 그대로 유지)
    #     """
    #     now = now.astimezone(self.tzinfo) if now else datetime.now(self.tzinfo)
    #     mode = intent.get("mode", "daily")
    #     src  = intent.get("source", "")
    #     out  = {"mode": mode, "source": src, "timezone": self.tz}

    #     if mode == "daily":
    #         d = intent.get("date") or now.date().isoformat()
    #         out.update({"date": d, "start": d, "end": d})

    #     elif mode == "weekly":
    #         base = now.date()
    #         if "저저번" in src: base = base - timedelta(days=14)
    #         elif "지난" in src or "저번" in src: base = base - timedelta(days=7)
    #         start = base - timedelta(days=base.weekday())
    #         end   = start + timedelta(days=6)
    #         out.update({"start": start.isoformat(), "end": end.isoformat()})

    #     elif mode == "monthly":
    #         base = now.date()
    #         if "지난달" in src:
    #             base = (base.replace(day=1) - timedelta(days=1)).replace(day=1)
    #         start = base.replace(day=1)
    #         next_month = (start + timedelta(days=32)).replace(day=1)
    #         end = (next_month - timedelta(days=1))
    #         out.update({"start": start.isoformat(), "end": end.isoformat()})

    #     elif mode == "range":
    #         out.update({"start": intent.get("start"), "end": intent.get("end")})

    #     else:
    #         d = now.date().isoformat()
    #         out.update({"mode":"daily","date":d,"start":d,"end":d})

    #     return out

    def run(self, query: str, now: Optional[datetime] = None) -> Dict[str, Any]:
        parsed = self.parse_intent(query)

        # 에러 모드면 그대로 리턴
        if parsed.get("mode") == "error":
            return parsed

        if self.debug:
            return {"debug": True, "intent": parsed}

        return parsed




