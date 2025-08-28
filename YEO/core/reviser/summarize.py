# %%
# summarize.py
import os, json, re
import google.generativeai as genai
from typing import Dict, Any, Optional

class Summarizer:
    """
    Data Engine 출력(PeriodReportJSON, DailyCardJSON)을 받아
    - LLM 요약문 생성
    - 결과 JSON 반환 및 파일 저장
    - (옵션) DB 매핑
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        debug: bool = False
    ):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Gemini API 키가 없습니다.")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model)
        self.debug = debug

    # ------------------------------
    # 내부 Gemini 호출
    # ------------------------------
    def _call_gemini(self, prompt: str) -> str:
        resp = self.model.generate_content(
            prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 256}
        )
        return getattr(resp, "text", "")

    # ------------------------------
    # Period 요약
    # ------------------------------
    def _extract_first_json(self, text: str) -> Optional[Dict[str, Any]]:
        """응답 문자열에서 첫 번째 {...} JSON 블록을 파싱"""
        if not text:
            return None
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def _fallback_brief(self, period: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 파싱 실패 시 짧은 보고 + 재시도 안내"""
        avg = period.get("average_sentiment_score", None)
        dom = period.get("dominant_emotions") or []
        top_emo = dom[0]["emotion"] if (isinstance(dom, list) and dom and "emotion" in dom[0]) else None

        parts = ["오류가 났어요."]
        if top_emo and avg is not None:
            parts.append(f"감정은 대략 {top_emo}, 평균 점수는 {avg}점이에요.")
        elif top_emo:
            parts.append(f"감정은 대략 {top_emo}예요.")
        elif avg is not None:
            parts.append(f"평균 점수는 {avg}점이에요.")
        parts.append("나중에 다시 시도해주세요.")
        return {"summary_message": " ".join(parts), "status": "fallback"}

    def summarize_period(self, period: Dict[str, Any], daily_cards: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        입력:
          - period: B 모듈 PeriodReportJSON (meta.status, trend.by_day 포함)
          - daily_cards: (선택) 일간 카드 사전
        출력:
          - {"summary_message": "...", "status": "ok|error|fallback"}
        """
        if not isinstance(period, dict):
            return {"summary_message": "오류가 났어요. 나중에 다시 시도해주세요.", "status": "error"}

        meta = period.get("meta") or {}
        status = meta.get("status", "ok")

        # 1) A/B에서 에러 전파된 경우: LLM 호출 없이 안내문 반환
        if status != "ok":
            return {
                "summary_message": "질의를 이해할 수 없어 요약을 만들 수 없습니다. 기간을 다시 지정해주세요.",
                "status": "error"
            }

        # 2) 정상 케이스: LLM에 요약 요청 (JSON만 반환하도록 강제)
        #    빈 날은 trend.by_day에 score/top=None으로 들어있으니 모델 안내에 명시
        try:
            trend = period.get("trend", {}) or {}
            by_day = trend.get("by_day", []) or []
            notable = trend.get("notable_changes", []) or []
            avg = period.get("average_sentiment_score", None)
            dom = period.get("dominant_emotions") or []

            # 프롬프트(간단 버전): 반드시 JSON만, 스키마 고정
            prompt = f"""
                너는 사용자의 감정 기록을 친근하게 요약해주는 상담 챗봇이야.  
                너의 말투는 따뜻하고 대화하듯 자연스러워야 해.  
                딱딱한 리포트 대신, 사용자가 이해하기 쉽게 세 문장 이내로 정리해줘.  

                아래 데이터를 참고해서:
                1. 전체적인 분위기를 먼저 설명하고,  
                2. 특정 날짜에 감정 점수나 주감정이 크게 변한 부분이 있으면 언급하고,  
                3. 마지막에 "좋은 시간", "힘든 시간", "기복이 큰 기간" 같은 총평으로 마무리해.  

                ⚠️ 출력은 반드시 JSON 형식으로만 하세요.  
                {{
                "summary_message": "<너가 작성한 친근한 요약 문장>"
                }}
                [DATA]
                    - 평균 점수: {period.get("average_sentiment_score")}
                    - 우세 감정 분포: {period.get("dominant_emotions")}
                    - 주요 변화 이벤트: {period.get("trend", {}).get("notable_changes")}
                    - 일자별 기록: {period.get("trend", {}).get("by_day")}
                """

            # 호출부: self.model 또는 헬퍼 사용 (환경에 맞춰 한 줄 수정)
            # 예시1) raw_text = self.model.generate_content(prompt).text
            # 예시2) raw_text = self._call_gemini(prompt)
            raw_text = self.model.generate_content(prompt).text  # <- 환경에 맞게 조정

            parsed = self._extract_first_json(raw_text)
            if parsed and isinstance(parsed, dict) and "summary_message" in parsed:
                out = {"summary_message": parsed["summary_message"], "status": "ok"}
                if getattr(self, "debug", False):
                    out["raw"] = raw_text
                return out

            # 3) 파싱 실패: 짧은 보고 + 재시도 안내
            fb = self._fallback_brief(period)
            if getattr(self, "debug", False):
                fb["raw"] = raw_text
            return fb

        except Exception:
            # LLM 호출 자체가 실패한 경우도 동일한 Fallback
            return self._fallback_brief(period)




