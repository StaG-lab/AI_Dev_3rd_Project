# ./core/analyzer/gemini_sentiment_aggregator.py
import google.generativeai as genai
import json
from typing import List, Dict, Any, Optional
from core.utils.analysis_logger import AnalysisLogger # AnalysisLogger 임포트

class GeminiSentimentAggregator:
    """
    여러 발화 세그먼트의 감정 분석 결과를 Gemini API를 통해 종합 분석하고
    HTML 카드 형식에 맞춰 필요한 정보를 생성합니다.
    """
    def __init__(self, api_key: str, logger: Optional[AnalysisLogger] = None):
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            generation_config={"response_mime_type": "application/json"}
        )
        self.logger = logger
        self._log_info("GeminiSentimentAggregator 초기화 완료.")

    def _log_info(self, message: str, data: Optional[Dict[str, Any]] = None):
        if self.logger:
            self.logger.log_info(f"[GeminiSentimentAggregator] {message}", data)

    def _log_warning(self, message: str, data: Optional[Dict[str, Any]] = None):
        if self.logger:
            self.logger.log_warning(f"[GeminiSentimentAggregator] {message}", data)

    def _log_error(self, message: str, data: Optional[Dict[str, Any]] = None):
        if self.logger:
            self.logger.log_error(f"[GeminiSentimentAggregator] {message}", data)

    def aggregate_sentiment(self, segment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        다수의 발화 세그먼트에서 도출된 이미지, 음성, 텍스트 감정 분석 결과를 종합하여
        Gemini에 최종 감정을 질의하고 JSON 형태로 반환합니다.

        Args:
            segment_results (List[Dict[str, Any]]): 각 세그먼트의 상세 분석 결과 리스트.
                                                    예: [{'segment_id': 1, 'start_time': ..., 'end_time': ..., 'transcribed_text': '...', 'visual_analysis': {...}, 'audio_analysis': {...}}, ...]

        Returns:
            Dict[str, Any]: Gemini가 종합 분석한 최종 감정 결과.
                            예: {
                                "sentiment_score": 75,
                                "dominant_overall_emotion": "기쁨",
                                "overall_emotion_message": "오늘 하루는 밝고 희망찹니다!",
                                "emotion_distribution": [
                                    {"emotion": "기쁨", "percentage": "60%"},
                                    {"emotion": "중립", "percentage": "20%"},
                                    {"emotion": "불안", "percentage": "10%"}
                                ]
                            }
        """
        if not segment_results:
            self._log_warning("분석할 세그먼트가 없습니다. 기본 결과 반환.")
            return self._default_empty_result("분석할 세그먼트가 없습니다.")

        self._log_info(f"총 {len(segment_results)}개 세그먼트 결과 종합 분석 시작.")
        summarized_results_for_gemini = []
        for sr in segment_results:
            summary = {
                "segment_id": sr.get("segment_id"),
                "start_time": sr.get("start_time"),
                "end_time": sr.get("end_time"),
                "transcribed_text": sr.get("transcribed_text", ""),
                "visual_analysis_dominant_emotion": sr.get("visual_analysis", {}).get("dominant_emotion", "N/A"),
                "visual_analysis_distribution": sr.get("visual_analysis", {}).get("distribution", {}),
                "audio_analysis_text_sentiment": sr.get("audio_analysis", {}).get("text_based_analysis", {}).get("sentiment", {}),
                "audio_analysis_text_emotions": sr.get("audio_analysis", {}).get("text_based_analysis", {}).get("emotions", {}),
                "audio_analysis_voice_emotions": sr.get("audio_analysis", {}).get("voice_based_analysis", {}).get("distribution", {}),
            }
            summarized_results_for_gemini.append(summary)
        
        self.logger.save_intermediate_result("summarized_segment_results_for_gemini", summarized_results_for_gemini)
        
        # Gemini 프롬프트 구성
        # HTML 카드 형식에 필요한 정보를 명확히 요청합니다.
        prompt = f"""
        당신은 영상 콘텐츠의 시각적, 청각적, 텍스트 정보를 종합하여 사람의 감정을 분석하는 전문가입니다.
        다음은 비디오에서 추출된 여러 발화 세그먼트의 상세한 감정 분석 결과입니다.
        각 세그먼트에는 이미지(시각), 음성(청각), 텍스트(STT) 기반의 감정 분석 정보가 포함되어 있습니다.

        이 모든 정보를 종합하여 영상 속 인물의 '전체적인 감정 상태'를 판단하고,
        '감정 온도'(0-100), '가장 지배적인 전체 감정', '전반적인 감정 상태를 요약하는 메시지',
        그리고 '가장 두드러지는 상위 3개의 감정 분포(퍼센티지)'를 JSON 형식으로 생성해주세요.

        분석 결과 데이터 (요약):
        {json.dumps(summarized_results_for_gemini, indent=2, ensure_ascii=False)}

        최종 결과는 반드시 아래의 JSON 형식으로만 반환해야 합니다.
        "sentiment_score"는 0에서 100 사이의 숫자로, 매우 부정적일수록 0에 가깝고 매우 긍정적일수록 100에 가깝습니다.
        "dominant_overall_emotion"은 전체 영상에서 가장 지배적인 감정을 한국어로 표현합니다 (예: "기쁨", "슬픔", "중립").
        "overall_emotion_message"는 현재 감정 온도와 어울리는 짧고 긍정적이거나 중립적인 한글 메시지입니다 (예: "오늘 하루는 밝고 희망찹니다!").
        "emotion_distribution"은 상위 3개의 주요 감정과 해당 퍼센티지를 포함하는 리스트입니다.
        
        JSON 형식:
        {{
          "sentiment_score": int, // 감정 온도 (0-100)
          "dominant_overall_emotion": "string", // 예: "기쁨"
          "overall_emotion_message": "string", // 예: "오늘 하루는 밝고 희망찹니다!"
          "overall_emotion_icon": "string", // 예: ☀️, 감정 온도를 대표할 수 있는 날씨 이모지를 사용하세요.
          "emotion_distribution": [
            {{"icon":"string", "emotion": "string", "percentage": "string"}}, // 예: {{"icon": "😊", "emotion": "기쁨", "percentage": "60%"}}
            {{"icon":"string", "emotion": "string", "percentage": "string"}},
            {{"icon":"string", "emotion": "string", "percentage": "string"}}
          ]
        }}
        """
        try:
            print("Gemini API에 종합 감정 분석 요청 중...")
            response = self.gemini_model.generate_content(prompt)
            json_response = json.loads(response.text)
            
            # 응답 스키마 검증 및 누락된 필드 기본값 처리
            final_result = {
                "sentiment_score": json_response.get("sentiment_score", 50),
                "dominant_overall_emotion": json_response.get("dominant_overall_emotion", "중립"),
                "overall_emotion_message": json_response.get("overall_emotion_message", "현재 감정은 중립적입니다."),
                "overall_emotion_icon": json_response.get("overall_emotion_icon", "☀️"),
                "emotion_distribution": json_response.get("emotion_distribution", [])
            }
            
            if not isinstance(final_result["emotion_distribution"], list):
                self._log_warning("Gemini 응답의 'emotion_distribution'이 리스트 형식이 아닙니다. 빈 리스트로 초기화합니다.", {"received_type": type(final_result["emotion_distribution"])})
                final_result["emotion_distribution"] = []
            
            final_result["emotion_distribution"] = final_result["emotion_distribution"][:3]

            for i, item in enumerate(final_result["emotion_distribution"]):
                if not isinstance(item, dict) or "emotion" not in item or "percentage" not in item:
                    self._log_warning(f"emotion_distribution 항목 {i}의 형식이 올바르지 않습니다. 기본값으로 대체합니다.", {"item": item})
                    final_result["emotion_distribution"][i] = {"emotion": "알 수 없음", "percentage": "0%"}
                elif not item["percentage"].endswith("%"):
                    try:
                        percent_value = float(item['percentage'].replace('%', ''))
                        if percent_value <= 1.0: 
                            item['percentage'] = f"{int(percent_value * 100)}%"
                        else: 
                            item['percentage'] = f"{int(percent_value)}%"
                    except ValueError:
                        self._log_warning(f"emotion_distribution 항목 {i}의 퍼센티지 변환 실패. 기본값 '0%'로 설정.", {"percentage_value": item['percentage']})
                        item['percentage'] = "0%" 
            
            while len(final_result["emotion_distribution"]) < 3:
                final_result["emotion_distribution"].append({"emotion": "N/A", "percentage": "0%"})

            self._log_info("Gemini API 응답 수신 및 파싱 완료.", {"final_aggregated_result": final_result})
            self.logger.save_intermediate_result("gemini_parsed_aggregation_result", final_result)
            return final_result

        except (json.JSONDecodeError, KeyError, Exception) as e:
            self._log_error(f"Gemini 종합 분석 응답 처리 중 에러 발생: {e}. 원본 응답 텍스트: {response.text if 'response' in locals() else 'N/A'}")
            return self._default_empty_result(f"Gemini API 에러: {e}")

    def _default_empty_result(self, error_message: str = "분석 데이터를 찾을 수 없음.") -> Dict[str, Any]:
        """분석 실패 또는 데이터 부재 시 반환할 기본 결과 구조."""
        return {
            "sentiment_score": 50,
            "dominant_overall_emotion": "중립",
            "overall_emotion_message": f"감정 분석에 실패했습니다. {error_message}",
            "overall_emotion_icon": "🎃",
            "emotion_distribution": [
                {"emotion": "중립", "percentage": "100%"},
                {"emotion": "N/A", "percentage": "0%"},
                {"emotion": "N/A", "percentage": "0%"}
            ],
            "error": error_message
        }
