# ./core/renderer/result_renderer.py

from jinja2 import Environment, FileSystemLoader
from typing import Dict, Any
from pathlib import Path

class ResultRenderer:
    """
    GeminiSentimentAggregator로부터 받은 감정 데이터를 HTML 템플릿에 Jinja2를 사용하여 렌더링합니다.
    """
    def __init__(self, template_dir: str, template_filename: str):
        self.template_dir = template_dir
        self.template_filename = template_filename
        
        try:
            # Jinja2 환경 설정
            self.env = Environment(loader=FileSystemLoader(self.template_dir), autoescape=True)
            self.template = self.env.get_template(self.template_filename)
        except Exception as e:
            print(f"오류: Jinja2 템플릿 로드 중 예외 발생: {e}")
            self.template = None # 템플릿 로드 실패 시 None으로 설정

    def render(self, data: Dict[str, Any]) -> str:
        """
        Gemini로부터 받은 데이터를 HTML 템플릿에 Jinja2를 사용하여 렌더링합니다.

        Args:
            data (Dict[str, Any]): GeminiSentimentAggregator로부터 받은 종합 감정 분석 결과.
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

        Returns:
            str: 동적 데이터가 채워진 최종 HTML 문자열.
        """
        if self.template is None:
            return "<p>Error: HTML template was not loaded successfully.</p>"

        try:
            # Jinja2 템플릿 렌더링
            return self.template.render(data)
        except Exception as e:
            print(f"오류: Jinja2 템플릿 렌더링 중 예외 발생: {e}")
            return f"<p>Error rendering HTML template: {e}</p>"