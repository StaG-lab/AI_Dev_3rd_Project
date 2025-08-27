# /analyzer.py

import os
from core.analyzer.video_analyzer import BatchVideoAnalyzer
from core.analyzer.gemini_sentiment_aggregator import GeminiSentimentAggregator 
from core.renderer.result_renderer import ResultRenderer
from core.utils.analysis_logger import AnalysisLogger # AnalysisLogger 임포트

import json
import argparse
from datetime import datetime
from pathlib import Path

if __name__ == "__main__":
    # Parse arguments 
    default_video_path = r"D:\Work_Dev\AI-Dev\Projects\datasets\video\video (5).mp4"
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default=default_video_path, help='분석할 동영상 파일 경로를 입력하세요.')
    parser.add_argument('--voice_model', type=str, default="wav2vec2", 
                        choices=["wav2vec2", "hubert-base", "wav2vec2_autumn"],
                        help='음성 감정 분석에 사용할 모델을 선택하세요. (wav2vec2, hubert-base, wav2vec2_autumn)')
    parser.add_argument('--min_speech_segment_duration', type=float, default=5.0,
                        help='최소 발화 세그먼트 지속 시간 (초). 이보다 짧은 세그먼트는 인접 세그먼트와 병합됩니다.')
    args = parser.parse_args()

    # --- AnalysisLogger 초기화 ---
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_log_filename = f"./logs/detailed_analysis_log_{current_timestamp}.json"
    analysis_logger = AnalysisLogger()
    analysis_logger.log_info(f"분석 시작: {datetime.now().isoformat()}", {"arguments": vars(args)})

    # --- 설정 ---
    VIDEO_FILE_PATH = args.video_path # 분석할 동영상 파일 경로
    IMAGE_MODEL_WEIGHTS = "infrastructure/models/emonet_100_2_trained.pth" # 훈련된 이미지 모델 가중치

    # API 키 로드
    GEMINI_API_KEY = None
    try:
        with open(".ignore/API.json", "r") as f:
            api_info = json.load(f)
        api_key_gemini = api_info.get("API_GEMINI")
        if not api_key_gemini or not api_key_gemini.get('key'):
            raise ValueError("API_GEMINI 키가 API.json에 없거나 유효하지 않습니다.")
        GEMINI_API_KEY = api_key_gemini['key'] # 실제 API 키 값만 사용
    except FileNotFoundError:
        print("오류: .ignore/API.json 파일을 찾을 수 없습니다. Gemini API 키를 설정해주세요.")
        exit(1)
    except json.JSONDecodeError:
        print("오류: API.json 파일 형식이 올바르지 않습니다.")
        exit(1)
    except ValueError as e:
        print(f"오류: {e}")
        exit(1)

    # API 키 로드
    GEMINI_API_KEY = None
    try:
        analysis_logger.log_info("API 키 로드 시도.")
        with open(".ignore/API.json", "r") as f:
            api_info = json.load(f)
        api_key_gemini = api_info.get("API_GEMINI")
        if not api_key_gemini or not api_key_gemini.get('key'):
            raise ValueError("API_GEMINI 키가 API.json에 없거나 유효하지 않습니다.")
        GEMINI_API_KEY = api_key_gemini['key']
        analysis_logger.log_info("Gemini API 키 로드 성공.")
    except FileNotFoundError:
        error_msg = "오류: .ignore/API.json 파일을 찾을 수 없습니다. Gemini API 키를 설정해주세요."
        analysis_logger.log_error(error_msg)
        print(error_msg)
        exit(1)
    except json.JSONDecodeError:
        error_msg = "오류: API.json 파일 형식이 올바르지 않습니다."
        analysis_logger.log_error(error_msg)
        print(error_msg)
        exit(1)
    except ValueError as e:
        error_msg = f"오류: {e}"
        analysis_logger.log_error(error_msg)
        print(error_msg)
        exit(1)

    # --- 분석기 생성 ---
    analysis_logger.log_info("BatchVideoAnalyzer 인스턴스 생성 시작.")
    batch_analyzer = BatchVideoAnalyzer(
        image_model_name="emonet",
        image_model_weights_path=IMAGE_MODEL_WEIGHTS,
        api_key=GEMINI_API_KEY,
        voice_model_name=args.voice_model,
        min_speech_segment_duration=args.min_speech_segment_duration,
        logger=analysis_logger # AnalysisLogger 인스턴스 전달
    )
    analysis_logger.log_info("BatchVideoAnalyzer 인스턴스 생성 완료.")
    
    # --- 분석 실행 (세그먼트별 결과 획득) ---
    print("\n--- 비디오 분석 시작 (세그먼트별) ---")
    analysis_logger.log_info(f"비디오 파일 '{VIDEO_FILE_PATH}' 분석 시작.")
    analysis_results_from_segments = batch_analyzer.analyze(VIDEO_FILE_PATH)
    analysis_logger.log_info("비디오 분석 완료.", {"results_summary": analysis_results_from_segments.get("total_segments", 0)})
    analysis_logger.save_intermediate_result("batch_video_analysis_full_results", analysis_results_from_segments)
    
    # --- Gemini를 이용한 최종 감정 종합 분석 ---
    print("\n--- Gemini를 이용한 최종 감정 종합 분석 시작 ---")
    analysis_logger.log_info("GeminiSentimentAggregator 인스턴스 생성 시작.")
    gemini_aggregator = GeminiSentimentAggregator(api_key=GEMINI_API_KEY, logger=analysis_logger) # AnalysisLogger 인스턴스 전달
    analysis_logger.log_info("GeminiSentimentAggregator 인스턴스 생성 완료.")
    
    analysis_logger.log_info("세그먼트 분석 결과를 Gemini로 전달하여 최종 종합 감정 분석 요청.")
    final_aggregated_sentiment = gemini_aggregator.aggregate_sentiment(
        analysis_results_from_segments.get("segment_analyses", [])
    )
    analysis_logger.log_info("Gemini 최종 감정 종합 분석 완료.", {"final_sentiment_score": final_aggregated_sentiment.get("sentiment_score")})
    analysis_logger.save_intermediate_result("final_aggregated_sentiment_result", final_aggregated_sentiment)


    # --- 최종 종합 결과 출력 (JSON) ---
    print("\n\n--- 최종 종합 분석 결과 (JSON) ---")
    print(json.dumps(final_aggregated_sentiment, indent=2, ensure_ascii=False))

    json_output_filename = f"./reports/analysis/analysis_result_{current_timestamp}.json"
    with open(json_output_filename, "w", encoding='utf-8') as f:
        json.dump(final_aggregated_sentiment, f, ensure_ascii=False, indent=2)
    print(f"최종 종합 분석 결과가 '{json_output_filename}'에 저장되었습니다.")
    analysis_logger.log_info(f"최종 종합 분석 결과 JSON 파일 저장 완료: '{json_output_filename}'")

    # --- HTML 카드 렌더링 및 저장 ---
    print("\n--- HTML 카드 렌더링 시작 ---")
    analysis_logger.log_info("HTML 카드 렌더링 시작.")
    html_template_path = "./templates/card_template_02.html"
    
    # card_design.html 파일이 현재 스크립트와 동일한 경로에 있는지 확인하고 상대 경로를 사용
    current_script_dir = Path(__file__).parent
    full_html_template_path = current_script_dir / html_template_path

    # Jinja2 템플릿 로드를 위해 템플릿 파일의 디렉토리와 파일명을 분리하여 ResultRenderer에 전달
    template_dir = str(full_html_template_path.parent)
    template_filename = full_html_template_path.name

    result_renderer = ResultRenderer(template_dir=template_dir, template_filename=template_filename)
    rendered_html_content = result_renderer.render(final_aggregated_sentiment)
    
    html_output_filename = f"./result_cards/emotion_card_{current_timestamp}.html"
    with open(html_output_filename, "w", encoding='utf-8') as f:
        f.write(rendered_html_content)
    analysis_logger.log_info(f"HTML 감정 카드 파일 저장 완료: '{html_output_filename}'")

    # --- 상세 로그 및 중간 결과 파일 저장 ---
    analysis_logger.log_info("모든 상세 로그 및 중간 결과 JSON 파일 저장 시작.")
    analysis_logger.save_to_file(detailed_log_filename)
    print(f"\n모든 상세 로그 및 중간 결과는 '{detailed_log_filename}'에 저장되었습니다.")
    analysis_logger.log_info("모든 분석 및 로깅 프로세스 완료.")
