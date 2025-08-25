# /analyzer.py

import os
from core.analyzer.video_analyzer import VideoAnalyzer
import json
import argparse
from datetime import datetime

if __name__ == "__main__":
    # Parse arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default="./datasets/video/video_3.mp4", help='분석할 동영상 파일 경로를 입력하세요.')
    args = parser.parse_args()

    # --- 설정 ---
    VIDEO_FILE_PATH = args.video_path # 분석할 동영상 파일 경로
    IMAGE_MODEL_WEIGHTS = "infrastructure/models/weights/checkpoints/emonet_50_2_percent_trained_2.pth" # 훈련된 이미지 모델 가중치
    # .ignore폴더의 API.json에서 API_CHATGPT 키를 가져옵니다.
    with open(".ignore/API.json", "r") as f:
        api_info = json.load(f)
    API_KEY = api_info.get("API_GEMINI")

    # --- 분석기 생성 ---
    # voice_model_name을 "distilhubert"로 변경하여 다른 모델 테스트 가능
    analyzer = VideoAnalyzer(
        image_model_name="emonet",
        image_model_weights_path=IMAGE_MODEL_WEIGHTS,
        api_key=API_KEY['key'],
        voice_model_name="wav2vec2"  # "wav2vec2", "hubert-base", "hubert-large", "hubert-xlarge"
    )
    
    # --- 분석 실행 ---
    result = analyzer.analyze(VIDEO_FILE_PATH)
    
    # --- 결과 출력 ---
    print("\n\n--- 최종 분석 결과 ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # analysis_result.json 파일명에 분석 날짜 시간 추가
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"analysis_result_{timestamp}.json"
    with open(output_filename, "w", encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)