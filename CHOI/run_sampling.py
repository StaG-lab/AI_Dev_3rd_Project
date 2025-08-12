# /run_sampling.py

from pathlib import Path
from infrastructure.data.data_manager import sample_dataset

if __name__ == "__main__":
    # -----------------------------------------------------------------
    # ❗️ 중요: 아래 경로를 자신의 실제 데이터 위치에 맞게 수정해주세요.
    # -----------------------------------------------------------------
    # [cite_start]'데이터 수집 계획서'에 명시된 "한국인 감정인식을 위한 복합 영상" 데이터를 사용합니다[cite: 18].
    
    # 예시: D드라이브 datasets 폴더에 데이터가 있는 경우
    BASE_PATH = Path("D:/Work_Dev/AI-Dev/Projects/datasets/korean_emotion_complex_vision")
    
    # 실제 원본 이미지와 라벨 폴더 경로
    # AI Hub 데이터 구조에 따라 경로가 다를 수 있으니 확인 후 수정 필요
    SOURCE_IMAGE_DIR = BASE_PATH / "Training" / "images"
    SOURCE_LABEL_DIR = BASE_PATH / "Training" / "labels"

    # -----------------------------------------------------------------
    # 샘플링된 데이터가 저장될 위치
    OUTPUT_DIR = Path("./datasets/korean_emotion_complex_vision_1_percent")
    
    # 샘플링 비율 (1%)
    SAMPLE_RATE = 0.01
    
    # ✅ 데이터셋에 맞는 모드를 명시적으로 지정
    DATA_STRUCTURE_MODE = 'file_per_emotion'
    VAL_SPLIT_RATIO = 0.2  # 검증 데이터 비율 (20%)
    print("="*50)
    print(f"샘플링을 시작합니다.")
    print(f"  - 원본 이미지: {SOURCE_IMAGE_DIR}")
    print(f"  - 원본 라벨: {SOURCE_LABEL_DIR}")
    print(f"  - 결과물 저장 위치: {OUTPUT_DIR}")
    print(f"  - 샘플링 비율: {SAMPLE_RATE * 100}%")
    print("="*50)

    sample_dataset(
        source_image_dir=SOURCE_IMAGE_DIR,
        source_label_dir=SOURCE_LABEL_DIR,
        output_dir=OUTPUT_DIR,
        sample_rate=SAMPLE_RATE,
        mode=DATA_STRUCTURE_MODE,
        val_split_ratio=VAL_SPLIT_RATIO
    )