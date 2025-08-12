# /preprocess_images.py (디버깅 버전)

import json
from pathlib import Path
from PIL import Image, ImageOps
import cv2
import numpy as np
from tqdm import tqdm

def run_intelligent_preprocessing(source_data_dir: Path, dest_data_dir: Path, final_size=224):
    
    face_cascade_path = Path("./infrastructure/models/haarcascade_frontalface_default.xml")
    if not face_cascade_path.exists():
        raise FileNotFoundError(f"Haar Cascade 파일을 찾을 수 없습니다: {face_cascade_path}")
    face_cascade = cv2.CascadeClassifier(str(face_cascade_path))

    # --- [디버깅 1] 라벨 파일 로드 확인 ---
    source_label_dir = source_data_dir.parent.parent / "labels" # 샘플링된 원본 라벨 폴더 (경로 수정)
    all_labels = {}
    for label_file in source_label_dir.glob("*_sampled.json"):
        with open(label_file, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                all_labels[item['filename']] = item
                
    print(f"[디버그] 총 {len(all_labels)}개의 라벨 정보를 로드했습니다.")
    if all_labels:
        print(f"[디버그] 로드된 라벨 Key 예시: {next(iter(all_labels.keys()))}")
    # ------------------------------------

    image_files = [p for p in source_data_dir.glob('**/*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    # --- [디버깅 2] 처리된 파일 카운터 ---
    processed_count = 0
    # ------------------------------------

    for img_path in tqdm(image_files, desc=f"Processing {source_data_dir.name}"):
        try:
            # --- [디버깅 3] 파일명 매칭 확인 ---
            label_info = all_labels.get(img_path.name)
            if not label_info:
                # 라벨을 찾지 못했다면, 어떤 파일인지 출력하고 다음으로 넘어감
                # print(f"[디버그] 라벨 매칭 실패: {img_path.name}") # 너무 많이 출력될 수 있어 일단 주석 처리
                continue
            # ------------------------------------

            # --- 이미지 및 라벨 처리 (이전과 동일) ---
            # ... (얼굴 탐지, crop, resize, 좌표 변환 로직) ...
            
            # --- 저장 ---
            # ... (파일 저장 로직) ...
            
            # --- [디버깅 4] 처리 성공 카운트 증가 ---
            processed_count += 1
            # ------------------------------------

        except Exception as e:
            print(f"처리 중 에러 발생 {img_path}: {e}")

    # --- [디버깅 5] 최종 결과 출력 ---
    print(f"총 {len(image_files)}개의 이미지 파일 중, {processed_count}개가 성공적으로 처리 및 저장되었습니다.")
    # ------------------------------------


if __name__ == "__main__":
    # 경로를 샘플링된 원본 폴더로 다시 확인
    SOURCE_DATA_DIR_ROOT = Path("./datasets/korean_emotion_complex_vision_1_percent") 
    DEST_DATA_DIR_ROOT = Path("./datasets/korean_emotion_complex_vision_1_percent_intelligent_processed")
    
    # 경로 재확인 및 수정
    train_source_dir = SOURCE_DATA_DIR_ROOT / "train"
    val_source_dir = SOURCE_DATA_DIR_ROOT / "val"

    if not train_source_dir.exists() or not val_source_dir.exists():
        print(f"[오류] 원본 데이터 폴더를 찾을 수 없습니다: {train_source_dir} 또는 {val_source_dir}")
    else:
        run_intelligent_preprocessing(train_source_dir, DEST_DATA_DIR_ROOT / "train")
        run_intelligent_preprocessing(val_source_dir, DEST_DATA_DIR_ROOT / "val")
        print("Intelligent preprocessing complete.")