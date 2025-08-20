# /preprocess_images_hybrid.py

import json
from pathlib import Path
from PIL import Image, ImageOps
import cv2
import numpy as np
from tqdm import tqdm

def run_hybrid_preprocessing(source_data_dir: Path, dest_data_dir: Path, final_size=224):
    
    # --- 1. 모델 로드 ---
    # 모델 1: Haar Cascade
    haar_path = Path("./infrastructure/models/haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(str(haar_path))

    # 모델 2: DNN Face Detector
    proto_path = Path("./infrastructure/models/deploy.prototxt")
    model_path = Path("./infrastructure/models/res10_300x300_ssd_iter_140000.caffemodel")
    dnn_net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
    
    # 원본 라벨 파일 로드
    source_label_dir = source_data_dir / "labels"
    all_labels = {}
    if not source_label_dir.exists():
        print(f"[오류] 라벨 디렉토리를 찾을 수 없습니다: {source_label_dir}")
        return
    
    for label_file in source_label_dir.glob("*_sampled.json"):
        with open(label_file, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                all_labels[item['filename']] = item

    # 이미지 파일 순회 및 처리
    image_files = list(source_data_dir.glob('**/*.*')) # 모든 이미지 확장자 처리
    failed = 0
    for img_path in tqdm(image_files, desc=f"Processing {source_data_dir.name}"):
        try:
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            # --- 이미지 처리 ---
            img = Image.open(img_path)
            
            # EXIF 정보 기반으로 이미지 자동 회전
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            
            # OpenCV에서 사용하기 위해 PIL 이미지를 numpy 배열로 변환
            cv_img = np.array(img)
            
            # --- 얼굴 탐지 (하이브리드 방식) ---
            faces = []
            
            # 단계 1: 튜닝된 Haar Cascade로 먼저 시도
            gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
            detected_faces_haar = face_cascade.detectMultiScale(gray, scaleFactor=1.42, minNeighbors=9, minSize=(350, 350))
            if len(detected_faces_haar) > 0:
                faces = detected_faces_haar

            # 단계 2: Haar가 실패하면 DNN 모델로 재시도
            if len(faces) == 0:
                failed += 1
                h, w = cv_img.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(cv_img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                dnn_net.setInput(blob)
                detections = dnn_net.forward()
                
                # 가장 신뢰도 높은 결과 하나를 선택
                best_detection_idx = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, best_detection_idx, 2]
                
                if confidence > 0.5: # 신뢰도가 50% 이상인 경우에만 인정
                    box = detections[0, 0, best_detection_idx, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    faces = np.array([[startX, startY, endX - startX, endY - startY]])
            
            if len(faces) > 0:
                # 문제 1 해결: 얼굴을 중심으로 자르기
                x, y, w, h = faces[0] # 가장 큰 얼굴 하나만 사용
                
                # 얼굴 영역을 정사각형으로 만들고, 약간의 여백(padding) 추가
                center_x, center_y = x + w // 2, y + h // 2
                size = max(w, h)
                padding = int(size * 0.4)
                size += padding
                
                crop_x1 = max(0, center_x - size // 2)
                crop_y1 = max(0, center_y - size // 2)
                crop_x2 = min(img.width, center_x + size // 2)
                crop_y2 = min(img.height, center_y + size // 2)
                
                img_cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            else:
                # 얼굴 탐지 실패 시, 중앙을 자르는 예전 방식으로 대체
                side = min(img.width, img.height)
                img_cropped = img.crop(((img.width - side) // 2, (img.height - side) // 2,
                                        (img.width + side) // 2, (img.height + side) // 2))
                crop_x1, crop_y1 = (img.width - side) // 2, (img.height - side) // 2

            # 최종 크기로 리사이즈
            img_resized = img_cropped.resize((final_size, final_size), Image.LANCZOS)
            
            # --- 라벨 처리 ---
            label_info = all_labels.get(img_path.name)
            if not label_info: continue

            # 문제 3 해결: 바운딩 박스 좌표 변환
            box = label_info['annot_A']['boxes'] # annot_A를 기준으로 함
            orig_box = [box['minX'], box['minY'], box['maxX'], box['maxY']]
            
            # 1. Crop에 맞춰 좌표 이동
            new_box_x1 = orig_box[0] - crop_x1
            new_box_y1 = orig_box[1] - crop_y1
            new_box_x2 = orig_box[2] - crop_x1
            new_box_y2 = orig_box[3] - crop_y1

            # 2. Resize에 맞춰 좌표 스케일링
            scale = final_size / img_cropped.width
            final_box = [coord * scale for coord in [new_box_x1, new_box_y1, new_box_x2, new_box_y2]]
            
            # --- 저장 ---
            # 이미지 저장
            relative_path = img_path.relative_to(source_data_dir)
            dest_img_path = (dest_data_dir / relative_path).with_suffix('.jpg')
            dest_img_path.parent.mkdir(parents=True, exist_ok=True)
            img_resized.save(dest_img_path, 'JPEG', quality=95)

            # 변환된 라벨 저장
            processed_label = {
                'emotion': label_info['faceExp_uploader'],
                'bbox_224px': final_box
            }
            dest_label_path = dest_img_path.with_suffix('.json')
            with open(dest_label_path, 'w', encoding='utf-8') as f:
                json.dump(processed_label, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"처리 중 에러 발생 {img_path}: {e}")
    print(f"처리 완료. Haar Cascade 실패한 이미지 수: {failed}")

if __name__ == "__main__":
    SOURCE_DATA_DIR_ROOT = Path("./datasets/korean_emotion_complex_vision_1_percent")
    DEST_DATA_DIR_ROOT = Path("./datasets/korean_emotion_complex_vision_1_percent_hybrid_processed")
    
    run_hybrid_preprocessing(SOURCE_DATA_DIR_ROOT / "train", DEST_DATA_DIR_ROOT / "train")
    run_hybrid_preprocessing(SOURCE_DATA_DIR_ROOT / "val", DEST_DATA_DIR_ROOT / "val")

    print("Hybrid preprocessing complete.")