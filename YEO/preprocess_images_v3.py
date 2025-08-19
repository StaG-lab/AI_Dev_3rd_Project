# /preprocess_images_verified.py

import json
from pathlib import Path
from PIL import Image, ImageOps
import cv2
import numpy as np
from tqdm import tqdm
from collections import Counter
import shutil

def _majority_label(item: dict):
    """annot_A/B/C.faceExp 다수결; 3/3 또는 2/1만 유효. 1/1/1은 (None,1,[]) 리턴"""
    labels, who = [], []
    for k in ("annot_A","annot_B","annot_C"):
        lab = (item.get(k) or {}).get("faceExp")
        if lab is not None:
            labels.append(lab); who.append(k)
    cnt = Counter(labels)
    if not cnt:
        return None, 0, []
    lab, n = cnt.most_common(1)[0]
    if n == 1:  # 1/1/1
        return None, 1, []
    voters = [k for k in who if (item.get(k) or {}).get("faceExp") == lab]
    return lab, n, voters  # n in {2,3}

# def _merge_bbox_from_voters(item: dict, voters: list):
#     """다수결에 참여한 annot의 bbox 평균. 결측이면 None."""
#     xs, ys, Xs, Ys = [], [], [], []
#     for k in voters:
#         b = (item.get(k) or {}).get("boxes") or {}
#         xs.append(b.get("minX")); ys.append(b.get("minY"))
#         Xs.append(b.get("maxX")); Ys.append(b.get("maxY"))
#     xs = [v for v in xs if v is not None]
#     ys = [v for v in ys if v is not None]
#     Xs = [v for v in Xs if v is not None]
#     Ys = [v for v in Ys if v is not None]
#     if not xs or not ys or not Xs or not Ys:
#         return None
#     return {
#         "minX": sum(xs)/len(xs), "minY": sum(ys)/len(ys),
#         "maxX": sum(Xs)/len(Xs), "maxY": sum(Ys)/len(Ys)
#     }

def run_verified_preprocessing(source_data_dir: Path, dest_data_dir: Path, final_size=224, confidence_threshold=0.7):
    split_name = source_data_dir.name
    # [v3] 라벨 불합의(1/1/1) 보관 폴더: <dest_root>/<split>_disagreed/<원본감정>
    disagreed_root = dest_data_dir.parent / f"{dest_data_dir.name}_disagreed"
    disagreed_root.mkdir(parents=True, exist_ok=True)
    
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
    
    # 검증 실패 이미지를 보관할 폴더 경로 설정
    discard_dir = dest_data_dir.parent / (dest_data_dir.name + "_discarded")
    discard_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    discarded_count = 0  # 검증 실패로 폐기된 이미지 카운터
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
            
            """
            기쁨_1 오분류 33개 scaleFactor=1.1, minNeighbors=5, minSize=(200, 200)
기쁨_2 오분류 81개 scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
기쁨_3 오분류 24개 scaleFactor=1.1, minNeighbors=5, minSize=(250, 250)
기쁨_4 오분류 17개 scaleFactor=1.1, minNeighbors=5, minSize=(300, 300)
기쁨_5 오분류 12개 scaleFactor=1.1, minNeighbors=5, minSize=(350, 350)
기쁨_6 오분류 12개 scaleFactor=1.1, minNeighbors=5, minSize=(400, 400)
기쁨_7 오분류 11개 scaleFactor=1.1, minNeighbors=6, minSize=(350, 350)
기쁨_8 오분류 10개 scaleFactor=1.1, minNeighbors=7, minSize=(350, 350)
기쁨_9 오분류 7개 scaleFactor=1.1, minNeighbors=8, minSize=(350, 350)
기쁨_10 오분류 6개 scaleFactor=1.1, minNeighbors=9, minSize=(350, 350)
기쁨_11 오분류 7개 scaleFactor=1.1, minNeighbors=10, minSize=(350, 350)
기쁨_12 오분류 8개 scaleFactor=1.09, minNeighbors=9, minSize=(350, 350)
기쁨_13 오분류 13개 scaleFactor=1.08, minNeighbors=9, minSize=(350, 350)
기쁨_14 오분류 23개 scaleFactor=1.05, minNeighbors=9, minSize=(350, 350)
---------------
총합_15 총오분류 9개 scaleFactor=1.2, minNeighbors=9, minSize=(350, 350) 
기쁨:0, 당황:2, 분노:0, 불안:1, 상처:1, 슬픔:4, 중립:1

총합_16 오분류 3개 scaleFactor=1.3, minNeighbors=9, minSize=(350, 350)
기쁨:1, 당황:0, 분노:0, 불안:0, 상처:0, 슬픔:2, 중립:0

총합_17 오분류 1개 scaleFactor=1.4, minNeighbors=9, minSize=(350, 350)
기쁨:0, 당황:0, 분노:0, 불안:0, 상처:0, 슬픔:1, 중립:0

총합_18 오분류 2개 scaleFactor=1.5, minNeighbors=9, minSize=(350, 350)
기쁨:0, 당황:1, 분노:0, 불안:0, 상처:0, 슬픔:1, 중립:0

총합_19 오분류 1개 scaleFactor=1.41, minNeighbors=9, minSize=(350, 350)
기쁨:0, 당황0:, 분노:0, 불안:0, 상처:0, 슬픔1:, 중립:0

총합_20 오분류 1개 scaleFactor=1.42, minNeighbors=9, minSize=(350, 350)   ------------------------------------선택
기쁨:0, 당황0:, 분노:0, 불안:0, 상처:0, 슬픔1:, 중립:0 (이전보다 Haar Cascade 실패횟수가 10회 줄어듬)
validate 사진에도 오분류 0개

총합_21 오분류 1개 scaleFactor=1.43, minNeighbors=9, minSize=(350, 350)
기쁨:0, 당황0:, 분노:0, 불안:0, 상처:0, 슬픔1:, 중립:0 (이전보다 Haar Cascade 실패횟수가 2회 줄어듬)
validate 사진(당황)에 1장의 추가 오분류가 있었음.

총합_22 오분류 1개 scaleFactor=1.44, minNeighbors=9, minSize=(350, 350)
기쁨:0, 당황0:, 분노:0, 불안:0, 상처:0, 슬픔1:, 중립:0 (이전보다 Haar Cascade 실패횟수가 25회 늘어남)



            
            """
            
            if len(detected_faces_haar) > 0:
                faces = detected_faces_haar

            # 단계 2: Haar가 실패하면 DNN 모델로 재시도
            if len(faces) == 0:
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
                        
            # [v3] 어노테이터 다수결(업로더 무시). 1/1/1은 disagreed로 보관 후 continue
            orig_emotion = img_path.parent.name  # 원본 폴더(업로더 기준)
            maj, agree_n, voters = _majority_label(label_info)
            if agree_n == 1:
                # 1/1/1 → <dest>_disagreed/<split>/<원본폴더>/ 로 원본과 1:1 json 함께 보관
                disagreed_dir = disagreed_root / orig_emotion
                disagreed_dir.mkdir(parents=True, exist_ok=True)
                # 원본 이미지/라벨 1:1 복사(라벨 파일이 있을 때만)
                src_img = img_path
                src_json = img_path.with_suffix(".json")
                shutil.copy2(src_img, disagreed_dir / src_img.name)
                if src_json.exists():
                    shutil.copy2(src_json, (disagreed_dir / src_img.name).with_suffix(".json"))
                # 메타 로그(선택): disagreed_dir/<filename>.v3meta.json
                with open((disagreed_dir / src_img.stem).with_suffix(".v3meta.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "reason": "annot_all_diff(1/1/1)",
                        "from_json_emotion": orig_emotion,
                        "uploader": label_info.get("faceExp_uploader"),
                        "A": (label_info.get("annot_A") or {}).get("faceExp"),
                        "B": (label_info.get("annot_B") or {}).get("faceExp"),
                        "C": (label_info.get("annot_C") or {}).get("faceExp"),
                    }, f, ensure_ascii=False, indent=2)
                continue
            final_emotion = maj if maj is not None else label_info.get("faceExp_uploader")

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
            
            # --- [DNN 기반 최종 검증 단계] ---
            # 최종 이미지를 DNN 모델 입력 형식으로 변환
            # (Pillow 이미지를 OpenCV BGR 형식으로 변환)
            final_cv_img_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

            # 검증을 위해 DNN 모델이 기대하는 300x300 크기로 임시 리사이즈
            resized_for_dnn = cv2.resize(final_cv_img_bgr, (300, 300))
            # 300x300 이미지로 blob 생성
            blob = cv2.dnn.blobFromImage(resized_for_dnn, 1.0, (300, 300), (104.0, 177.0, 123.0))

            dnn_net.setInput(blob)
            detections = dnn_net.forward()
            
            # 가장 높은 신뢰도를 찾음
            confidence = detections[0, 0, 0, 2]

            # 신뢰도가 설정된 임계값 이상인 경우에만 저장
            if confidence > confidence_threshold:
                # 이미지 저장
                relative_path = img_path.relative_to(source_data_dir)
                # dest_img_path = (dest_data_dir / relative_path).with_suffix('.jpg')
                # dest_img_path.parent.mkdir(parents=True, exist_ok=True)
                # [v3] 저장 폴더를 다수결 감정으로 강제(업로더/원본 폴더 무시)
                dest_img_path = (dest_data_dir / final_emotion / img_path.name).with_suffix(".jpg")
                dest_img_path.parent.mkdir(parents=True, exist_ok=True)
                img_resized.save(dest_img_path, 'JPEG', quality=95)

                # 변환된 라벨 저장
                processed_label = {
                    "emotion": final_emotion,
                    "agree_n": int(agree_n),
                    "from_json_emotion": orig_emotion
                }
                dest_label_path = dest_img_path.with_suffix('.json')
                with open(dest_label_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_label, f, ensure_ascii=False, indent=2)
                processed_count += 1
            else:
                # --- [검증 실패 시 처리] ---
                # 폐기하지 않고 별도의 폴더에 이미지만 저장
                relative_path = img_path.relative_to(source_data_dir)
                discard_img_path = (discard_dir / relative_path).with_suffix('.jpg')
                discard_img_path.parent.mkdir(parents=True, exist_ok=True)
                img_resized.save(discard_img_path, 'JPEG', quality=95)
                discarded_count += 1

        except Exception as e:
            print(f"처리 중 에러 발생 {img_path}: {e}")
    print(f"\n총 {len(image_files)}개의 이미지 파일 처리 시도.")
    print(f"  - 성공적으로 처리 및 저장된 이미지: {processed_count}개")
    print(f"  - 최종 검증 실패로 폐기된 이미지: {discarded_count}개")

if __name__ == "__main__":
    SOURCE_DATA_DIR_ROOT = Path("./datasets/korean_emotion_complex_vision_5_percent_split")
    DEST_DATA_DIR_ROOT = Path("./datasets/v3_korean_emotion_complex_vision_5_percent_verified_processed")
    
    run_verified_preprocessing(SOURCE_DATA_DIR_ROOT / "train", DEST_DATA_DIR_ROOT / "train")
    run_verified_preprocessing(SOURCE_DATA_DIR_ROOT / "val", DEST_DATA_DIR_ROOT / "val")

    print("Verified preprocessing complete.")