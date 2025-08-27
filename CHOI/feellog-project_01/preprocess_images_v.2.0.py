import json
from pathlib import Path
from PIL import Image, ImageOps
import cv2
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
import shutil
import multiprocessing
from functools import partial
import random
import argparse

# --- 전역 모델 변수 ---
# 워커 초기화 시 설정됨
face_cascade = None
dnn_net = None

# --- Helper Functions ---
def _majority_label(item: dict):
    """annot_A/B/C.faceExp 다수결; 3/3 또는 2/1만 유효. 1/1/1은 (None,1,[]) 리턴"""
    labels, who = [], []
    for k in ("annot_A", "annot_B", "annot_C"):
        lab = (item.get(k) or {}).get("faceExp")
        if lab is not None:
            labels.append(lab)
            who.append(k)
    cnt = Counter(labels)
    if not cnt:
        return None, 0, []
    lab, n = cnt.most_common(1)[0]
    if n == 1:  # 1/1/1
        return None, 1, []
    return lab, n, []

# --- Worker Initializer ---
def init_worker(haar_path_str, proto_path_str, model_path_str):
    """각 워커 프로세스가 시작될 때 모델을 로드하여 전역 변수로 설정."""
    global face_cascade, dnn_net
    face_cascade = cv2.CascadeClassifier(haar_path_str)
    dnn_net = cv2.dnn.readNetFromCaffe(proto_path_str, model_path_str)

# --- 단일 책임 원칙에 따른 함수 분리 ---
def load_labels(source_root: Path) -> dict:
    """모든 레이블 파일을 로드하여 단일 딕셔너리로 반환."""
    all_labels = {}
    label_files = list(source_root.glob("**/labels/*_sampled.json"))
    for label_file in tqdm(label_files, desc="레이블 파일 로드 중"):
        with open(label_file, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                all_labels[item['filename']] = item
    return all_labels

def group_images_by_person(source_root: Path) -> dict:
    """이미지 파일들을 스캔하여 person_id를 기준으로 그룹화."""
    person_to_images = defaultdict(list)
    image_files = [p for p in source_root.glob('**/*.*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png'] and 'labels' not in p.parts]
    for img_path in image_files:
        person_id = img_path.name.split('_')[0]
        person_to_images[person_id].append(img_path)
    return person_to_images

def split_persons(person_to_images: dict) -> dict:
    """인물 ID를 기준으로 Train/Val/Test 세트로 분할."""
    person_ids = list(person_to_images.keys())
    random.shuffle(person_ids)

    total_images = sum(len(files) for files in person_to_images.values())
    train_target = int(total_images * 0.7)
    val_target = int(total_images * 0.2)
    
    person_to_split = {}
    train_count, val_count = 0, 0

    for person_id in person_ids:
        count = len(person_to_images[person_id])
        if train_count < train_target:
            person_to_split[person_id] = 'train'
            train_count += count
        elif val_count < val_target:
            person_to_split[person_id] = 'val'
            val_count += count
        else:
            person_to_split[person_id] = 'test'
            
    return person_to_split

def prepare_tasks(person_to_images: dict, person_to_split: dict, all_labels: dict) -> list:
    """병렬 처리를 위한 작업 목록(인자 튜플) 생성."""
    tasks = []
    for person_id, images in person_to_images.items():
        split_name = person_to_split.get(person_id)
        if not split_name:
            continue
        for img_path in images:
            label_info = all_labels.get(img_path.name)
            if label_info:
                # (이미지 경로, 레이블 정보, 목적지 스플릿 이름)
                tasks.append((img_path, label_info, split_name))
    return tasks
    
# --- 리팩토링된 이미지 처리 워커 ---
def process_and_save_image(task_args: tuple, config: dict):
    """
    단일 이미지 파일을 전처리하고 최종 목적지에 직접 저장.
    IPC 오버헤드를 줄이기 위해 필요한 정보만 task_args로 받음.
    """
    img_path, label_info, split_name = task_args
    try:
        # 설정값 언패킹
        dest_root = config['dest_root']
        final_size = config['final_size']
        confidence_threshold = config['confidence_threshold']

        # --- 라벨 처리 ---
        orig_emotion = img_path.parent.name
        maj, agree_n, _ = _majority_label(label_info)

        if agree_n == 1:
            target_dir = dest_root / "_disagreed" / orig_emotion
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, target_dir / img_path.name)
            
            meta_path = (target_dir / img_path.stem).with_suffix(".v3meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({ 
                    "reason": "annot_all_diff(1/1/1)", 
                    "from_json_emotion": orig_emotion,
                    "uploader": label_info.get("faceExp_uploader"), 
                    "A": (label_info.get("annot_A") or {}).get("faceExp"),
                    "B": (label_info.get("annot_B") or {}).get("faceExp"), 
                    "C": (label_info.get("annot_C") or {}).get("faceExp"),
                }, f, ensure_ascii=False, indent=2)
            return 'disagreed', (img_path, "1/1/1 disagreement")

        final_emotion = maj if maj is not None else label_info.get("faceExp_uploader")

        # --- 이미지 처리 및 얼굴 탐지 ---
        img = Image.open(img_path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        cv_img = np.array(img)
        
        faces = []
        confidence = 0.0

        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        detected_faces_haar = face_cascade.detectMultiScale(gray, scaleFactor=1.42, minNeighbors=9, minSize=(350, 350))

        if len(detected_faces_haar) > 0:
            faces = detected_faces_haar
            confidence = 1.0
        else:
            h, w = cv_img.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(cv_img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            dnn_net.setInput(blob)
            detections = dnn_net.forward()
            best_detection_idx = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, best_detection_idx, 2]
            
            if confidence > 0.5:
                box = detections[0, 0, best_detection_idx, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces = np.array([[startX, startY, endX - startX, endY - startY]])

        # 얼굴 영역 또는 중앙 크롭
        if len(faces) == 0:
            side = min(img.width, img.height)
            crop_box = ((img.width - side) // 2, (img.height - side) // 2, (img.width + side) // 2, (img.height + side) // 2)
        else:
            x, y, w, h = faces[0]
            center_x, center_y = x + w // 2, y + h // 2
            size = max(w, h) + int(max(w, h) * 0.4)
            crop_x1, crop_y1 = max(0, center_x - size // 2), max(0, center_y - size // 2)
            crop_x2, crop_y2 = min(img.width, center_x + size // 2), min(img.height, center_y + size // 2)
            crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
            
        img_cropped = img.crop(crop_box)
        img_resized = img_cropped.resize((final_size, final_size), Image.LANCZOS)
    
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
        best_detection_idx = np.argmax(detections[0, 0, :, 2])
        
        # 가장 높은 신뢰도를 찾음
        confidence = detections[0, 0, best_detection_idx, 2]
    
        # --- 검증 및 저장 ---
        target_dir_base = dest_root
        if confidence <= confidence_threshold:
            target_dir = target_dir_base / "_discarded" / split_name / final_emotion
            status = 'discarded'
        else:
            target_dir = target_dir_base / split_name / final_emotion
            status = 'processed'

        # 최종 목적지에 바로 저장 (I/O 단일화)
        dest_img_path = target_dir / img_path.name
        dest_img_path.parent.mkdir(parents=True, exist_ok=True)
        img_resized.save(dest_img_path, 'JPEG', quality=95)

        # 메타데이터 저장
        if status == 'processed':
            processed_label = {"emotion": final_emotion, "agree_n": int(agree_n), "from_json_emotion": orig_emotion}
            with open(dest_img_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(processed_label, f, ensure_ascii=False, indent=2)
        
        return status, img_path.name

    except Exception as e:
        return 'error', (img_path, e)

# --- Main Runner ---
def run_preprocess(source_root: Path, dest_root: Path, final_size=256, confidence_threshold=0.7):
    # 0. 초기 설정
    if dest_root.exists():
        print(f"기존 '{dest_root}' 폴더를 삭제합니다.")
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True)

    # --- 1단계: 데이터 분할 계획 수립 (I/O 없이 메모리에서 수행) ---
    print("\n--- 1단계: 데이터 분할 계획 수립 ---")
    all_labels = load_labels(source_root)
    person_to_images = group_images_by_person(source_root)
    person_to_split = split_persons(person_to_images)
    tasks = prepare_tasks(person_to_images, person_to_split, all_labels)
    
    # 분할 계획 요약
    splits_summary = defaultdict(lambda: {'persons': 0, 'images': 0})
    for person_id, split in person_to_split.items():
        splits_summary[split]['persons'] += 1
        splits_summary[split]['images'] += len(person_to_images[person_id])
    
    print("\n분할 계획:")
    for split, counts in splits_summary.items():
        print(f"  - {split.upper()}: {counts['persons']}명, {counts['images']}개 이미지")
    print(f"총 {len(tasks)}개의 이미지 처리 작업을 생성했습니다.")


    # --- 2단계: 데이터 병렬 전처리 ---
    print("\n--- 2단계: 데이터 병렬 전처리 ---")
    haar_path = Path("./infrastructure/models/haarcascade_frontalface_default.xml")
    proto_path = Path("./infrastructure/models/deploy.prototxt")
    model_path = Path("./infrastructure/models/res10_300x300_ssd_iter_140000.caffemodel")

    config = {
        'dest_root': dest_root,
        'final_size': final_size,
        'confidence_threshold': confidence_threshold,
    }

    num_processes = multiprocessing.cpu_count()
    print(f"{num_processes}개의 워커로 병렬 처리를 시작합니다...")
    
    worker_func = partial(process_and_save_image, config=config)
    results_counter = Counter()
    errors = []

    init_args = (str(haar_path), str(proto_path), str(model_path))
    
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=init_args) as pool:
        pbar = tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc="이미지 처리 중")
        for status, data in pbar:
            results_counter[status] += 1
            if status == 'error':
                errors.append(data)

    # --- 3단계: 최종 결과 요약 ---
    print("\n--- 전처리 결과 요약 ---")
    print(f"  - 성공: {results_counter['processed']}개")
    print(f"  - 폐기 (신뢰도 미달): {results_counter['discarded']}개")
    print(f"  - 의견 불일치: {results_counter['disagreed']}개")
    print(f"  - 오류: {results_counter['error']}개")
    # 'skipped'는 task 생성 단계에서 걸러지므로 0이 됨

    if errors:
        print("\n[오류 발생 파일]")
        for img_path, e in errors:
            print(f"  - {img_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling', type=int, default=1, help='샘플링 수치를 1부터 100사이의 정수로 입력하세요.')
    args = parser.parse_args()

    if not (1 <= args.sampling <= 100):
        raise ValueError("샘플링 수치는 1부터 100사이의 정수여야 합니다.")
        
    sampling = args.sampling
    
    SOURCE_DATA_DIR_ROOT = Path(f"./datasets/KECV_{sampling}_percent") 
    DEST_DATA_DIR_ROOT = Path(f"./datasets/KECV_{sampling}_percent_FaceCrop_PersonSplit_256")
    
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    run_preprocess(SOURCE_DATA_DIR_ROOT, DEST_DATA_DIR_ROOT)

    print("\n모든 분할 및 전처리 작업이 완료되었습니다.")