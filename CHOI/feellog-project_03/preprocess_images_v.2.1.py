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
import torch
from torch import nn
from core.models.emonet import EmoNet


# --- 전역 모델 변수 ---
# 워커 초기화 시 설정됨
face_cascade = None
dnn_net = None
emonet_model = None # EmoNet 모델을 위한 전역 변수
device = None       # PyTorch 디바이스 설정을 위한 전역 변수
emotion_classes = {0:"기쁨", 1:"당황", 2:"분노", 3:"불안", 4:"상처", 5:"슬픔", 6:"중립"} # EmoNet 출력 매핑

# --- Worker Initializer ---
def init_worker(haar_path_str, proto_path_str, model_path_str, emonet_path_str):
    """각 워커 프로세스가 시작될 때 모든 모델을 로드하여 전역 변수로 설정."""
    global face_cascade, dnn_net, emonet_model, device

    # 얼굴 탐지 모델 로드
    face_cascade = cv2.CascadeClassifier(haar_path_str)
    dnn_net = cv2.dnn.readNetFromCaffe(proto_path_str, model_path_str)
    
    # EmoNet 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_dict = torch.load(emonet_path_str, map_location=device)
    # state_dict 키에서 'module.' 접두사 제거 (DataParallel로 학습된 경우)
    if next(iter(state_dict)).startswith('module.'):
        state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
        
    emonet_model = EmoNet(n_expressions=len(emotion_classes)+1)
    emonet_model.load_state_dict(state_dict, strict=False)
    emonet_model.to(device)
    emonet_model.eval()
    #print(f"Worker {multiprocessing.current_process().pid}: EmoNet loaded on {device}")


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
    
# --- 이미지 처리 워커 ---
def process_and_save_image(task_args: tuple, config: dict):
    """
    단일 이미지 파일을 전처리하고, EmoNet으로 감정을 분류하여 최종 목적지에 저장.
    """
    img_path, label_info, split_name = task_args
    try:
        # 설정값 언패킹
        dest_root = config['dest_root']
        final_size = config['final_size']
        confidence_threshold = config['confidence_threshold']
        
        # 원본 json의 감정 정보는 메타데이터 저장을 위해 유지
        orig_emotion = img_path.parent.name
        
        # --- 이미지 처리 및 얼굴 탐지  ---
        img = Image.open(img_path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        cv_img = np.array(img)
        
        faces = []
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        detected_faces_haar = face_cascade.detectMultiScale(gray, scaleFactor=1.42, minNeighbors=9, minSize=(350, 350))

        if len(detected_faces_haar) > 0:
            faces = detected_faces_haar
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
    
        # DNN 기반 최종 검증 단계
        final_cv_img_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
        resized_for_dnn = cv2.resize(final_cv_img_bgr, (300, 300))
        blob = cv2.dnn.blobFromImage(resized_for_dnn, 1.0, (300, 300), (104.0, 177.0, 123.0))

        dnn_net.setInput(blob)
        detections = dnn_net.forward()
        best_detection_idx = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, best_detection_idx, 2]
    
        # --- 검증 및 EmoNet 기반 감정 분류 후 저장 ---
        if confidence <= confidence_threshold:
            target_dir = dest_root / "_discarded" / split_name / orig_emotion
            status = 'discarded'
            
            # 폐기 파일 저장
            dest_img_path = target_dir / img_path.name
            dest_img_path.parent.mkdir(parents=True, exist_ok=True)
            img_resized.save(dest_img_path, 'JPEG', quality=95)
            
        else:
            # EmoNet 감정 분류 로직
            with torch.no_grad():
                # 이미지를 EmoNet 입력 형식에 맞게 변환
                image_tensor = torch.as_tensor(np.array(img_resized), dtype=torch.float32).permute(2, 0, 1).to(device)
                image_tensor /= 255.0 # [0, 1] 범위로 정규화
                
                # 모델 추론
                output = emonet_model(image_tensor.unsqueeze(0))
                
                # 결과 해석
                predicted_class_idx = torch.argmax(nn.functional.softmax(output["expression"], dim=1)).cpu().item()
                final_emotion = emotion_classes.get(predicted_class_idx, "알수없음")

            target_dir = dest_root / split_name / final_emotion
            status = 'processed'
            
            # 최종 목적지에 이미지 저장
            dest_img_path = target_dir / img_path.name
            dest_img_path.parent.mkdir(parents=True, exist_ok=True)
            img_resized.save(dest_img_path, 'JPEG', quality=95)

            '''
            # 메타데이터 저장
            processed_label = {
                "emotion_model": "Emonet_50p",
                "predicted_emotion": final_emotion,
                "original_folder_emotion": orig_emotion
            }
            with open(dest_img_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(processed_label, f, ensure_ascii=False, indent=2)
            '''
        return status, img_path.name

    except Exception as e:
        return 'error', (img_path, str(e))


# --- Main Runner ---
def run_preprocess(source_root: Path, dest_root: Path, emonet_path: Path, final_size=256, confidence_threshold=0.7):
    # 0. 초기 설정
    if dest_root.exists():
        print(f"기존 '{dest_root}' 폴더를 삭제합니다.")
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True)

    # 1단계: 데이터 분할 계획 수립
    print("\n--- 1단계: 데이터 분할 계획 수립 ---")
    all_labels = load_labels(source_root)
    person_to_images = group_images_by_person(source_root)
    person_to_split = split_persons(person_to_images)
    tasks = prepare_tasks(person_to_images, person_to_split, all_labels)
    
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
    # 모델 경로 설정
    haar_path = Path("./infrastructure/models/haarcascade_frontalface_default.xml")
    proto_path = Path("./infrastructure/models/deploy.prototxt")
    model_path = Path("./infrastructure/models/res10_300x300_ssd_iter_140000.caffemodel")

    # EmoNet 모델 파일 존재 여부 확인
    if not emonet_path.exists():
        raise FileNotFoundError(f"EmoNet 모델 파일을 찾을 수 없습니다: {emonet_path}")

    config = {
        'dest_root': dest_root,
        'final_size': final_size,
        'confidence_threshold': confidence_threshold,
    }

    num_processes = int(multiprocessing.cpu_count() / 2)
    print(f"{num_processes}개의 워커로 병렬 처리를 시작합니다...")
    
    worker_func = partial(process_and_save_image, config=config)
    results_counter = Counter()
    errors = []

    # initializer에 EmoNet 모델 경로 추가
    init_args = (str(haar_path), str(proto_path), str(model_path), str(emonet_path))
    
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=init_args) as pool:
        pbar = tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc="이미지 처리 중")
        for status, data in pbar:
            results_counter[status] += 1
            if status == 'error':
                errors.append(data)

    # --- 3단계: 최종 결과 요약 ---
    print("\n--- 전처리 결과 요약 ---")
    print(f"  - 성공 (EmoNet 분류 완료): {results_counter['processed']}개")
    print(f"  - 폐기 (얼굴 탐지 신뢰도 미달): {results_counter['discarded']}개")
    print(f"  - 오류: {results_counter['error']}개")

    if errors:
        print("\n[오류 발생 파일]")
        for img_path, e in errors:
            print(f"  - {img_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KECV 데이터셋 전처리 스크립트 (EmoNet 기반)")
    parser.add_argument('--sampling', type=int, default=1, help='샘플링 수치를 1부터 100사이의 정수로 입력하세요.')
    parser.add_argument('--confidence', type=float, default=0.7, help='얼굴 탐지 최소 신뢰도 (0.0 ~ 1.0)')
    parser.add_argument('--emonet_path', type=str, default='./infrastructure/models/weights/checkpoints/emonet_50_2_percent_trained_2.pth', help='사용할 모델 경로')
    args = parser.parse_args()

    if not (1 <= args.sampling <= 100):
        raise ValueError("샘플링 수치는 1부터 100사이의 정수여야 합니다.")
        
    sampling = args.sampling
    
    SOURCE_DATA_DIR_ROOT = Path(f"./datasets/KECV_{sampling}_percent") 
    DEST_DATA_DIR_ROOT = Path(f"./datasets/KECV_{sampling}_percent_FC_PS_EmoNet_256")
    EMONET_PATH = Path(args.emonet_path)

    
    try:
        # 'spawn' 시작 방식은 자식 프로세스가 부모의 리소스를 상속받지 않아 더 안정적임
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    run_preprocess(
        SOURCE_DATA_DIR_ROOT, 
        DEST_DATA_DIR_ROOT, 
        EMONET_PATH,
        confidence_threshold=args.confidence,
    )

    print("\n모든 분할 및 전처리 작업이 완료되었습니다.")