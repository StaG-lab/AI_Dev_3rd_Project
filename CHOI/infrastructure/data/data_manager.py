# /infrastructure/data/data_manager.py

import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

def _sample_by_file_per_image(source_image_dir, source_label_dir, output_dir, sample_rate):
    """[내부 함수] 1이미지-1라벨 구조를 샘플링합니다."""
    # (기존 data_sampler.py의 로직과 동일)
    emotion_groups = defaultdict(list)
    for label_path in source_label_dir.glob("*.json"):
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            emotion = data.get("emotion")
            if emotion:
                emotion_groups[emotion].append(label_path.name)

    output_source_dir = output_dir / "source"
    output_label_dir = output_dir / "labels"
    output_source_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    for emotion, files in emotion_groups.items():
        num_to_sample = int(len(files) * sample_rate)
        if num_to_sample == 0 and len(files) > 0: num_to_sample = 1
        sampled_files = random.sample(files, num_to_sample)
        for filename in sampled_files:
            shutil.copy(source_label_dir / filename, output_label_dir / filename)
            shutil.copy(source_image_dir / Path(filename).with_suffix('.jpg'), output_source_dir / Path(filename).with_suffix('.jpg'))

def _sample_by_file_per_emotion(source_image_dir, source_label_dir, output_dir, sample_rate, val_split_ratio=0.2):
    """[내부 함수] 감정별 통합 라벨 구조를 샘플링하고 train/val으로 분리합니다."""
    
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    for label_path in source_label_dir.glob("*.json"):
        emotion = label_path.stem
        with open(label_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
            image_filenames = [item.get("filename") for item in labels if item.get("filename")]
        
        num_to_sample = int(len(image_filenames) * sample_rate)
        if num_to_sample == 0 and len(image_filenames) > 0: num_to_sample = 1
        
        sampled_filenames = random.sample(image_filenames, num_to_sample)
            
        # 분리하기 전에 파일 리스트를 무작위로 섞습니다.
        random.shuffle(sampled_filenames)
        
        # 훈련 세트와 검증 세트로 리스트를 나눕니다. (예: 80% 훈련, 20% 검증)
        split_point = int(len(sampled_filenames) * (1 - val_split_ratio))
        train_files = sampled_filenames[:split_point]
        val_files = sampled_filenames[split_point:]
        
        # 목적지 폴더를 생성합니다.
        (train_dir / emotion).mkdir(parents=True, exist_ok=True)
        (val_dir / emotion).mkdir(parents=True, exist_ok=True)
        
        label_train_data = []  # 샘플링된 데이터를 저장할 리스트
        label_train_dir = train_dir / "labels"
        label_train_dir.mkdir(exist_ok=True)
        
        label_val_data = []  # 샘플링된 데이터를 저장할 리스트
        label_val_dir = val_dir / "labels"
        label_val_dir.mkdir(exist_ok=True)

        # 파일들을 train 폴더로 복사합니다.
        for filename in train_files:
            source_path = source_image_dir / filename
            if source_path.exists():
                shutil.copy(source_path, train_dir / emotion / filename)
                
                # 추가: filename과 일치하는 JSON 요소를 저장
                matching_data = next((item for item in labels if item.get("filename") == filename), None)
                if matching_data:
                    label_train_data.append(matching_data)

        # {emotion}_sampled.json 파일에 저장
        if label_train_data:
            label_train_dir_path = label_train_dir / f"{emotion}_sampled.json"
            with open(label_train_dir_path, 'w', encoding='utf-8') as f:
                json.dump(label_train_data, f, ensure_ascii=False, indent=4)

        # 파일들을 val 폴더로 복사합니다.
        for filename in val_files:
            source_path = source_image_dir / filename
            if source_path.exists():
                shutil.copy(source_path, val_dir / emotion / filename)        

                # 추가: filename과 일치하는 JSON 요소를 저장
                matching_data = next((item for item in labels if item.get("filename") == filename), None)
                if matching_data:
                    label_val_data.append(matching_data)

        # {emotion}_sampled.json 파일에 저장
        if label_val_data:
            label_val_dir_path = label_val_dir / f"{emotion}_sampled.json"
            with open(label_val_dir_path, 'w', encoding='utf-8') as f:
                json.dump(label_val_data, f, ensure_ascii=False, indent=4)
                
def sample_dataset(source_image_dir, source_label_dir, output_dir, sample_rate, mode, val_split_ratio=0.2):
    """
    데이터 구조에 맞는 샘플링 로직을 선택하여 실행합니다.

    Args:
        ...
        mode (str): 샘플링 모드 ('file_per_image' 또는 'file_per_emotion').
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"'{mode}' 모드로 데이터 샘플링을 시작합니다.")

    if mode == 'file_per_image':
        _sample_by_file_per_image(source_image_dir, source_label_dir, output_dir, sample_rate)
    elif mode == 'file_per_emotion':
        _sample_by_file_per_emotion(source_image_dir, source_label_dir, output_dir, sample_rate, val_split_ratio)
    else:
        raise ValueError(f"알 수 없는 모드입니다: {mode}. 'file_per_image' 또는 'file_per_emotion'을 사용하세요.")
    
    print("샘플링 완료!")