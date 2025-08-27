# /core/data/dataset.py (수정)

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import json
#import cv2
#import numpy as np

'''
class EmotionDataset(Dataset):
    """
    샘플링된 데이터 폴더로부터 이미지와 라벨을 불러오는 커스텀 데이터셋.
    - data_dir
        - 기쁨/
            - happy_1.jpg
        - 슬픔/
            - sad_1.jpg
        - labels/
            - 기쁨_sampled.json
            - 슬픔_sampled.json
    위와 같은 구조를 읽도록 최적화되었습니다.
    """
    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = data_dir
        
        # 1. 샘플링된 라벨 파일들(*_sampled.json)을 기반으로 이미지 경로와 라벨을 로드합니다.
        self.image_paths = []
        self.labels = []
        self.classes = []
        
        label_files_path = data_dir / "labels"
        
        for label_file in sorted(label_files_path.glob("*_sampled.json")):
            emotion = label_file.stem.replace("_sampled", "") # '기쁨_sampled' -> '기쁨'
            if emotion not in self.classes:
                self.classes.append(emotion)
            
            with open(label_file, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
                for item in label_data:
                    filename = item.get("filename")
                    if filename:
                        # 이미지 경로 조합: (상위폴더)/감정폴더/이미지파일명
                        image_path = self.data_dir / emotion / filename
                        self.image_paths.append(image_path)
                        self.labels.append(self.classes.index(emotion))

        # 2. 클래스 이름과 숫자 라벨을 매핑하는 사전을 만듭니다.
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 3. 이미지 변환(transform)을 설정합니다.
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
'''

# 폴더 구조가 변경되어서 클래스를 새로 정의.

class EmotionDataset(Dataset):
    """
    전처리가 완료된 데이터 폴더로부터 이미지와 라벨을 불러옵니다.
    폴더 이름 자체를 클래스(라벨)로 사용합니다.
    - data_dir
        - 기쁨/
            - image_1.jpg
            - image_1.json
        - 슬픔/
            - image_2.jpg
            - image_2.json
    """
    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = data_dir
        
        # 1. 모든 이미지 파일(.jpg)의 경로를 찾습니다.
        self.image_paths = sorted(list(data_dir.glob('*/*.jpg')))
        
        if not self.image_paths:
            raise FileNotFoundError(f"지정된 경로에서 이미지를 찾을 수 없습니다: {data_dir}")

        # 2. 이미지 경로의 부모 폴더 이름을 클래스로 사용합니다.
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 3. 각 이미지에 해당하는 숫자 라벨 리스트를 생성합니다.
        self.labels = [self.class_to_idx[path.parent.name] for path in self.image_paths]

        # 4. 이미지 변환(transform) 설정
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                # 전처리가 완료된 224x224 이미지이므로 Resize는 선택사항.
                # 다만, 데이터 증강(Augmentation)을 위해 남겨둘 수 있습니다.
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 현재는 이미지와 감정 라벨만 반환. 추후 bbox_224px 좌표가 필요하면 여기서 json 파일을 읽어 함께 반환.
        return image, label
    
        '''
        # 이미지를 PIL이 아닌 OpenCV로 불러와 NumPy 배열로 변환
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Albumentations transform은 NumPy 배열을 입력으로 받습니다.
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
            
        # NumPy 배열을 PyTorch 텐서로 변환
        # (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return image, label
        '''