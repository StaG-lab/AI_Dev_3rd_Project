# /train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from core.models.model_factory import create_model
from core.data.dataset import EmotionDataset
from core.training.trainer import train_model

if __name__ == '__main__':
    # 설정값 정의
    # 장치 설정: 사용 가능한 경우 GPU(cuda)를, 그렇지 않으면 CPU를 사용
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    DATA_DIR = Path("./datasets/korean_emotion_complex_vision_1_percent_preprocessed")
    MODEL_NAME = 'resnet18'
    NUM_CLASSES = 7  # 데이터셋의 클래스 수에 맞게 조정해야 합니다. ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    
    # 훈련용과 검증용 데이터셋을 각각 생성합니다.
    train_dataset = EmotionDataset(data_dir=DATA_DIR / "train")
    val_dataset = EmotionDataset(data_dir=DATA_DIR / "val")
    
    # 데이터로더를 각각 생성합니다. (검증용은 섞을 필요가 없습니다)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    NUM_CLASSES = len(train_dataset.classes)
    
    print("데이터 준비 완료!")
    print(f"훈련 데이터셋 크기: {len(train_dataset)}")
    print(f"클래스 수: {NUM_CLASSES} -> {train_dataset.classes}")

    # 모델, 손실 함수, 옵티마이저 준비
    model = create_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES)
    # 모델을 지정된 장치로 이동
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"'{MODEL_NAME}' 모델, 손실 함수, 옵티마이저 준비 완료!")

    # 모델 훈련 시작
    print("\n모델 훈련을 시작합니다...")
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, DEVICE, num_epochs=NUM_EPOCHS)
    
    # 훈련된 모델 저장 (옵션)
    # torch.save(trained_model.state_dict(), f'{MODEL_NAME}_trained.pth')
    # print("훈련된 모델 가중치가 저장되었습니다.")