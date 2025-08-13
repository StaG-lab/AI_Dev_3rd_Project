# %%
# 모델을 다운로드 받고 모델이 제대로 작동하는지 확인하는 명령어입니다.
# 가상환경이 활성화된 터미널에서 아래 명령어를 실행하세요.
# python -m unittest tests/test_model.py
#-------------------------
# Ran 1 test in 8.179s
# OK
#-------------------------
# 위와 같은 결과를 얻었다면 모델이 정상적으로 작동하는 것입니다.
# 코드 중 담당하신 MODEL_NAME에 해당하는 주석(#)을 제거하고 실행해 주세요.

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from pathlib import Path

from core.models.model_factory import create_model
from core.data.dataset import EmotionDataset
from core.training.trainer import train_model

if __name__ == '__main__':
    # 설정값 정의
    # 장치 설정: 사용 가능한 경우 GPU(cuda)를, 그렇지 않으면 CPU를 사용
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    DATA_DIR = Path("./datasets/korean_emotion_complex_vision_5_percent_verified_processed")
    # 사용하고자 하는 모델 하나만 남기고 다른 MODEL_NAME 앞에 # 붙여서 주석처리
    #MODEL_NAME = 'resnet18'             #철원
    #MODEL_NAME = 'resnet50' 
    # MODEL_NAME = 'mobilenet_v3_small'  #승현님
    MODEL_NAME = 'shufflenet_v2'       #철원
    #MODEL_NAME = 'efficientnet_v2_s'   #규진님
    #MODEL_NAME = 'squeezenet'          #승희님
    
    NUM_CLASSES = 7  # 데이터셋의 클래스 수에 맞게 조정해야 합니다. ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10 # 10번 연속 성능 개선이 없으면 조기 종료
    
    # 데이터 증강을 포함한 훈련용 Transform 정의
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
        transforms.RandomRotation(15),           # -15도 ~ 15도 사이로 랜덤 회전
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 밝기, 대비, 채도 조절
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 증강이 없는 검증/테스트용 Transform 정의
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 각 데이터셋에 맞는 Transform 적용
    train_dataset = EmotionDataset(data_dir=DATA_DIR / "train", transform=train_transform)
    val_dataset = EmotionDataset(data_dir=DATA_DIR / "val", transform=val_transform)

    # 훈련용과 검증용 데이터셋을 각각 생성. 
    train_dataset = EmotionDataset(data_dir=DATA_DIR / "train", transform=train_transform)
    val_dataset = EmotionDataset(data_dir=DATA_DIR / "val", transform=val_transform)

    # 데이터로더를 각각 생성. (검증용은 섞을 필요가 없음)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    NUM_CLASSES = len(train_dataset.classes)
    
    print("데이터 준비 완료!")
    print(f"훈련 데이터셋 크기: {len(train_dataset)}")
    print(f"클래스 수: {NUM_CLASSES} -> {train_dataset.classes}")

    # 모델, 손실 함수, 옵티마이저 준비
    model = create_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES)
    # 모델을 지정된 장치로 이동
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        #weight_decay=1e-4, #과적합 방지를 위한 가중치 감쇠를 넣었으나 오히려 학습에 방해가 되고 있음.
        lr=LEARNING_RATE 
        ) 

    print(f"'{MODEL_NAME}' 모델, 손실 함수, 옵티마이저 준비 완료!")

    # 모델 훈련 시작
    print("\n모델 훈련을 시작합니다...")
    trained_model = train_model(model, 
                                train_loader, 
                                val_loader, 
                                criterion, 
                                optimizer, 
                                DEVICE, 
                                num_epochs=NUM_EPOCHS, 
                                patience=EARLY_STOPPING_PATIENCE)

    # 훈련된 모델 저장 (옵션)
    # torch.save(trained_model.state_dict(), f'{MODEL_NAME}_trained.pth')
    # print("훈련된 모델 가중치가 저장되었습니다.")

# %%



