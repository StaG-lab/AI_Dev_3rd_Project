import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import os
from core.models.model_factory import create_model
from core.data.dataset import EmotionDataset
from core.training.trainer_speedup import train_model

from torch.optim.lr_scheduler import StepLR

if __name__ == '__main__':
    # CUDA 성능 플래그 최적화
    torch.backends.cudnn.benchmark = True
    # TF32 텐서 코어 사용을 허용하여 Ampere 아키텍처 이상 GPU에서 연산 속도 향상
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # 설정값 정의
    # 장치 설정: 사용 가능한 경우 GPU(cuda)를, 그렇지 않으면 CPU를 사용
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    DATA_DIR = Path("./datasets/korean_emotion_complex_vision_5_percent_verified_processed")
    # 사용하고자 하는 모델 하나만 남기고 다른 MODEL_NAME 앞에 # 붙여서 주석처리
    #MODEL_NAME = 'resnet18'             #철원
    #MODEL_NAME = 'resnet50' 
    #MODEL_NAME = 'mobilenet_v3_small'  #승현님
    #MODEL_NAME = 'shufflenet_v2'       #철원
    MODEL_NAME = 'efficientnet_v2_s'   #규진님
    #MODEL_NAME = 'squeezenet'          #승희님
    
    NUM_CLASSES = 7  # 데이터셋의 클래스 수에 맞게 조정해야 합니다. ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
    BATCH_SIZE = 64  # 배치 크기를 늘려 GPU 메모리 사용 최적화
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10 # 10번 연속 성능 개선이 없으면 조기 종료
    STEPS_PER_EPOCH = None # 빠른 테스트를 위해 에폭당 배치 수를 제한하려면 숫자로 변경 (예: 100)

    
    # 데이터 증강을 포함한 훈련용 Transform 정의
    #train_transform = transforms.Compose([
    #    transforms.Resize((224, 224)),
    #    transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
    #    transforms.RandomRotation(15),           # -15도 ~ 15도 사이로 랜덤 회전
    #    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 밝기, 대비, 채도 조절
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    #])
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # TrivialAugmentWide 추가, 이미지에 다양한 변형(자르기, 색상 왜곡, 회전 등)을 알아서 최적의 강도로 적용, 과적합 방지.
        transforms.TrivialAugmentWide(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 증강이 없는 검증/테스트용 Transform 정의
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 훈련용과 검증용 데이터셋을 각각 생성.
    train_dataset = EmotionDataset(data_dir=DATA_DIR / "train", transform=train_transform)
    val_dataset = EmotionDataset(data_dir=DATA_DIR / "val", transform=val_transform)

    # 데이터로더를 각각 생성. (검증용은 섞을 필요가 없음)
    #train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # DataLoader I/O 튜닝
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        # CPU 코어를 최대한 활용하여 데이터를 미리 GPU 메모리로 올리는 작업을 병렬 처리
        num_workers=min(8, os.cpu_count()), 
        pin_memory=True, # GPU로의 데이터 전송 속도 향상
        persistent_workers=True, # 워커 프로세스를 계속 유지하여 오버헤드 감소
        prefetch_factor=2, # 각 워커가 미리 로드할 배치 수
        drop_last=True # 마지막 배치가 배치 사이즈보다 작을 경우 버려서 연산 일관성 유지
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

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
        weight_decay=1e-4, #과적합 방지를 위한 정규화 기법(Weight Decay), 학습을 방해함으로서 과적합 방지.
        lr=LEARNING_RATE 
        ) 
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    
    print(f"'{MODEL_NAME}' 모델, 손실 함수, 옵티마이저 준비 완료!")

    # 모델 훈련 시작
    print("\n모델 훈련을 시작합니다...")
    trained_model = train_model(model, 
                                train_loader, 
                                val_loader, 
                                criterion, 
                                optimizer, 
                                scheduler,
                                DEVICE, 
                                num_epochs=NUM_EPOCHS, 
                                patience=EARLY_STOPPING_PATIENCE,
                                steps_per_epoch=STEPS_PER_EPOCH
                                )

    # 훈련된 모델 저장 (옵션)
    # torch.save(trained_model.state_dict(), f'{MODEL_NAME}_trained.pth')
    # print("훈련된 모델 가중치가 저장되었습니다.")