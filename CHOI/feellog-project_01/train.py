import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import os

from core.models.model_factory import create_model
from core.data.dataset import EmotionDataset
from core.training.trainer import train_model

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
    
    sampling_percent = 100
    DATA_DIR = Path(f"./datasets/KECV_{sampling_percent}_percent_FaceCrop")
    # 사용하고자 하는 모델 하나만 남기고 다른 MODEL_NAME 앞에 # 붙여서 주석처리
    #MODEL_NAME = 'resnet18'             #철원
    #MODEL_NAME = 'resnet50' 
    #MODEL_NAME = 'mobilenet_v3_small'  #승현님
    #MODEL_NAME = 'shufflenet_v2'       #철원
    #MODEL_NAME = 'efficientnet_v2_s'   #규진님
    #MODEL_NAME = 'squeezenet'          #승희님
    #MODEL_NAME = 'emotionnet'           # 감정 인식 전용 모델
    MODEL_NAME = 'emonet'               # 경량화된 감정 인식 모델

    NUM_CLASSES = 7  # 데이터셋의 클래스 수에 맞게 조정해야 합니다. ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
    BATCH_SIZE = 64  # 배치 크기를 늘려 GPU 메모리 사용 최적화
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10 # 10번 연속 성능 개선이 없으면 조기 종료
    STEPS_PER_EPOCH = None # 빠른 테스트를 위해 에폭당 배치 수를 제한하려면 숫자로 변경 (예: 100)
    train_transform = None
    val_transform = None
    
    if MODEL_NAME == 'emotionnet':
        # 48x48 크기, 흑백(Grayscale), 정규화
        # RandomResizedCrop + TrivialAugmentWide (강력한 데이터 증강 방법)
        train_transform = transforms.Compose([
            #transforms.Resize((48, 48)),
            # 원본 이미지의 80% ~ 100% 사이를 무작위로 잘라 48x48 크기로 만듦
            transforms.RandomResizedCrop(size=48, scale=(0.8, 1.0)),
            # 잘라낸 이미지에 최적의 증강 정책을 자동으로 적용
            transforms.TrivialAugmentWide(),
            # 흑백으로 변환
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) # 흑백 이미지 정규화
        ])
        val_transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) # 흑백 이미지는 채널이 1개
        ])

    elif MODEL_NAME == 'emonet':
        # 데이터 증강을 포함한 훈련용 Transform 정의
        train_transform = transforms.Compose([
            #transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.TrivialAugmentWide(), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 증강이 없는 검증/테스트용 Transform 정의
        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    else:
        # 데이터 증강을 포함한 훈련용 Transform 정의
        train_transform = transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
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
    START_EPOCH = 0
    
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)   # 7 에폭마다 학습률을 0.1배로 감소

    CHECKPOINT_PATH = f'./infrastructure/models/weights/checkpoints/{MODEL_NAME}_{sampling_percent}_percent_trained.pth'
    if os.path.exists(CHECKPOINT_PATH):
        print("체크포인트를 불러옵니다...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        START_EPOCH = checkpoint['epoch'] + 1 # 다음 에폭부터 시작
        print(f"체크포인트 로드 완료! {START_EPOCH} 에폭부터 훈련을 재개합니다.")
    else:
        print("체크포인트가 존재하지 않습니다. 처음부터 훈련을 시작합니다.")
    
    #model = torch.compile(model)   # Windows 환경에서 에러 발생
    #print("모델 컴파일 완료!")
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
                                start_epoch=START_EPOCH,
                                patience=EARLY_STOPPING_PATIENCE,
                                steps_per_epoch=STEPS_PER_EPOCH
                                )

    # 훈련된 모델 저장 (옵션)
    torch.save(trained_model.state_dict(), f'./infrastructure/models/weights/checkpoints/{MODEL_NAME}_{sampling_percent}_percent_trained.pth')
    print("훈련된 모델 가중치가 저장되었습니다.")