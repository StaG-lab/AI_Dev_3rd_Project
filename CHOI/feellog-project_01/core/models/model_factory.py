# /core/models/model_factory.py

import torch
import torch.nn as nn
from torchvision import models
from .emotionnet import EmotionNet
from .emonet import EmoNet
from pathlib import Path


def create_model(model_name: str, num_classes: int, pretrained: bool = True):
    """
    지정된 이름의 모델을 생성하고, 전이 학습을 위해 마지막 레이어를 수정합니다.

    Args:
        model_name (str): 생성할 모델의 이름 (예: 'resnet18').
        num_classes (int): 최종 분류할 클래스의 수.
        pretrained (bool): ImageNet으로 사전 학습된 가중치를 사용할지 여부.

    Returns:
        torch.nn.Module: 생성 및 수정된 PyTorch 모델.
    """
    model = None
    
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "squeezenet":
        weights = models.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.squeezenet1_1(weights=weights)
        # SqueezeNet은 마지막 분류기가 Conv2d
        in_channels = model.classifier[1].in_channels
        model.classifier[1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        model.num_classes = num_classes

    elif model_name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "shufflenet_v2":
        weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.shufflenet_v2_x1_0(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)    
        
    elif model_name == "emotionnet":
        # EmotionNet은 사전 훈련된 가중치가 없으므로 pretrained 인자는 사용하지 않습니다.
        model = EmotionNet(num_classes=num_classes)     
        
    # EmoNet 생성 및 가중치 로드 로직 추가
    elif model_name == "emonet":
        # EmoNet은 우리 데이터셋의 7개 클래스에 맞게 새로 생성
        model = EmoNet(num_classes=num_classes, n_expressions=8)
        if pretrained:
            print("사전 훈련된 EmoNet 가중치를 불러옵니다 (Fine-tuning)...")
            # 가중치 파일 경로
            weights_path = Path("./infrastructure/models/weights/emonet_8.pth")
            if weights_path.exists():
                # 원본 모델(8개 클래스)의 가중치를 불러옴
                state_dict = torch.load(weights_path)
                model.load_state_dict(state_dict, strict=False)
            else:
                print(f"[경고] EmoNet 가중치 파일을 찾을 수 없습니다: {weights_path}")
    
    else:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}")

    return model