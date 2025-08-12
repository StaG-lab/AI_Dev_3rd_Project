# /core/models/model_factory.py

import torch
import torch.nn as nn
from torchvision import models

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
        
    else:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}")

    return model