import torch.nn as nn
import torch.optim as optim
from core.models.model_factory import create_model

# 1. 모델 준비 (이전 단계에서 만든 팩토리 사용)
model = create_model(model_name='resnet18', num_classes=7)

# 2. 손실 함수 정의
# 다중 클래스 분류 문제이므로 CrossEntropyLoss를 사용합니다.
criterion = nn.CrossEntropyLoss()

# 3. 옵티마이저 정의
# model.parameters()는 옵티마이저에게 학습시킬 모든 파라미터를 알려줍니다.
# lr=0.001은 학습률(learning rate)로, 모델을 얼마나 큰 보폭으로 업데이트할지 정합니다.
# 0.001은 일반적으로 좋은 출발점으로 여겨집니다.
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("모델, 손실 함수, 옵티마이저가 준비되었습니다.")
print("\nModel:", model.__class__.__name__)
print("Loss Function:", criterion)
print("Optimizer:", optimizer)