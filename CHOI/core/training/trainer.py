# /core/training/trainer.py

import torch
import time

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    """
    모델 훈련을 위한 메인 루프.

    Args:
        model: 훈련시킬 PyTorch 모델.
        train_loader: 훈련 데이터로더.
        val_loader: 검증 데이터로더.
        criterion: 손실 함수.
        optimizer: 옵티마이저.
        num_epochs (int): 총 훈련 에폭 수.
    """
    since = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # === 1. 훈련 단계 ===
        model.train()  # 모델을 훈련 모드로 설정
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')

        # === 2. 검증 단계 ===
        model.eval()   # 모델을 평가 모드로 설정
        val_loss = 0.0
        val_corrects = 0

        # 검증 단계에서는 기울기를 계산할 필요가 없습니다.
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    return model