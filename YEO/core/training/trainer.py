# /core/training/trainer.py

import torch
import time
import copy

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, patience=10):
    """
    모델 훈련을 위한 메인 루프.

    Args:
        model: 훈련시킬 PyTorch 모델.
        train_loader: 훈련 데이터로더.
        val_loader: 검증 데이터로더.
        criterion: 손실 함수.
        optimizer: 옵티마이저.
        num_epochs (int): 총 훈련 에폭 수.
        patience (int): 조기 종료를 위한 인내 에폭 수.
    """
    since = time.time()

    # ✅ 조기 종료를 위한 변수 초기화
    best_val_loss = float('inf')  # 가장 좋았던 검증 손실 값을 저장 (낮을수록 좋음)
    patience_counter = 0          # 성능 개선이 없는 에폭 수를 카운트
    best_model_wts = copy.deepcopy(model.state_dict()) # 최고 성능 모델의 가중치를 저장

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
        
        # 조기 종료 로직
        if val_loss < best_val_loss:
            # 검증 손실이 개선되었을 경우
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f'  -> Val Loss 개선됨! ({best_val_loss:.4f}) 모델 저장.')
        else:
            # 검증 손실이 개선되지 않았을 경우
            patience_counter += 1
            print(f'  -> Val Loss 개선되지 않음. EarlyStopping Counter: {patience_counter}/{patience}')

        if patience_counter >= patience:
            print(f'\nEarly stopping! {patience} 에폭 동안 성능 개선이 없었습니다.')
            break # 훈련 루프 탈출
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Loss: {best_val_loss:.4f}')

    # 가장 성능이 좋았던 모델의 가중치를 로드하여 반환
    model.load_state_dict(best_model_wts)
    return model