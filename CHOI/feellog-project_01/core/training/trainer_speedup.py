# /core/training/trainer.py

import torch
import time
import copy
#from torch.cuda.amp import autocast, GradScaler # AMP를 위해 import
from torch.amp.grad_scaler import GradScaler 
from torch.amp.autocast_mode import autocast 

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=100, patience=10, log_interval=20, steps_per_epoch=None):
    """
    모델 훈련을 위한 메인 루프.

    Args:
        model: 훈련시킬 PyTorch 모델.
        train_loader: 훈련 데이터로더.
        val_loader: 검증 데이터로더.
        criterion: 손실 함수.
        optimizer: 옵티마이저.
        scheduler: 학습률 스케줄러.
        device: 학습에 사용할 장치 (CPU 또는 GPU).
        num_epochs (int): 총 훈련 에폭 수.
        patience (int): 조기 종료를 위한 인내 에폭 수.
        log_interval (int): 로그 출력을 위한 배치 간격.
        steps_per_epoch (int, optional): 각 에폭당 훈련 스텝 수 (빠른 테스트 용도).
    """
    since = time.time()
    
    # AMP (Automatic Mixed Precision)
    # GradScaler는 반정밀도(float16) 연산 시 기울기(gradient)가 너무 작아지는 것을 방지
    use_amp = (device.type == 'cuda')
    scaler = GradScaler(enabled=use_amp)
    # ------------------------------------------------

    # 조기 종료를 위한 변수 초기화               
    best_val_loss = float('inf')  # 가장 좋았던 검증 손실 값을 저장 (낮을수록 좋음)
    best_train_loss = float('inf')  # 가장 좋았던 훈련 손실 값을 저장 (낮을수록 좋음)
    best_val_acc = float('-inf')  # 가장 좋았던 검증 정확도를 저장 (높을수록 좋음)
    best_train_acc = float('-inf')  # 가장 좋았던 훈련 정확도를 저장 (높을수록 좋음)
    
    saved_val_loss = float('inf')  # 가장 좋았던 검증 손실 값을 저장 (낮을수록 좋음)
    saved_train_loss = float('inf')  # saved_val_loss 기준으로 가장 좋았던 훈련 손실 값을 저장 (낮을수록 좋음)
    saved_val_acc = float('-inf')  # saved_val_loss 기준으로 가장 좋았던 검증 정확도를 저장 (높을수록 좋음)
    saved_train_acc = float('-inf')  # saved_val_loss 기준으로 가장 좋았던 훈련 정확도를 저장 (높을수록 좋음)
    saved_epoch = 0 # saved_val_loss 기준으로 최고 성능을 기록한 에폭
    
    patience_counter = 0          # 성능 개선이 없는 에폭 수를 카운트
    saved_model_wts = copy.deepcopy(model.state_dict()) # 최고 성능 모델의 가중치를 저장
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # === 1. 훈련 단계 ===
        model.train()  # 모델을 훈련 모드로 설정
        running_loss = 0.0
        running_corrects = 0
        
        # 실제 처리한 샘플 수를 추적할 변수 추가
        samples_seen = 0
        
        # enumerate를 step으로 변경하여 배치 번호 추적
        for step, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            # AMP 적용, autocast 컨텍스트에 device 타입 명시 및 CUDA 환경에서만 활성화
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
                
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # 처리한 샘플 수 업데이트
            samples_seen += inputs.size(0)
            
            # 배치 단위 진행률 로그
            if (step + 1) == 20 or (step + 1) == len(train_loader) or (step + 1) == steps_per_epoch:
                batch_loss = loss.item()
                batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
                print(f'  [Batch {step+1}/{len(train_loader)}] Train Loss: {batch_loss:.4f} Acc: {batch_acc:.4f}')
                
            # 선택적 배치 수 제한
            if steps_per_epoch and (step + 1) >= steps_per_epoch:
                print(f'  -> Reached steps_per_epoch ({steps_per_epoch}), moving to next epoch.')
                break

        #train_loss = running_loss / len(train_loader.dataset)
        #train_acc = running_corrects.double() / len(train_loader.dataset)        
        #print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        
        # 에폭 통계 계산 시 전체 데이터셋 크기 대신, 실제 처리한 샘플 수로 나눔
        epoch_loss = running_loss / samples_seen
        epoch_acc = running_corrects.double() / samples_seen
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
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
        if val_loss < saved_val_loss:
            # 검증 손실이 개선되었을 경우
            saved_val_loss = val_loss
            saved_val_acc = val_acc
            saved_train_loss = epoch_loss
            saved_train_acc = epoch_acc
            saved_epoch = epoch + 1
            saved_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f'  -> Val Loss 개선됨! ({saved_val_loss:.4f}) 모델 저장.')
        else:
            # 검증 손실이 개선되지 않았을 경우
            patience_counter += 1
            print(f'  -> Val Loss 개선되지 않음. EarlyStopping Counter: {patience_counter}/{patience}')

        if patience_counter >= patience:
            print(f'\nEarly stopping! {patience} 에폭 동안 성능 개선이 없었습니다.')
            break # 훈련 루프 탈출
        
        if best_val_acc < val_acc:
            best_val_acc = val_acc
        if best_train_acc < epoch_acc:
            best_train_acc = epoch_acc
        if best_val_loss > val_loss:
            best_val_loss = val_loss
        if best_train_loss > epoch_loss:
            best_train_loss = epoch_loss
        
        scheduler.step() #스케쥴러 업데이트
    
    time_elapsed = time.time() - since
    print('-'*50)
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Saved Epoch: {saved_epoch}')
    print('-'*50)
    print(f'Saved Train Loss: {saved_train_loss:.4f}')
    print(f'Saved Train Acc: {saved_train_acc:.4f}')
    print(f'Saved Val Loss: {saved_val_loss:.4f}')
    print(f'Saved Val Acc: {saved_val_acc:.4f}')
    print('-'*50)
    print(f'Best Train Loss: {best_train_loss:.4f}')
    print(f'Best Train Acc: {best_train_acc:.4f}')
    print(f'Best Val Loss: {best_val_loss:.4f}')
    print(f'Best Val Acc: {best_val_acc:.4f}')
    print('-'*50)
    
    # 가장 성능이 좋았던 모델의 가중치를 로드하여 반환
    model.load_state_dict(saved_model_wts)
    return model