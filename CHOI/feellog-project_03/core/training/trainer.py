# /core/training/trainer.py

import torch
import time
import copy
from torch.amp.grad_scaler import GradScaler 
from torch.amp.autocast_mode import autocast 
import shutil
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
from datetime import datetime
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
                num_epochs=100, patience=10, log_interval=20, steps_per_epoch=None,
                misclassified_dir: Path = None, start_epoch=0, accumulation_steps=1):
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
        misclassified_dir: 오분류된 이미지를 저장할 경로.
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
    
    saved_metrics = {}
    
    for epoch in range(start_epoch, num_epochs):
        torch.cuda.empty_cache()
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # === 1. 훈련 단계 ===
        model.train()  # 모델을 훈련 모드로 설정
        optimizer.zero_grad()
        running_loss = 0.0
        running_corrects = 0
        
        # 실제 처리한 샘플 수를 추적할 변수 추가
        samples_seen = 0
        
        for step, batch in enumerate(train_loader):
            # 비어있는 배치가 전달되면 건너뜀
            if not batch:
                continue
            
            # Hugging Face 모델의 입력을 위해 batch를 device로 바로 이동
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = inputs.pop("labels") # 라벨 분리
            
            with autocast(device_type=device.type, enabled=use_amp):
                # Hubert처럼 transformers 모델의 경우 **inputs 사용
                # (EmoNet이 아닌 경우에도 통합적으로 처리 가능하지만, 모델 이름으로 분기)
                if model._get_name() in ["HubertForSequenceClassification", "AutoModelForAudioClassification"]:  # Hubert 모델 분기 추가
                    outputs = model(**inputs)  # 내장 loss 사용
                    loss = outputs.loss
                    # Accumulation을 위해 loss를 스텝 수로 나눠줌
                    logits = outputs.logits
                elif model._get_name() == "EmotionFineTuningModel":
                        logits = model(**inputs)
                        loss = criterion(logits, labels)
                elif model._get_name() == "EmoNet":
                    outputs_dict = model(inputs)  # EmoNet 전용: dict 출력 기대
                    emotion_preds = outputs_dict['expression']
                    loss = criterion(emotion_preds, labels)
                    logits = emotion_preds  # preds 계산용
                else:
                    # 일반 모델: positional 입력 (하지만 Hubert은 여기에 오면 안 됨)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    logits = outputs
                    
                loss = loss / accumulation_steps
                
            scaler.scale(loss).backward()
            
            # accumulation_steps 마다 모델 업데이트
            if (step + 1) % accumulation_steps == 0:
                # 그래디언트 폭발을 막기 위해 클리핑 적용
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            _, preds = torch.max(logits, 1)
            
            # batch_size 구하기: 모델 유형에 따라 분기
            if isinstance(inputs, dict):
                batch_size = inputs['input_values'].size(0)  # Hubert: dict, 'input_values' 키 사용
            else:
                batch_size = inputs.size(0)  # EmoNet: 텐서
            
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data)
            
            samples_seen += batch_size
            
            # 배치 단위 진행률 로그
            if (step + 1) == 20 or (step + 1) == len(train_loader) or (step + 1) == steps_per_epoch:
                batch_loss = loss.item()
                if isinstance(inputs, dict):
                    batch_size = inputs['input_values'].size(0)
                else:
                    batch_size = inputs.size(0)
                batch_acc = torch.sum(preds == labels.data).double() / batch_size
                print(f'  [Batch {step+1}/{len(train_loader)}] Train Loss: {batch_loss:.4f} Acc: {batch_acc:.4f}')
                
            # 선택적 배치 수 제한
            if steps_per_epoch and (step + 1) >= steps_per_epoch:
                print(f'  -> Reached steps_per_epoch ({steps_per_epoch}), moving to next epoch.')
                break
        
        if (step + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        # 에폭 통계 계산 시 전체 데이터셋 크기 대신, 실제 처리한 샘플 수로 나눔
        epoch_loss = running_loss / samples_seen
        epoch_acc = running_corrects.double() / samples_seen
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # === 2. 검증 단계 ===
        model.eval()   # 모델을 평가 모드로 설정
        val_loss = 0.0
        val_corrects = 0
        
        # 클래스별 성능 측정을 위한 변수 초기화
        num_classes = len(val_loader.dataset.classes)
        class_correct = list(0. for i in range(num_classes))
        class_total = list(0. for i in range(num_classes))
        
        # F1 Score 계산을 위해 모든 예측과 라벨을 저장할 리스트
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                # 비어있는 배치가 전달되면 건너뜀
                if not batch:
                    continue
                
                inputs = {k: v.to(device) for k, v in batch.items()}
                labels = inputs.pop("labels")
                
                with autocast(device_type=device.type, enabled=use_amp):
                    if model._get_name() in ["HubertForSequenceClassification", "AutoModelForAudioClassification"]:
                        outputs = model(**inputs)
                        loss = outputs.loss
                        logits = outputs.logits
                    elif model._get_name() == "EmotionFineTuningModel":
                        logits = model(**inputs)
                        loss = criterion(logits, labels)
                    elif model._get_name() == "EmoNet":
                        outputs_dict = model(inputs)
                        emotion_preds = outputs_dict['expression']
                        loss = criterion(emotion_preds, labels)
                        logits = emotion_preds
                    else:
                        outputs = model(**inputs)
                        loss = criterion(outputs, labels)
                        logits = outputs
                
                _, preds = torch.max(logits, 1)
                        
                if isinstance(inputs, dict):
                    batch_size = inputs['input_values'].size(0)
                else:
                    batch_size = inputs.size(0)
                
                val_loss += loss.item() * batch_size
                val_corrects += torch.sum(preds == labels.data)
                
                # F1 Score 계산을 위해 CPU로 옮겨 저장
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 클래스별 정확도 계산
                corrects = (preds == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += corrects[i].item()
                    class_total[label] += 1
                
                # 오답 이미지 저장 로직
                if misclassified_dir:
                    # 예측이 틀린 이미지들의 인덱스를 찾음
                    mismatched_indices = (preds != labels).nonzero(as_tuple=True)[0]
                    for idx in mismatched_indices:
                        true_label_name = val_loader.dataset.classes[labels[idx].item()]
                        # pred_label_name = val_loader.dataset.classes[preds[idx].item()]
                        # 유효성 검사
                        pred_class_index = preds[idx].item()
                        if 0 <= pred_class_index < len(val_loader.dataset.classes):
                            pred_label_name = val_loader.dataset.classes[pred_class_index]
                        else:
                            # 유효하지 않은 인덱스에 대한 처리
                            #print(f"Warning: Invalid predicted class index {pred_class_index} found.")
                            pred_label_name = "Unknown"
                        original_path = Path(image_paths[idx])

                        # 저장 경로: {오답폴더}/{에폭}/true_{실제라벨}_pred_{예측라벨}/이미지명.jpg
                        #save_dir = misclassified_dir / f"epoch_{epoch+1}" / f"true_{true_label_name}_pred_{pred_label_name}"
                        # 저장 경로: {오답폴더}/true_{실제라벨}_pred_{예측라벨}/이미지명.jpg
                        save_dir = misclassified_dir / f"true_{true_label_name}_pred_{pred_label_name}"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        
                        shutil.copy2(original_path, save_dir / original_path.name)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Macro-F1: {macro_f1:.4f}')
        
        # 조기 종료 로직
        if val_acc > saved_val_acc or epoch == start_epoch:
            # 검증 정확도가 개선되었을 경우
            saved_val_loss = val_loss
            saved_val_acc = val_acc
            saved_train_loss = epoch_loss
            saved_train_acc = epoch_acc
            saved_epoch = epoch + 1
            saved_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f'  -> Val Acc 개선됨! ({saved_val_acc:.4f}) 모델 저장.')

            # 1. 클래스 이름 리스트 가져오기
            class_names = val_loader.dataset.classes
            
            # 2. 오차 행렬 계산
            cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
            
            # 3. 클래스별 정확도 및 오탐지 정보 딕셔너리 생성
            class_accuracies = {}
            per_class_misclassification = {}
            
            for i, class_name in enumerate(class_names):
                correct_count = cm[i, i]
                total_count = class_total[i]
                accuracy = 100 * correct_count / total_count if total_count > 0 else 0
                class_accuracies[class_name] = f"{accuracy:.2f}%"
                
                # 4. 해당 클래스의 오탐지 정보 계산
                misclassified_info = {}
                misclassified_total = total_count - correct_count
                
                for j, pred_class_name in enumerate(class_names):
                    if i != j:
                        misclassified_count = cm[i, j]
                        misclassified_ratio = (misclassified_count / misclassified_total * 100) if misclassified_total > 0 else 0
                        misclassified_info[pred_class_name] = {
                            "count": int(misclassified_count),
                            "ratio": f"{misclassified_ratio:.2f}%"
                        }
                
                per_class_misclassification[class_name] = {
                    "total_count": int(total_count),
                    "misclassified_total": int(misclassified_total),
                    "details": misclassified_info
                }

            # 5. 최종 메트릭스 딕셔너리에 추가
            saved_metrics = {
                'train_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'best_epoch': saved_epoch,
                'train_loss': f"{saved_train_loss:.4f}",
                'train_accuracy': f"{saved_train_acc.item():.4f}",
                'val_loss': f"{saved_val_loss:.4f}",
                'val_accuracy': f"{saved_val_acc.item():.4f}",
                'macro_f1_score': f"{macro_f1:.4f}",
                'per_class_accuracy': class_accuracies,
                'per_class_misclassification': per_class_misclassification
            }
        else:
            patience_counter += 1
            print(f'  -> val_acc 개선되지 않음. EarlyStopping Counter: {patience_counter}/{patience}')

        if patience_counter >= patience:
            print(f'\nEarly stopping! {patience} 에폭 동안 성능 개선이 없었습니다.')
            break
        
        # Best metrics update
        if 'best_val_acc' not in locals() or best_val_acc < val_acc:
            best_val_acc = val_acc
        if 'best_train_acc' not in locals() or best_train_acc < epoch_acc:
            best_train_acc = epoch_acc
        if 'best_val_loss' not in locals() or best_val_loss > val_loss:
            best_val_loss = val_loss
        if 'best_train_loss' not in locals() or best_train_loss > epoch_loss:
            best_train_loss = epoch_loss

        #scheduler.step()
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)  # Pass the validation loss
        else:
            scheduler.step()  # No argument needed for other schedulers
            
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
    
    # 가장 성능이 좋았던 모델의 가중치를 로드하여 반환
    model.load_state_dict(saved_model_wts)
    return model, saved_metrics