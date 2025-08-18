# /core/training/trainer.py

import torch
import time
import copy
from torch.cuda.amp import autocast, GradScaler  # [A1] AMP 유틸 사용

def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs=100, patience=10,
                use_amp=True, amp_in_val=True,              # [A2] AMP 스위치
                grad_clip=None, log_interval=50,            # [A3] 로그/클리핑
                save_best_path=None,                        # [A4] 베스트 ckpt 저장
                set_tf32=True, cudnn_benchmark=True,
                # unfreeze_at=0, lr_backbone=None,
                print_metrics=('acc','macro_f1','per_class_f1','confmat'),
                ):       # [A5] 성능 플래그
    """
    모델 훈련을 위한 메인 루프.

    Args:
        model: 훈련시킬 PyTorch 모델.
        train_loader: 훈련 데이터로더.
        val_loader: 검증 데이터로더.
        criterion: 손실 함수.
        optimizer: 옵티마이저.
        device: torch.device (cuda/cpu).
        num_epochs (int): 총 훈련 에폭 수.
        patience (int): 조기 종료 인내 에폭 수.
        use_amp (bool): 훈련 시 AMP 사용 여부. [A2]
        amp_in_val (bool): 검증에서도 AMP 허용 여부. [A2]
        grad_clip (float|None): grad norm 클리핑 값. [A3]
        log_interval (int): 배치 로그 주기(배치마다 출력 간격). [A3]
        save_best_path (str|None): best val loss 시 ckpt 경로. [A4]
        set_tf32 (bool): TF32 허용(매트멀/커드NN). [A5]
        cudnn_benchmark (bool): cudnn.benchmark 활성화. [A5]
    """
    since = time.time()

    # ✅ 백엔드 성능 플래그 설정(저장/적용)
    prev_benchmark = torch.backends.cudnn.benchmark          # [A6]
    prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32 # [A6]
    prev_tf32_cudnn = getattr(torch.backends.cudnn, "allow_tf32", None)  # [A6]
    if cudnn_benchmark:                                      # [A7]
        torch.backends.cudnn.benchmark = True
    if set_tf32:                                             # [A8]
        torch.backends.cuda.matmul.allow_tf32 = True
        if prev_tf32_cudnn is not None:
            torch.backends.cudnn.allow_tf32 = True

    # ✅ AMP 스위치 및 스케일러
    amp_enabled = bool(use_amp and torch.cuda.is_available() and getattr(device, "type", "cpu") == "cuda")  # [A9]
    scaler = GradScaler(enabled=amp_enabled)                 # [A9]
    non_blocking_io = amp_enabled                            # [A10] pin_memory=True일 때 효과적

    # ✅ 조기 종료 변수 초기화(원본 그대로)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    unfreezed = False

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # === 1. 훈련 단계 ===
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        # # === 에폭 시작 ===
        # if (unfreeze_at and (epoch + 1) == unfreeze_at) and (not unfreezed):
        #     to_add = []
        #     for name, p in model.named_parameters():
        #         # 헤드가 아닌(이미 freeze되어 있던) 파라미터 전부 활성화
        #         if not any(k in name for k in ['fc', 'classifier']) and (not p.requires_grad):
        #             p.requires_grad = True
        #             to_add.append(p)
        #     if to_add:
        #         # 백본용 파라미터 그룹 추가(헤드:백본 = 10:1 가정, 호출부에서 lr_backbone 지정)
        #         backbone_group = {'params': to_add, 'lr': lr_backbone or max(pg['lr'] for pg in optimizer.param_groups)/10.0}
        #         if hasattr(optimizer, 'defaults') and 'weight_decay' in optimizer.defaults:
        #             backbone_group['weight_decay'] = optimizer.defaults['weight_decay']
        #         optimizer.add_param_group(backbone_group)
        #         unfreezed = True
        #         print(f'>>> Unfreezed backbone: +{len(to_add)} params, lr={optimizer.param_groups[-1]["lr"]:.2e}')

        for step, (inputs, labels) in enumerate(train_loader):  # [A11] step 사용
            inputs = inputs.to(device, non_blocking=non_blocking_io)   # [A12]
            labels = labels.to(device, non_blocking=non_blocking_io)   # [A12]
            optimizer.zero_grad(set_to_none=True)                       # [A13]

            # AMP forward
            with autocast(enabled=amp_enabled):                         # [A14]
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            # backward/step (AMP-aware)
            if amp_enabled:                                             # [A15]
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # 배치 진행 로그(선택)
            if log_interval and (step + 1) % log_interval == 0:         # [A16]
                seen = (step + 1) * inputs.size(0)
                train_loss_so_far = running_loss / min(seen, len(train_loader.dataset))
                train_acc_so_far = running_corrects.double() / min(seen, len(train_loader.dataset))
                print(f"[E{epoch+1} B{step+1}] loss={train_loss_so_far:.4f} acc={train_acc_so_far:.3f}")

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')

        num_classes = getattr(train_loader.dataset, 'classes', None)
        num_classes = len(num_classes) if num_classes else (outputs.shape[1] if 'outputs' in locals() else 7)
        confmat = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)

        # === 2. 검증 단계 ===
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=non_blocking_io)     # [A17]
                labels = labels.to(device, non_blocking=non_blocking_io)     # [A17]
                with autocast(enabled=amp_enabled and amp_in_val):           # [A18]
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

                        # ✅ 매 배치마다 혼동행렬 누적
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confmat[t.long(), p.long()] += 1

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')

        # ✅ 에폭 끝에 F1/ConfMat 계산·출력 (원하면 print_metrics 플래그로 가드)
        tp = torch.diag(confmat).float()
        fp = confmat.sum(0).float() - tp
        fn = confmat.sum(1).float() - tp
        f1_per_class = (2*tp) / (2*tp + fp + fn + 1e-9)
        macro_f1 = f1_per_class.mean().item()
        print(f'Macro-F1: {macro_f1:.4f}')
        print('Per-class F1:', ', '.join([f'{i}:{f1_per_class[i].item():.3f}' for i in range(num_classes)]))
        print('ConfMat (rows=true, cols=pred):\n', confmat.cpu().numpy())

        # 조기 종료 로직 + optional best 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f'  -> Val Loss 개선됨! ({best_val_loss:.4f})')
            if save_best_path is not None:                                   # [A19]
                torch.save({'model': model.state_dict(),
                            'epoch': epoch+1,
                            'val_loss': best_val_loss}, save_best_path)
        else:
            patience_counter += 1
            print(f'  -> Val Loss 개선되지 않음. EarlyStopping Counter: {patience_counter}/{patience}')

        if patience_counter >= patience:
            print(f'\nEarly stopping! {patience} 에폭 동안 성능 개선이 없었습니다.')
            break

       

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Loss: {best_val_loss:.4f}')

    # ✅ 백엔드 플래그 복원
    torch.backends.cudnn.benchmark = prev_benchmark            # [A20]
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32_matmul   # [A20]
    if prev_tf32_cudnn is not None:                            # [A20]
        torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn

    # 가장 성능이 좋았던 모델의 가중치를 로드하여 반환
    model.load_state_dict(best_model_wts)
    return model
