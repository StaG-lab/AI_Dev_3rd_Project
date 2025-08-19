## 0. 최소 하드웨어 및 환경 요건

- **GPU**: NVIDIA GPU (Ampere 아키텍처 이상 권장 — RTX 30 시리즈, A100 등)
- **CUDA**: 11.8 이상 (PyTorch 버전과 호환)
- **cuDNN**: 8.7 이상 (PyTorch 빌드와 맞는 버전)
- **Python**: 3.9\~3.11 (프로젝트 .venv 환경과 동일)
- **PyTorch**: CUDA 지원 빌드 (pip/conda 설치 시 `+cu118` 등 명시)
- **운영체제**: Ubuntu 20.04+ 또는 Windows 10/11 (WSL2 권장)
- **메모리**: GPU 8GB 이상 (AMP 사용 시 더 적게도 가능), 시스템 RAM 16GB 이상

---

## 1. CUDA 성능 플래그 최적화

**코드 예시**

```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch.backends.cudnn, "allow_tf32"):
    torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
```

**효과**: CNN 계열 학습 속도 10\~30% 향상 (입력 크기 고정 시)

---

## 2. AMP (Automatic Mixed Precision)

**코드 예시**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**효과**: GPU 메모리 ~~50% 절감, 연산 속도 20~~40% 향상

---

## 3. DataLoader I/O 튜닝

**코드 예시**

```python
train_loader = DataLoader(
    dataset, batch_size=64, shuffle=True,
    num_workers=min(8, os.cpu_count()),
    pin_memory=True, persistent_workers=True,
    prefetch_factor=2, drop_last=True
)
```

**효과**: I/O 병목 제거로 속도 10\~20% 향상

---

## 4. 배치 단위 진행률 로그

**코드 예시**

```python
if (step + 1) % log_interval == 0:
    print(f"[E{epoch+1} B{step+1}] loss={running_loss/seen:.4f} acc={correct/seen:.3f}")
```

**효과**: 학습 진행 상황 조기 파악, 이상 탐지 효율 증가

---

## 5. 선택적 배치 수 제한 (steps\_per\_epoch)

**코드 예시**

```python
for step, (x, y) in enumerate(train_loader):
    ...
    if steps_per_epoch and (step+1) >= steps_per_epoch:
        break
```

**효과**: 빠른 반복 실험 및 하이퍼파라미터 튜닝에 유리

---

## 종합 효능 요약

| 기술                     | 속도 향상   | 메모리 절감 | 안정성            | 주요 용도       |
| ---------------------- | ------- | ------ | -------------- | ----------- |
| cudnn.benchmark + TF32 | 10\~30% | -      | 안정             | CNN·고정크기 입력 |
| AMP                    | 20\~40% | \~50%  | 안정(GradScaler) | 큰 모델·고해상도   |
| DataLoader 튜닝          | 10\~20% | -      | 안정             | I/O 병목 제거   |
| 배치 로그                  | -       | -      | 높음             | 조기 디버깅      |
| steps\_per\_epoch      | -       | -      | 높음             | 빠른 실험       |

