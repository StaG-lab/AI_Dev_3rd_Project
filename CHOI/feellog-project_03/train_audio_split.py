import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from pathlib import Path
import mlflow
import optuna
import json
import os
import shutil
import numpy as np
from datetime import datetime
import audiomentations as A

from core.data.audio_dataset import AudioDataset
from core.training.trainer import train_model
from core.data.DataCollatorForAudio import DataCollatorForAudio

# 라벨에 따라 다른 증강을 적용하는 래퍼(wrapper) 클래스
class ClassAwareAugment:
    def __init__(self, minority_classes, strong_augment, weak_augment):
        self.minority_classes = minority_classes
        self.strong_augment = strong_augment
        self.weak_augment = weak_augment

    def __call__(self, samples: np.ndarray, sample_rate: int, emotion: str):
        if emotion in self.minority_classes:
            return self.strong_augment(samples=samples, sample_rate=sample_rate)
        else:
            return self.weak_augment(samples=samples, sample_rate=sample_rate)

def objective(trial: optuna.Trial):
    """Optuna가 최적화할 목표 함수 (단일 실행)"""
    now_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    # MLflow는 Optuna의 Trial ID와 연동하여 각 실행을 기록
    with mlflow.start_run(run_name=f"trial_{trial.number}_{now_date}"):
        mlflow.log_params(trial.params)

        # 하이퍼파라미터 및 설정 
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        SAMPLING_PERCENT = 5
        NUM_EPOCHS = 5
        PATIENCE = 3
        MODEL_NAME = trial.suggest_categorical("model_name", ["hubert-base"])
        LR = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
        BACKBONE_LR_SCALE = trial.suggest_float("backbone_lr_scale", 0.05, 0.2, log=True)
        BATCH_SIZE = trial.suggest_categorical("batch_size", [4, 8])
        ACCUMULATION_STEPS = trial.suggest_int("accumulation_steps", 1, 4)

        mlflow.log_param("sampling_percent", SAMPLING_PERCENT)
        mlflow.log_params(trial.params)
        
        # 데이터 준비
        DATA_DIR = Path(f"./datasets/audio_sampling_sets/dataset_{SAMPLING_PERCENT}_percent")
        
        # 소수/다수 클래스 정의 (분석 결과 기반)
        minority_classes = ['surprise', 'disgust', 'fear']
        
        # 소수 클래스에 적용할 강력한 증강 파이프라인
        strong_augment = A.Compose([
            A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            A.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            A.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        ])

        # 다수 클래스에 적용할 약한 증강 파이프라인 (또는 A.Compose([])로 비워둘 수 있음)
        weak_augment = A.Compose([
            A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.2),
        ])

        # 차등 증강 적용기 생성
        train_augmenter = ClassAwareAugment(
            minority_classes=minority_classes,
            strong_augment=strong_augment,
            weak_augment=weak_augment
        )

        # 훈련셋에는 차등 증강 적용, 검증셋에는 미적용
        train_dataset = AudioDataset(metadata_path=DATA_DIR / "train.csv", audio_dir=DATA_DIR / "train", transform=train_augmenter)
        val_dataset = AudioDataset(metadata_path=DATA_DIR / "val.csv", audio_dir=DATA_DIR / "val", transform=None)
        
        # 모델 로드
        model_id = f"team-lucid/{MODEL_NAME}-korean"
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        model = AutoModelForAudioClassification.from_pretrained(
            model_id, num_labels=len(train_dataset.classes),
            label2id=train_dataset.class_to_idx, id2label=train_dataset.idx_to_class,
            ignore_mismatched_sizes=True
        ).to(DEVICE)
        
        # =================================================================
        # === 1단계: 머리(Classifier) 훈련 ===
        # =================================================================
        print("\n--- 파인튜닝 1단계 시작: 분류층(Head) 훈련 ---")
        
        # '몸통'에 해당하는 Hubert 모델의 모든 파라미터를 동결
        for param in model.hubert.parameters():
            param.requires_grad = False
            
        # 훈련시킬 파라미터는 '머리' 부분 뿐임
        head_params = [p for name, p in model.named_parameters() if not name.startswith(model.base_model_prefix)]
        optimizer_head = optim.AdamW(head_params, lr=LR)
            
        # 옵티마이저 및 스케줄러 준비
        # 모델의 base_model_prefix를 사용하여 동적으로 파라미터 분리
        backbone_prefix = model.base_model_prefix 
        backbone_params = [p for name, p in model.named_parameters() if name.startswith(backbone_prefix)]
        classifier_params = [p for name, p in model.named_parameters() if not name.startswith(backbone_prefix)]
        
        optimizer_grouped_parameters = [
            {'params': backbone_params, 'lr': LR * BACKBONE_LR_SCALE},
            {'params': classifier_params, 'lr': LR}
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        # 클래스 가중치 및 데이터로더
        # 훈련 데이터셋의 라벨 분포를 기반으로 가중치 계산
        class_names = train_dataset.classes
        labels = [train_dataset.class_to_idx[emotion] for emotion in train_dataset.df['emotion']]
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
        
        print(f"클래스 가중치 적용: { {name: f'{w:.2f}' for name, w in zip(class_names, class_weights)} }")
        
        # 데이터 콜레이터 및 로더
        data_collator = DataCollatorForAudio(feature_extractor=feature_extractor, padding=True)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator, num_workers=2, pin_memory=True)

        # 손실 함수에 클래스 가중치 적용
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # 3~5 에폭 정도만 '머리'를 훈련시켜 안정화
        head_epochs=5
        train_model(
            model, train_loader, val_loader, criterion, 
            optimizer_head, scheduler, DEVICE, 
            num_epochs=head_epochs, patience=3, start_epoch=0
        )
        
        # =================================================================
        # === 2단계: 전체 모델 미세 조정 ===
        # =================================================================
        print("\n--- 파인튜닝 2단계 시작: 전체 모델 미세 조정 ---")
        
        # 동결했던 '몸통' 파라미터를 모두 학습 가능하도록 해동
        for param in model.hubert.parameters():
            param.requires_grad = True
            
        # 차등 학습률을 적용한 옵티마이저 생성
        backbone_params = model.hubert.parameters()
        optimizer_full = optim.AdamW([
            {'params': backbone_params, 'lr': LR * BACKBONE_LR_SCALE},
            {'params': head_params, 'lr': LR}
        ])
        
        # 새로운 옵티마이저로 스케줄러 다시 정의
        scheduler_full = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_full, T_max=50)
        
        # 체크포인트 로드
        CHECKPOINT_DIR = Path("./checkpoints")
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        CHECKPOINT_PATH = CHECKPOINT_DIR / f'{MODEL_NAME}_{SAMPLING_PERCENT}_percent_best.pth'
        start_epoch = 0
        
        if CHECKPOINT_PATH.exists():
            print(f"체크포인트를 불러옵니다: {CHECKPOINT_PATH}")
            checkpoint = torch.load(CHECKPOINT_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"체크포인트 로드 완료! {start_epoch} 에폭부터 훈련을 재개합니다.")
        else:
            print("체크포인트가 존재하지 않습니다. 처음부터 훈련을 시작합니다.")

        # --- 7. 모델 훈련 ---
        best_wts_model, best_metrics = train_model(
            model, train_loader, val_loader, criterion,
            optimizer_full, scheduler_full, DEVICE, num_epochs=NUM_EPOCHS+start_epoch+head_epochs, start_epoch=start_epoch+head_epochs, 
            patience=PATIENCE, accumulation_steps=ACCUMULATION_STEPS
            )
        
        # 훈련된 모델 저장
        torch.save({'epoch': best_metrics['best_epoch'],
                     'model_state_dict': best_wts_model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler_state_dict': scheduler.state_dict()}, CHECKPOINT_PATH)

        # MLflow에 모델 저장
        MLFLOW_CHECKPOINT_PATH = CHECKPOINT_DIR / f'{MODEL_NAME}_{SAMPLING_PERCENT}_percent_best_{now_date}.pth'
        shutil.copy2(CHECKPOINT_PATH, MLFLOW_CHECKPOINT_PATH)
        mlflow.log_artifact(MLFLOW_CHECKPOINT_PATH, artifact_path="model")
        
        print("훈련된 모델 가중치가 저장되었습니다.")
        
        # --- 8. 결과 기록 ---
        if best_metrics:
            mlflow.log_metrics({
                "best_epoch": best_metrics['best_epoch'],
                "best_train_loss": float(best_metrics['train_loss']),
                "best_train_accuracy": float(best_metrics['train_accuracy']),
                "best_val_loss": float(best_metrics['val_loss']),
                "best_val_accuracy": float(best_metrics['val_accuracy']),
                "best_macro_f1": float(best_metrics['macro_f1_score']),
            })
            REPORT_PATH = CHECKPOINT_DIR / f"{MODEL_NAME}_{SAMPLING_PERCENT}_percent_report_trial_{trial.number}_{now_date}.json"
            with open(REPORT_PATH, 'w', encoding='utf-8') as f:
                json.dump(best_metrics, f, ensure_ascii=False, indent=4)
            mlflow.log_artifact(REPORT_PATH, artifact_path="reports")

        return float(best_metrics.get('val_accuracy', 0))

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://127.0.0.1:5003")
    mlflow.set_experiment("Audio Emotion Finetuning")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2) # n번의 다른 조합으로 실험
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Best Val Accuracy): {trial.value}")
    print(f"  Params: {trial.params}")