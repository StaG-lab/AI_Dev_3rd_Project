import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModel 
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
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F

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

# 어텐션 풀링을 수행하는 헤드
class AttentionHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        # 각 시간 단계의 "중요도"를 학습하기 위한 레이어
        self.attention_weights = nn.Linear(input_size, 1)
        # 최종 분류를 위한 레이어
        self.classifier = nn.Linear(input_size, num_classes)

    def forward(self, features): # features shape: [batch, seq_len, hidden_size]
        # 1. 각 시간 단계별 중요도(attention score) 계산
        attention_scores = self.attention_weights(features).squeeze(-1)
        
        # 2. Softmax를 통해 확률적인 가중치로 변환
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 3. 계산된 가중치를 원래 특징에 곱하여 가중 평균 계산 (어텐션 풀링)
        #    -> 중요한 부분의 특징은 강조되고, 중요하지 않은 부분은 억제됨
        weighted_features = torch.sum(features * attention_weights.unsqueeze(-1), dim=1)
        
        # 4. 최종적으로 가중 평균된 특징을 사용하여 감정 분류
        logits = self.classifier(weighted_features)
        return logits
    
# 커스텀 분류기 헤드 정의
class CustomClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes, dropout_prob=0.5):
        super().__init__()
        self.dense = nn.Linear(input_size, input_size // 2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(input_size // 2, num_classes)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

# HuBERT 몸통과 커스텀 헤드를 결합한 최종 모델을 정의.
class EmotionFineTuningModel(nn.Module):
    def __init__(self, model_id, num_labels):
        super().__init__()
        # '몸통' 부분인 기본 HuBERT 모델을 로드
        self.base_model = AutoModel.from_pretrained(model_id)
        
        # 커스텀 헤드를 AttentionHead로 교체
        self.classifier = AttentionHead(self.base_model.config.hidden_size, num_labels)
        self.base_model_prefix = "base_model"
        '''
        # '머리' 부분인 커스텀 분류기 생성
        self.classifier = CustomClassificationHead(self.base_model.config.hidden_size, num_labels)
        # 나중에 파라미터 분리를 위해 몸통의 이름을 저장
        self.base_model_prefix = "base_model"
        '''
    def forward(self, input_values, attention_mask=None):
        outputs = self.base_model(input_values=input_values, attention_mask=attention_mask)
        # torch.mean을 사용한 평균 풀링을 제거
        # pooled_features = torch.mean(outputs.last_hidden_state, dim=1)
        
        # 시퀀스 전체를 AttentionHead에 전달
        logits = self.classifier(outputs.last_hidden_state)
        return logits  

def objective(trial: optuna.Trial):
    """Optuna가 최적화할 목표 함수 (단일 실행)"""
    now_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    # MLflow는 Optuna의 Trial ID와 연동하여 각 실행을 기록
    with mlflow.start_run(run_name=f"trial_{trial.number}_{now_date}"):
        # 하이퍼파라미터 및 설정 
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        SAMPLING_PERCENT = 5
        
        # 1단계와 2단계의 에폭 수를 명확히 분리하여 정의
        HEAD_TUNE_EPOCHS = 15   # 1단계에서 '머리'만 훈련시킬 에폭 수
        FULL_TUNE_EPOCHS = 35   # 2단계에서 전체 모델을 훈련시킬 에폭 수
        PATIENCE = 10           # 조기 종료 '인내심'도 충분히 늘려줌
        
        # if model_name == "wav2vec2":
        # model_id = "inseong00/wav2vec2-large-xlsr-korean-autumn"
        
        MODEL_NAME = trial.suggest_categorical("model_name", ["hubert-base"])
        # 초기 학습 불안정성을 줄이기 위해 학습률 범위를 약간 낮춤
        EARLY_LR = trial.suggest_float("lr", 1e-6, 2e-5, log=True)
        #EARLY_LR = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
        #LATE_LR = trial.suggest_float("lr", 5e-6, 5e-5, log=True)
        #EARLY_LR = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
        BACKBONE_LR_SCALE = trial.suggest_float("backbone_lr_scale", 0.05, 0.2, log=True)
        BATCH_SIZE = trial.suggest_categorical("batch_size", [4, 8])
        ACCUMULATION_STEPS = trial.suggest_int("accumulation_steps", 1, 4)
        
        mlflow.log_params(trial.params)
        mlflow.log_param("sampling_percent", SAMPLING_PERCENT)
        
        
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
        
        
        # 새로운 EmotionFineTuningModel을 생성
        model_id = f"team-lucid/{MODEL_NAME}-korean"
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        model = EmotionFineTuningModel(model_id, num_labels=len(train_dataset.classes)).to(DEVICE)
    
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
        
        '''
        # Optuna를 사용할 땐 필요없음.
        if trial.number == 0: 
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
        '''
        # =================================================================
        # === 1단계: 커스텀 헤드 훈련 ===
        # =================================================================
        print(f"\n--- Trial {trial.number}, 파인튜닝 1단계 시작: 커스텀 헤드 훈련 ---")
        
        for name, param in model.named_parameters():
            if name.startswith(model.base_model_prefix):
                param.requires_grad = False
        
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer_head = optim.AdamW(head_params, lr=EARLY_LR)
        # 1단계는 간단한 스케줄러 사용
        scheduler_head = torch.optim.lr_scheduler.LinearLR(optimizer_head, start_factor=1.0, end_factor=0.1, total_iters=HEAD_TUNE_EPOCHS)
        train_model(
            model, train_loader, val_loader, criterion, optimizer_head, scheduler_head, DEVICE,
            num_epochs=HEAD_TUNE_EPOCHS, patience=5, accumulation_steps=ACCUMULATION_STEPS
        )
        
        # =================================================================
        # === 2단계: 전체 모델 미세 조정 ===
        # =================================================================
        print(f"\n--- Trial {trial.number}, 파인튜닝 2단계 시작: 전체 모델 미세 조정 ---")
        
        # 동결했던 '몸통' 파라미터를 모두 학습 가능하도록 해동
        for param in model.parameters():
            param.requires_grad = True
            
        # 차등 학습률을 적용한 전체 모델용 옵티마이저 생성
        backbone_params = model.base_model.parameters()
        classifier_params = model.classifier.parameters()
        
        optimizer_full = optim.AdamW([
            {'params': backbone_params, 'lr': EARLY_LR * BACKBONE_LR_SCALE},
            {'params': classifier_params, 'lr': EARLY_LR}
        ])
        
        # Warmup을 포함한 스케줄러 생성
        num_training_steps = len(train_loader) * FULL_TUNE_EPOCHS
        num_warmup_steps = int(num_training_steps * 0.1) # 첫 10% 스텝 동안 워밍업
        
        scheduler_full = get_linear_schedule_with_warmup(
            optimizer_full,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        #scheduler_full = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_full, T_max=FULL_TUNE_EPOCHS)
        
        best_model, best_metrics = train_model(
            model, train_loader, val_loader, criterion, optimizer_full, scheduler_full, DEVICE,
            num_epochs=FULL_TUNE_EPOCHS,
            patience=PATIENCE, 
            start_epoch=0,
            accumulation_steps=ACCUMULATION_STEPS
        )
        
        '''
        # 훈련된 모델은 MLflow에 아티팩트로 저장 (추후에 결과가 좋아지면 저장)
        if best_wts_model:
            CHECKPOINT_DIR = Path("./checkpoints")
            CHECKPOINT_DIR.mkdir(exist_ok=True)
            BEST_MODEL_PATH = CHECKPOINT_DIR / f"trial_{trial.number}_best_model.pth"
            
            torch.save({'model_state_dict': best_wts_model.state_dict(),
                     'optimizer_state_dict': optimizer_full.state_dict(),
                     'scheduler_state_dict': scheduler_full.state_dict()}, BEST_MODEL_PATH)
            
            mlflow.log_artifact(BEST_MODEL_PATH, artifact_path="model")
        '''            
        # --- 8. 결과 기록 ---
        if best_metrics:
            REPORT_DIR = Path("./reports")
            REPORT_DIR.mkdir(exist_ok=True)
            mlflow.log_metrics({
                "best_train_loss": float(best_metrics['train_loss']),
                "best_train_accuracy": float(best_metrics['train_accuracy']),
                "best_val_loss": float(best_metrics['val_loss']),
                "best_val_accuracy": float(best_metrics['val_accuracy']),
                "best_macro_f1": float(best_metrics['macro_f1_score']),
            })
            REPORT_PATH = REPORT_DIR / f"{MODEL_NAME}_{SAMPLING_PERCENT}_percent_report_trial_{trial.number}_{now_date}.json"
            with open(REPORT_PATH, 'w', encoding='utf-8') as f:
                json.dump(best_metrics, f, ensure_ascii=False, indent=4)
            mlflow.log_artifact(REPORT_PATH, artifact_path="reports")

        return float(best_metrics.get('val_accuracy', 0))

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://127.0.0.1:5003")
    mlflow.set_experiment("Audio Emotion Finetuning")
    
    # 연구 기록을 저장할 데이터베이스 파일과 연구 이름을 정의
    STUDY_NAME = "audio-finetune-study-v1" # 연구에 고유한 이름을 부여
    STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db" # SQLite 데이터베이스 파일로 저장
    
    # storage와 study_name을 지정하고, load_if_exists=True로 설정
    # 기존 연구가 있으면 불러오고, 없으면 새로 생성
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME,
        direction="maximize",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=1) # n번의 다른 조합으로 실험
    
    print("\n--- Hyperparameter Optimization Finished ---")
    print(f"Total trials in this study: {len(study.trials)}")
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Best Val Accuracy): {trial.value}")
    print(f"  Params: {trial.params}")