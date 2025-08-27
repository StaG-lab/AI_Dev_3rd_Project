import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, DataCollatorWithPadding

import pandas as pd
from pathlib import Path
import mlflow
import optuna

from core.data.audio_dataset import AudioDataset
from core.training.trainer import train_model
from core.data.DataCollatorForAudio import DataCollatorForAudio


def objective(trial: optuna.Trial):
    """Optuna가 최적화할 목표 함수"""
    with mlflow.start_run():
        # --- 하이퍼파라미터 및 모델 제안 ---
        #model_id = "team-lucid/hubert-large-korean"
        #model_name = "hubert-large"

        model_id = "team-lucid/hubert-base-korean"
        model_name = "hubert-base"
        #model_name = "wav2vec2"
        #model_name = trial.suggest_categorical("model_name", ["wav2vec2", "hubert-large"])
        
        lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
        mlflow.log_params(trial.params)

        # --- 데이터 준비 ---
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        DATA_DIR = Path("./datasets/audio_sampling_sets/dataset_5_percent")
        
        # train_dataset에는 train 폴더 경로를, val_dataset에는 val 폴더 경로를 정확히 전달
        train_dataset = AudioDataset(metadata_path=DATA_DIR / "train.csv", audio_dir=DATA_DIR / "train")
        val_dataset = AudioDataset(metadata_path=DATA_DIR / "val.csv", audio_dir=DATA_DIR / "val")

        # --- 모델 및 Feature Extractor 로드 ---
        if model_name == "wav2vec2":
            model_id = "inseong00/wav2vec2-large-xlsr-korean-autumn"
        elif model_name == "hubert-large":
            model_id = "team-lucid/hubert-large-korean"
        elif model_name == "hubert-base":
            model_id = "team-lucid/hubert-base-korean"
        else:
            raise ValueError("지원하지 않는 모델 이름입니다.")
            
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        
        model = AutoModelForAudioClassification.from_pretrained(
            model_id,
            num_labels=len(train_dataset.classes),
            label2id=train_dataset.class_to_idx,
            id2label=train_dataset.idx_to_class,
            ignore_mismatched_sizes=True # 사전 훈련된 모델의 분류층과 크기가 달라도 에러 없이 로드
        ).to(DEVICE)
        
        
        # 데이터 콜레이터 및 로더
        # 새로 만든 DataCollator 클래스를 사용
        data_collator = DataCollatorForAudio(feature_extractor=feature_extractor, padding=True)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
        val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=data_collator)

        # DataLoader 생성 시, collate_fn 인자로 data_collator 객체를 직접 전달
        #train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
        #val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=data_collator)

        # --- 5. 훈련 준비 및 시작 ---
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss() # trainer 내부에서 사용되지 않지만 형식상 전달
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        best_model, best_metrics = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, DEVICE, num_epochs=10, patience=3)
        
        val_accuracy = float(best_metrics.get('val_accuracy', 0))
        mlflow.log_metric("best_val_accuracy", val_accuracy)
        
        return val_accuracy # Optuna는 정확도를 '최대화'하는 것을 목표로 함

if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10) # 10번의 다른 조합으로 실험
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Best Val Accuracy): {trial.value}")
    print(f"  Params: {trial.params}")