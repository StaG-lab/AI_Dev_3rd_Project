# /core/data/audio_dataset.py

import torch
import torchaudio
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, metadata_path: Path, audio_dir: Path, transform=None):
        self.audio_dir = audio_dir
        self.df = pd.read_csv(metadata_path)
        self.df['emotion'] = self.df['emotion'].str.lower()
        self.classes = sorted(self.df['emotion'].unique())
        self.class_to_idx = {label: i for i, label in enumerate(self.classes)}
        self.idx_to_class = {i: label for i, label in enumerate(self.classes)}
        
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = self.audio_dir / row['path']
        label_str = row['emotion']
        
        try:
            # 원본 waveform과 sample_rate, 그리고 label만 반환
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # transform이 지정된 경우에만 증강 적용
            if self.transform:
                # audiomentations는 numpy 배열을 입력으로 받음
                waveform_np = waveform.squeeze(0).numpy()
                # 라벨(label_str)을 transform에 함께 전달
                augmented_np = self.transform(samples=waveform_np, sample_rate=sample_rate, emotion=label_str)
                waveform = torch.from_numpy(augmented_np).unsqueeze(0)
            
            return {"waveform": waveform.squeeze(0), "sample_rate": sample_rate, "label": self.class_to_idx[label_str]}
        except Exception as e:
            print(f"Warning: Failed to load {audio_path}. Skipping. Error: {e}")
            return None # 실패 시 None 반환