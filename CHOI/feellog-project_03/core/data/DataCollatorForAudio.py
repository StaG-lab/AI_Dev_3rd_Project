# /core/data/DataCollatorForAudio.py

import torch
from dataclasses import dataclass
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from typing import Dict, List, Union

@dataclass
class DataCollatorForAudio:
    feature_extractor: PreTrainedFeatureExtractor
    padding: Union[bool, str] = True
    
    def __call__(self, features: List[Dict[str, Union[torch.Tensor, int]]]) -> Dict[str, torch.Tensor]:
        # None 값을 가진 실패한 샘플들을 걸러냄
        valid_features = [f for f in features if f is not None]
        if not valid_features:
            return {}
        
        waveforms_np = [f["waveform"].numpy() for f in valid_features]
        labels = [f["label"] for f in valid_features]
        
        # NumPy 배열 리스트를 전달하여 패딩 및 변환 수행
        processed_batch = self.feature_extractor(
            waveforms_np, 
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=self.padding,
            do_normalize=False
        )
        
        # 라벨도 텐서로 변환하여 배치에 추가
        processed_batch["labels"] = torch.tensor(labels)
        
        return processed_batch