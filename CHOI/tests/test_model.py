# /tests/test_model.py (수정)

import unittest
import torch
import torch.nn as nn
from core.models.model_factory import create_model

class TestModelFactory(unittest.TestCase):
    def test_create_different_models(self):
        """
        다양한 종류의 모델을 생성하고, 분류기의 출력 노드 수가
        지정한 클래스 수와 일치하는지 테스트합니다.
        """
        num_classes = 7
        # 테스트해볼 모델 이름 리스트
        model_names = ['resnet18', 'mobilenet_v3_small']

        for model_name in model_names:
            with self.subTest(model=model_name): # 각 모델별로 서브테스트 실행
                model = create_model(model_name=model_name, num_classes=num_classes)
                
                # 1. 반환된 객체가 PyTorch 모델이 맞는지 확인
                self.assertIsInstance(model, nn.Module)
                
                # 2. 모델 종류에 따라 마지막 레이어의 이름이 다르므로, 이를 확인하고 테스트
                if "resnet" in model_name:
                    final_layer = model.fc
                elif "mobilenet" in model_name:
                    final_layer = model.classifier[-1]
                else:
                    self.fail(f"{model_name}의 마지막 레이어를 확인할 수 없습니다.")
                
                self.assertEqual(final_layer.out_features, num_classes)

if __name__ == '__main__':
    unittest.main()