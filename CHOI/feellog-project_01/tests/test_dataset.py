# /tests/test_dataset.py

import unittest
import torch
import shutil
from pathlib import Path
from core.data.dataset import EmotionDataset
from PIL import Image
from infrastructure.data.data_manager import sample_dataset
import json

class TestEmotionDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 전체에서 한번만 실행됩니다."""
        cls.data_dir = Path("test_data_for_dataset/sampled")
        
        # 1. 이미지 폴더 생성
        happy_img_dir = cls.data_dir / "기쁨"
        sad_img_dir = cls.data_dir / "슬픔"
        happy_img_dir.mkdir(parents=True, exist_ok=True)
        sad_img_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. 테스트용 실제 이미지 파일 생성
        test_image = Image.new('RGB', (10, 10))
        test_image.save(happy_img_dir / "happy_1.jpg")
        test_image.save(happy_img_dir / "happy_2.jpg")
        test_image.save(sad_img_dir / "sad_1.jpg")
        test_image.save(sad_img_dir / "sad_2.jpg")

        # 3. 샘플링된 라벨(.json) 폴더 및 파일 생성 (사용자가 수정한 로직과 동일한 결과물)
        label_dir = cls.data_dir / "labels"
        label_dir.mkdir(exist_ok=True)
        
        happy_labels = [{"filename": "happy_1.jpg"}, {"filename": "happy_2.jpg"}]
        with open(label_dir / "기쁨_sampled.json", "w", encoding="utf-8") as f:
            json.dump(happy_labels, f)

        sad_labels = [{"filename": "sad_1.jpg"}, {"filename": "sad_2.jpg"}]
        with open(label_dir / "슬픔_sampled.json", "w", encoding="utf-8") as f:
            json.dump(sad_labels, f)
    
    @classmethod
    def tearDownClass(cls):
        """테스트 클래스가 모두 끝나고 한번만 실행됩니다."""
        shutil.rmtree("test_data_for_dataset")

    def test_dataset_length(self):
        """데이터셋의 전체 길이를 정확히 반환하는지 테스트합니다."""
        dataset = EmotionDataset(data_dir=self.data_dir)
        self.assertEqual(len(dataset), 4) # 기쁨 2 + 슬픔 2 = 4

    def test_getitem(self):
        """데이터셋에서 한 개의 아이템을 올바르게 가져오는지 테스트합니다."""
        dataset = EmotionDataset(data_dir=self.data_dir)
        
        # 첫 번째 아이템을 가져옵니다.
        image, label = dataset[0]

        # 이미지의 타입이 torch.Tensor인지 확인
        self.assertIsInstance(image, torch.Tensor)
        # 이미지의 형태(shape)가 [채널, 높이, 너비]인지 확인 (예: [3, 224, 224])
        # transform을 적용하기 때문에 정확한 크기를 가정합니다.
        self.assertEqual(image.shape[0], 3) 
        
        # 라벨의 타입이 integer인지 확인
        self.assertIsInstance(label, int)

    def test_label_mapping(self):
        """감정(폴더명)이 숫자 라벨로 잘 변환되는지 테스트합니다."""
        dataset = EmotionDataset(data_dir=self.data_dir)
        # '기쁨' -> 0, '슬픔' -> 1 (또는 그 반대)
        self.assertIn(dataset.class_to_idx['기쁨'], [0, 1])
        self.assertIn(dataset.class_to_idx['슬픔'], [0, 1])
        self.assertEqual(len(dataset.class_to_idx), 2)

if __name__ == '__main__':
    unittest.main()