# /tests/test_data_sampler.py

import unittest
import os
import shutil
import json
from pathlib import Path

from infrastructure.data.data_manager import sample_dataset

class TestDataSampler(unittest.TestCase):

    def setUp(self):
        """테스트를 실행하기 전에 매번 호출되는 메서드"""
        # 테스트를 위한 가짜 원본 데이터 폴더와 출력 폴더를 설정
        self.source_dir = Path("test_data/source")
        self.output_dir = Path("test_data/output")
        self.label_dir = Path("test_data/labels")

        # 기존에 폴더가 있다면 삭제하고 새로 생성
        for d in [self.source_dir, self.output_dir, self.label_dir]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        # 테스트용 가짜 데이터 파일 생성
        # 총 200개의 데이터 (이미지 100개, 라벨 100개)
        # 감정 비율: happy 60%, sad 40%
        for i in range(100):
            if i < 60:
                emotion = "happy"
            else:
                emotion = "sad"

            # 이미지 파일 (빈 파일)
            (self.source_dir / f"image_{i}.jpg").touch()
            # 라벨 파일 (json)
            with open(self.label_dir / f"image_{i}.json", "w") as f:
                json.dump({"emotion": emotion}, f)

    def tearDown(self):
        """테스트가 끝난 후 매번 호출되는 메서드"""
        # 테스트 중에 생성된 폴더들을 정리
        shutil.rmtree("test_data")

    def test_sampling_logic(self):
        """실제 샘플링 로직을 테스트하는 핵심 메서드"""
        # 10% 샘플링을 실행 (총 200개 파일 중 20개 샘플링 예상)
        sample_rate = 0.1 # 10%
        VAL_SPLIT_RATIO = 0.2
        # ✅ 통합된 함수를 'file_per_image' 모드로 호출
        sample_dataset(
            source_image_dir=self.source_dir,
            source_label_dir=self.label_dir,
            output_dir=self.output_dir,
            sample_rate=sample_rate,
            mode='file_per_image',
            val_split_ratio=VAL_SPLIT_RATIO
        )

        # 1. 출력 폴더가 생성되었는지 확인
        self.assertTrue(self.output_dir.exists())

        # 2. 샘플링된 파일 수가 올바른지 확인 (이미지 10개, 라벨 10개)
        output_files = list(self.output_dir.glob('**/*.*'))
        self.assertEqual(len(output_files), 20) # 100개 이미지의 10%는 10개 * 2(이미지+라벨)

        # 3. 감정 라벨 비율이 유지되는지 확인
        # happy: 6개, sad: 4개여야 함
        emotion_counts = {"happy": 0, "sad": 0}
        for label_file in self.output_dir.glob('labels/*.json'):
            with open(label_file, 'r') as f:
                data = json.load(f)
                emotion_counts[data['emotion']] += 1
        
        self.assertEqual(emotion_counts['happy'], 6) # 60개의 10%
        self.assertEqual(emotion_counts['sad'], 4)   # 40개의 10%

if __name__ == '__main__':
    unittest.main()