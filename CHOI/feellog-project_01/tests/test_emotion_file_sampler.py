# /tests/test_emotion_file_sampler.py

import unittest
import os
import json
import shutil
from pathlib import Path

from infrastructure.data.data_manager import sample_dataset
from run_sampling import VAL_SPLIT_RATIO

class TestEmotionFileSampler(unittest.TestCase):

    def setUp(self):
        """테스트 환경을 설정합니다."""
        # 기본 경로 설정
        self.source_img_dir = Path("test_data/source_images")
        self.source_label_dir = Path("test_data/source_labels")
        self.output_dir = Path("test_data/output")

        # 테스트 폴더 초기화
        for d in [self.source_img_dir, self.source_label_dir, self.output_dir]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        # 1. 테스트용 가짜 이미지 파일들을 생성합니다. (총 100개)
        for i in range(60):
            (self.source_img_dir / f"happy_{i}.jpg").touch()
        for i in range(40):
            (self.source_img_dir / f"sad_{i}.jpg").touch()

        # 2. 테스트용 가짜 라벨 파일을 생성합니다. (감정별 1개 파일)
        # 기쁨.json
        happy_labels = [{"filename": f"happy_{i}.jpg"} for i in range(60)]
        with open(self.source_label_dir / "기쁨.json", "w", encoding="utf-8") as f:
            json.dump(happy_labels, f, ensure_ascii=False, indent=2)

        # 슬픔.json
        sad_labels = [{"filename": f"sad_{i}.jpg"} for i in range(40)]
        with open(self.source_label_dir / "슬픔.json", "w", encoding="utf-8") as f:
            json.dump(sad_labels, f, ensure_ascii=False, indent=2)

    def tearDown(self):
        """테스트 환경을 정리합니다."""
        shutil.rmtree("test_data")

    def test_sampling_from_emotion_files(self):
        """감정별 파일 기반 샘플링 로직을 테스트합니다."""
        sample_rate = 0.1  # 10%
        VAL_SPLIT_RATIO = 0.2
    
        # ✅ 통합된 함수를 'file_per_emotion' 모드로 호출
        sample_dataset(
            source_image_dir=self.source_img_dir,
            source_label_dir=self.source_label_dir,
            output_dir=self.output_dir,
            sample_rate=sample_rate,
            mode='file_per_emotion',
            val_split_ratio=VAL_SPLIT_RATIO
        )

        # 1. 출력 폴더 내에 감정별 하위 폴더가 생성되었는지 확인
        output_happy_dir = self.output_dir / "기쁨"
        output_sad_dir = self.output_dir / "슬픔"
        self.assertTrue(output_happy_dir.exists() and output_happy_dir.is_dir())
        self.assertTrue(output_sad_dir.exists() and output_sad_dir.is_dir())

        # 2. 각 감정 폴더에 올바른 개수의 파일이 샘플링되었는지 확인
        num_happy_files = len(list(output_happy_dir.glob("*.jpg")))
        num_sad_files = len(list(output_sad_dir.glob("*.jpg")))

        self.assertEqual(num_happy_files, 6)  # 60개의 10%
        self.assertEqual(num_sad_files, 4)    # 40개의 10%

if __name__ == '__main__':
    unittest.main()