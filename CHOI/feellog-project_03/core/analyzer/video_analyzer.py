# ./core/analyzer/video_analyzer.py

import torch
from torchvision import transforms
from pathlib import Path
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
import numpy as np
import json
from PIL import Image
import os
import time

# 우리 프로젝트의 핵심 모듈들을 import
from core.models.model_factory import create_model
from core.analyzer.audio_analyzer import VoiceAnalyzer

class VideoAnalyzer:
    def __init__(self, image_model_name: str, image_model_weights_path: str, api_key: str, voice_model_name: str = "wav2vec2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 이미지 감정 분석 모델 로드
        print("이미지 감정 분석 모델을 로드합니다...")
        self.image_model = create_model(model_name=image_model_name, num_classes=7, pretrained=False)
        self.image_model.load_state_dict(torch.load(image_model_weights_path))
        self.image_model.to(self.device)
        self.image_model.eval()
        
        # 이미지 전처리 transform 정의 (EmoNet 기준)
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # DNN 얼굴 탐지기 모델 로드
        print("DNN 얼굴 탐지기를 로드합니다...")
        proto_path = Path("./infrastructure/models/deploy.prototxt")
        model_path = Path("./infrastructure/models/res10_300x300_ssd_iter_140000.caffemodel")
        self.face_net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
        
        # 2. 음성 감정 분석기 로드
        print("음성 감정 분석기를 로드합니다...")
        self.voice_analyzer = VoiceAnalyzer(api_key=api_key, voice_model_name=voice_model_name)
        
        self.emotion_labels = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']

    def extract_frames(self, video_path: Path, num_frames: int = 30) -> list:
        """비디오에서 N개의 프레임을 균일한 간격으로 추출합니다."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        cap.release()
        return frames

    def _detect_and_crop_face(self, frame_pil: Image.Image, confidence_threshold=0.5) -> Image.Image:
        """PIL 이미지를 입력받아 얼굴을 탐지하고, 얼굴 부분만 잘라낸 PIL 이미지를 반환합니다."""
        cv_img = np.array(frame_pil)
        h, w = cv_img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(cv_img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        best_confidence = 0
        best_box = None
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > best_confidence:
                best_confidence = confidence
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                best_box = box.astype("int")

        if best_confidence > confidence_threshold:
            (startX, startY, endX, endY) = best_box
            # 약간의 여백을 줘서 얼굴이 너무 꽉 차지 않게 함
            padding_x = int((endX - startX) * 0.1)
            padding_y = int((endY - startY) * 0.1)
            startX = max(0, startX - padding_x)
            startY = max(0, startY - padding_y)
            endX = min(w, endX + padding_x)
            endY = min(h, endY + padding_y)
            return frame_pil.crop((startX, startY, endX, endY))
        else:
            return None # 얼굴 탐지 실패
    
    def analyze_image_emotions(self, frames: list) -> dict:
        """추출된 프레임들의 감정을 분석하고 종합합니다."""
        all_preds = []
        with torch.no_grad():
            for frame in frames:
                # 얼굴을 먼저 탐지하고 잘라냄
                face_crop = self._detect_and_crop_face(frame)
                if face_crop:
                    img_tensor = self.image_transform(frame).unsqueeze(0).to(self.device)
                    outputs_dict = self.image_model(img_tensor)
                    emotion_preds_tensor = outputs_dict['expression']
                    _, preds = torch.max(emotion_preds_tensor, 1)
                    all_preds.append(preds.item())
        
        # 가장 많이 예측된 감정을 찾아 반환
        if not all_preds:
            return {"dominant_emotion": "N/A", "distribution": {}}
            
        dominant_emotion_idx = max(set(all_preds), key=all_preds.count)
        distribution = {self.emotion_labels[i]: all_preds.count(i) / len(all_preds) for i in set(all_preds)}
        
        return {
            "dominant_emotion": self.emotion_labels[dominant_emotion_idx],
            "distribution": distribution
        }

    def analyze(self, video_path_str: str, output_dir: str = "temp"):
        """하나의 비디오 파일에 대한 전체 이미지/음성 감정 분석을 수행합니다."""
        total_start_time = time.perf_counter()
        timings = {}
        
        video_path = Path(video_path_str)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. 음성 추출
        print(f"'{video_path.name}'에서 음성을 추출합니다...")
        audio_path = output_path / f"{video_path.stem}.wav"
        voice_analysis = {"error": "Audio track not found or could not be extracted."} # 기본값 설정
        
        try:
            with VideoFileClip(video_path_str) as video_clip:
                if video_clip.audio is None:
                    print("[경고] 원본 비디오에 오디오 트랙이 없습니다.")
                else:
                    video_clip.audio.write_audiofile(str(audio_path), codec='pcm_s16le', logger=None)
                    
                    # 파일이 정상적으로 생성되었고 비어있지 않은지 확인
                    if audio_path.exists() and os.path.getsize(str(audio_path)) > 0:
                        # 3. 음성 감정 분석
                        print("음성 감정 분석을 시작합니다...")
                        voice_analysis = self.voice_analyzer.analyze(str(audio_path))
                    else:
                        print("[경고] 오디오 파일이 비어있거나 생성되지 않았습니다.")
                    
        except Exception as e:
            print(f"[에러] 음성 추출 중 문제 발생: {e}")

        # 2. 프레임 추출
        print(f"'{video_path.name}'에서 프레임을 추출합니다...")
        frames = self.extract_frames(video_path)
        
        # 3. 음성 감정 분석
        print("음성 감정 분석을 시작합니다...")
        start_time = time.perf_counter()
        voice_analysis = self.voice_analyzer.analyze(str(audio_path))
        timings["audio_analysis_seconds"] = time.perf_counter() - start_time
        
        # 4. 이미지 감정 분석
        start_time = time.perf_counter()
        print("이미지 감정 분석을 시작합니다...")
        image_analysis = self.analyze_image_emotions(frames)
        timings["image_analysis_seconds"] = time.perf_counter() - start_time

        # 5. 결과 종합
        total_time_elapsed = time.perf_counter() - total_start_time
        timings["total_elapsed_seconds"] = total_time_elapsed
        final_result = {
            "video_file": video_path.name,
            "visual_analysis": image_analysis,
            "audio_analysis": voice_analysis,
            "performance": timings
        }
        
        # 임시 오디오 파일 삭제
        if audio_path.exists():
            audio_path.unlink()
                
        return final_result