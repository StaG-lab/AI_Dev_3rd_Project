# ./core/analyzer/video_analyzer.py

import torch
from torchvision import transforms
from pathlib import Path
import cv2
from moviepy import VideoFileClip
import numpy as np
import json
from PIL import Image
import os
import time
from typing import List, Dict, Union, Any, Optional
from core.utils.analysis_logger import AnalysisLogger # AnalysisLogger 임포트

# 우리 프로젝트의 핵심 모듈들을 import
from core.models.model_factory import create_model
from core.analyzer.audio_analyzer import VoiceAnalyzer # VoiceAnalyzer는 이제 세그먼트 단위 분석 담당
from core.analyzer.speech_segmenter import SpeechSegmenter # 새로운 SpeechSegmenter 임포트

class BatchVideoAnalyzer: # 클래스 이름을 BatchVideoAnalyzer로 변경
    def __init__(self, image_model_name: str, image_model_weights_path: str, api_key: str, voice_model_name: str = "wav2vec2", min_speech_segment_duration: float = 5.0, logger: Optional[AnalysisLogger] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        
        # 1. 이미지 감정 분석 모델 로드
        self._log_info("이미지 감정 분석 모델을 로드합니다...")
        self.image_model = create_model(model_name=image_model_name, num_classes=7, pretrained=False)
        self.image_model.load_state_dict(torch.load(image_model_weights_path))
        self.image_model.to(self.device)
        self.image_model.eval()
        self._log_info("이미지 감정 분석 모델 로드 완료.")
        
        # 이미지 전처리 transform 정의 (EmoNet 기준)
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # DNN 얼굴 탐지기 모델 로드
        self._log_info("DNN 얼굴 탐지기를 로드합니다...")
        proto_path = Path("./infrastructure/models/deploy.prototxt")
        model_path = Path("./infrastructure/models/res10_300x300_ssd_iter_140000.caffemodel")
        self.face_net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
        self._log_info("DNN 얼굴 탐지기 로드 완료.")
        
        # 2. 음성 발화 세그먼트 추출기 로드
        self._log_info("음성 발화 세그먼트 추출기(SpeechSegmenter)를 로드합니다...")
        self.speech_segmenter = SpeechSegmenter(min_segment_duration=min_speech_segment_duration, logger=logger) # 최소 발화 지속 시간 및 로거 전달
        self._log_info(f"SpeechSegmenter 설정: 최소 발화 지속 시간 = {min_speech_segment_duration}초.")

        # 3. 음성 감정 분석기 로드
        self._log_info("음성 감정 분석기(VoiceAnalyzer)를 로드합니다...")
        self.voice_analyzer = VoiceAnalyzer(api_key=api_key, voice_model_name=voice_model_name) # VoiceAnalyzer에는 로거를 직접 전달하지 않음 (내부에서 로깅하지 않도록 설계)
        self._log_info("음성 감정 분석기 로드 완료.")
        
        self.emotion_labels = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']

    def _log_info(self, message: str, data: Optional[Dict[str, Any]] = None):
        if self.logger:
            self.logger.log_info(f"[BatchVideoAnalyzer] {message}", data)

    def _log_warning(self, message: str, data: Optional[Dict[str, Any]] = None):
        if self.logger:
            self.logger.log_warning(f"[BatchVideoAnalyzer] {message}", data)

    def _log_error(self, message: str, data: Optional[Dict[str, Any]] = None):
        if self.logger:
            self.logger.log_error(f"[BatchVideoAnalyzer] {message}", data)

    def extract_frames(self, video_path: Path, start_sec: float = 0, end_sec: float = None, num_frames: int = 10) -> List[Image.Image]:
        """
        비디오에서 지정된 시간 구간(start_sec ~ end_sec) 내 N개의 프레임을 균일한 간격으로 추출합니다.
        end_sec가 None이면, 비디오 끝까지 추출합니다.
        """
        self._log_info(f"비디오 프레임 추출 시작: {video_path}, 구간 {start_sec:.2f}s - {end_sec if end_sec is not None else 'end'}s, 목표 {num_frames} 프레임.")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self._log_error(f"비디오 파일 열기 실패: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame_idx = int(start_sec * fps)
        
        if end_sec is None:
            end_frame_idx = total_frames_in_video - 1
        else:
            end_frame_idx = int(end_sec * fps)
        
        start_frame_idx = max(0, start_frame_idx)
        end_frame_idx = min(total_frames_in_video - 1, end_frame_idx)

        if end_frame_idx <= start_frame_idx:
            self._log_warning(f"유효한 프레임 구간이 없거나 너무 짧습니다. ({start_sec:.2f}s - {end_sec if end_sec is not None else 'end'}s)")
            cap.release()
            return []

        actual_num_frames_to_extract = max(1, num_frames)

        frame_indices = np.linspace(start_frame_idx, end_frame_idx, actual_num_frames_to_extract, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            else:
                self._log_warning(f"프레임 {idx} 추출 실패.", {"video_path": video_path})
        cap.release()
        self._log_info(f"{len(frames)}개 프레임 추출 완료.")
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
            padding_x = int((endX - startX) * 0.1)
            padding_y = int((endY - startY) * 0.1)
            startX = max(0, startX - padding_x)
            startY = max(0, startY - padding_y)
            endX = min(w, endX + padding_x)
            endY = min(h, endY + padding_y)
            return frame_pil.crop((startX, startY, endX, endY))
        else:
            return None # 얼굴 탐지 실패
    
    def analyze_image_emotions(self, frames: List[Image.Image]) -> Dict[str, Union[str, Dict[str, float]]]:
        """
        추출된 프레임들의 감정을 분석하고 종합합니다.
        얼굴이 탐지되지 않은 프레임은 분석에서 제외됩니다.
        """
        self._log_info(f"이미지 감정 분석 시작. 총 {len(frames)}개 프레임.")
        all_preds = []
        valid_frames_count = 0
        with torch.no_grad():
            for i, frame in enumerate(frames):
                face_crop = self._detect_and_crop_face(frame)
                if face_crop:
                    valid_frames_count += 1
                    img_tensor = self.image_transform(face_crop).unsqueeze(0).to(self.device)
                    outputs_dict = self.image_model(img_tensor)
                    emotion_preds_tensor = outputs_dict['expression']
                    _, preds = torch.max(emotion_preds_tensor, 1)
                    all_preds.append(preds.item())
                else:
                    self._log_warning(f"프레임 {i+1}에서 얼굴을 탐지하지 못했습니다. 해당 프레임은 이미지 감정 분석에서 제외됩니다.")
        
        if not all_preds:
            self._log_warning("분석할 유효한 얼굴 프레임이 없어 이미지 감정 분석 결과를 'N/A'로 반환합니다.")
            return {"dominant_emotion": "N/A", "distribution": {label: 0.0 for label in self.emotion_labels}}
            
        dominant_emotion_idx = max(set(all_preds), key=all_preds.count)
        
        distribution = {label: 0.0 for label in self.emotion_labels}
        for i_pred in set(all_preds):
            distribution[self.emotion_labels[i_pred]] = all_preds.count(i_pred) / len(all_preds)
        
        result = {
            "dominant_emotion": self.emotion_labels[dominant_emotion_idx],
            "distribution": distribution
        }
        self._log_info(f"이미지 감정 분석 완료. 탐지된 얼굴 프레임: {valid_frames_count}/{len(frames)}", result)
        return result

    def analyze(self, video_path_str: str, output_dir: str = "temp") -> Dict[str, Any]:
        """
        하나의 비디오 파일에 대한 전체 이미지/음성/텍스트 감정 분석을 발화 시점별로 수행하고 종합합니다.
        """
        total_start_time = time.perf_counter()
        timings = {"overall_processing": {}, "segment_processing": []}
        
        video_path = Path(video_path_str)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        all_segment_results = [] # 모든 세그먼트 분석 결과를 담을 리스트

        # 1. 비디오에서 전체 오디오 추출 (SpeechSegmenter 사용을 위함)
        self._log_info(f"'{video_path.name}'에서 전체 오디오를 추출합니다...")
        full_audio_path = output_path / f"{video_path.stem}_full.wav"
        full_audio_extraction_start = time.perf_counter()
        
        main_video_clip = None 
        try:
            main_video_clip = VideoFileClip(video_path_str)
            if main_video_clip.audio is None:
                self._log_warning("[경고] 원본 비디오에 오디오 트랙이 없습니다. 음성 분석을 건너뜁니다.")
                timings["overall_processing"]["full_audio_extraction_seconds"] = 0.0
                
                image_analysis_for_full_video = self.analyze_image_emotions(self.extract_frames(video_path, num_frames=30))
                
                return {
                    "video_file": video_path.name,
                    "visual_analysis_overall": image_analysis_for_full_video,
                    "audio_analysis_overall": {"error": "No audio track in video."},
                    "segment_analyses": [],
                    "performance": timings
                }
            else:
                main_video_clip.audio.write_audiofile(str(full_audio_path), codec='pcm_s16le', logger=None)
                if not (full_audio_path.exists() and os.path.getsize(str(full_audio_path)) > 0):
                    raise Exception("오디오 파일이 비어있거나 생성되지 않았습니다.")
            timings["overall_processing"]["full_audio_extraction_seconds"] = time.perf_counter() - full_audio_extraction_start
            self._log_info("전체 오디오 추출 완료.")

        except Exception as e:
            self._log_error(f"전체 오디오 추출 중 문제 발생: {e}", {"video_path": video_path_str})
            timings["overall_processing"]["full_audio_extraction_seconds"] = time.perf_counter() - full_audio_extraction_start
            
            image_analysis_for_full_video = self.analyze_image_emotions(self.extract_frames(video_path, num_frames=30))
            
            return {
                "video_file": video_path.name,
                "visual_analysis_overall": image_analysis_for_full_video,
                "audio_analysis_overall": {"error": f"Failed to extract audio: {e}"},
                "segment_analyses": [],
                "performance": timings
            }
        # main_video_clip.close()는 모든 세그먼트 처리 후에 한 번만 호출하도록 finally 블록에서 수행.


        # 2. SpeechSegmenter를 사용하여 발화 세그먼트 추출
        self._log_info("발화 세그먼트를 추출합니다...")
        speech_segmentation_start = time.perf_counter()
        segments = self.speech_segmenter.get_speech_segments(str(full_audio_path))
        timings["overall_processing"]["speech_segmentation_seconds"] = time.perf_counter() - speech_segmentation_start
        self._log_info(f"총 {len(segments)}개의 발화 세그먼트 추출 완료.")

        # 3. 각 발화 세그먼트에 대해 이미지, 음성, 텍스트 감정 분석 수행
        self._log_info("각 발화 세그먼트에 대한 감정 분석을 시작합니다...")
        for i, segment in enumerate(segments):
            segment_id = i + 1
            segment_start = segment['start']
            segment_end = segment['end']
            segment_text = segment['text']
            
            if segment_end <= segment_start:
                self._log_warning(f"세그먼트 {segment_id} ({segment_start:.2f}s - {segment_end:.2f}s) 길이가 0 또는 음수입니다.")
                continue

            self._log_info(f"\n--- 세그먼트 {segment_id} ({segment_start:.2f}s - {segment_end:.2f}s) 분석 시작 ---")
            segment_total_start_time = time.perf_counter()
            segment_timings = {"segment_id": segment_id}

            # 3.1. 세그먼트 오디오 파일 크롭
            cropped_audio_path = output_path / f"{video_path.stem}_segment_{segment_id}.wav"
            segment_audio_analysis_result = None # 초기화
            
            audio_crop_start = time.perf_counter()
            
            try:
                if main_video_clip and main_video_clip.audio is not None:
                    # 수정된 부분: 먼저 비디오 클립을 자른 후, 잘린 비디오 클립에서 오디오를 추출합니다.
                    segment_audio_clip = main_video_clip.subclipped(segment_start, segment_end)
                    segment_audio_clip.audio.write_audiofile(str(cropped_audio_path), codec='pcm_s16le', logger=None)
                    
                    if not (cropped_audio_path.exists() and os.path.getsize(str(cropped_audio_path)) > 0):
                        raise Exception("세그먼트 오디오 파일이 비어있거나 생성되지 않았습니다.")
                else:
                    raise Exception("원본 비디오에 오디오 트랙이 없습니다 (초기 추출 단계에서 확인됨).")
                
            except Exception as e:
                self._log_error(f"세그먼트 {segment_id} 오디오 크롭 실패: {e}. 음성 분석에 실패했습니다.", {"segment_id": segment_id, "start": segment_start, "end": segment_end})
                cropped_audio_path = None 
                segment_audio_analysis_result = {
                    "text_based_analysis": self.voice_analyzer.analyze_emotion_from_text(segment_text),
                    "voice_based_analysis": {"error": "Failed to crop audio for voice analysis."}
                }
            
            segment_timings["audio_cropping_seconds"] = time.perf_counter() - audio_crop_start

            # 3.2. 음성 감정 분석 (텍스트 기반 및 음성 특징 기반)
            if cropped_audio_path and cropped_audio_path.exists():
                self._log_info(f"세그먼트 {segment_id} 음성 감정 분석 시작.")
                segment_audio_analysis_result = self.voice_analyzer.analyze_segment(str(cropped_audio_path), segment_text)
                segment_timings.update(segment_audio_analysis_result.pop("timings", {})) 
                self._log_info(f"세그먼트 {segment_id} 음성 감정 분석 완료.", {"result": segment_audio_analysis_result})
            else:
                self._log_warning(f"세그먼트 {segment_id} 오디오 파일이 없어 음성 특징 기반 분석은 건너뛰고 텍스트 감정 분석만 수행합니다.")
            
            # 3.3. 세그먼트 이미지 프레임 추출 및 감정 분석
            self._log_info(f"세그먼트 {segment_id} 이미지 감정 분석 시작.")
            image_analysis_start = time.perf_counter()
            
            segment_duration = segment_end - segment_start
            frames_to_extract = max(1, int(segment_duration * 10)) # 최대 초당 10프레임
            
            segment_frames = self.extract_frames(video_path, segment_start, segment_end, num_frames=frames_to_extract)
            segment_image_analysis = self.analyze_image_emotions(segment_frames)
            segment_timings["image_analysis_seconds"] = time.perf_counter() - image_analysis_start
            self._log_info(f"세그먼트 {segment_id} 이미지 감정 분석 완료. 추출 프레임 수: {len(segment_frames)}.", {"result": segment_image_analysis})

            # 3.4. 세그먼트별 결과 취합
            segment_result = {
                "segment_id": segment_id,
                "start_time": segment_start,
                "end_time": segment_end,
                "transcribed_text": segment_text,
                "visual_analysis": segment_image_analysis,
                "audio_analysis": segment_audio_analysis_result,
                "performance": segment_timings
            }
            all_segment_results.append(segment_result)
            timings["segment_processing"].append(segment_timings) 
            self.logger.save_intermediate_result(f"segment_analysis_{segment_id}", segment_result)
            
            # 임시 세그먼트 오디오 파일 삭제
            if cropped_audio_path and cropped_audio_path.exists():
                try:
                    cropped_audio_path.unlink()
                    self._log_info(f"임시 세그먼트 오디오 파일 삭제 완료: {cropped_audio_path}")
                except OSError as e:
                    self._log_warning(f"임시 세그먼트 오디오 파일 삭제 실패 ({cropped_audio_path}): {e}")

            self._log_info(f"--- 세그먼트 {segment_id} 분석 완료 (소요 시간: {time.perf_counter() - segment_total_start_time:.2f}초) ---")

        total_time_elapsed = time.perf_counter() - total_start_time
        timings["overall_processing"]["total_elapsed_seconds"] = total_time_elapsed

        final_result = {
            "video_file": video_path.name,
            "total_segments": len(all_segment_results),
            "segment_analyses": all_segment_results,
            "performance": timings
        }
        
        if full_audio_path.exists():
            try:
                full_audio_path.unlink()
                self._log_info(f"임시 전체 오디오 파일 삭제 완료: {full_audio_path}")
            except OSError as e:
                self._log_warning(f"임시 전체 오디오 파일 삭제 실패 ({full_audio_path}): {e}")
        
        if main_video_clip:
            main_video_clip.close()
            self._log_info("메인 비디오 클립 닫기 완료.")
                
        self._log_info("모든 비디오 분석 단계 완료.")
        return final_result