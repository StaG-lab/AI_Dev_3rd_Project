# ./core/analyzer/audio_analyzer.py

import google.generativeai as genai
import torch
from transformers import Wav2Vec2ForSequenceClassification, HubertForSequenceClassification, AutoFeatureExtractor
import json
import torchaudio 
import torchaudio.transforms as T
import time
from typing import Dict, Any

class VoiceAnalyzer: # 클래스 이름을 VoiceAnalyzer로 유지하되, 내부 역할 변경
    def __init__(self, api_key: str, voice_model_name: str = "wav2vec2"):
        self.target_sr = 16000 # 오디오 리샘플링을 위한 목표 샘플링 레이트
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            generation_config={"response_mime_type": "application/json"}
        )
        
        self.feature_extractor = None
        self.voice_model = None
        model_id = ""

        if voice_model_name == "wav2vec2":
            model_id = "jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition2_data_rebalance"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            self.voice_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to("cuda")
        elif voice_model_name == "hubert-base":
            model_id = "team-lucid/hubert-base-korean"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            self.voice_model = HubertForSequenceClassification.from_pretrained(model_id).to("cuda")
        elif voice_model_name == "wav2vec2_autumn":
            model_id = "inseong00/wav2vec2-large-xlsr-korean-autumn"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            self.voice_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to("cuda")
        else:
            raise ValueError(f"지원하지 않는 모델 이름입니다: {voice_model_name}")
        
        print(f"음성 특징 분석 모델로 '{model_id}'를 로드합니다.")
        self.voice_model_name = voice_model_name
        self.voice_model_id = model_id

    def analyze_emotion_from_text(self, text: str) -> dict:
        """텍스트를 Gemini로 분석하여 감정 스코어를 JSON으로 반환합니다."""
        if not text.strip():
            return {
                "sentiment": {"긍정": 0.0, "부정": 0.0},
                "emotions": {"기쁨": 0.0, "당황": 0.0, "분노": 0.0, "불안": 0.0, "상처": 0.0, "슬픔": 0.0, "중립": 1.0}
            }
        
        prompt = f"""
        당신은 텍스트에서 감정을 분석하는 전문가입니다.
        다음 텍스트를 분석하여 {{긍정, 부정}}과 {{기쁨, 당황, 분노, 불안, 상처, 슬픔, 중립}}의 강도를 0.0에서 1.0 사이의 수치로 표현해야 합니다.
        결과는 반드시 아래의 JSON 형식으로만 반환해야 합니다.
        모든 감정의 합은 1이 될 필요는 없지만, 각 감정 스코어는 0.0에서 1.0 사이여야 합니다.

        분석할 텍스트: "{text}"

        JSON 형식:
        {{
          "sentiment": {{"긍정": float, "부정": float}},
          "emotions": {{
            "기쁨": float, "당황": float, "분노": float, "불안": float, 
            "상처": float, "슬픔": float, "중립": float
          }}
        }}
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            json_response = json.loads(response.text)
            
            expected_emotions = ["기쁨", "당황", "분노", "불안", "상처", "슬픔", "중립"]
            for emo in expected_emotions:
                if emo not in json_response.get("emotions", {}):
                    json_response["emotions"][emo] = 0.0
            if "긍정" not in json_response.get("sentiment", {}):
                json_response["sentiment"]["긍정"] = 0.0
            if "부정" not in json_response.get("sentiment", {}):
                json_response["sentiment"]["부정"] = 0.0

            return json_response

        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Gemini 텍스트 감정 분석 응답 처리 중 에러 발생: {e}. 원본 응답 텍스트: {response.text if 'response' in locals() else 'N/A'}")
            return {
                "sentiment": {"긍정": 0.0, "부정": 0.0},
                "emotions": {"기쁨": 0.0, "당황": 0.0, "분노": 0.0, "불안": 0.0, "상처": 0.0, "슬픔": 0.0, "중립": 0.0},
                "error": f"Failed to parse Gemini response: {e}"
            }

    def analyze_emotion_from_voice(self, audio_path: str) -> dict:
        """오디오 파형 자체를 분석하여 감정 스코어를 반환합니다. (세그먼트 오디오 파일 경로 입력)"""
        try:
            waveform, original_sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"오디오 파일 로드 실패: {audio_path}, 에러: {e}")
            return {"error": "Failed to load audio file", "distribution": {}}

        if original_sr != self.target_sr:
            resampler = T.Resample(original_sr, self.target_sr)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        speech_array = waveform.squeeze(0).numpy()

        inputs = self.feature_extractor(speech_array, sampling_rate=self.target_sr, return_tensors="pt", padding=True).to("cuda")
        
        with torch.no_grad():
            logits = self.voice_model(**inputs).logits

        scores = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
        
        labels = getattr(self.voice_model.config, "id2label", None)
        if labels is None:
            if self.voice_model_id == "jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition2_data_rebalance":
                labels = {0: '기쁨', 1: '당황', 2: '분노', 3: '불안', 4: '상처', 5: '슬픔', 6: '중립'}
            elif self.voice_model_id == "inseong00/wav2vec2-large-xlsr-korean-autumn":
                # 이 모델의 정확한 레이블 순서 확인 후 업데이트 필요
                # 임시 순서: 실제 사용 시 모델 허브 또는 훈련 코드에서 확인 필수
                labels = {0: '기쁨', 1: '슬픔', 2: '분노', 3: '불안', 4: '중립', 5: '당황', 6: '상처'} 
            else:
                print(f"[경고] {self.voice_model_id}에 대한 id2label 정보를 찾을 수 없습니다. scores 길이 ({len(scores)})에 기반한 기본 순서 사용.")
                labels = {i: f"emotion_{i}" for i in range(len(scores))}

        distribution = {labels[i]: 0.0 for i in range(len(labels))}
        for i, score in enumerate(scores):
            if i < len(labels): # labels 딕셔너리에 해당하는 인덱스가 있는지 확인
                distribution[labels[i]] = float(score)

        return {"distribution": distribution}

    def analyze_segment(self, audio_segment_path: str, text_segment: str) -> Dict[str, Any]:
        """
        단일 발화 세그먼트에 대한 음성 특징 기반 감정 분석 및 텍스트 기반 감정 분석을 수행합니다.
        
        Args:
            audio_segment_path (str): 세그먼트 오디오 파일 경로.
            text_segment (str): 세그먼트 텍스트.
            
        Returns:
            Dict[str, Any]: 세그먼트 분석 결과.
        """
        segment_timings = {}

        # 텍스트 기반 감정 분석
        start_time_text = time.perf_counter()
        text_analysis_result = self.analyze_emotion_from_text(text_segment)
        segment_timings["text_analysis_seconds"] = time.perf_counter() - start_time_text
        
        # 음성 특징 기반 감정 분석
        start_time_voice = time.perf_counter()
        voice_analysis_result = self.analyze_emotion_from_voice(audio_segment_path)
        segment_timings["voice_analysis_seconds"] = time.perf_counter() - start_time_voice
        
        return {
            "text_based_analysis": text_analysis_result,
            "voice_based_analysis": voice_analysis_result,
            "timings": segment_timings
        }
