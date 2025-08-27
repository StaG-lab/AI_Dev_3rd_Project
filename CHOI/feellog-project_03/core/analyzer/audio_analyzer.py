# ./core/analyzer/audio_analyzer.py
'''
import inspect
try:
    import google.generativeai as genai
    path = inspect.getfile(genai)
    print("--- DEBUG START ---")
    print(f"audio_analyzer.py에서 import한 google.generativeai 경로:")
    print(f"▶ {path}")
    print("--- DEBUG END ---")
except Exception as e:
    print(f"--- DEBUG: FAILED TO IMPORT IN audio_analyzer.py ---")
    print(e)
# -----------------------------------------------------------------
'''
from faster_whisper import WhisperModel
import google.generativeai as genai
import torch
from transformers import Wav2Vec2ForSequenceClassification, HubertForSequenceClassification, Wav2Vec2Processor
#from transformers import AutoModelForSequenceClassification, AutoProcessor, AutoModel
from transformers import AutoFeatureExtractor
import json
import torchaudio 
import torchaudio.transforms as T
import time

class VoiceAnalyzer:
    def __init__(self, api_key: str, voice_model_name: str = "wav2vec2"):
        # Faster Whisper 모델 로드 (STT)
        self.stt_model = WhisperModel("medium", device="cuda", compute_type="float16")
        self.target_sr = 16000
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest", # 최신의 빠르고 효율적인 모델 사용
            generation_config={"response_mime_type": "application/json"} # JSON 출력 모드 설정
        )
        # 선택된 음성 특징 모델 로드
        if voice_model_name == "wav2vec2":
            model_id = "inseong00/wav2vec2-large-xlsr-korean-autumn"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            self.voice_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to("cuda")
        elif voice_model_name == "hubert-base":
            model_id = "team-lucid/hubert-base-korean"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            self.voice_model = HubertForSequenceClassification.from_pretrained(model_id).to("cuda")
        elif voice_model_name == "hubert-large":
            model_id = "team-lucid/hubert-large-korean"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            self.voice_model = HubertForSequenceClassification.from_pretrained(model_id).to("cuda")
        elif voice_model_name == "hubert-xlarge":
            model_id = "team-lucid/hubert-xlarge-korean"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            self.voice_model = HubertForSequenceClassification.from_pretrained(model_id).to("cuda")
        else:
            raise ValueError("지원하지 않는 모델 이름입니다.")
        
        print(f"음성 특징 분석 모델로 '{model_id}'를 로드합니다...")
        self.voice_model_name = voice_model_name

    def transcribe(self, audio_path: str) -> str:
        """오디오 파일을 텍스트로 변환합니다."""
        segments, _ = self.stt_model.transcribe(audio_path, beam_size=5)
        return " ".join([seg.text for seg in segments])

    # 4. 텍스트 감정 분석 메서드를 Gemini API를 사용하도록 수정
    def analyze_emotion_from_text(self, text: str) -> dict:
        """텍스트를 Gemini로 분석하여 감정 스코어를 JSON으로 반환합니다."""
        prompt = f"""
        당신은 텍스트에서 감정을 분석하는 전문가입니다.
        다음 텍스트를 분석하여 {{긍정, 부정}}과 {{기쁨, 당황, 분노, 불안, 상처, 슬픔, 중립}}의 강도를 0.0에서 1.0 사이의 수치로 표현해야 합니다.
        결과는 반드시 아래의 JSON 형식으로만 반환해야 합니다.

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
            return json.loads(response.text)
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Gemini 응답 처리 중 에러 발생: {e}")
            return {"error": "Failed to parse Gemini response"}

    def analyze_emotion_from_voice(self, audio_path: str) -> dict:
        """오디오 파형 자체를 분석하여 감정 스코어를 반환합니다."""
        # 1. torchaudio로 오디오 파일 로드
        # waveform: 오디오 데이터 (Tensor), original_sr: 원본 샘플링 레이트
        try:
            waveform, original_sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"오디오 파일 로드 실패: {audio_path}, 에러: {e}")
            return {"error": "Failed to load audio file"}

        # 2. 샘플링 레이트 변환 (필요한 경우)
        # 모델이 기대하는 16000Hz가 아니면 리샘플링 수행
        if original_sr != self.target_sr:
            resampler = T.Resample(original_sr, self.target_sr)
            waveform = resampler(waveform)

        # 3. 오디오 데이터를 모델 입력 형식에 맞게 준비
        # 스테레오(채널 2개)일 경우 모노(채널 1개)로 평균내어 변환
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 프로세서가 1차원 배열을 기대하므로 차원 축소
        speech_array = waveform.squeeze(0).numpy()

        # 4. 모델 추론 (이하 로직은 동일)
        inputs = self.feature_extractor(speech_array, sampling_rate=self.target_sr, return_tensors="pt", padding=True).to("cuda")
        
        with torch.no_grad():
            #logits = self.voice_model(inputs.input_values, attention_mask=inputs.attention_mask).logits
            logits = self.voice_model(**inputs).logits

        scores = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
        labels = self.voice_model.config.id2label
        return {labels[i]: float(score) for i, score in enumerate(scores)}

    def analyze(self, audio_path: str) -> dict:
        """음성 파일 하나에 대한 전체 감정 분석을 수행합니다."""
        timings = {}
        
        # 디버깅을 위한 print 문 추가
        
        print("  -> (1/3) 음성 -> 텍스트 변환(STT) 시작...")
        start_time = time.perf_counter()
        transcribed_text = self.transcribe(audio_path)
        timings["1_stt_seconds"] = time.perf_counter() - start_time
        print(f"  -> (1/3) STT 완료. 텍스트: '{transcribed_text[:50]}...'")
        
        print("  -> (2/3) 텍스트 기반 감정 분석 시작...")
        start_time = time.perf_counter()
        text_analysis_result = self.analyze_emotion_from_text(transcribed_text)
        timings["2_text_analysis_seconds"] = time.perf_counter() - start_time
        print("  -> (2/3) 텍스트 분석 완료.")
        
        print("  -> (3/3) 음성 특징 기반 감정 분석 시작...")
        start_time = time.perf_counter()
        voice_analysis_result = self.analyze_emotion_from_voice(audio_path)
        timings["3_voice_analysis_seconds"] = time.perf_counter() - start_time
        print("  -> (3/3) 음성 특징 분석 완료.")
        
        return {
            "transcribed_text": transcribed_text,
            "text_based_analysis": text_analysis_result,
            "voice_based_analysis": voice_analysis_result,
            "timings": timings
        }