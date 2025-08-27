# ./core/analyzer/speech_segmenter.py

from faster_whisper import WhisperModel
import time
from typing import List, Dict, Union, Any, Optional
from core.utils.analysis_logger import AnalysisLogger # AnalysisLogger 임포트

class SpeechSegmenter:
    """
    오디오 파일에서 음성 발화 세그먼트를 감지하고 텍스트로 변환합니다.
    """
    def __init__(self, model_size: str = "medium", min_segment_duration: float = 5.0, logger: Optional[AnalysisLogger] = None):
        self.model_size = model_size
        self.min_segment_duration = min_segment_duration
        self.logger = logger
        
        self._log_info(f"STT 모델 (FasterWhisper, size='{model_size}') 로드 중...")
        # device="cuda"로 고정하여 GPU 사용
        self.stt_model = WhisperModel(model_size, device="cuda", compute_type="float16")
        self._log_info("STT 모델 로드 완료.")
        self._log_info(f"SpeechSegmenter 설정: 최소 발화 지속 시간 = {min_segment_duration}초.")

    def _log_info(self, message: str, data: Optional[Dict[str, Any]] = None):
        if self.logger:
            self.logger.log_info(f"[SpeechSegmenter] {message}", data)

    def _log_warning(self, message: str, data: Optional[Dict[str, Any]] = None):
        if self.logger:
            self.logger.log_warning(f"[SpeechSegmenter] {message}", data)

    def _log_error(self, message: str, data: Optional[Dict[str, Any]] = None):
        if self.logger:
            self.logger.log_error(f"[SpeechSegmenter] {message}", data)

    def _merge_short_segments(self, segments: List[Dict[str, Union[float, str]]]) -> List[Dict[str, Union[float, str]]]:
        """
        주어진 세그먼트 리스트에서 최소 지속 시간보다 짧은 세그먼트들을 인접 세그먼트와 병합합니다.
        최소 길이를 보장하기 위해 인접한 세그먼트와 합치는 전략을 사용합니다.
        """
        if not segments:
            self._log_info("병합할 세그먼트가 없습니다.")
            return []

        filtered_segments = [s for s in segments if (s['end'] - s['start']) > 0]
        if not filtered_segments:
            self._log_warning("필터링 후 유효한 세그먼트가 없습니다.")
            return []

        merged_segments = [filtered_segments[0]]
        for i in range(1, len(filtered_segments)):
            current_segment = filtered_segments[i]
            
            last_merged_segment_duration = merged_segments[-1]['end'] - merged_segments[-1]['start']

            if last_merged_segment_duration < self.min_segment_duration:
                self._log_info(f"세그먼트 병합: 이전 세그먼트 ({merged_segments[-1]['start']:.2f}-{merged_segments[-1]['end']:.2f})와 현재 세그먼트 ({current_segment['start']:.2f}-{current_segment['end']:.2f})")
                merged_segments[-1]['end'] = current_segment['end']
                merged_segments[-1]['text'] += " " + current_segment['text']
            else:
                merged_segments.append(current_segment)
        
        final_segments = []
        if not merged_segments:
            return []
        
        final_segments.append(merged_segments[0])
        for i in range(1, len(merged_segments)):
            current = merged_segments[i]
            prev = final_segments[-1]
            
            if (prev['end'] - prev['start']) < self.min_segment_duration and len(final_segments) > 0: # 이전 세그먼트가 여전히 짧고, 병합 가능한 경우
                self._log_info(f"최종 세그먼트 병합: 이전 세그먼트 ({prev['start']:.2f}-{prev['end']:.2f})와 현재 세그먼트 ({current['start']:.2f}-{current['end']:.2f})")
                prev['end'] = current['end']
                prev['text'] += " " + current['text']
            else:
                final_segments.append(current)
                
        if len(final_segments) == 1 and (final_segments[0]['end'] - final_segments[0]['start']) < self.min_segment_duration:
            self._log_warning(f"모든 세그먼트 병합 후에도 단일 세그먼트가 최소 지속 시간보다 짧습니다 ({final_segments[0]['end'] - final_segments[0]['start']:.2f}s). 더 이상 병합할 대상이 없습니다.")
            
        return final_segments


    def get_speech_segments(self, audio_path: str) -> List[Dict[str, Union[float, str]]]:
        """
        오디오 파일에서 발화 세그먼트를 추출하고 각 세그먼트의 시작, 종료 시간 및 텍스트를 반환합니다.
        추출된 세그먼트 중 최소 지속 시간보다 짧은 세그먼트들을 병합합니다.
        """
        self._log_info(f"오디오 ({audio_path})에서 발화 세그먼트 추출 시작...")
        start_time = time.perf_counter()
        
        try:
            segments_raw, info = self.stt_model.transcribe(audio_path, beam_size=5, language="ko")
        except Exception as e:
            self._log_error(f"STT 모델 트랜스크라이브 중 에러 발생: {e}", {"audio_path": audio_path})
            return []
        
        initial_segments = []
        for segment in segments_raw:
            initial_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
        
        self._log_info(f"초기 발화 세그먼트 {len(initial_segments)}개 추출 완료.", {"segments": initial_segments})
        self.logger.save_intermediate_result("initial_speech_segments", initial_segments)

        # 최소 지속 시간 규칙을 적용하여 세그먼트 병합
        processed_segments = self._merge_short_segments(initial_segments)
        
        duration = time.perf_counter() - start_time
        self._log_info(f"오디오 ({audio_path})에서 최종 {len(processed_segments)}개 발화 세그먼트 추출 완료 (소요 시간: {duration:.2f}초).", {"final_segments": processed_segments})
        self.logger.save_intermediate_result("processed_speech_segments", processed_segments)

        return processed_segments