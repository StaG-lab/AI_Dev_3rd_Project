# ./core/utils/analysis_logger.py

import json
from datetime import datetime
from typing import List, Dict, Any, Optional

class AnalysisLogger:
    """
    분석 과정에서 발생하는 모든 로그 메시지와 중간 결과값을 수집하고 저장하는 클래스.
    """
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.intermediate_results: Dict[str, Any] = {}
        self.analysis_start_time = datetime.now()

    def _add_log_entry(self, level: str, message: str, data: Optional[Dict[str, Any]] = None):
        """내부 로그 항목 추가."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        if data:
            log_entry["data"] = data
        self.logs.append(log_entry)

    def log_info(self, message: str, data: Optional[Dict[str, Any]] = None):
        """정보 로그를 추가합니다."""
        self._add_log_entry("INFO", message, data)

    def log_warning(self, message: str, data: Optional[Dict[str, Any]] = None):
        """경고 로그를 추가합니다."""
        self._add_log_entry("WARNING", message, data)

    def log_error(self, message: str, data: Optional[Dict[str, Any]] = None):
        """에러 로그를 추가합니다."""
        self._add_log_entry("ERROR", message, data)

    def save_intermediate_result(self, key: str, result: Any):
        """중간 결과값을 저장합니다."""
        self.intermediate_results[key] = result
        self.log_info(f"중간 결과 저장: {key}", {"result_preview": str(result)[:200] + "..." if len(str(result)) > 200 else str(result)})

    def get_full_log_data(self) -> Dict[str, Any]:
        """모든 로그와 중간 결과값을 포함하는 전체 데이터를 반환합니다."""
        return {
            "analysis_started_at": self.analysis_start_time.isoformat(),
            "logs": self.logs,
            "intermediate_results": self.intermediate_results,
            "analysis_finished_at": datetime.now().isoformat()
        }

    def save_to_file(self, filename: str):
        """모든 로그와 중간 결과값을 JSON 파일로 저장합니다."""
        full_data = self.get_full_log_data()
        try:
            with open(filename, "w", encoding='utf-8') as f:
                json.dump(full_data, f, ensure_ascii=False, indent=2)
            self.log_info(f"상세 분석 로그 및 중간 결과가 '{filename}'에 저장되었습니다.")
        except Exception as e:
            self.log_error(f"상세 로그 파일 저장 실패: {e}", {"filename": filename})