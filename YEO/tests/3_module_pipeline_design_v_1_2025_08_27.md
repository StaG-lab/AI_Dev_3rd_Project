좋아, 7개 모듈을 **3개**로 깔끔하게 합쳐서 “바로 코드로 옮길 수 있는” 설계안으로 정리했어. (규칙·스키마·ERD 매핑은 지난 문서 기준 유지)

# 1) Module A — Intent & Query
역할: 사용자 질의를 **기간 의도**로 정규화하고(LLM/Gemini), 스캔할 날짜 범위를 만들어 다음 단계로 전달.

- 입력
  - `query_text: str` (사용자 문장)
- 처리
  - **Intent 파싱**: `mode ∈ {daily, weekly, monthly, range}` + `date|start|end` 산출, KST 기준(Asia/Seoul) 해석
  - **날짜 보정 규칙**: 모호하면 일반 해석 + 근거 날짜 명시
- 출력(계약)
  ```json
  {
    "mode": "daily|weekly|monthly|range",
    "date": "YYYY-MM-DD",        // mode=daily
    "start": "YYYY-MM-DD",       // mode∈{weekly,monthly,range}
    "end": "YYYY-MM-DD"
  }
  ```
- 핵심 인터페이스
  - `parse_intent(query:str) -> IntentJSON`
  - (옵션) `resolve_dates(intent:IntentJSON, now_kst:date) -> DateRange`
- 비고
  - A 모듈은 **LLM 어댑터**를 안쪽에 숨겨서, 외부에선 순수 함수처럼 보이게 유지(테스트 편의)

---

# 2) Module B — Data Engine (Scan → Normalize → Aggregate)
역할: 기간 범위에 해당하는 `analysis_result_*.json`을 **수집/파싱/정규화/집계**하여 일간 카드들과 기간 리포트를 만든다.

- 입력
  - `date_range: {date|start|end}`
  - `root_dir: "./reports/analysis"` (기본)
- 처리 단계
  1) **Scanner**  
     - 패턴: `analysis_result_*.json`  
     - 일자 추출 우선순위: (1) 파일명 `YYYYMMDD`, (2) 내부 `generated_at` 등  
     - 결과: `dict[date] -> list[Path]`
  2) **Parser/Normalizer**  
     - 스키마: `sentiment_score:int(0~100)`, `dominant_overall_emotion:str`, `emotion_distribution:[{icon, emotion, percentage:"45%"}]`  
     - Top-K만 오는 분포를 **표준 7감정**(기쁨/당황/분노/불안/상처/슬픔/중립)으로 확장하고 합 1로 정규화(누락=0)
  3) **Aggregator**  
     - **일간**: 점수=중앙값(동률→평균), 주감정=다수결(동률→전일 연속성/점수 상위 파일 Top), 분포=합산 후 정규화 → **일간 카드 JSON** 산출  
     - **기간**: 평균점수(정수 반올림), 우세감정 TopK, `trend.by_day`, 급변(Δscore≥10 또는 Top1 전환) → **기간 리포트 JSON**
- 출력(계약)
  - **DailyCardJSON[]** & **PeriodReportJSON** (예시 스키마는 문서 6.1/6.2 참고)
- 핵심 인터페이스
  - `scan(date_range) -> dict[str, list[Path]]`
  - `load_and_normalize(path) -> NormalizedRecord`  // 7감정 풀셋, 합=1
  - `aggregate_daily(paths:list[Path]) -> DailyCardJSON`
  - `aggregate_period(cards:list[DailyCardJSON]) -> PeriodReportJSON`
- 비고
  - 이 모듈은 **로컬 규칙 기반**만 사용(LLM 불필요), `analysis_result`만 의존하며 `detailed_*`는 스킵

---

# 3) Module C — Summarizer & Export (LLM + Report/DB Adapter)
역할: B 모듈의 통계를 받아 **요약문(JSON)**을 생성하고, 필요 시 **저장·전달 포맷**(파일/DB/HTTP)에 맞게 패키징한다.

- 입력
  - `DailyCardJSON[]`, `PeriodReportJSON`
- 처리
  1) **LLM Summarizer**  
     - 프롬프트: 기간/일간 각각 전용 템플릿 사용, 출력은 `{ "summary_message": "..." }` 단일 JSON  
  2) **Exporter / Adapter**  
     - 로컬 결과물: 카드/리포트 JSON 파일 저장(디폴트)  
     - **ERD 매핑 어댑터**(옵션): `analysis_tbl`, `report_tbl` 컬럼 규칙에 따라 변환(Top-K→7감정 확장 포함)
- 출력(계약)
  - `{"period": PeriodReportJSON + summary_message, "daily": [DailyCardJSON + summary_message] }`
  - (옵션) `DBRow{analysis_tbl}`, `DBRow{report_tbl}`
- 핵심 인터페이스
  - `summarize_period(stats:PeriodReportJSON) -> {"summary_message": str}`
  - `summarize_daily(card:DailyCardJSON) -> {"summary_message": str}`
  - `export_to_files(period, cards, out_dir)`
  - `map_to_erd(period, cards) -> {analysis_rows, report_rows}`  // 실제 저장은 호출측 책임
- 비고
  - 사용자-facing 결과는 `report_tbl`에 적재 가능(요약/카드/디테일 분리 필드)

---

## 디렉터리/파일 구조(권장)
```
app/
  module_a_intent/
    __init__.py
    intent_parser.py      # parse_intent, resolve_dates
  module_b_engine/
    __init__.py
    scanner.py            # scan
    io_normalize.py       # load_and_normalize(Top-K→7감정)
    aggregator.py         # aggregate_daily / aggregate_period
  module_c_summary_export/
    __init__.py
    summarizer.py         # summarize_period / summarize_daily
    exporter.py           # export_to_files
    erd_adapter.py        # map_to_erd (analysis_tbl / report_tbl 규칙)
main.py                   # A→B→C 오케스트레이션
config.py                 # 경로(root_dir), 라벨맵, 임계값(Δscore=10 등)
```

## 공통 규약(요약)
- **데이터 소스**: `./reports/analysis/analysis_result_*.json`만 사용(융합 결과, 경량)
- **Label 표준**: 7감정 풀셋으로 확장·정규화, 합=1
- **집계 규칙**: 일간=중앙값/다수결/합산정규화, 기간=평균·TopK·trend·급변
- **요약 출력**: `summary_message` JSON 단일 객체(일간/기간 각각)
- **ERD 연계**: `analysis_tbl`, `report_tbl` 매핑 규칙 명시(Null 복제 금지, 시간열은 없으면 빈 배열)

## 실패/엣지 처리
- 스캔 결과 없음 → 빈 결과 + `"기간 데이터 없음"` 메시지로 반환(Exporter 단계에서 그대로 저장)
- 퍼센트 파싱 불가 → 해당 감정만 제외 후 남은 값 정규화, `aggregation_notes`에 기록
- 스키마 일부 누락 → 있는 필드만 사용(느슨한 파싱), 경고 로그 남김
- `days_requested` vs `days_covered`를 별도 노출(기간 구멍 가시화)

---

### 이렇게 쓰면 됨 (흐름)
1) A: `intent = parse_intent(query)` → `date_range`  
2) B: `files_by_day = scan(date_range)` → `daily_cards = [aggregate_daily(...)]` → `period = aggregate_period(daily_cards)`  
3) C: `period_sum = summarize_period(period)` / `daily_sums = [summarize_daily(c) ...]` → `export_to_files(...)` → (옵션) `map_to_erd(...)`

