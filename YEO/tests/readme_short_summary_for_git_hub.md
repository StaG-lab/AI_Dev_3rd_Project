# 🎧 오디오 전처리 파이프라인 — 핵심 요약 (GitHub용)

## 2. 통합 메타데이터 생성
3개 차수(4차, 5차, 5_2차)의 CSV를 하나의 DataFrame으로 병합.

- **컬럼명 표준화:** `4번감정세기` → `4번 감정세기` 등 불일치 수정.
- **라벨 표준화:**
```python
label_map = {
  "Sadness": "sad", "Angry": "angry", "Disgust": "disgust",
  "Fear": "fear", "Neutral": "neutral",
  "Happiness": "happiness", "Surprise": "surprise",
  # 소문자/변형 포함 매핑
  "sad": "sad", "sadness": "sad",
  "anger": "angry", "angry": "angry",
  "disgust": "disgust", "fear": "fear",
  "neutral": "neutral", "happiness": "happiness", "surprise": "surprise",
}
```
- **다수결 기반 최종 라벨(`final_emotion`) 생성**
  - 동점 발생 시 **상황(context)** 사용
  - 그래도 불가 → **샘플 drop**

---

## 3. 품질 필터링
**파일 단위 검증**
- 손상 여부(열기 실패, `duration==0`)
- 샘플레이트/채널/비트뎁스 확인

**오디오 품질 지표**
- 신호 대 잡음비(SNR)
- Clipping 여부

**길이 필터링**
- `duration < 0.2s` drop

> 결과: 약 **42,408 → 36,467** (약 **14%** drop)  
> ※ 클래스별 손실률은 별도 로깅으로 확인 가능

---

## 4. 오디오 세그먼트화 (8초 단위)
- **타겟 샘플레이트:** 16kHz (mono 다운믹스)
- **로직:**
  - `> 8s`: 8초 단위로 나눔, **잔여 ≥ 4s → pad**, **< 4s → drop**
  - `< 8s`: **pad**하여 1개 세그먼트 생성
- **특징:** 원본 row 수 **이상**으로만 증가(줄어들지 않음)
- **출력 예:**
```
{wav_id}__seg_000.wav
{wav_id}__seg_001.wav
...
```

---

## 5. 정규화
- 현재: **Peak Normalize**로 통일
- 학습 시 **DataCollator**에서는 `do_normalize=False` (중복 방지)
- 필요 시 옵션화: `--normalize rms/peak/none`

---

## 6. 최종 산출물
**폴더 구조**
```
./datasets/KES_processed/
    ├─ metadata.csv
    ├─ {wav_id}__seg_000.wav
    ├─ {wav_id}__seg_001.wav
    └─ ...
```
**`metadata.csv` 주요 컬럼**
```
path, final_emotion, [발화문, 상황, 나이, 성별...]
```

---

## 7. 이후 파이프라인 연계
- **`sample_and_split_audio.py`**: KES_processed → stratified split(train/val/test), 5%/50%/100% 샘플링 지원
- **`audio_dataset.py`**: CSV 불러와 waveform+label 반환, transform 증강 적용 가능
- **`DataCollatorForAudio.py`**: batch padding, attention mask 생성
- **`train_audio_split.py`**: class-aware augmentation, 2단계 파인튜닝(HuBERT/Wav2Vec2)

---

## ✅ 요약
- **라벨 표준화 → 다수결 `final_emotion` → 품질 필터링 → 8초 세그먼트화 → Peak Normalize → `metadata.csv` 저장**
- 출력은 **`KES_processed` 단일 폴더**에 wav와 CSV 동시 저장
- **wav2vec2 / HuBERT** 학습에 바로 투입 가능

