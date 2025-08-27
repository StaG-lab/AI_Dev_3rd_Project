import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import os

# --- 설정 (Configuration) ---
# 전처리가 완료된 데이터셋의 메타데이터와 오디오 폴더 경로
SOURCE_METADATA = Path("./datasets/KES_processed/metadata.csv")
SOURCE_AUDIO_DIR = Path("./datasets/KES_processed")

# 최종적으로 샘플링되고 분할된 데이터셋을 저장할 기본 폴더
OUTPUT_BASE_DIR = Path("./datasets/audio_sampling_sets")

# 샘플링할 비율 (단위: %)
SAMPLING_PERCENTAGES = [5, 50, 100]

# 데이터를 분할할 비율
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# 재현성을 위한 랜덤 시드
RANDOM_STATE = 42

def process_and_copy_split(df: pd.DataFrame, split_name: str, source_dir: Path, dest_dir: Path):
    """데이터프레임에 해당하는 오디오 파일을 복사하고, CSV를 저장합니다."""
    
    split_dest_dir = dest_dir / split_name
    split_dest_dir.mkdir(parents=True, exist_ok=True)
    
    # 새로운 메타데이터 CSV 저장
    df.to_csv(dest_dir / f"{split_name}.csv", index=False, encoding='utf-8-sig')
    
    print(f"  -> Copying {len(df)} files for '{split_name}' split...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        source_path = source_dir / row['path']
        # 대상 폴더 구조는 감정별 하위 폴더를 유지
        relative_dest_path = row['path']
        dest_path = split_dest_dir / relative_dest_path
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if source_path.exists():
            shutil.copy(source_path, dest_path)
        else:
            print(f"Warning: Source file not found and skipped: {source_path}")

def main():
    """메인 샘플링 및 분할 파이프라인"""
    
    if not SOURCE_METADATA.exists():
        print(f"Error: Source metadata file not found at {SOURCE_METADATA}")
        return
        
    df_master = pd.read_csv(SOURCE_METADATA)
    print(f"Master metadata loaded: {len(df_master)} total files.")

    # 최종 보고서 데이터를 저장할 리스트 초기화
    report_data = []
    
    for percent in SAMPLING_PERCENTAGES:
        print(f"\n--- Processing {percent}% Sample ---")
        
        output_dir_percent = OUTPUT_BASE_DIR / f"dataset_{percent}_percent"
        output_dir_percent.mkdir(parents=True, exist_ok=True)
        
        # 계층적 샘플링
        if percent == 100:
            sampled_df = df_master
        else:
            # train_test_split을 사용하여 계층적 샘플링 수행
            sampled_df, _ = train_test_split(
                df_master,
                train_size=percent / 100.0,
                stratify=df_master['emotion'],
                random_state=RANDOM_STATE
            )
        
        print(f"Sampled {len(sampled_df)} files for {percent}% dataset.")
        
        # 1단계 필터링: 첫 번째 분할을 위해 최소 3개의 샘플이 있는 클래스만 남김
        emotion_counts_sampled = sampled_df['emotion'].value_counts()
        classes_to_keep_1 = emotion_counts_sampled[emotion_counts_sampled >= 3].index
        filtered_df_1 = sampled_df[sampled_df['emotion'].isin(classes_to_keep_1)]
        
        if len(filtered_df_1) < len(sampled_df):
            print(f"  -> [Warning] 1차 분할 전, 샘플 수가 3개 미만인 클래스의 데이터 {len(sampled_df) - len(filtered_df_1)}개를 제외했습니다.")
        
        # 1차 분할
        train_df, temp_df = train_test_split(
            filtered_df_1,
            test_size=(VAL_RATIO + TEST_RATIO),
            stratify=filtered_df_1['emotion'],
            random_state=RANDOM_STATE
        )

        # 2단계 필터링: 두 번째 분할을 위해 최소 2개의 샘플이 있는 클래스만 남김
        emotion_counts_temp = temp_df['emotion'].value_counts()
        classes_to_keep_2 = emotion_counts_temp[emotion_counts_temp >= 2].index
        filtered_temp_df = temp_df[temp_df['emotion'].isin(classes_to_keep_2)]

        if len(filtered_temp_df) < len(temp_df):
            print(f"  -> [Warning] 2차 분할 전, 샘플 수가 2개 미만인 클래스의 데이터 {len(temp_df) - len(filtered_temp_df)}개를 제외했습니다.")

        # 2차 분할
        # temp_df를 val과 test로 나누려면 최소 2개의 샘플이 필요
        if len(filtered_temp_df) >= 2:
            relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
            val_df, test_df = train_test_split(
                filtered_temp_df, # temp_df -> filtered_temp_df 로 변경
                test_size=relative_test_ratio,
                stratify=filtered_temp_df['emotion'], # temp_df -> filtered_temp_df 로 변경
                random_state=RANDOM_STATE
            )
        else:
            # 만약 필터링 후 temp_df에 데이터가 거의 없다면, val/test를 비움
            val_df = pd.DataFrame(columns=temp_df.columns)
            test_df = pd.DataFrame(columns=temp_df.columns)
        
        # 보고서 데이터 집계
        # 각 데이터프레임과 이름을 튜플로 묶어 반복 처리
        splits_to_process = [('train', train_df), ('val', val_df), ('test', test_df)]
        
        for split_name, df_split in splits_to_process:
            # 'emotion' 컬럼의 값별로 개수를 셈
            emotion_counts = df_split['emotion'].value_counts()
            for emotion, count in emotion_counts.items():
                report_data.append({
                    'sampling_percentage': f"{percent}%",
                    'split': split_name,
                    'emotion': emotion.lower(),
                    'file_count': count
                })

        # 각 분할별로 파일 복사 및 메타데이터 저장
        process_and_copy_split(train_df, "train", SOURCE_AUDIO_DIR, output_dir_percent)
        process_and_copy_split(val_df, "val", SOURCE_AUDIO_DIR, output_dir_percent)
        process_and_copy_split(test_df, "test", SOURCE_AUDIO_DIR, output_dir_percent)

    # 최종 보고서 생성 및 저장
    if report_data:
        report_df = pd.DataFrame(report_data)
        report_path = OUTPUT_BASE_DIR / "sampling_report.csv"
        report_df.to_csv(report_path, index=False, encoding='utf-8-sig')
        print(f"\nSampling report successfully generated at: {report_path}")

    print("\nAll sampling and splitting processes are complete.")

if __name__ == "__main__":
    main()