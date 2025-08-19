# %%
# subject_split_si.py
from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# %%
# === 설정 ===
SRC = Path("./datasets/korean_emotion_complex_vision_10_percent_verified_processed")  # 10% 전처리본 루트
DST = Path("./datasets/korean_emotion_complex_vision_10_percent_SI")                  # 새 SI 출력 루트
SEED = 42
VAL_RATIO = 0.2
IMG_SUFFIXES = (".jpg", ".jpeg", ".png")  # 필요시 확장

# %%
def subject_id_from_name(fname: str) -> str:
    # 규칙: 첫 '_' 이전 토큰이 subject id
    # 예) "0a3a21be68..._남_20_기쁨_..." -> "0a3a21be68..."
    return fname.split('_', 1)[0]

# %%
def scan_images(src: Path):
    rows = []
    for split in ("train", "val"):
        split_dir = src / split
        if not split_dir.exists():
            continue
        for img_path in split_dir.rglob("*"):
            if img_path.suffix.lower() in IMG_SUFFIXES and img_path.is_file():
                emotion = img_path.parent.name  # 부모 폴더명이 감정 라벨
                fname = img_path.name
                sid = subject_id_from_name(fname)
                rows.append({"path": img_path, "emotion": emotion, "subject": sid})
    # 혹시 train/val 없이 바로 감정 폴더가 있는 구조도 지원
    if not rows:
        for img_path in src.rglob("*"):
            if img_path.suffix.lower() in IMG_SUFFIXES and img_path.is_file():
                # .../<emotion>/<file>
                emotion = img_path.parent.name
                fname = img_path.name
                sid = subject_id_from_name(fname)
                rows.append({"path": img_path, "emotion": emotion, "subject": sid})
    return pd.DataFrame(rows)

# %%
def copy_pair(src_img: Path, dst_img: Path):
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    # 파일 복사(빠르게 하고 싶으면 shutil.copy2 대신 하드링크/심볼릭링크로 바꿀 수 있음)
    shutil.copy2(src_img, dst_img)
    # 동일 이름의 .json 라벨이 있으면 같이 복사
    src_json = src_img.with_suffix(".json")
    if src_json.exists():
        shutil.copy2(src_json, dst_img.with_suffix(".json"))

# %%
def main():
    df = scan_images(SRC)
    assert not df.empty, f"이미지 못 찾음: {SRC}"
    print(f"[scan] total images = {len(df)} | classes={sorted(df.emotion.unique())} | subjects={df.subject.nunique()}")

    # StratifiedGroupKFold: 클래스 분포 유지 + subject 단위 분리
    n_splits = max(2, min(10, round(1 / VAL_RATIO)))  # 대략적인 분할 수 추정
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(sgkf.split(df["path"], df["emotion"], groups=df["subject"]))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    # 교차검증: subject 중복 없어야 함
    assert set(train_df.subject).isdisjoint(set(val_df.subject)), "train/val에 동일 subject가 섞였습니다!"

    # 복사
    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        for _, row in split_df.iterrows():
            # 출력 경로: <DST>/<split>/<emotion>/<filename>
            dst_img = DST / split_name / row["emotion"] / row["path"].name
            copy_pair(row["path"], dst_img)
            
    # 분포 로그
    def pct(dfs):
        return (dfs["emotion"].value_counts(normalize=True).sort_index().round(3)).to_dict()
    print(f"[done] train={len(train_df)} val={len(val_df)} | train_dist={pct(train_df)} | val_dist={pct(val_df)}")
    print(f"[subjects] train={train_df.subject.nunique()} val={val_df.subject.nunique()} | overlap=0")

# %%
if __name__ == "__main__":
    main()

# %%



