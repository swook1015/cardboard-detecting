from pathlib import Path
import shutil

# ============================================================
# 경로 설정
# ============================================================

# parcel2d_yolo 루트
DATASET_ROOT = Path(
    r"C:\Users\11sungwook\OneDrive - Customparts\바탕 화면\2026_Q3_box\parcel2d_yolo"
)

# Claude가 준 stale 파일 목록
TXT_PATH = "C:/Users/11sungwook/OneDrive - Customparts/바탕 화면/2026_Q3_box/Claude outputs/STALE_FILES_TO_DELETE.txt"

# 삭제 대신 일단 여기로 이동
BACKUP_DIR = DATASET_ROOT / "stale_backup"

# ============================================================
# 실행
# ============================================================

BACKUP_DIR.mkdir(parents=True, exist_ok=True)

moved = 0
missing = 0
skipped = 0

with open(TXT_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()

    # 실제 PNG 경로가 적힌 줄만 처리
    if not line.lower().endswith(".png"):
        continue

    # txt에는 images\train\xxx.png 형식으로 들어 있음
    relative_path = Path(line.replace("\\", "/"))

    src = DATASET_ROOT / relative_path
    dst = BACKUP_DIR / src.name

    if not src.exists():
        print(f"[없음] {src}")
        missing += 1
        continue

    if dst.exists():
        print(f"[이미 존재] {dst}")
        skipped += 1
        continue

    shutil.move(str(src), str(dst))

    print(f"[이동] {src.name}")
    moved += 1

print()
print("=" * 60)
print("완료")
print(f"이동 성공 : {moved}")
print(f"파일 없음 : {missing}")
print(f"이미 존재 : {skipped}")
print("=" * 60)