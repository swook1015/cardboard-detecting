#!/usr/bin/env python3
"""
Parcel2D YOLO26X-Pose Dataset Visualizer

FINAL YOLO label 파일을 다시 읽어서
bbox + 8 keypoints + skeleton을 이미지 위에 시각화한다.

사용 방법:
    1. 아래 main()의 DATASET_DIR 수정
    2. NUM_SAMPLES, SPLIT 필요하면 수정
    3. 그냥 실행

    python visualize_yolo26_pose_dataset.py
"""

import os
import random
import json

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Keypoint 설정
# ============================================================

KEYPOINT_NAMES = [
    "front_intersect3_inside",   # 0
    "front_intersect2",          # 1
    "front_intersect3_left",     # 2
    "front_intersect3_right",    # 3
    "back_intersect3_outside",   # 4
    "back_hidden",               # 5
    "back_intersect2_left",      # 6
    "back_intersect2_right",     # 7
]

NUM_KEYPOINTS = 8


# ============================================================
# Cuboid skeleton 연결 관계
# ============================================================

SKELETON_EDGES = [
    (4, 6),
    (4, 7),
    (0, 4),
    (2, 6),
    (3, 7),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (5, 6),
    (5, 7),
    (1, 5),
]


# ============================================================
# 색상 설정
# ============================================================

KP_COLORS = [
    (255, 0, 0),       # 0 red
    (255, 128, 0),     # 1 orange
    (255, 255, 0),     # 2 yellow
    (0, 255, 0),       # 3 green
    (0, 255, 255),     # 4 cyan
    (0, 128, 255),     # 5 blue
    (128, 0, 255),     # 6 purple
    (255, 0, 255),     # 7 magenta
]

SKELETON_COLOR = (255, 255, 255)

INSTANCE_COLORS = [
    (0, 255, 0),
    (255, 165, 0),
    (0, 200, 255),
    (255, 0, 200),
]


# ============================================================
# 로그 출력
# ============================================================

def log(msg):
    print(msg, flush=True)


# ============================================================
# YOLO Pose label 읽기
#
# 한 줄 구조:
#
# class_id
# cx cy w h
# kp0_x kp0_y kp0_v
# ...
# kp7_x kp7_y kp7_v
#
# 총 29-field
# ============================================================

def load_label_file(path):

    instances = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            expected_fields = 5 + NUM_KEYPOINTS * 3

            if len(parts) != expected_fields:
                log(
                    f"[WARNING] 잘못된 label 형식: {path} "
                    f"({len(parts)} fields, expected {expected_fields})"
                )
                continue

            values = [float(p) for p in parts]

            class_id = int(values[0])

            cx, cy, bw, bh = values[1:5]

            kp_values = values[5:]

            keypoints = []

            for k in range(NUM_KEYPOINTS):

                kx = kp_values[3 * k]
                ky = kp_values[3 * k + 1]
                kv = kp_values[3 * k + 2]

                keypoints.append(
                    (kx, ky, kv)
                )

            instances.append(
                {
                    "class_id": class_id,
                    "bbox": (cx, cy, bw, bh),
                    "keypoints": keypoints,
                }
            )

    return instances


# ============================================================
# Font
# ============================================================

def get_font(size=14):

    font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]

    for font_path in font_candidates:

        if os.path.isfile(font_path):

            try:
                return ImageFont.truetype(
                    font_path,
                    size
                )

            except Exception:
                pass

    return ImageFont.load_default()


# ============================================================
# Annotation 그리기
# ============================================================

def draw_annotations(image, instances, font):

    img = image.copy().convert("RGB")

    draw = ImageDraw.Draw(img)

    width, height = img.size


    for inst_idx, inst in enumerate(instances):

        # --------------------------------------------------------
        # 객체별 bbox 색상
        # --------------------------------------------------------

        instance_color = INSTANCE_COLORS[
            inst_idx % len(INSTANCE_COLORS)
        ]


        # --------------------------------------------------------
        # bbox
        # YOLO normalized cx, cy, w, h
        # → pixel x1, y1, x2, y2
        # --------------------------------------------------------

        cx, cy, bw, bh = inst["bbox"]

        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height

        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height


        draw.rectangle(
            [x1, y1, x2, y2],
            outline=instance_color,
            width=3
        )


        draw.text(
            (x1 + 4, max(0, y1 - 18)),
            f"inst{inst_idx}",
            fill=instance_color,
            font=font
        )


        # --------------------------------------------------------
        # keypoint normalized → pixel 좌표
        # --------------------------------------------------------

        keypoints = inst["keypoints"]

        pixel_points = []

        for k in range(NUM_KEYPOINTS):

            kx, ky, kv = keypoints[k]

            px = kx * width
            py = ky * height

            pixel_points.append(
                (px, py)
            )


        # --------------------------------------------------------
        # Skeleton 먼저 그리기
        # --------------------------------------------------------

        for a, b in SKELETON_EDGES:

            draw.line(
                [
                    pixel_points[a],
                    pixel_points[b]
                ],
                fill=SKELETON_COLOR,
                width=2
            )


        # --------------------------------------------------------
        # Keypoint 그리기
        # --------------------------------------------------------

        for k in range(NUM_KEYPOINTS):

            px, py = pixel_points[k]

            kv = keypoints[k][2]

            kp_color = KP_COLORS[
                k % len(KP_COLORS)
            ]

            radius = 6


            draw.ellipse(
                [
                    px - radius,
                    py - radius,
                    px + radius,
                    py + radius
                ],
                fill=kp_color,
                outline=(0, 0, 0)
            )


            label = f"{k}:v{int(kv)}"

            draw.text(
                (px + 8, py - 8),
                label,
                fill=kp_color,
                font=font
            )


    return img


# ============================================================
# 원본 + annotation 비교 이미지
# ============================================================

def make_comparison(original, annotated):

    width, height = original.size

    comparison = Image.new(
        "RGB",
        (width * 2 + 10, height),
        (30, 30, 30)
    )

    comparison.paste(
        original.convert("RGB"),
        (0, 0)
    )

    comparison.paste(
        annotated.convert("RGB"),
        (width + 10, 0)
    )

    return comparison


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # 사용자 설정
    # ========================================================

    DATASET_DIR = r"C:\Users\11sungwook\OneDrive - Customparts\바탕 화면\2026_Q3_box\parcel2d_yolo"

    # 몇 장 시각화할지
    NUM_SAMPLES = 20

    # 랜덤 seed
    SEED = None

    # "train", "val", "test", "all"
    SPLIT = "all"

    # 결과 저장 폴더
    OUTPUT_DIR = os.path.join(
        DATASET_DIR,
        "inspection_samples"
    )


    # ========================================================
    # 기본 경로 확인
    # ========================================================

    if not os.path.isdir(DATASET_DIR):

        print(
            f"[ERROR] dataset 폴더가 존재하지 않습니다:\n"
            f"{DATASET_DIR}"
        )

        return


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    # ========================================================
    # 사용할 split 결정
    # ========================================================

    if SPLIT == "all":

        splits_to_scan = [
            "train",
            "val",
            "test"
        ]

    else:

        splits_to_scan = [
            SPLIT
        ]


    # ========================================================
    # 이미지 + 라벨 쌍 수집
    # ========================================================

    candidates = []


    for split in splits_to_scan:

        image_dir = os.path.join(
            DATASET_DIR,
            "images",
            split
        )

        label_dir = os.path.join(
            DATASET_DIR,
            "labels",
            split
        )


        if not os.path.isdir(image_dir):

            print(
                f"[WARNING] 이미지 폴더 없음: "
                f"{image_dir}"
            )

            continue


        if not os.path.isdir(label_dir):

            print(
                f"[WARNING] 라벨 폴더 없음: "
                f"{label_dir}"
            )

            continue


        for filename in sorted(
            os.listdir(image_dir)
        ):

            if not filename.lower().endswith(".png"):
                continue


            uuid = os.path.splitext(
                filename
            )[0]


            label_path = os.path.join(
                label_dir,
                f"{uuid}.txt"
            )


            if os.path.isfile(label_path):

                candidates.append(
                    (split, uuid)
                )


    print()
    print("=" * 70)
    print(f"전체 후보 scene 수 : {len(candidates)}")
    print(f"대상 split         : {splits_to_scan}")
    print("=" * 70)


    if len(candidates) == 0:

        print(
            "[ERROR] 이미지와 label이 모두 존재하는 "
            "scene을 찾지 못했습니다."
        )

        return


    # ========================================================
    # 랜덤 샘플링
    # ========================================================

    rng = random.Random(SEED)

    sample_count = min(
        NUM_SAMPLES,
        len(candidates)
    )


    sampled = rng.sample(
        candidates,
        sample_count
    )


    # ========================================================
    # 시각화
    # ========================================================

    font = get_font(14)

    results = []


    for i, (split, uuid) in enumerate(sampled):

        index_string = f"{i:02d}"


        # ----------------------------------------------------
        # 파일 경로
        # ----------------------------------------------------

        image_path = os.path.join(
            DATASET_DIR,
            "images",
            split,
            f"{uuid}.png"
        )


        label_path = os.path.join(
            DATASET_DIR,
            "labels",
            split,
            f"{uuid}.txt"
        )


        # ----------------------------------------------------
        # label 읽기
        # ----------------------------------------------------

        instances = load_label_file(
            label_path
        )


        # ----------------------------------------------------
        # 이미지 읽기
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")


        width, height = image.size


        # ----------------------------------------------------
        # annotation 이미지 생성
        # ----------------------------------------------------

        annotated = draw_annotations(
            image,
            instances,
            font
        )


        annotated_path = os.path.join(
            OUTPUT_DIR,
            f"{index_string}_{uuid}_annotated.jpg"
        )


        annotated.save(
            annotated_path,
            quality=92
        )


        # ----------------------------------------------------
        # 원본 + annotation 비교 이미지
        # ----------------------------------------------------

        comparison = make_comparison(
            image,
            annotated
        )


        comparison_path = os.path.join(
            OUTPUT_DIR,
            f"{index_string}_{uuid}_comparison.jpg"
        )


        comparison.save(
            comparison_path,
            quality=92
        )


        # ----------------------------------------------------
        # 콘솔 출력
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print(f"[{index_string}]")
        print(f"UUID       : {uuid}")
        print(f"Split      : {split}")
        print(f"Image Size : {width} x {height}")
        print(f"Instances  : {len(instances)}")


        for inst_idx, inst in enumerate(instances):

            cx, cy, bw, bh = inst["bbox"]


            print()
            print(f"Object {inst_idx}")


            print(
                f"BBox "
                f"cx={cx:.4f}, "
                f"cy={cy:.4f}, "
                f"w={bw:.4f}, "
                f"h={bh:.4f}"
            )


            for k, (kx, ky, kv) in enumerate(
                inst["keypoints"]
            ):

                print(
                    f"  KP{k} "
                    f"({KEYPOINT_NAMES[k]}) "
                    f"x={kx:.4f}, "
                    f"y={ky:.4f}, "
                    f"v={int(kv)} "
                    f"-> pixel=({kx * width:.1f}, "
                    f"{ky * height:.1f})"
                )


        # ----------------------------------------------------
        # summary용 결과 저장
        # ----------------------------------------------------

        results.append(
            {
                "index": index_string,
                "uuid": uuid,
                "split": split,
                "num_instances": len(instances),
                "annotated_file": os.path.basename(
                    annotated_path
                ),
                "comparison_file": os.path.basename(
                    comparison_path
                ),
            }
        )


    # ========================================================
    # JSON summary 저장
    # ========================================================

    summary_path = os.path.join(
        OUTPUT_DIR,
        "_visualization_summary.json"
    )


    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "seed": SEED,
                "split": SPLIT,
                "num_samples": sample_count,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False
        )


    # ========================================================
    # 완료 출력
    # ========================================================

    print()
    print("=" * 70)
    print("시각화 완료")
    print("=" * 70)
    print(f"생성 scene 수 : {len(results)}")
    print(f"저장 위치     : {OUTPUT_DIR}")
    print(f"Summary      : {summary_path}")
    print("=" * 70)


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    main()