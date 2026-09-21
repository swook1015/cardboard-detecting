def main():

    # ============================================================
    # 사용자 설정
    # ============================================================

    # YOLO dataset 루트 폴더
    DATASET_DIR = r"C:\Users\11sungwook\OneDrive - Customparts\바탕 화면\2026_Q3_box\parcel2d_yolo"

    # 랜덤으로 몇 장 시각화할지
    NUM_SAMPLES = 10

    # 랜덤 시드
    SEED = 42

    # "train", "val", "test", "all" 중 선택
    SPLIT = "all"

    # 결과 저장 폴더
    OUTPUT_DIR = os.path.join(DATASET_DIR, "inspection_samples")


    # ============================================================
    # 여기부터 실행부
    # ============================================================

    root = DATASET_DIR
    output_dir = OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    if SPLIT == "all":
        splits_to_scan = ["train", "val", "test"]
    else:
        splits_to_scan = [SPLIT]

    # 이미지와 라벨이 둘 다 존재하는 scene 수집
    candidates = []

    for split in splits_to_scan:

        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)

        if not os.path.isdir(img_dir):
            print(f"[없음] 이미지 폴더: {img_dir}")
            continue

        if not os.path.isdir(lbl_dir):
            print(f"[없음] 라벨 폴더: {lbl_dir}")
            continue

        for fname in sorted(os.listdir(img_dir)):

            if not fname.lower().endswith(".png"):
                continue

            uuid = os.path.splitext(fname)[0]

            label_path = os.path.join(
                lbl_dir,
                f"{uuid}.txt"
            )

            if os.path.isfile(label_path):
                candidates.append(
                    (split, uuid)
                )

    print(f"\n전체 후보 이미지 수: {len(candidates)}")

    if len(candidates) == 0:
        print("시각화할 이미지가 없습니다.")
        return


    # ============================================================
    # 랜덤 샘플 선택
    # ============================================================

    rng = random.Random(SEED)

    sample_count = min(
        NUM_SAMPLES,
        len(candidates)
    )

    sampled = rng.sample(
        candidates,
        sample_count
    )


    # ============================================================
    # 시각화
    # ============================================================

    font = get_font(14)
    results = []

    for i, (split, uuid) in enumerate(sampled):

        idx_str = f"{i:02d}"

        img_path = os.path.join(
            root,
            "images",
            split,
            f"{uuid}.png"
        )

        lbl_path = os.path.join(
            root,
            "labels",
            split,
            f"{uuid}.txt"
        )

        # 라벨 읽기
        instances = load_label_file(lbl_path)

        # 이미지 읽기
        image = Image.open(img_path)

        w, h = image.size


        # --------------------------------------------------------
        # Annotation 이미지 생성
        # --------------------------------------------------------

        annotated = draw_annotations(
            image,
            instances,
            font
        )

        annotated_path = os.path.join(
            output_dir,
            f"{idx_str}_{uuid}_annotated.jpg"
        )

        annotated.save(
            annotated_path,
            quality=92
        )


        # --------------------------------------------------------
        # 원본 + annotation 비교 이미지
        # --------------------------------------------------------

        comparison = make_comparison(
            image,
            annotated
        )

        comparison_path = os.path.join(
            output_dir,
            f"{idx_str}_{uuid}_comparison.jpg"
        )

        comparison.save(
            comparison_path,
            quality=92
        )


        # --------------------------------------------------------
        # 콘솔 출력
        # --------------------------------------------------------

        print()
        print("=" * 70)
        print(f"[{idx_str}]")
        print(f"UUID       : {uuid}")
        print(f"Split      : {split}")
        print(f"Image Size : {w} x {h}")
        print(f"Instances  : {len(instances)}")

        for inst_idx, inst in enumerate(instances):

            cx, cy, bw, bh = inst["bbox"]

            print()
            print(f"Object {inst_idx}")

            print(
                f"BBox: "
                f"cx={cx:.4f}, "
                f"cy={cy:.4f}, "
                f"w={bw:.4f}, "
                f"h={bh:.4f}"
            )

            for k, (kx, ky, kv) in enumerate(inst["keypoints"]):

                print(
                    f"KP{k} "
                    f"{KEYPOINT_NAMES[k]} "
                    f"x={kx:.4f}, "
                    f"y={ky:.4f}, "
                    f"v={int(kv)}"
                )


        results.append({
            "index": idx_str,
            "uuid": uuid,
            "split": split,
            "num_instances": len(instances),
            "annotated_file": os.path.basename(annotated_path),
            "comparison_file": os.path.basename(comparison_path),
        })


    # ============================================================
    # 결과 출력
    # ============================================================

    print()
    print("=" * 70)
    print("시각화 완료")
    print("=" * 70)

    print(f"생성 이미지 수 : {len(results)}")
    print(f"저장 위치      : {output_dir}")


    # ============================================================
    # 결과 summary JSON 저장
    # ============================================================

    import json

    summary_path = os.path.join(
        output_dir,
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
                "results": results
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Summary 저장 : {summary_path}")


if __name__ == "__main__":
    main()