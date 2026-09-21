#!/usr/bin/env python3
"""
convert_parcel2d_to_yolo26_pose.py
====================================
Converts Parcel2D Real merge_dataset into YOLO26X-Pose format.

Input:
    parcel2d_real/merge_dataset/
        <UUID>/
            00_rgb.png
            annotations.json

Output:
    parcel2d_yolo/
        images/{train,val,test}/<UUID>.png
        labels/{train,val,test}/<UUID>.txt
        data.yaml
        source_manifest.json
        split_manifest.json

Keypoint ordering (TAMPAR semantic, consistent with register_datasets.py):
    KP0: front_intersect3_inside
    KP1: front_intersect2
    KP2: front_intersect3_left
    KP3: front_intersect3_right
    KP4: back_intersect3_outside
    KP5: back_hidden
    KP6: back_intersect2_left
    KP7: back_intersect2_right

Visibility convention (COCO-style):
    2 = labeled and visible
    1 = labeled but not visible (occluded)
    0 = not labeled

Since all Parcel2D Real annotations have all 8 keypoints visible (verified via
top-level COCO JSON: 100% vis=2), all keypoints are assigned visibility=2.
"""

import argparse
import hashlib
import json
import math
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)

KP_NAMES = [
    "front_intersect3_inside",
    "front_intersect2",
    "front_intersect3_left",
    "front_intersect3_right",
    "back_intersect3_outside",
    "back_hidden",
    "back_intersect2_left",
    "back_intersect2_right",
]

SKELETON = [
    # Front face
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 3),
    # Back face
    (4, 6), (4, 7),
    (5, 6), (5, 7),
    # Side edges
    (0, 4), (2, 6), (3, 7), (1, 5),
]

IMAGE_W = 1280
IMAGE_H = 960

# ─────────────────────────────────────────────
# Scene scanning
# ─────────────────────────────────────────────

def scan_merge_dataset(merge_dir: Path) -> Dict:
    """Scan merge_dataset and validate all scenes."""
    merge_dir = Path(merge_dir)
    results = {
        "valid_scenes": [],
        "invalid_scenes": [],
        "stats": {},
    }

    all_subdirs = sorted([p for p in merge_dir.iterdir() if p.is_dir()])
    uuid_dirs = [p for p in all_subdirs if UUID_PATTERN.match(p.name)]
    non_uuid = [p for p in all_subdirs if not UUID_PATTERN.match(p.name)]

    for scene_dir in uuid_dirs:
        uuid = scene_dir.name
        rgb_path = scene_dir / "00_rgb.png"
        ann_path = scene_dir / "annotations.json"

        issues = []

        if not rgb_path.exists():
            issues.append("missing_rgb")
        if not ann_path.exists():
            issues.append("missing_ann")
            results["invalid_scenes"].append({"uuid": uuid, "issues": issues})
            continue

        # Parse annotation
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                ann_data = json.load(f)
        except Exception as e:
            issues.append(f"corrupt_json: {e}")
            results["invalid_scenes"].append({"uuid": uuid, "issues": issues})
            continue

        scene_annotations = ann_data.get("annotations", [])
        if len(scene_annotations) == 0:
            issues.append("empty_annotations")

        valid_anns = []
        for ann in scene_annotations:
            kp = ann.get("keypoints", [])
            bbox = ann.get("bbox", [])
            if len(kp) != 16:
                issues.append(f"invalid_kp_len_{ann.get('id')}={len(kp)}")
                continue
            if len(bbox) != 4:
                issues.append(f"invalid_bbox_{ann.get('id')}={len(bbox)}")
                continue
            if bbox[2] <= 0 or bbox[3] <= 0:
                issues.append(f"zero_bbox_{ann.get('id')}")
                continue
            valid_anns.append(ann)

        if issues:
            results["invalid_scenes"].append({"uuid": uuid, "issues": issues})
        else:
            # Get image dimensions from annotation JSON or use default
            images_info = ann_data.get("images", [])
            if images_info:
                img_w = images_info[0].get("width", IMAGE_W)
                img_h = images_info[0].get("height", IMAGE_H)
                file_name = images_info[0].get("file_name", "")
            else:
                img_w, img_h = IMAGE_W, IMAGE_H
                file_name = ""

            results["valid_scenes"].append({
                "uuid": uuid,
                "scene_dir": str(scene_dir),
                "rgb_path": str(rgb_path),
                "ann_path": str(ann_path),
                "annotations": valid_anns,
                "img_w": img_w,
                "img_h": img_h,
                "file_name": file_name,
            })

    results["stats"] = {
        "total_dirs": len(all_subdirs),
        "uuid_dirs": len(uuid_dirs),
        "non_uuid_dirs": len(non_uuid),
        "valid_scenes": len(results["valid_scenes"]),
        "invalid_scenes": len(results["invalid_scenes"]),
    }
    return results


# ─────────────────────────────────────────────
# Source split lookup (val vs test provenance)
# ─────────────────────────────────────────────

def build_source_manifest(
    merge_dir: Path,
    val_dir: Path,
    test_dir: Path,
    valid_scenes: List[Dict],
) -> Dict[str, str]:
    """Map UUID -> 'validation' or 'test' based on original source directories."""
    val_uuids = set(
        d.name for d in val_dir.iterdir()
        if d.is_dir() and UUID_PATTERN.match(d.name)
    )
    test_uuids = set(
        d.name for d in test_dir.iterdir()
        if d.is_dir() and UUID_PATTERN.match(d.name)
    )

    manifest = {}
    for scene in valid_scenes:
        uuid = scene["uuid"]
        if uuid in val_uuids:
            manifest[uuid] = "validation"
        elif uuid in test_uuids:
            manifest[uuid] = "test"
        else:
            manifest[uuid] = "unknown"

    return manifest


# ─────────────────────────────────────────────
# Train/Val/Test split
# ─────────────────────────────────────────────

def assign_splits(
    valid_scenes: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, str]:
    """Assign each UUID to train/val/test split."""
    rng = random.Random(seed)
    uuids = [s["uuid"] for s in valid_scenes]
    shuffled = uuids[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_train -= 1
        n_test = 1

    split_map = {}
    for uuid in shuffled[:n_train]:
        split_map[uuid] = "train"
    for uuid in shuffled[n_train:n_train + n_val]:
        split_map[uuid] = "val"
    for uuid in shuffled[n_train + n_val:]:
        split_map[uuid] = "test"

    return split_map


# ─────────────────────────────────────────────
# Annotation conversion
# ─────────────────────────────────────────────

def normalize_bbox(bbox: List[float], w: int, h: int) -> Tuple[float, float, float, float]:
    """Convert COCO [x,y,bw,bh] to YOLO [cx,cy,nw,nh] normalized."""
    x, y, bw, bh = [float(v) for v in bbox]
    cx = (x + bw / 2.0) / w
    cy = (y + bh / 2.0) / h
    nw = bw / w
    nh = bh / h
    return cx, cy, nw, nh


def build_yolo_label_row(
    ann: Dict,
    img_w: int,
    img_h: int,
    class_id: int = 0,
    visibility: int = 2,
) -> Optional[List[float]]:
    """
    Convert one annotation to YOLO pose label row.
    Returns [class_id, cx, cy, bw, bh, kp0x, kp0y, kp0v, ..., kp7x, kp7y, kp7v]
    Total: 5 + 8*3 = 29 values.
    """
    bbox = ann.get("bbox", [])
    kp = ann.get("keypoints", [])

    if len(bbox) != 4 or len(kp) != 16:
        return None

    if bbox[2] <= 0 or bbox[3] <= 0:
        return None

    try:
        cx, cy, nw, nh = normalize_bbox(bbox, img_w, img_h)
    except (ValueError, ZeroDivisionError):
        return None

    # Validate bbox range
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < nw <= 1.0 and 0.0 < nh <= 1.0):
        return None

    # Build keypoint values: [x_norm, y_norm, vis] for each of 8 KPs
    kp_values = []
    for i in range(0, 16, 2):
        x_raw = float(kp[i])
        y_raw = float(kp[i + 1])

        x_norm = x_raw / img_w
        y_norm = y_raw / img_h

        if math.isnan(x_norm) or math.isnan(y_norm) or math.isinf(x_norm) or math.isinf(y_norm):
            return None

        # Clamp to [0, 1] with small tolerance
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))

        kp_values.extend([x_norm, y_norm, float(visibility)])

    row = [float(class_id), cx, cy, nw, nh] + kp_values
    assert len(row) == 29, f"Expected 29 values, got {len(row)}"
    return row


def format_label_row(row: List[float]) -> str:
    """Format label row as string with adequate precision."""
    parts = []
    parts.append(str(int(row[0])))  # class_id
    for v in row[1:5]:             # bbox
        parts.append(f"{v:.6f}")
    for i in range(0, 24, 3):      # 8 keypoints
        parts.append(f"{row[5 + i]:.6f}")   # x
        parts.append(f"{row[6 + i]:.6f}")   # y
        parts.append(str(int(row[7 + i])))  # vis
    return " ".join(parts)


# ─────────────────────────────────────────────
# Image hash
# ─────────────────────────────────────────────

def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ─────────────────────────────────────────────
# Main build function
# ─────────────────────────────────────────────

def build_dataset(
    merge_dir: Path,
    output_dir: Path,
    parcel2d_root: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    overwrite: bool = False,
):
    merge_dir = Path(merge_dir)
    output_dir = Path(output_dir)
    parcel2d_root = Path(parcel2d_root)
    val_dir = parcel2d_root / "validation"
    test_dir = parcel2d_root / "test"

    print("=" * 70)
    print("PARCEL2D REAL -> YOLO26X-POSE DATASET BUILDER")
    print("=" * 70)
    print(f"Input  : {merge_dir}")
    print(f"Output : {output_dir}")
    print(f"Split  : train={train_ratio:.0%} / val={val_ratio:.0%} / test={1-train_ratio-val_ratio:.0%}")
    print(f"Seed   : {seed}")
    print()

    # 1. Scan merge_dataset
    print("[1/8] Scanning merge_dataset ...")
    scan = scan_merge_dataset(merge_dir)
    stats = scan["stats"]
    valid_scenes = scan["valid_scenes"]
    invalid_scenes = scan["invalid_scenes"]
    print(f"  Total dirs       : {stats['total_dirs']}")
    print(f"  UUID dirs        : {stats['uuid_dirs']}")
    print(f"  Valid scenes     : {stats['valid_scenes']}")
    print(f"  Invalid scenes   : {stats['invalid_scenes']}")
    if invalid_scenes:
        for s in invalid_scenes[:5]:
            print(f"  INVALID: {s['uuid'][:12]}... {s['issues']}")

    # 2. Source manifest (val vs test provenance)
    print("[2/8] Building source manifest ...")
    source_manifest = build_source_manifest(merge_dir, val_dir, test_dir, valid_scenes)
    val_count = sum(1 for v in source_manifest.values() if v == "validation")
    test_count = sum(1 for v in source_manifest.values() if v == "test")
    print(f"  From validation  : {val_count}")
    print(f"  From test        : {test_count}")

    # 3. Assign splits
    print("[3/8] Assigning train/val/test splits ...")
    split_map = assign_splits(valid_scenes, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    split_counts = defaultdict(int)
    for v in split_map.values():
        split_counts[v] += 1
    print(f"  train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}")

    # 4. Create output directories
    print("[4/8] Creating output directories ...")
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 5. Convert and copy
    print("[5/8] Converting annotations and copying images ...")
    total_instances = 0
    skipped_instances = 0
    split_instance_counts = defaultdict(int)
    split_manifest = {}
    image_hashes = {}

    for scene in valid_scenes:
        uuid = scene["uuid"]
        split = split_map.get(uuid, "train")
        img_w = scene["img_w"]
        img_h = scene["img_h"]

        # Copy image
        src_rgb = Path(scene["rgb_path"])
        dst_img = output_dir / "images" / split / f"{uuid}.png"
        if not dst_img.exists() or overwrite:
            shutil.copy2(str(src_rgb), str(dst_img))

        img_hash = md5_file(dst_img)
        image_hashes[uuid] = img_hash

        # Generate label
        label_rows = []
        for ann in scene["annotations"]:
            row = build_yolo_label_row(ann, img_w, img_h, class_id=0, visibility=2)
            if row is not None:
                label_rows.append(row)
                total_instances += 1
                split_instance_counts[split] += 1
            else:
                skipped_instances += 1

        dst_label = output_dir / "labels" / split / f"{uuid}.txt"
        with open(dst_label, "w", encoding="utf-8") as f:
            for row in label_rows:
                f.write(format_label_row(row) + "\n")

        split_manifest[uuid] = {
            "uuid": uuid,
            "split": split,
            "source_split": source_manifest.get(uuid, "unknown"),
            "instances": len(label_rows),
            "image_hash": img_hash,
            "file_name": scene.get("file_name", ""),
        }

    print(f"  Total instances written : {total_instances}")
    print(f"  Skipped instances       : {skipped_instances}")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {split_instance_counts[split]} instances")

    # 6. Write data.yaml
    print("[6/8] Writing data.yaml ...")
    data_yaml_path = output_dir / "data.yaml"
    yaml_content = (
        f"# Parcel2D Real - YOLO26X-Pose Dataset\n"
        f"# Generated by convert_parcel2d_to_yolo26_pose.py\n"
        f"#\n"
        f"# Keypoint ordering (TAMPAR semantic):\n"
        f"#   KP0: front_intersect3_inside\n"
        f"#   KP1: front_intersect2\n"
        f"#   KP2: front_intersect3_left\n"
        f"#   KP3: front_intersect3_right\n"
        f"#   KP4: back_intersect3_outside\n"
        f"#   KP5: back_hidden\n"
        f"#   KP6: back_intersect2_left\n"
        f"#   KP7: back_intersect2_right\n"
        f"#\n"
        f"# NOTE: flip_idx is not set (set to []) because the semantic meaning\n"
        f"# of left/right keypoints does NOT follow simple left-right flip rules\n"
        f"# (front vs back face distinction breaks on horizontal flip).\n"
        f"# Disable horizontal flip augmentation (fliplr=0) during training.\n"
        f"\n"
        f"path: {str(output_dir.resolve()).replace(chr(92), '/')}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"names:\n"
        f"  0: parcel\n"
        f"\n"
        f"kpt_shape: [8, 3]\n"
    )
    data_yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"  Written: {data_yaml_path}")

    # 7. Write source_manifest.json
    print("[7/8] Writing manifests ...")
    source_manifest_path = output_dir / "source_manifest.json"
    with open(source_manifest_path, "w", encoding="utf-8") as f:
        json.dump(source_manifest, f, indent=2, ensure_ascii=False)
    print(f"  Written: {source_manifest_path}")

    split_manifest_path = output_dir / "split_manifest.json"
    with open(split_manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2, ensure_ascii=False)
    print(f"  Written: {split_manifest_path}")

    # 8. Summary
    print("[8/8] Build complete.")
    print()
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Total valid scenes : {len(valid_scenes)}")
    for split in ["train", "val", "test"]:
        n_imgs = len(list((output_dir / "images" / split).glob("*.png")))
        n_lbls = len(list((output_dir / "labels" / split).glob("*.txt")))
        print(f"  {split:5s}: {n_imgs} images, {n_lbls} labels, {split_instance_counts[split]} instances")
    print(f"Total instances    : {total_instances}")
    print(f"Skipped instances  : {skipped_instances}")
    print(f"Output dir         : {output_dir}")

    return {
        "valid_scenes": len(valid_scenes),
        "split_counts": dict(split_counts),
        "instance_counts": dict(split_instance_counts),
        "total_instances": total_instances,
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert Parcel2D Real merge_dataset to YOLO26X-Pose format."
    )
    parser.add_argument(
        "--merge-dir",
        type=Path,
        default=Path("parcel2d_real/merge_dataset"),
        help="Path to merge_dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("parcel2d_yolo"),
        help="Output directory for YOLO dataset",
    )
    parser.add_argument(
        "--parcel2d-root",
        type=Path,
        default=Path("parcel2d_real"),
        help="Parcel2D Real root (for source provenance lookup)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files"
    )
    args = parser.parse_args()

    build_dataset(
        merge_dir=args.merge_dir,
        output_dir=args.output_dir,
        parcel2d_root=args.parcel2d_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
