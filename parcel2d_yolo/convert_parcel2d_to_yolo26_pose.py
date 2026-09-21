#!/usr/bin/env python3
"""
Parcel2D Real -> YOLO26X-Pose Dataset Converter

Converts merge_dataset/<UUID>/annotations.json + 00_rgb.png scenes into a
YOLO26X-Pose training dataset with 8 keypoints per instance.

Split: scene-level random split (seeded shuffle over all scene UUIDs, then
sliced train/val/test). An earlier version used a MODEL-GROUP split
(connected components over shared `model` IDs) to try to keep any one 3D
box model out of more than one split, but that was abandoned: physical-box
identity cannot be reliably determined across scenes from the annotation
metadata alone (different model filenames may refer to the same physical
box geometry). The scene (UUID) is now the split unit; all instances of a
scene stay together and there is no UUID overlap across splits.

Keypoint semantic order (fixed, must never be reordered):
  KP0 = front_intersect3_inside
  KP1 = front_intersect2
  KP2 = front_intersect3_left
  KP3 = front_intersect3_right
  KP4 = back_intersect3_outside
  KP5 = back_hidden
  KP6 = back_intersect2_left
  KP7 = back_intersect2_right

Visibility: v=2 for ALL keypoints in ALL scenes (see dataset_build_report.md
for the justification). KP5 (back_hidden) is NOT set to v=1 despite its name.

category_id is always 0 ("parcel") - single class dataset.

Usage:
    python convert_parcel2d_to_yolo26_pose.py --source SOURCE_DIR --output OUTPUT_DIR [--split-only]

    --split-only    Only build the scene/model graph and compute the split;
                     skip image copying and label writing. Useful to validate
                     the split before RGB images have been staged.
"""
import argparse
import json
import math
import os
import random
import re
import shutil
import sys
from collections import defaultdict

KEYPOINT_NAMES = [
    "front_intersect3_inside",
    "front_intersect2",
    "front_intersect3_left",
    "front_intersect3_right",
    "back_intersect3_outside",
    "back_hidden",
    "back_intersect2_left",
    "back_intersect2_right",
]

NUM_KEYPOINTS = 8
SEED = 42
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

MODEL_ID_RE = re.compile(r"model_(\d+)")


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Step 1: Scan merge_dataset for scene UUID directories
# ---------------------------------------------------------------------------
def scan_merge_dataset(source_dir):
    scenes = []
    for entry in sorted(os.listdir(source_dir)):
        scene_dir = os.path.join(source_dir, entry)
        if not os.path.isdir(scene_dir):
            continue
        # UUID validation (basic format check)
        if len(entry) != 36 or entry.count("-") != 4:
            log(f"WARNING: {entry} does not look like a UUID, skipping")
            continue
        ann_path = os.path.join(scene_dir, "annotations.json")
        if not os.path.isfile(ann_path):
            log(f"WARNING: {entry} has no annotations.json, skipping")
            continue
        scenes.append(entry)
    return scenes


# ---------------------------------------------------------------------------
# Step 2: Parse annotation for a scene
# ---------------------------------------------------------------------------
def extract_model_id(model_field):
    """Extract numeric model ID from a model field like
    'validation/<uuid>/model_109.obj' or 'test/<uuid>/model_124.obj'.
    Model IDs are GLOBAL/flat numeric IDs shared across the old
    validation/test source split, which is exactly why we cannot reuse
    that split and must build a model-group graph instead.
    """
    m = MODEL_ID_RE.search(model_field)
    if not m:
        return None
    return f"model_{m.group(1)}"


def parse_scene_annotation(scene_dir, uuid):
    ann_path = os.path.join(scene_dir, "annotations.json")
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Image dims from the "images" list (COCO-style)
    images = data.get("images", [])
    if not images:
        raise ValueError(f"{uuid}: no 'images' entry in annotations.json")
    img_info = images[0]
    img_w = img_info["width"]
    img_h = img_info["height"]

    annotations = data.get("annotations", data.get("annotation"))
    if annotations is None:
        # Some schemas may store a single instance directly at top level
        if "bbox" in data:
            annotations = [data]
        else:
            raise ValueError(f"{uuid}: no 'annotations' field found")
    if isinstance(annotations, dict):
        annotations = [annotations]

    instances = []
    for inst in annotations:
        category_id = inst.get("category_id", 0)
        bbox = inst["bbox"]  # COCO xywh
        # Prefer 'keypoints', fall back to 'keypoints2d_left' (identical per spec)
        kps_raw = inst.get("keypoints")
        if kps_raw is None:
            kps_raw = inst.get("keypoints2d_left")
        if kps_raw is None:
            raise ValueError(f"{uuid}: no 'keypoints'/'keypoints2d_left' field")
        if len(kps_raw) != NUM_KEYPOINTS * 2:
            raise ValueError(
                f"{uuid}: expected {NUM_KEYPOINTS * 2} keypoint values, got {len(kps_raw)}"
            )
        model_field = inst.get("model", "")
        model_id = extract_model_id(model_field)
        instances.append(
            {
                "category_id": category_id,
                "bbox": bbox,
                "keypoints": kps_raw,
                "model_field": model_field,
                "model_id": model_id,
            }
        )

    return {
        "uuid": uuid,
        "img_w": img_w,
        "img_h": img_h,
        "instances": instances,
    }


# ---------------------------------------------------------------------------
# Step 3: Scene-level random split
# ---------------------------------------------------------------------------
# NOTE: an earlier version of this script used a MODEL-GROUP split (connected
# components over shared 3D model IDs extracted from annotation "model"
# fields) to try to prevent the same physical box from appearing in more
# than one split. That approach was abandoned: physical-box identity cannot
# be reliably determined across scenes from the annotation metadata alone,
# because different model filenames may correspond to the same physical box
# geometry (and vice versa). Since the model-ID grouping was not trustworthy,
# it has been removed along with the model-overlap validation it enabled.
# The split unit is now simply the scene (UUID): a plain seeded random split,
# with all instances of a scene kept together by construction and zero UUID
# overlap across splits by construction.
def assign_splits_scene_level(scene_data, train_n, val_n, test_n, seed=SEED):
    """Simple scene-level random split: shuffle the sorted UUID list with a
    seeded RNG, then slice into train/val/test by count. Returns
    (assignment, counts) with the same shape as before."""
    all_uuids = sorted(scene_data.keys())
    total = len(all_uuids)
    if train_n + val_n + test_n != total:
        raise ValueError(
            f"train+val+test ({train_n + val_n + test_n}) != total scenes ({total})"
        )

    rng = random.Random(seed)
    shuffled = all_uuids[:]
    rng.shuffle(shuffled)

    assignment = {
        "train": shuffled[0:train_n],
        "val": shuffled[train_n:train_n + val_n],
        "test": shuffled[train_n + val_n:train_n + val_n + test_n],
    }
    counts = {k: len(v) for k, v in assignment.items()}
    return assignment, counts


# ---------------------------------------------------------------------------
# Step 5: bbox normalization
# ---------------------------------------------------------------------------
def normalize_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    bw = w / img_w
    bh = h / img_h
    return cx, cy, bw, bh


def normalize_keypoints(kps_raw, img_w, img_h):
    """kps_raw is a flat list of 16 values [x0,y0,x1,y1,...,x7,y7].
    Returns list of (x_norm, y_norm, v) with v always 2."""
    out = []
    for i in range(NUM_KEYPOINTS):
        x = kps_raw[2 * i]
        y = kps_raw[2 * i + 1]
        out.append((x / img_w, y / img_h, 2))
    return out


def build_yolo_label_row(bbox, img_w, img_h, kps_raw):
    cx, cy, bw, bh = normalize_bbox(bbox, img_w, img_h)
    kps_norm = normalize_keypoints(kps_raw, img_w, img_h)
    values = [0, cx, cy, bw, bh]  # class_id=0 always
    for (kx, ky, kv) in kps_norm:
        values.extend([kx, ky, kv])
    assert len(values) == 5 + NUM_KEYPOINTS * 3 == 29
    return values


def format_label_row(values):
    parts = [str(values[0])]
    for v in values[1:]:
        if isinstance(v, int):
            parts.append(str(v))
        else:
            parts.append(f"{v:.6f}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Path to parcel2d_real/merge_dataset")
    ap.add_argument("--output", required=True, help="Path to output parcel2d_yolo directory")
    ap.add_argument("--split-only", action="store_true",
                     help="Only compute split_manifest.json + source_manifest.json; skip image copy/label writing")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    source_dir = args.source
    output_dir = args.output

    log(f"Scanning {source_dir} ...")
    uuids = scan_merge_dataset(source_dir)
    log(f"Found {len(uuids)} scene UUID directories")

    scene_data = {}
    parse_errors = []
    for uuid in uuids:
        scene_dir = os.path.join(source_dir, uuid)
        try:
            scene_data[uuid] = parse_scene_annotation(scene_dir, uuid)
        except Exception as e:
            parse_errors.append((uuid, str(e)))

    if parse_errors:
        log(f"ERROR: {len(parse_errors)} scenes failed to parse:")
        for uuid, err in parse_errors[:20]:
            log(f"  {uuid}: {err}")
        if len(parse_errors) > 20:
            log(f"  ... and {len(parse_errors) - 20} more")
        sys.exit(1)

    total_scenes = len(scene_data)
    total_instances = sum(len(sd["instances"]) for sd in scene_data.values())
    log(f"Parsed {total_scenes} scenes, {total_instances} instances")

    # Collect model_ids per scene purely for informational manifest metadata
    # (NOT used for split assignment - see assign_splits_scene_level docstring).
    scene_models = {}
    for uuid, sd in scene_data.items():
        scene_models[uuid] = {inst["model_id"] for inst in sd["instances"] if inst["model_id"]}

    # Scene-level random split
    train_n = round(SPLIT_RATIOS["train"] * total_scenes)
    val_n = round(SPLIT_RATIOS["val"] * total_scenes)
    test_n = total_scenes - train_n - val_n
    assignment, counts = assign_splits_scene_level(
        scene_data, train_n, val_n, test_n, seed=args.seed
    )
    log(f"Split scene counts: train={counts['train']} val={counts['val']} test={counts['test']}")

    split_of_uuid = {}
    for split_name, uuid_list in assignment.items():
        for u in uuid_list:
            split_of_uuid[u] = split_name

    # --- Validate zero UUID overlap (structurally guaranteed, but double check) ---
    all_assigned = [u for lst in assignment.values() for u in lst]
    if len(all_assigned) != len(set(all_assigned)):
        log("FATAL: duplicate UUID assignment across splits!")
        sys.exit(1)
    if set(all_assigned) != set(scene_data.keys()):
        log("FATAL: UUID assignment does not cover exactly all scanned scenes!")
        sys.exit(1)
    log("Verified: zero UUID overlap across splits, full coverage of all scanned scenes")

    # --- Per-split stats ---
    split_stats = {}
    for split_name, uuid_list in assignment.items():
        n_scenes = len(uuid_list)
        n_instances = sum(len(scene_data[u]["instances"]) for u in uuid_list)
        split_stats[split_name] = {
            "scenes": n_scenes,
            "instances": n_instances,
        }
        log(f"  {split_name}: scenes={n_scenes} instances={n_instances}")

    # --- Write split_manifest.json ---
    os.makedirs(output_dir, exist_ok=True)
    split_manifest = {
        "seed": args.seed,
        "split_method": "scene_level_random",
        "split_method_note": (
            "Simple scene-level random split (random.Random(seed).shuffle over "
            "sorted UUIDs, then sliced train/val/test). An earlier model-group "
            "split (connected components over shared 3D model IDs) was "
            "abandoned because physical-box identity could not be reliably "
            "determined across scenes from the annotation metadata (different "
            "model filenames may refer to the same physical box geometry), so "
            "model-based leakage avoidance was not trustworthy. All instances "
            "of a scene stay together because the scene (UUID) is the split "
            "unit; there is no UUID overlap across splits by construction."
        ),
        "target_counts": {"train": train_n, "val": val_n, "test": test_n},
        "total_scenes": total_scenes,
        "total_instances": total_instances,
        "split_stats": split_stats,
        "assignment": {k: sorted(v) for k, v in assignment.items()},
    }
    split_manifest_path = os.path.join(output_dir, "split_manifest.json")
    with open(split_manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)
    log(f"Wrote {split_manifest_path}")

    # --- Write source_manifest.json (provenance) ---
    source_manifest = {}
    for uuid, sd in scene_data.items():
        source_manifest[uuid] = {
            "split": split_of_uuid[uuid],
            "img_w": sd["img_w"],
            "img_h": sd["img_h"],
            "num_instances": len(sd["instances"]),
            "model_ids": sorted(scene_models[uuid]),
            "model_fields": [inst["model_field"] for inst in sd["instances"]],
        }
    source_manifest_path = os.path.join(output_dir, "source_manifest.json")
    with open(source_manifest_path, "w", encoding="utf-8") as f:
        json.dump(source_manifest, f, indent=2)
    log(f"Wrote {source_manifest_path}")

    if args.split_only:
        log("--split-only set: skipping image copy and label writing.")
        return

    # --- Create output dirs ---
    for split_name in ("train", "val", "test"):
        os.makedirs(os.path.join(output_dir, "images", split_name), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split_name), exist_ok=True)

    # --- Copy images + write labels ---
    copy_errors = []
    for uuid, sd in scene_data.items():
        split_name = split_of_uuid[uuid]
        src_img = os.path.join(source_dir, uuid, "00_rgb.png")
        dst_img = os.path.join(output_dir, "images", split_name, f"{uuid}.png")
        if not os.path.isfile(src_img):
            copy_errors.append(uuid)
            continue
        shutil.copyfile(src_img, dst_img)

        label_lines = []
        for inst in sd["instances"]:
            values = build_yolo_label_row(inst["bbox"], sd["img_w"], sd["img_h"], inst["keypoints"])
            label_lines.append(format_label_row(values))
        dst_label = os.path.join(output_dir, "labels", split_name, f"{uuid}.txt")
        with open(dst_label, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines) + "\n")

    if copy_errors:
        log(f"WARNING: {len(copy_errors)} scenes missing 00_rgb.png, skipped: {copy_errors[:10]}"
            f"{'...' if len(copy_errors) > 10 else ''}")

    # --- Write data.yaml ---
    data_yaml_path = os.path.join(output_dir, "data.yaml")
    yaml_content = f"""# Parcel2D YOLO26X-Pose dataset
# Keypoint order (fixed semantic identity - DO NOT REORDER):
#   0: front_intersect3_inside
#   1: front_intersect2
#   2: front_intersect3_left
#   3: front_intersect3_right
#   4: back_intersect3_outside
#   5: back_hidden       (visibility is still v=2, NOT v=1, per dataset_build_report.md)
#   6: back_intersect2_left
#   7: back_intersect2_right
# Split: scene-level random split (seeded shuffle over all scene UUIDs,
# then sliced train/val/test). See split_manifest.json for full provenance.

path: {output_dir}
train: images/train
val: images/val
test: images/test

names:
  0: parcel

kpt_shape: [8, 3]
"""
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    log(f"Wrote {data_yaml_path}")

    log("Conversion complete.")


if __name__ == "__main__":
    main()
