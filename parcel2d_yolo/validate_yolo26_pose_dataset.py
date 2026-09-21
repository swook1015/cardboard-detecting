#!/usr/bin/env python3
"""
Parcel2D YOLO26X-Pose Dataset Validator

Validates a built parcel2d_yolo/ dataset:
  - image/label counts per split
  - missing/orphan labels
  - UUID overlap across splits (scene-level random split; the scene/UUID is
    the split unit, so no scene appears in more than one split)
  - field count per label line (must be 29)
  - class ID (must be 0)
  - bbox range [0,1] and positive width/height
  - 8 keypoints present with x/y in [0,1]
  - visibility == 2 for all keypoints
  - NaN/Inf checks
  - duplicate images/labels (by content hash)
  - empty/malformed labels

Usage:
    python validate_yolo26_pose_dataset.py --dataset /path/to/parcel2d_yolo
"""
import argparse
import hashlib
import json
import math
import os
import sys

NUM_KEYPOINTS = 8
EXPECTED_FIELDS = 5 + NUM_KEYPOINTS * 3  # 29


def log(msg):
    print(msg, flush=True)


def file_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to parcel2d_yolo directory")
    args = ap.parse_args()

    root = args.dataset
    errors = []
    warnings = []

    splits = ["train", "val", "test"]

    # --- 1. image/label counts ---
    counts = {}
    uuid_by_split = {}
    for split in splits:
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)
        if not os.path.isdir(img_dir):
            errors.append(f"Missing images dir: {img_dir}")
            continue
        if not os.path.isdir(lbl_dir):
            errors.append(f"Missing labels dir: {lbl_dir}")
            continue
        imgs = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
        lbls = sorted(f for f in os.listdir(lbl_dir) if f.endswith(".txt"))
        img_uuids = {os.path.splitext(f)[0] for f in imgs}
        lbl_uuids = {os.path.splitext(f)[0] for f in lbls}
        counts[split] = {"images": len(imgs), "labels": len(lbls)}
        uuid_by_split[split] = img_uuids

        missing_labels = img_uuids - lbl_uuids
        orphan_labels = lbl_uuids - img_uuids
        if missing_labels:
            errors.append(f"[{split}] {len(missing_labels)} images missing labels: {sorted(missing_labels)[:5]}")
        if orphan_labels:
            errors.append(f"[{split}] {len(orphan_labels)} orphan labels (no image): {sorted(orphan_labels)[:5]}")

        log(f"[{split}] images={len(imgs)} labels={len(lbls)}")

    # --- 2. UUID overlap across splits ---
    all_uuid_sets = list(uuid_by_split.values())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            if s1 not in uuid_by_split or s2 not in uuid_by_split:
                continue
            overlap = uuid_by_split[s1] & uuid_by_split[s2]
            if overlap:
                errors.append(f"UUID overlap between {s1} and {s2}: {len(overlap)} scenes: {sorted(overlap)[:5]}")
    if not errors:
        log("UUID overlap check: PASS (zero overlap across splits)")

    # --- 3. split_manifest.json sanity (scene-level random split; no model-group checks) ---
    split_manifest_path = os.path.join(root, "split_manifest.json")
    if os.path.isfile(split_manifest_path):
        with open(split_manifest_path) as f:
            split_manifest = json.load(f)
        split_method = split_manifest.get("split_method")
        if split_method != "scene_level_random":
            warnings.append(
                f"split_manifest.json split_method is {split_method!r}, "
                f"expected 'scene_level_random'"
            )
        else:
            log("Split method check: PASS (scene_level_random)")
    else:
        warnings.append(f"split_manifest.json not found at {split_manifest_path}")

    # --- 4. Per-label-file validation ---
    total_instances = 0
    empty_labels = []
    malformed_labels = []
    field_count_errors = []
    class_id_errors = []
    bbox_range_errors = []
    bbox_size_errors = []
    keypoint_range_errors = []
    visibility_errors = []
    nan_inf_errors = []

    for split in splits:
        lbl_dir = os.path.join(root, "labels", split)
        if not os.path.isdir(lbl_dir):
            continue
        for fname in sorted(os.listdir(lbl_dir)):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(lbl_dir, fname)
            with open(path) as f:
                content = f.read()
            lines = [l for l in content.splitlines() if l.strip()]
            if not lines:
                empty_labels.append(f"{split}/{fname}")
                continue
            for li, line in enumerate(lines):
                parts = line.split()
                if len(parts) != EXPECTED_FIELDS:
                    field_count_errors.append(f"{split}/{fname}:{li} has {len(parts)} fields, expected {EXPECTED_FIELDS}")
                    malformed_labels.append(f"{split}/{fname}:{li}")
                    continue
                try:
                    values = [float(p) for p in parts]
                except ValueError:
                    malformed_labels.append(f"{split}/{fname}:{li} (non-numeric field)")
                    continue

                total_instances += 1
                class_id = values[0]
                if class_id != 0:
                    class_id_errors.append(f"{split}/{fname}:{li} class_id={class_id}, expected 0")

                cx, cy, bw, bh = values[1:5]
                for name, v in [("cx", cx), ("cy", cy), ("bw", bw), ("bh", bh)]:
                    if math.isnan(v) or math.isinf(v):
                        nan_inf_errors.append(f"{split}/{fname}:{li} bbox {name}={v}")
                    elif not (0.0 <= v <= 1.0):
                        bbox_range_errors.append(f"{split}/{fname}:{li} bbox {name}={v} out of [0,1]")
                if bw <= 0 or bh <= 0:
                    bbox_size_errors.append(f"{split}/{fname}:{li} bbox has non-positive size (bw={bw}, bh={bh})")

                kp_values = values[5:]
                for k in range(NUM_KEYPOINTS):
                    kx, ky, kv = kp_values[3 * k], kp_values[3 * k + 1], kp_values[3 * k + 2]
                    for name, v in [("x", kx), ("y", ky)]:
                        if math.isnan(v) or math.isinf(v):
                            nan_inf_errors.append(f"{split}/{fname}:{li} kp{k}.{name}={v}")
                        elif not (0.0 <= v <= 1.0):
                            keypoint_range_errors.append(f"{split}/{fname}:{li} kp{k}.{name}={v} out of [0,1]")
                    if kv != 2:
                        visibility_errors.append(f"{split}/{fname}:{li} kp{k}.v={kv}, expected 2")

    log(f"Total instances validated: {total_instances}")

    for name, lst in [
        ("Empty label files", empty_labels),
        ("Malformed label lines", malformed_labels),
        ("Field count errors", field_count_errors),
        ("Class ID errors", class_id_errors),
        ("Bbox range errors", bbox_range_errors),
        ("Bbox non-positive size errors", bbox_size_errors),
        ("Keypoint range errors", keypoint_range_errors),
        ("Visibility errors (v != 2)", visibility_errors),
        ("NaN/Inf errors", nan_inf_errors),
    ]:
        if lst:
            errors.append(f"{name}: {len(lst)} occurrences, e.g. {lst[:5]}")
        else:
            log(f"{name}: PASS (0 occurrences)")

    # --- 5. Duplicate images/labels (by content hash) ---
    log("Checking for duplicate images (content hash)...")
    img_hashes = {}
    dup_images = []
    for split in splits:
        img_dir = os.path.join(root, "images", split)
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith(".png"):
                continue
            path = os.path.join(img_dir, fname)
            h = file_sha1(path)
            key = f"{split}/{fname}"
            if h in img_hashes:
                dup_images.append((key, img_hashes[h]))
            else:
                img_hashes[h] = key
    if dup_images:
        warnings.append(f"Duplicate images found: {len(dup_images)} pairs, e.g. {dup_images[:5]}")
    else:
        log("Duplicate image check: PASS (no exact-duplicate images)")

    log("Checking for duplicate label files (content hash)...")
    lbl_hashes = {}
    dup_labels = []
    for split in splits:
        lbl_dir = os.path.join(root, "labels", split)
        if not os.path.isdir(lbl_dir):
            continue
        for fname in sorted(os.listdir(lbl_dir)):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(lbl_dir, fname)
            h = file_sha1(path)
            key = f"{split}/{fname}"
            if h in lbl_hashes:
                dup_labels.append((key, lbl_hashes[h]))
            else:
                lbl_hashes[h] = key
    if dup_labels:
        warnings.append(f"Duplicate label content found: {len(dup_labels)} pairs (note: identical labels for different scenes may be coincidental, not necessarily an error), e.g. {dup_labels[:5]}")
    else:
        log("Duplicate label check: PASS (no exact-duplicate label files)")

    # --- Summary ---
    log("")
    log("=" * 70)
    log("VALIDATION SUMMARY")
    log("=" * 70)
    log(f"Splits: {counts}")
    log(f"Total label instances: {total_instances}")
    log(f"Errors: {len(errors)}")
    for e in errors:
        log(f"  ERROR: {e}")
    log(f"Warnings: {len(warnings)}")
    for w in warnings:
        log(f"  WARNING: {w}")

    if errors:
        log("")
        log("RESULT: FAIL")
        sys.exit(1)
    else:
        log("")
        log("RESULT: PASS")


if __name__ == "__main__":
    main()
