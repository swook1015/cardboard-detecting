#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def split_image_ids(image_ids: List[int], train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    shuffled = image_ids[:]
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * train_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def parse_annotation_json(json_path: Path) -> Tuple[Dict[int, dict], Dict[int, dict]]:
    data = load_json(json_path)
    images_by_id = {img['id']: img for img in data.get('images', [])}
    ann_by_image = {}
    for ann in data.get('annotations', []):
        image_id = ann.get('image_id')
        ann_by_image.setdefault(image_id, []).append(ann)
    return images_by_id, ann_by_image


def normalise_xywh(bbox: List[float], image_w: int, image_h: int) -> Tuple[float, float, float, float]:
    if len(bbox) < 4:
        raise ValueError(f'Invalid bbox: {bbox}')
    x, y, w, h = [float(v) for v in bbox]
    if w <= 0 or h <= 0:
        raise ValueError(f'Invalid bbox size: {bbox} for image {image_w}x{image_h}')
    cx = (x + w / 2.0) / image_w
    cy = (y + h / 2.0) / image_h
    nw = w / image_w
    nh = h / image_h
    return cx, cy, nw, nh


def normalise_keypoint(kp: List[float], image_w: int, image_h: int) -> Tuple[float, float, int]:
    if len(kp) < 3:
        raise ValueError(f'Invalid keypoint triplet: {kp}')
    x, y, v = float(kp[0]), float(kp[1]), float(kp[2])
    if v == 0:
        return 0.0, 0.0, 0
    x_norm = x / image_w
    y_norm = y / image_h
    if math.isnan(x_norm) or math.isnan(y_norm) or math.isinf(x_norm) or math.isinf(y_norm):
        raise ValueError(f'NaN/Inf keypoint encountered: {kp}')
    return x_norm, y_norm, int(v)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def is_valid_label_tuple(values: List[float]) -> bool:
    return len(values) == 8 * 3 and all(math.isfinite(v) for v in values)


def write_yolo_label(label_path: Path, class_id: int, bbox: Tuple[float, float, float, float], keypoints: List[float]):
    if not is_valid_label_tuple(keypoints):
        raise ValueError(f'Keypoints invalid for label: {label_path}')
    if not (0.0 <= bbox[0] <= 1.0 and 0.0 <= bbox[1] <= 1.0 and 0.0 <= bbox[2] <= 1.0 and 0.0 <= bbox[3] <= 1.0):
        raise ValueError(f'BBox out of range: {bbox}')
    parts = [str(class_id), f'{bbox[0]:.6f}', f'{bbox[1]:.6f}', f'{bbox[2]:.6f}', f'{bbox[3]:.6f}']
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0 <= v <= 2):
            raise ValueError(f'Keypoint out of range: [{x}, {y}, {v}]')
        parts.extend([f'{x:.6f}', f'{y:.6f}', str(int(v))])
    label_path.write_text(' '.join(parts) + '\n', encoding='utf-8')


def convert_split(source_json: Path, output_root: Path, split_name: str, dataset_root: Path, class_id: int = 0, include_test: bool = False):
    images_by_id, ann_by_image = parse_annotation_json(source_json)
    image_dir = output_root / 'images' / split_name
    label_dir = output_root / 'labels' / split_name
    ensure_dir(image_dir)
    ensure_dir(label_dir)

    for image_id, image_meta in sorted(images_by_id.items()):
        file_name = image_meta.get('file_name')
        if not file_name:
            raise ValueError(f'Image {image_id} is missing file_name')
        src_image = dataset_root / file_name
        if not src_image.exists():
            raise ValueError(f'Missing source image: {src_image}')

        rel_parent = Path(file_name).parent
        dst_image_dir = image_dir / rel_parent
        ensure_dir(dst_image_dir)
        dst_image = dst_image_dir / src_image.name
        if not dst_image.exists():
            dst_image.write_bytes(src_image.read_bytes())

        objs = ann_by_image.get(image_id, [])
        label_path = label_dir / rel_parent / f'{src_image.stem}.txt'
        ensure_dir(label_path.parent)
        if len(objs) == 0:
            label_path.write_text('', encoding='utf-8')
            continue

        lines = []
        for ann in objs:
            image_w = int(image_meta.get('width', 0))
            image_h = int(image_meta.get('height', 0))
            if image_w <= 0 or image_h <= 0:
                raise ValueError(f'Image size invalid for image {image_id}: {image_w}x{image_h}')
            bbox = ann.get('bbox', [])
            if len(bbox) != 4:
                raise ValueError(f'Annotation {ann.get("id")} has invalid bbox: {bbox}')
            if bbox[2] <= 0 or bbox[3] <= 0:
                raise ValueError(f'Zero/negative bbox for annotation {ann.get("id")}: {bbox}')
            cx, cy, bw, bh = normalise_xywh(bbox, image_w, image_h)
            keypoints = []
            raw_keypoints = ann.get('keypoints', [])
            if len(raw_keypoints) != 24:
                raise ValueError(f'Annotation {ann.get("id")} keypoints length is {len(raw_keypoints)}, expected 24')
            for idx in range(0, len(raw_keypoints), 3):
                x_norm, y_norm, vis = normalise_keypoint(raw_keypoints[idx:idx + 3], image_w, image_h)
                if vis == 0:
                    x_norm = 0.0
                    y_norm = 0.0
                keypoints.extend([x_norm, y_norm, vis])
            lines.append([class_id, cx, cy, bw, bh, *keypoints])

        if lines:
            with open(label_path, 'w', encoding='utf-8') as f:
                for row in lines:
                    f.write(' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in row) + '\n')



def build_dataset(root: Path, output_root: Path, train_ratio: float = 0.8, seed: int = 42):
    dataset_root = root
    validation_json = dataset_root / 'parcel2d_real_validation.json'
    test_json = dataset_root / 'parcel2d_real_test.json'
    if not validation_json.exists():
        raise FileNotFoundError(f'No validation JSON found at {validation_json}')

    images_by_id, _ = parse_annotation_json(validation_json)
    image_ids = sorted(images_by_id.keys())
    train_ids, val_ids = split_image_ids(image_ids, train_ratio=train_ratio, seed=seed)

    train_json = validation_json
    val_json = validation_json
    # Recreate train/val by filtering the validation JSON itself.
    train_images = [images_by_id[i] for i in train_ids]
    val_images = [images_by_id[i] for i in val_ids]
    train_ann_by_image = {}
    val_ann_by_image = {}
    data = load_json(validation_json)
    for ann in data.get('annotations', []):
        if ann.get('image_id') in train_ids:
            train_ann_by_image.setdefault(ann['image_id'], []).append(ann)
        elif ann.get('image_id') in val_ids:
            val_ann_by_image.setdefault(ann['image_id'], []).append(ann)

    output_root.mkdir(parents=True, exist_ok=True)
    ensure_dir(output_root / 'images' / 'train')
    ensure_dir(output_root / 'images' / 'val')
    ensure_dir(output_root / 'images' / 'test')
    ensure_dir(output_root / 'labels' / 'train')
    ensure_dir(output_root / 'labels' / 'val')
    ensure_dir(output_root / 'labels' / 'test')

    def write_split(split_name: str, images: List[dict], ann_by_image: Dict[int, List[dict]]):
        for img in images:
            src = dataset_root / img['file_name']
            rel_parent = Path(img['file_name']).parent
            dst_dir = output_root / 'images' / split_name / rel_parent
            ensure_dir(dst_dir)
            dst = dst_dir / src.name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
            label_dir = output_root / 'labels' / split_name / rel_parent
            ensure_dir(label_dir)
            label_path = label_dir / (src.stem + '.txt')
            anns = ann_by_image.get(img['id'], [])
            if not anns:
                label_path.write_text('', encoding='utf-8')
                continue
            rows = []
            for ann in anns:
                image_w = int(img.get('width', 0))
                image_h = int(img.get('height', 0))
                bbox = ann.get('bbox', [])
                if len(bbox) != 4:
                    raise ValueError(f'Annotation {ann.get("id")} invalid bbox {bbox}')
                cx, cy, bw, bh = normalise_xywh(bbox, image_w, image_h)
                keypoints = []
                kp = ann.get('keypoints', [])
                if len(kp) != 24:
                    raise ValueError(f'Annotation {ann.get("id")} invalid length {len(kp)}')
                for j in range(0, len(kp), 3):
                    x_norm, y_norm, vis = normalise_keypoint(kp[j:j + 3], image_w, image_h)
                    if vis == 0:
                        x_norm = 0.0
                        y_norm = 0.0
                    keypoints.extend([x_norm, y_norm, vis])
                rows.append([0, cx, cy, bw, bh, *keypoints])
            with open(label_path, 'w', encoding='utf-8') as f:
                for row in rows:
                    f.write(' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in row) + '\n')

    write_split('train', train_images, train_ann_by_image)
    write_split('val', val_images, val_ann_by_image)

    if test_json.exists():
        test_images_by_id, test_ann_by_image = parse_annotation_json(test_json)
        test_images = [test_images_by_id[i] for i in sorted(test_images_by_id.keys())]
        for img in test_images:
            src = dataset_root / img['file_name']
            rel_parent = Path(img['file_name']).parent
            dst_dir = output_root / 'images' / 'test' / rel_parent
            ensure_dir(dst_dir)
            dst = dst_dir / src.name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
            label_dir = output_root / 'labels' / 'test' / rel_parent
            ensure_dir(label_dir)
            label_path = label_dir / (src.stem + '.txt')
            anns = test_ann_by_image.get(img['id'], [])
            if not anns:
                label_path.write_text('', encoding='utf-8')
                continue
            rows = []
            for ann in anns:
                image_w = int(img.get('width', 0))
                image_h = int(img.get('height', 0))
                bbox = ann.get('bbox', [])
                cx, cy, bw, bh = normalise_xywh(bbox, image_w, image_h)
                kp = ann.get('keypoints', [])
                keypoints = []
                for j in range(0, len(kp), 3):
                    x_norm, y_norm, vis = normalise_keypoint(kp[j:j + 3], image_w, image_h)
                    if vis == 0:
                        x_norm = 0.0
                        y_norm = 0.0
                    keypoints.extend([x_norm, y_norm, vis])
                rows.append([0, cx, cy, bw, bh, *keypoints])
            with open(label_path, 'w', encoding='utf-8') as f:
                for row in rows:
                    f.write(' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in row) + '\n')

    data_yaml = output_root / 'data.yaml'
    data_yaml.write_text(
        'path: C:/Users/11sungwook/OneDrive - Customparts/바탕 화면/2026_Q3_box/parcel2d_yolo\n'
        'train: images/train\n'
        'val: images/val\n'
        'test: images/test\n\n'
        'names:\n'
        '  0: parcel\n\n'
        'kpt_shape: [8, 3]\n',
        encoding='utf-8'
    )

    print(f'Converted dataset created at {output_root}')
    return output_root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Parcel2D Real COCO-style annotations to YOLO pose labels.')
    parser.add_argument('--dataset-root', type=Path, default=Path('parcel2d_real'), help='Root directory containing the extracted Parcel2D Real dataset.')
    parser.add_argument('--output-root', type=Path, default=Path('parcel2d_yolo'), help='Destination directory for the YOLO pose dataset.')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train split ratio when creating a train/val split from the validation set.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible split.')
    args = parser.parse_args()
    build_dataset(args.dataset_root, args.output_root, train_ratio=args.train_ratio, seed=args.seed)
