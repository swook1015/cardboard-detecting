#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from collections import Counter, defaultdict


def parse_label_file(path: Path):
    if not path.exists():
        return []
    lines = path.read_text(encoding='utf-8').strip().splitlines()
    rows = []
    for line in lines:
        if not line.strip():
            continue
        values = [float(v) for v in line.split()]
        if len(values) != 5 + 8 * 3:
            raise ValueError(f'Unexpected label length in {path}: {len(values)} values')
        rows.append(values)
    return rows


def validate_dataset(dataset_root: Path):
    images_root = dataset_root / 'images'
    labels_root = dataset_root / 'labels'
    total_images = 0
    total_instances = 0
    split_stats = defaultdict(lambda: {'images': 0, 'instances': 0, 'visible': 0, 'occluded': 0, 'hidden': 0})
    keypoint_counter = defaultdict(Counter)

    for split in ['train', 'val', 'test']:
        img_dir = images_root / split
        label_dir = labels_root / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file():
                continue
            total_images += 1
            split_stats[split]['images'] += 1
            label_path = label_dir / f'{img_path.stem}.txt'
            rows = parse_label_file(label_path)
            if not label_path.exists():
                raise FileNotFoundError(f'Missing label for {img_path}')
            if not rows:
                continue
            for row in rows:
                total_instances += 1
                split_stats[split]['instances'] += 1
                class_id = int(row[0])
                if class_id < 0:
                    raise ValueError(f'Negative class id in {label_path}: {class_id}')
                bbox = row[1:5]
                if not all(0.0 <= v <= 1.0 for v in bbox):
                    raise ValueError(f'BBox out of range in {label_path}: {bbox}')
                if bbox[2] <= 0.0 or bbox[3] <= 0.0:
                    raise ValueError(f'Zero/negative bbox size in {label_path}: {bbox}')
                for idx in range(0, 8 * 3, 3):
                    x, y, vis = row[5 + idx:5 + idx + 3]
                    if not all(math.isfinite(v) for v in (x, y, vis)):
                        raise ValueError(f'NaN/Inf encountered in {label_path}: {row}')
                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                        raise ValueError(f'Keypoint x/y out of range in {label_path}: {(x, y, vis)}')
                    if vis not in (0, 1, 2):
                        raise ValueError(f'Invalid visibility value in {label_path}: {vis}')
                    keypoint_counter[split][idx // 3] += 1 if vis == 2 else 0
                    if vis == 2:
                        split_stats[split]['visible'] += 1
                    elif vis == 1:
                        split_stats[split]['occluded'] += 1
                    elif vis == 0:
                        split_stats[split]['hidden'] += 1

    print('Dataset validation summary')
    print('========================')
    print(f'Total images: {total_images}')
    print(f'Total instances: {total_instances}')
    for split in ['train', 'val', 'test']:
        if split_stats[split]['images'] > 0:
            print(f'{split}: images={split_stats[split]["images"]}, instances={split_stats[split]["instances"]}, visible={split_stats[split]["visible"]}, occluded={split_stats[split]["occluded"]}, hidden={split_stats[split]["hidden"]}')

    print('\nKeypoint visibility summary by index:')
    for idx in range(8):
        counts = {split: keypoint_counter.get(split, {}).get(idx, 0) for split in ['train', 'val', 'test'] if split in keypoint_counter}
        print(f'kp{idx}: {counts}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate the generated YOLO pose dataset.')
    parser.add_argument('--dataset-root', type=Path, default=Path('parcel2d_yolo'))
    args = parser.parse_args()
    validate_dataset(args.dataset_root)
