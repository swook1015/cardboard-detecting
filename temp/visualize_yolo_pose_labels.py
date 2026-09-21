#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    'bbox': (0, 255, 255),
    'visible': (0, 255, 0),
    'occluded': (0, 165, 255),
    'hidden': (128, 128, 128),
    'text': (255, 255, 255),
    'skeleton': (255, 0, 255),
}

SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

NAME_MAP = {idx: f'kp{idx}' for idx in range(8)}


def read_label_rows(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        vals = [float(v) for v in line.split()]
        if len(vals) != 5 + 8 * 3:
            continue
        rows.append(vals)
    return rows


def rows_to_objects(rows, image_w, image_h):
    objs = []
    for row in rows:
        cls = int(row[0])
        cx, cy, bw, bh = row[1:5]
        x0 = (cx - bw / 2.0) * image_w
        y0 = (cy - bh / 2.0) * image_h
        x1 = (cx + bw / 2.0) * image_w
        y1 = (cy + bh / 2.0) * image_h
        kps = []
        for idx in range(8):
            x = row[5 + idx * 3]
            y = row[6 + idx * 3]
            v = int(row[7 + idx * 3])
            px = x * image_w
            py = y * image_h
            kps.append((px, py, v, idx))
        objs.append({
            'class_id': cls,
            'bbox': (x0, y0, x1, y1),
            'keypoints': kps,
        })
    return objs


def draw_pose_image(img_path: Path, label_path: Path, out_path: Path):
    try:
        with Image.open(img_path) as img:
            image = np.array(img.convert('RGB'))
    except Exception as e:
        raise FileNotFoundError(f'Could not read image: {img_path} ({e})')

    h, w = image.shape[:2]
    objects = rows_to_objects(read_label_rows(label_path), w, h)
    rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for obj in objects:
        x0, y0, x1, y1 = [int(v) for v in obj['bbox']]
        cv2.rectangle(rgb, (x0, y0), (x1, y1), COLORS['bbox'], 2)
        for px, py, vis, idx in obj['keypoints']:
            color = COLORS['visible'] if vis == 2 else COLORS['occluded'] if vis == 1 else COLORS['hidden']
            cv2.circle(rgb, (int(px), int(py)), 5, color, -1)
            cv2.putText(rgb, str(idx), (int(px) + 8, int(py) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1, cv2.LINE_AA)
            cv2.putText(rgb, NAME_MAP.get(idx, f'kp{idx}'), (int(px) + 8, int(py) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1, cv2.LINE_AA)
        for a, b in SKELETON:
            pa = next((item for item in obj['keypoints'] if item[3] == a), None)
            pb = next((item for item in obj['keypoints'] if item[3] == b), None)
            if pa is None or pb is None:
                continue
            x1p, y1p, v1, _ = pa
            x2p, y2p, v2, _ = pb
            if v1 == 0 and v2 == 0:
                continue
            cv2.line(rgb, (int(x1p), int(y1p)), (int(x2p), int(y2p)), COLORS['skeleton'], 1)
    cv2.imwrite(str(out_path), rgb)


def main():
    parser = argparse.ArgumentParser(description='Visualize YOLO pose labels for a random subset of images.')
    parser.add_argument('--dataset-root', type=Path, default=Path('parcel2d_yolo'))
    parser.add_argument('--output-dir', type=Path, default=Path('inspection_samples'))
    parser.add_argument('--samples-per-split', type=int, default=5)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    for split in ['train', 'val', 'test']:
        img_dir = args.dataset_root / 'images' / split
        label_dir = args.dataset_root / 'labels' / split
        if not img_dir.exists():
            continue
        images = []
        for p in img_dir.rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                images.append(p)
        sample_images = random.sample(images, min(args.samples_per_split, len(images))) if len(images) > 0 else []
        for img in sample_images:
            rel_path = img.relative_to(img_dir)
            label_path = label_dir / rel_path.with_suffix('.txt')
            out_path = output_dir / f'{split}_{rel_path.parent.name}_{img.stem}_pose.jpg'
            draw_pose_image(img, label_path, out_path)
    print(f'Visual inspection images saved to {output_dir}')


if __name__ == '__main__':
    main()
