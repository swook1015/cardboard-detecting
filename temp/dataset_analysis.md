# Parcel2D Real dataset analysis

## 1) Actual dataset layout

The dataset is present in the workspace as a local archive and extracted folder:

- extracted root: `parcel2d_real/`
- annotation JSON files:
  - `parcel2d_real/parcel2d_real_validation.json`
  - `parcel2d_real/parcel2d_real_test.json`
  - `parcel2d_real/omni3d_json/parcel2d_real_validation.json`
  - `parcel2d_real/omni3d_json/parcel2d_real_test.json`
- image folders:
  - `parcel2d_real/validation/`
  - `parcel2d_real/test/`

This is a COCO-style annotation dataset, not a YOLO label set.

## 2) Structure and content

The JSON root contains:

- `images`: list of image records
- `annotations`: list of object annotations
- `categories`: list of object types

Observed categories:

- `id: 0` -> `normal box`
- `id: 1` -> `damaged box`

Example image record:

- `id`: image identifier
- `width`: 1280
- `height`: 960
- `file_name`: e.g. `validation/c1fc0fd0-d90e-4b3d-82e9-1fb3138b6e33/00_rgb.png`

Example annotation record contains:

- `id`, `image_id`, `category_id`, `category_name`
- `bbox`: `[x, y, width, height]`
- `keypoints`: `[x1, y1, v1, x2, y2, v2, ...]` with length `24`
- `num_keypoints`: `8`
- `iscrowd`: `0`
- `visibility`: `True` (a top-level image/object property, not keypoint visibility)
- `bbox_mode`: `1`

## 3) Bounding box format

The dataset uses standard COCO `xywh` boxes:

- `bbox = [x, y, width, height]`
- Example: `[461, 275, 222, 259]`

This was confirmed from the actual annotation JSON in the extracted file.

## 4) Keypoints and visibility

A real annotation instance contains:

```
"keypoints": [
  587.77, 436.96, 2.0,
  483.84, 493.98, 2.0,
  461.42, 397.47, 2.0,
  598.06, 533.08, 2.0,
  681.99, 304.57, 2.0,
  579.45, 370.44, 2.0,
  569.12, 275.01, 2.0,
  682.52, 400.53, 2.0
]
```

This means:

- 8 keypoints total
- each keypoint is stored as `(x, y, visibility)`
- each visibility value is a float: `2.0`, `1.0` or `0.0`
- the data does not include symbolic names like `left_front_top` etc.

The actual visibility convention is the standard COCO-style one:

- `2` = visible
- `1` = occluded / not fully visible
- `0` = not labeled / hidden / absent

The dataset stores `num_keypoints = 8` and each keypoint triplet is ordered in the annotation array; the dataset therefore has a source-defined ordering that is authoritative in the raw annotation and we must preserve it exactly.

## 5) Keypoint ordering

The raw keypoint array does not include human-readable names.

Because this dataset does not provide semantic names in the JSON, the only authoritative source ordering is the native array order from the dataset itself. We therefore maintain the exact source ordering as `kp0` ... `kp7` and do not invent left/right/front/back semantics from a guess.

This preserves object identity across augmentation and avoids label corruption.

## 6) Hidden / occluded / unlabeled handling

The raw annotation keypoints use `visibility` values consistent with COCO semantics:

- visible = `2`
- occluded = `1`
- not labeled = `0`

For Ultralytics pose labels, the same semantics are preserved as `x, y, v` where `v` is in `{0,1,2}`. For `v=0`, the keypoint is treated as absent and the safest conversion is to set `(x, y) = (0.0, 0.0)` while keeping `v=0`.

## 7) Multi-object images

Several images contain multiple parcel objects (for example, the extracted `validation` and `test` directories include multiple scene layouts and the JSON annotations include multiple object instances per image). Therefore the conversion must emit one YOLO pose row per detected object instance, and multiple rows are valid in a single label file.

## 8) Split status

The extracted dataset has:

- `validation/` directory
- `test/` directory
- no explicit `train/` split in the source archive

For conversion, the pipeline keeps the original `validation`/`test` split when present and creates a reproducible `train/val` split from the validation set when a training split is required.

## 9) Decision for this pipeline

The conversion pipeline will preserve the raw source order exactly and expose it as:

- `kp0` ... `kp7`

The script will not invent semantic names or arbitrary left/right/front/back assignments unless the source documentation is later found to define them explicitly.

## 10) Additional note about model naming

The installed local environment is using `ultralytics 8.4.155`. This version does not expose a `YOLO26X-Pose` model name. The actual official pose models in this environment are `yolov8...-pose` and `yolo11...-pose` variants. This is an authoritative environment fact and the training script must use the available official model names, not an assumed `YOLO26` name.
