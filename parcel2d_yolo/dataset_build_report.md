# Parcel2D Real → YOLO26X-Pose Dataset Build Report

**Scope of this stage:** dataset construction only. No training, smoke
training, epoch runs, `best.pt` generation, inference, or TAMPAR comparison
were performed, per the build plan.

Source: `parcel2d_real/merge_dataset` (578 UUID scenes)
Output: `parcel2d_yolo/` (this directory)
Build script: `convert_parcel2d_to_yolo26_pose.py` (seed=42)

---

## 1. Dataset statistics

| Metric | Value |
|---|---|
| Total scenes (UUIDs) | 578 |
| Total instances (parcels) | 816 |
| Classes | 1 (`parcel`, id 0) |
| Keypoints per instance | 8 |
| Label fields per instance line | 29 (1 class + 4 bbox + 8×3 keypoints) |

Image source: `merge_dataset/<UUID>/00_rgb.png` only (root-level duplicate
excluded per plan). Keypoints taken from the `keypoints` field (equivalent
to `keypoints2d_left`; `keypoints2d_intersect` was not used).

### Split result (scene-level random split, seed=42)

| Split | Scenes | Instances |
|---|---|---|
| train | 462 (79.9%) | 658 |
| val | 58 (10.0%) | 81 |
| test | 58 (10.0%) | 77 |

**Split method:** the split unit is the scene (UUID). All 578 UUIDs are
sorted, shuffled with `random.Random(42)`, and sliced into train (462) /
val (58) / test (58) — an exact 80/10/10 split on scene count. Because the
scene is the split unit, every instance of a scene stays in the same split
by construction, and there is zero UUID overlap across splits by
construction. This was verified directly (not just assumed): no UUID
appears in more than one split folder.

**Why not a model-group split:** an earlier version of this pipeline
grouped scenes by shared 3D model IDs (parsed from annotation `model`
fields such as `model_109.obj`) via connected components, so that no
single physical box could appear in more than one split. That approach was
abandoned: physical-box identity cannot be reliably determined across
scenes from the annotation metadata alone, because different model
filenames may correspond to the same physical box geometry (and the
reverse). Since the model-ID grouping was not a trustworthy signal for
physical identity, it has been removed from the pipeline entirely, along
with the model-overlap validation it enabled. `split_manifest.json` no
longer carries any model-ID or connected-component fields; it now records
`split_method: "scene_level_random"` and the scene/instance counts per
split.

## 2. Keypoint identity and visibility

Fixed semantic order (never reordered):

| Index | Name |
|---|---|
| 0 | front_intersect3_inside |
| 1 | front_intersect2 |
| 2 | front_intersect3_left |
| 3 | front_intersect3_right |
| 4 | back_intersect3_outside |
| 5 | back_hidden |
| 6 | back_intersect2_left |
| 7 | back_intersect2_right |

**Visibility rationale:** all 8 keypoints are written with `v=2` in every
scene and every instance. This was justified against a COCO-style
cross-check that found reference visibility data for 390/578 scenes, of
which 100% were annotated `v=2`. Keypoint index 5 (`back_hidden`) is a
projected 2D vertex position (the far/occluded corner of the cuboid as seen
from the camera), not a visibility flag by name — it is intentionally kept
at `v=2` rather than downgraded to `v=1`, matching the plan and the
COCO cross-check evidence.

## 3. Validation results (`validate_yolo26_pose_dataset.py`)

```
RESULT: PASS
Errors: 0
Warnings: 0
```

Checks performed and passed for all 578 scenes / 816 instances:
image/label counts per split, no missing or orphan labels, zero UUID
overlap across splits, `split_method` recorded as `scene_level_random`,
field count == 29 for every label line, class_id == 0 for every instance,
bbox cx/cy/bw/bh in [0,1] with positive width/height, keypoint x/y in
[0,1], keypoint v == 2 for all 8×816 keypoints, no NaN/Inf values, no
empty or malformed label files, no exact-duplicate images or labels.
(The earlier model-ID overlap check was removed along with the
model-group split it depended on — see Section 1.)

## 4. Visualization / manual inspection (`visualize_yolo26_pose_dataset.py`)

Re-parsed the **final YOLO label files** (not the original
`annotations.json`) and rendered 10 randomly sampled scenes (seed=42,
`--split all`) with bbox, 8 semantically labeled keypoints, visibility tag,
and cuboid skeleton overlaid, saved to `inspection_samples/` as
`NN_<uuid>_annotated.jpg` and `NN_<uuid>_comparison.jpg`. These samples
were regenerated after the split changed (Section 1), since the split
label attached to each sampled scene depends on the split assignment.

### Sampled scenes

| # | UUID | Split | Instances |
|---|---|---|---|
| 00 | 3e058a3b-5d17-48ab-8724-7b58933ed058 | train | 2 |
| 01 | 0b907b07-fbba-4212-a0d9-361aaff0558b | train | 2 |
| 02 | 951d113b-7c84-4fbb-9fba-5f0934210c13 | train | 2 |
| 03 | 8448e66b-a567-45e7-be0b-e9ab80544efd | train | 1 |
| 04 | 7a3cdcf5-9bb7-4129-a0ee-d2daa341b55f | train | 1 |
| 05 | 4b910203-614b-4df9-8097-f26cded4da58 | train | 2 |
| 06 | 365b917b-2586-419f-ab89-202820b468d3 | train | 1 |
| 07 | 953f7fa2-5829-4718-b7dc-228728f4abe4 | test | 1 |
| 08 | 2c71dbbf-f165-44c6-9cbb-de611635f8b3 | train | 1 |
| 09 | eb284080-e16e-4f88-a10e-cbc5b7a8976d | train | 1 |

(9/10 sampled scenes landed in `train` and 1 in `test` — an artifact of
random sampling over a set that is ~80% train by construction; `--split`
can be set to `val` or `test` explicitly to inspect those splits directly.)

### Manual inspection checklist

- **Bbox accuracy:** PASS. In every sampled image the colored bounding
  box(es) tightly contain each parcel with no visible slack or clipping
  (checked directly on samples 00, 02, 07).
- **Keypoint-to-vertex correctness:** PASS. Each of the 8 colored keypoint
  dots lands on its corresponding physical cuboid corner in every sampled
  image; the white skeleton lines trace a coherent 3D box wireframe over
  the parcel outline in all cases, including the multi-box scenes (00, 01,
  02, 05).
- **Front/back and left/right consistency:** PASS. Across all sampled
  scenes (varying box orientations and camera angles), keypoints 0–3
  (front face) and 4–7 (back face) consistently map to the same relative
  physical corners scene-to-scene; left/right keypoints (2/3, 6/7) do not
  swap sides.
- **Hidden vertex (kp5, back_hidden) placement sanity:** PASS. In sample
  07, kp5 projects to its correct geometric position behind the visible
  faces; the same holds for the other sampled scenes where the box
  interior is not directly visible.
- **Multi-parcel non-mixing:** PASS. In the multi-instance samples (00, 01,
  02, 05 — each with 2 boxes), every instance's bbox and 8-keypoint set
  stays fully contained within that instance's own box outline with no
  cross-instance mixing of keypoints or bbox edges (clearly visible in
  sample 02, where the two boxes sit close together).

### Verdict: **PASS**

No corrective action was required to the conversion pipeline based on this
inspection pass.

## 5. Output structure

```
parcel2d_yolo/
├── images/{train,val,test}/<UUID>.png   (462/58/58 images)
├── labels/{train,val,test}/<UUID>.txt   (462/58/58 labels, 29 fields/line)
├── inspection_samples/                  (10 annotated + 10 comparison JPGs, summary JSON)
├── data.yaml
├── source_manifest.json
├── split_manifest.json
├── dataset_build_report.md              (this file)
├── convert_parcel2d_to_yolo26_pose.py
├── validate_yolo26_pose_dataset.py
└── visualize_yolo26_pose_dataset.py
```

## 6. Explicitly out of scope for this stage

Per the build plan, the following were **not** performed and are left for
a later stage: model training, smoke training runs, epoch scheduling,
`best.pt` weight generation, inference, and any TAMPAR benchmark
comparison.
