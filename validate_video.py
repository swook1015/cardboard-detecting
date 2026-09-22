"""
Frame-by-frame YOLO Pose inference on test.mp4 using best.pt.

Draws bbox + 8 keypoints (kp0-kp7) on every frame, logs per-frame
predictions to CSV, and computes frame-to-frame keypoint displacement
to flag temporal instability (sudden jumps / possible vertex-identity
flips) for downstream stereo-triangulation sanity checking.

The model is NOT a tracker: every frame is inferred independently.
Displacement is a post-hoc stability signal, not an accuracy metric.

Does not touch any dataset/label files, does not compute mAP, does not
do GT matching, does not train or modify best.pt.

Usage:
    python validate_video.py
    python validate_video.py --weights best.pt --video test.mp4 --conf 0.25 --device cpu
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "best.pt"
DEFAULT_VIDEO = PROJECT_ROOT / "test.mp4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "validation_results"

NUM_KEYPOINTS = 8

KP_PALETTE = [
    (255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 200, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
]
BOX_COLOR = (0, 255, 0)
MISSING_COLOR = (0, 0, 255)
MULTI_COLOR = (0, 165, 255)

DISPLACEMENT_STD_MULT = 3.0
CONF_DROP_THRESHOLD = 0.3


def load_model(weights_path: Path) -> YOLO:
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    print("=" * 70)
    print("MODEL CHECK")
    print("=" * 70)
    print(f"Weights : {weights_path}")

    model = YOLO(str(weights_path), task="pose")

    print(f"Task    : {model.task}")

    if model.task != "pose":
        raise RuntimeError(f"Expected a pose model, got task={model.task}")

    print("=" * 70)
    return model


def select_primary_detection(result):
    """Return (det_dict_or_None, num_detections) for one frame's result.

    det_dict picks the highest-confidence box; kpts is an (8, 3) array
    of (x, y, conf) in pixel coordinates.
    """
    boxes = result.boxes
    kpts = result.keypoints

    if boxes is None or len(boxes) == 0 or kpts is None:
        return None, 0

    confs = boxes.conf.cpu().numpy()
    idx = int(np.argmax(confs))

    xyxy = boxes.xyxy.cpu().numpy()[idx]
    box_conf = float(confs[idx])
    kp_data = kpts.data.cpu().numpy()[idx]  # (8, 3)

    return {"box_conf": box_conf, "xyxy": xyxy, "kpts": kp_data}, len(boxes)


def draw_overlay(frame, det, num_detections, note=None):
    if det is not None:
        x1, y1, x2, y2 = det["xyxy"].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(
            frame, f"conf={det['box_conf']:.2f}", (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, BOX_COLOR, 1, cv2.LINE_AA,
        )

        for k in range(NUM_KEYPOINTS):
            kx, ky = det["kpts"][k][:2]
            color = KP_PALETTE[k % len(KP_PALETTE)]
            cv2.circle(frame, (int(kx), int(ky)), 4, color, -1)
            cv2.putText(
                frame, str(k), (int(kx) + 6, int(ky) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

        if num_detections > 1:
            cv2.putText(
                frame, f"detections={num_detections} (showing top-1 by conf)",
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                MULTI_COLOR, 1, cv2.LINE_AA,
            )
    else:
        cv2.putText(
            frame, "NO DETECTION", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, MISSING_COLOR, 2, cv2.LINE_AA,
        )

    if note:
        cv2.putText(
            frame, note, (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, MISSING_COLOR, 1, cv2.LINE_AA,
        )

    return frame


def run(args):
    weights_path = Path(args.weights)
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    suspicious_dir = output_dir / "suspicious_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    suspicious_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(weights_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video   : {video_path}")
    print(f"FPS     : {fps:.3f}  Size: {width}x{height}  Frames(hdr): {total_frames}")
    print("=" * 70)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_path = output_dir / "test_result.mp4"
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    csv_path = output_dir / "predictions.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    fieldnames = ["frame", "timestamp_ms", "box_conf", "x1", "y1", "x2", "y2"]
    for k in range(NUM_KEYPOINTS):
        fieldnames += [f"kp{k}_x", f"kp{k}_y", f"kp{k}_conf"]
    fieldnames.append("num_detections")
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    prev_kpts = None
    prev_conf = None
    all_dets = []       # index-aligned with frame_idx: det dict or None
    frame_records = []  # per-frame: has_detection, box_conf, displacement(8,), conf_drop

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.predict(
            frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False,
        )[0]

        det, num_det = select_primary_detection(result)
        all_dets.append(det)

        timestamp_ms = (frame_idx / fps) * 1000.0
        row = {"frame": frame_idx, "timestamp_ms": round(timestamp_ms, 2), "num_detections": num_det}

        if det is None:
            row.update({"box_conf": "", "x1": "", "y1": "", "x2": "", "y2": ""})
            for k in range(NUM_KEYPOINTS):
                row[f"kp{k}_x"] = ""
                row[f"kp{k}_y"] = ""
                row[f"kp{k}_conf"] = ""
            displacement = np.full(NUM_KEYPOINTS, np.nan)
            cur_kpts = None
            conf_drop = None
        else:
            x1, y1, x2, y2 = det["xyxy"]
            row["box_conf"] = round(det["box_conf"], 4)
            row["x1"], row["y1"], row["x2"], row["y2"] = round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)
            for k in range(NUM_KEYPOINTS):
                kx, ky, kc = det["kpts"][k]
                row[f"kp{k}_x"] = round(float(kx), 2)
                row[f"kp{k}_y"] = round(float(ky), 2)
                row[f"kp{k}_conf"] = round(float(kc), 4)

            cur_kpts = det["kpts"][:, :2]
            if prev_kpts is not None:
                displacement = np.linalg.norm(cur_kpts - prev_kpts, axis=1)
            else:
                displacement = np.full(NUM_KEYPOINTS, np.nan)

            conf_drop = (prev_conf - det["box_conf"]) if prev_conf is not None else None

        csv_writer.writerow(row)

        frame_records.append({
            "frame": frame_idx,
            "has_detection": det is not None,
            "box_conf": det["box_conf"] if det is not None else None,
            "displacement": displacement,
            "conf_drop": conf_drop,
        })

        annotated = draw_overlay(frame.copy(), det, num_det)
        writer.write(annotated)

        prev_kpts = cur_kpts if det is not None else None
        prev_conf = det["box_conf"] if det is not None else None

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"[{frame_idx}] frames processed")

    cap.release()
    writer.release()
    csv_file.close()

    num_frames = len(frame_records)
    print(f"Done: {num_frames} frames processed -> {out_video_path.name}, {csv_path.name}")

    # ---- summary.json ----
    disp_matrix = np.array([r["displacement"] for r in frame_records], dtype=float)  # (N, 8)

    summary = {
        "video": str(video_path),
        "weights": str(weights_path),
        "num_frames": num_frames,
        "num_detected_frames": sum(1 for r in frame_records if r["has_detection"]),
        "num_missing_frames": sum(1 for r in frame_records if not r["has_detection"]),
        "per_keypoint": {},
    }

    def stats(values):
        if len(values) == 0:
            return {"mean": None, "median": None, "max": None, "p95": None, "n": 0}
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
            "p95": float(np.percentile(values, 95)),
            "n": int(len(values)),
        }

    for k in range(NUM_KEYPOINTS):
        col = disp_matrix[:, k]
        summary["per_keypoint"][f"kp{k}"] = stats(col[~np.isnan(col)])

    all_valid = disp_matrix[~np.isnan(disp_matrix)]
    summary["overall_displacement"] = stats(all_valid)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- console report ----
    print("=" * 70)
    print("DISPLACEMENT SUMMARY (px, frame-to-frame)")
    print("=" * 70)
    print(f"{'kp':<6}{'mean':>10}{'median':>10}{'max':>10}{'p95':>10}{'n':>8}")
    for k in range(NUM_KEYPOINTS):
        s = summary["per_keypoint"][f"kp{k}"]
        if s["n"] == 0:
            print(f"kp{k:<4}{'--':>10}{'--':>10}{'--':>10}{'--':>10}{0:>8}")
        else:
            print(f"kp{k:<4}{s['mean']:>10.2f}{s['median']:>10.2f}{s['max']:>10.2f}{s['p95']:>10.2f}{s['n']:>8}")
    o = summary["overall_displacement"]
    if o["n"] > 0:
        print(f"{'all':<6}{o['mean']:>10.2f}{o['median']:>10.2f}{o['max']:>10.2f}{o['p95']:>10.2f}{o['n']:>8}")
    print(f"Detected frames: {summary['num_detected_frames']}/{num_frames}  "
          f"Missing: {summary['num_missing_frames']}/{num_frames}")

    # ---- suspicious frame selection ----
    overall_mean = o["mean"] or 0.0
    overall_std = float(np.std(all_valid)) if len(all_valid) else 0.0
    disp_threshold = overall_mean + DISPLACEMENT_STD_MULT * overall_std

    candidates = []
    for r in frame_records:
        reasons = []
        severity = 0.0

        if not r["has_detection"]:
            reasons.append("detection_failure")
            severity = max(severity, 1e9)
        else:
            disp = r["displacement"]
            if not np.all(np.isnan(disp)):
                max_disp = float(np.nanmax(disp))
                max_kp = int(np.nanargmax(disp))
                if disp_threshold > 0 and max_disp > disp_threshold:
                    reasons.append(f"large_displacement(kp{max_kp}={max_disp:.1f}px)")
                    severity = max(severity, max_disp)

            if r["conf_drop"] is not None and r["conf_drop"] > CONF_DROP_THRESHOLD:
                reasons.append(f"conf_drop({r['conf_drop']:.2f})")
                severity = max(severity, r["conf_drop"] * 100)

        if reasons:
            candidates.append({"frame": r["frame"], "reasons": reasons, "severity": severity})

    candidates.sort(key=lambda c: c["severity"], reverse=True)
    selected = candidates[: args.max_suspicious]
    selected_frames = {c["frame"]: c["reasons"] for c in selected}

    print("=" * 70)
    print(f"Suspicious frames flagged: {len(candidates)} "
          f"(saving top {len(selected)} by severity, cap={args.max_suspicious})")
    print(f"Displacement threshold (mean + {DISPLACEMENT_STD_MULT}*std): {disp_threshold:.2f}px")
    print("=" * 70)

    if selected_frames:
        cap2 = cv2.VideoCapture(str(video_path))
        idx2 = 0
        saved = 0
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            if idx2 in selected_frames:
                det = all_dets[idx2]
                num_det = 1 if det is not None else 0
                note = " | ".join(selected_frames[idx2])
                annotated = draw_overlay(frame.copy(), det, num_det, note=note)
                out_path = suspicious_dir / f"frame_{idx2:06d}.jpg"
                cv2.imwrite(str(out_path), annotated)
                saved += 1
            idx2 += 1
        cap2.release()
        print(f"Saved {saved} suspicious frame images -> {suspicious_dir}")

    print("=" * 70)
    print("Output:")
    print(f"  {out_video_path}")
    print(f"  {csv_path}")
    print(f"  {output_dir / 'summary.json'}")
    print(f"  {suspicious_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run best.pt YOLO-Pose inference over test.mp4.")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--video", type=str, default=str(DEFAULT_VIDEO))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu", help="'cpu', '0', etc.")
    parser.add_argument("--max-suspicious", type=int, default=150,
                         help="Max number of suspicious frame images to save.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
