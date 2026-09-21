import argparse
from pathlib import Path

from ultralytics import YOLO


def train_model(
    dataset_yaml: str,
    weights: str = "yolo26x-pose.pt",
    epochs: int = 200,
    imgsz: int = 640,
    batch: int = 2,
    device: str = "0",
    workers: int = 0,
    seed: int = 42,
    patience: int = 30,
):
    weights_path = Path(weights).resolve()

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights_path}"
        )

    if "yolo26" not in weights_path.name.lower():
        raise ValueError(
            f"Only YOLO26 weights are allowed: {weights_path.name}"
        )

    print("=" * 70)
    print("MODEL CHECK")
    print("=" * 70)
    print(f"Checkpoint : {weights_path}")
    print(f"Dataset    : {Path(dataset_yaml).resolve()}")
    print("=" * 70)

    model = YOLO(str(weights_path), task="pose")

    print(f"Model task : {model.task}")

    if model.task != "pose":
        raise RuntimeError(
            f"Expected pose model, but loaded task = {model.task}"
        )

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        seed=seed,

        optimizer="AdamW",
        lr0=0.01,

        # Early stopping
        patience=patience,

        project="runs/pose",
        name="train_yolo26",

        fliplr=False,
        flipud=False,
        cache=False,
        verbose=True,
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLO26 Pose on Parcel2D Real data."
    )

    parser.add_argument(
        "--data",
        type=str,
        default="parcel2d_yolo/data.yaml",
        help="Path to data.yaml",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="yolo26x-pose.pt",
        help="YOLO26 Pose checkpoint",
    )

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)

    # GPU 0 사용
    parser.add_argument("--device", type=str, default="0")

    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)

    args = parser.parse_args()

    train_model(
        dataset_yaml=args.data,
        weights=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
    )