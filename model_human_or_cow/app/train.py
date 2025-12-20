from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 on cow/human dataset")

    p.add_argument("--data", type=str, default="data/dataset/data.yaml", help="Path to data.yaml")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Base model (pretrained) or custom .pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="cpu", help='cpu | "0" for GPU0 | "0,1" for multi-gpu')

    p.add_argument("--project", type=str, default="artifacts", help="Where to save runs")
    p.add_argument("--name", type=str, default="train", help="Run name (subfolder in project)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"data.yaml not found: {data_path}\n"
            f"Tip: run preprocess first (python -m app.preprocess) and ensure paths are correct."
        )

    project_dir = Path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] data:    {data_path.resolve()}")
    print(f"[INFO] model:   {args.model}")
    print(f"[INFO] epochs:  {args.epochs}")
    print(f"[INFO] imgsz:   {args.imgsz}")
    print(f"[INFO] batch:   {args.batch}")
    print(f"[INFO] device:  {args.device}")
    print(f"[INFO] output:  {(project_dir / args.name).resolve()}")

    model = YOLO(args.model)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_dir),
        name=args.name,
    )

    best = project_dir / args.name / "weights" / "best.pt"
    last = project_dir / args.name / "weights" / "last.pt"

    print(f"[DONE] best: {best}")
    print(f"[DONE] last: {last}")


if __name__ == "__main__":
    main()
