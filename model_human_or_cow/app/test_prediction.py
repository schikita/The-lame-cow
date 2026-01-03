from __future__ import annotations

import argparse
from pathlib import Path

from predict import Predictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test for Predictor")
    p.add_argument("--weights", type=str, default="artifacts/train-seg/weights/best.pt")
    p.add_argument("--image", type=str, required=True, help="Path to a single test image")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    img = Path(args.image)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not img.exists():
        raise FileNotFoundError(f"Image not found: {img}")

    predictor = Predictor(weights=weights, device=args.device)

    preds = predictor.predict(source=img, conf=0.25, iou=0.7, save=False)
    assert isinstance(preds, list), "Predictor.predict must return list"
    assert len(preds) >= 1, "No result frames returned"

    first = preds[0]
    assert isinstance(first, list), "Frame predictions must be list"

    print(f"[OK] Loaded weights: {weights}")
    print(f"[OK] Predicted on: {img}")
    print(f"[OK] Detections on first frame: {len(first)}")

    for o in first[:10]:
        print(f"  - {o.cls_name} conf={o.conf:.3f} bbox={o.bbox_xyxy} seg={'yes' if o.seg_xy else 'no'}")


if __name__ == "__main__":
    main()
