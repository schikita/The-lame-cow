from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO


DEFAULT_NAMES = {0: "cow", 1: "human"}


@dataclass(frozen=True)
class DetectedObject:
    cls_id: int
    cls_name: str
    conf: float
    bbox_xyxy: tuple[float, float, float, float] | None = None
    seg_xy: list[float] | None = None


class Predictor:

    def __init__(
        self,
        weights: str | Path = "artifacts/train-seg/weights/best.pt",
        device: str = "cpu",
    ) -> None:
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(
                f"Weights not found: {self.weights}\n"
                f"Tip: train first (python train.py) or pass --weights <path_to_best.pt>"
            )

        self.model = YOLO(str(self.weights))
        self.device = device

        self.names: dict[int, str] = {}
        try:
            names_any: Any = getattr(self.model, "names", None)
            if isinstance(names_any, dict) and names_any:
                self.names = {int(k): str(v) for k, v in names_any.items()}
        except Exception:
            pass

        if not self.names:
            self.names = DEFAULT_NAMES

    def predict(
        self,
        source: str | Path,
        conf: float = 0.25,
        iou: float = 0.7,
        save: bool = False,
        save_dir: str | Path = "artifacts/predict",
    ) -> list[list[DetectedObject]]:
        """
        Returns list of predictions per frame/image.
        Each item is a list of DetectedObject.
        """
        source = str(source)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            device=self.device,
            save=save,
            project=str(save_dir),
            name="run",
        )

        parsed: list[list[DetectedObject]] = []

        for r in results:
            frame_objs: list[DetectedObject] = []

            boxes = getattr(r, "boxes", None)
            masks = getattr(r, "masks", None)

            if boxes is not None and len(boxes) > 0:
                cls_ids = boxes.cls
                confs = boxes.conf
                xyxy = boxes.xyxy

                cls_list = cls_ids.detach().cpu().tolist() if hasattr(cls_ids, "detach") else cls_ids.tolist()
                conf_list = confs.detach().cpu().tolist() if hasattr(confs, "detach") else confs.tolist()
                xyxy_list = xyxy.detach().cpu().tolist() if hasattr(xyxy, "detach") else xyxy.tolist()

                segs_per_det: list[list[float] | None] = [None] * len(cls_list)
                if masks is not None and getattr(masks, "xy", None) is not None:
                    xy_polys = masks.xy
                    for i, poly in enumerate(xy_polys):
                        try:
                            flat = []
                            for x, y in poly.tolist():
                                flat.extend([float(x), float(y)])
                            segs_per_det[i] = flat
                        except Exception:
                            segs_per_det[i] = None

                for i in range(len(cls_list)):
                    cid = int(cls_list[i])
                    name = self.names.get(cid, str(cid))
                    confv = float(conf_list[i])
                    x1, y1, x2, y2 = map(float, xyxy_list[i])
                    frame_objs.append(
                        DetectedObject(
                            cls_id=cid,
                            cls_name=name,
                            conf=confv,
                            bbox_xyxy=(x1, y1, x2, y2),
                            seg_xy=segs_per_det[i],
                        )
                    )

            parsed.append(frame_objs)

        return parsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict with trained YOLOv8-seg model (cow/human).")
    p.add_argument("--source", type=str, required=True, help="Image/video path, folder, URL, webcam id, etc.")
    p.add_argument("--weights", type=str, default="artifacts/train-seg/weights/best.pt", help="Path to best.pt")
    p.add_argument("--device", type=str, default="cpu", help='cpu | "0" for GPU0 | "0,1" for multi-gpu')
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--save", action="store_true", help="Save annotated outputs")
    p.add_argument("--out", type=str, default="artifacts/predict", help="Output dir for saved predictions")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    predictor = Predictor(weights=args.weights, device=args.device)

    preds = predictor.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        save_dir=args.out,
    )

    total = sum(len(x) for x in preds)
    print(f"[DONE] frames/images: {len(preds)} | total detections: {total}")

    for frame_idx, objs in enumerate(preds[:1]):
        print(f"[INFO] sample frame {frame_idx}: {len(objs)} detections")
        for o in objs[:10]:
            print(f"  - {o.cls_name} conf={o.conf:.3f} bbox={o.bbox_xyxy} seg={'yes' if o.seg_xy else 'no'}")


if __name__ == "__main__":
    main()
