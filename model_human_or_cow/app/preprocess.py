from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tqdm import tqdm
import yaml


Split = Literal["train", "val"]


@dataclass(frozen=True)
class Sample:
    img_path: Path
    class_name: str


CLASS_NAME_TO_ID = {
    "cow": 0,
    "human": 1,
}


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class YoloPreprocessor:

    def __init__(
        self,
        raw_root: str | Path = "data/raw",
        out_root: str | Path = "app/dataset",
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.out_root = Path(out_root)
        self.train_ratio = float(train_ratio)
        self.seed = int(seed)
        random.seed(self.seed)

    def run(self) -> None:
        self._prepare_dirs()
        samples = self._collect_samples()
        train_samples, val_samples = self._split(samples)

        print(f"[INFO] train: {len(train_samples)} | val: {len(val_samples)}")

        for s in tqdm(train_samples, desc="Export train"):
            self._export_sample(s, split="train")

        for s in tqdm(val_samples, desc="Export val"):
            self._export_sample(s, split="val")

        self._write_data_yaml()
        print(f"[DONE] Preprocess complete. Dataset in: {self.out_root.resolve()}")

    def _prepare_dirs(self) -> None:
        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            (self.out_root / sub).mkdir(parents=True, exist_ok=True)

    def _collect_samples(self) -> list[Sample]:
        samples: list[Sample] = []

        for class_name in CLASS_NAME_TO_ID.keys():
            class_dir = self.raw_root / class_name
            if not class_dir.exists():
                print(f"[WARN] Dir not found: {class_dir} (skip)")
                continue

            for img_path in class_dir.iterdir():
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue

                label_path = img_path.with_suffix(".txt")
                if not label_path.exists():
                    raise FileNotFoundError(f"Label not found for image: {img_path}")

                samples.append(Sample(img_path=img_path, class_name=class_name))

        if not samples:
            raise RuntimeError(
                f"No samples found. Check raw_root={self.raw_root} and structure:\n"
                f"  data/raw/cow/*.jpg + *.txt\n"
                f"  data/raw/human/*.jpg + *.txt\n"
            )

        print(f"[INFO] Found {len(samples)} images")
        return samples

    def _split(self, samples: list[Sample]) -> tuple[list[Sample], list[Sample]]:
        samples = samples[:]
        random.shuffle(samples)

        n_train = int(len(samples) * self.train_ratio)
        n_train = max(1, min(n_train, len(samples) - 1)) 

        return samples[:n_train], samples[n_train:]

    def _export_sample(self, sample: Sample, split: Split) -> None:
        class_id = CLASS_NAME_TO_ID[sample.class_name]

        img_dst = self.out_root / "images" / split / sample.img_path.name
        label_dst = self.out_root / "labels" / split / (sample.img_path.stem + ".txt")

        img_dst.write_bytes(sample.img_path.read_bytes())

        label_src = sample.img_path.with_suffix(".txt")
        converted = self._convert_label_file(label_src, class_id=class_id)

        label_dst.write_text(converted, encoding="utf-8")

    def _convert_label_file(self, label_path: Path, class_id: int) -> str:

        lines_out: list[str] = []

        raw = label_path.read_text(encoding="utf-8").strip()
        if not raw:
            return ""

        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln:
                continue

            parts = ln.split()
            nums: list[float] = []
            try:
                nums = [float(x) for x in parts]
            except ValueError:
                raise ValueError(f"Non-numeric label line in {label_path}: {ln}")

            if len(nums) < 5:
                raise ValueError(f"Too short label line in {label_path}: {ln}")
            
            coords = nums[1:]

            if len(coords) == 4:

                xc, yc, w, h = coords
                poly = self._bbox_to_poly(xc, yc, w, h)
                out = [str(class_id)] + [self._fmt(x) for x in poly]
                lines_out.append(" ".join(out))
            else:

                if len(coords) % 2 != 0:
                    raise ValueError(
                        f"Polygon coords count must be even in {label_path}. Line: {ln}"
                    )

                out = [str(class_id)] + [self._fmt(x) for x in coords]
                lines_out.append(" ".join(out))

        return "\n".join(lines_out) + "\n"

    @staticmethod
    def _bbox_to_poly(xc: float, yc: float, w: float, h: float) -> list[float]:

        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
        return [x1, y1, x2, y1, x2, y2, x1, y2]

    @staticmethod
    def _fmt(x: float) -> str:
        
        return f"{x:.10f}".rstrip("0").rstrip(".")

    def _write_data_yaml(self) -> None:
        data = {
            "path": str(self.out_root),
            "train": "images/train",
            "val": "images/val",
            "names": {v: k for k, v in CLASS_NAME_TO_ID.items()},
        }
        yaml_path = self.out_root / "data.yaml"
        yaml_path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")
        print(f"[INFO] Created data.yaml: {yaml_path.resolve()}")


if __name__ == "__main__":
    YoloPreprocessor(
        raw_root="data/raw",
        out_root="app/dataset",
        train_ratio=0.8,
        seed=42,
    ).run()
