from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tqdm import tqdm
import yaml


Split = Literal["train","val"]

@dataclass
class Sample:
    path:Path
    class_name:str
    split:Split

CLASS_NAME_TO_ID = {
    "cow" : 0,
    "human" : 1,
}

class YoloPreprocessor:
    def __init__(
            self,
            raw_root: str | Path = "raw_data",
            out_root: str | Path = "dataset",
            train_ratio:float = 0.8,
            seed: int = 42
   ) -> None :
        self.raw_root = Path(raw_root)
        self.out_root = Path(out_root)
        self.train_ratio = train_ratio
        self.seed = seed
        random.seed(seed)

    def run(self) ->  None:
        self._prepare_dirs()
        samples = self._collect_samples()
        self._split_and_export(samples)
        self._write_data_yaml()
        print("Preprocess complete succeful. Data set is in:  {self.out_root}")

    
    def _prepare_dirs(self) -> None:
        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            (self.out_root/sub).mkdir(parents = True, exist_ok=True)

    def _collect_samples(self) -> list[Sample]:
        samples : list[Sample] = []
        for class_name in CLASS_NAME_TO_ID.keys():
            class_dir = self.raw_root / class_name
            if not class_dir.exists():
                print(f"[WARN] Dir {class_dir} not found, skip" )
                continue

            for img_path in class_dir.glob("*.*"):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                samples.append(Sample(path=img_path, class_name=class_name,split="train"))

        print(f"Found {len(samples)} pictures")
        return samples
    
    def _split_and_export(self, samples: list[Sample]) -> None:
        random.shuffle(samples)

        n_train = int(len(samples) * self.train_ratio)
        train_samples = samples[:n_train]
        val_samples = samples[n_train:]

        print(f"train: {len(train_samples)}, val: {len(val_samples)}")

        for s in tqdm(train_samples, desc="Export train"):
            self._export_sample(s, split="train")

        for s in tqdm(val_samples, desc="Export val"):
            self._export_sample(s, split="val")

    def _export_sample(self, sample: Sample, split: Split) -> None:
        class_id = CLASS_NAME_TO_ID[sample.class_name]

        img_dst = self.out_root / "images" / split / sample.path.name

        label_dst = self.out_root / "labels" / split / (sample.path.stem + ".txt")

        img_dst.write_bytes(sample.path.read_bytes())

        # YOLO-разметка: <class_id> <x_center> <y_center> <width> <height>
        # Пока bbox на весь кадр (0.5, 0.5, 1, 1) — потом заменим на реальные
        line = f"{class_id} 0.5 0.5 1.0 1.0\n"
        label_dst.write_text(line, encoding="utf-8")

    def _write_data_yaml(self) -> None:
        data = {
            "path": str(self.out_root),
            "train": "images/train",
            "val": "images/val",
            "names": {v: k for k, v in CLASS_NAME_TO_ID.items()},
        }
        yaml_path = self.out_root / "data.yaml"
        yaml_path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")
        print("Создан data.yaml:", yaml_path)


if __name__ == "__main__":
    prep = YoloPreprocessor(
        raw_root="data/raw",
        out_root="data/dataset",
        train_ratio=0.8,
        seed=42,
    )
    prep.run()