"""
One-time conversion: HuggingFace Hub CIFAR-10 parquet (data/hf_cifar10/*.parquet,
downloaded by hand -- see this script's companion download command in
README.md) -> a flat uint8 npz cache data/hf_cifar10/{train,test}.npz.

Why not torchvision.datasets.CIFAR10(download=True) directly: its source
(www.cs.toronto.edu) was measured at <0.2 MB/s on this network -- 170MB
would take 5+ hours. huggingface.co served the same data (uoft-cs/cifar10,
verified same class order/count) at ~0.8 MB/s, a real difference. This
script converts once so diffusiongraph/data/cifar10.py can load a plain
numpy cache instead of decoding parquet+PIL on every run.
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image

from diffusiongraph.config import DATA_DIR

HF_DIR = DATA_DIR / "hf_cifar10"


def convert_split(parquet_path: Path, out_path: Path):
    df = pd.read_parquet(parquet_path)
    img_col = "img" if "img" in df.columns else "image"
    n = len(df)
    images = np.empty((n, 32, 32, 3), dtype=np.uint8)
    labels = np.empty((n,), dtype=np.int64)
    for i, row in enumerate(df.itertuples(index=False)):
        img_field = getattr(row, img_col)
        # HF parquet stores the Image feature as a dict-like {'bytes':..., 'path':...}
        raw = img_field["bytes"] if isinstance(img_field, dict) else img_field
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        images[i] = np.array(img, dtype=np.uint8)
        labels[i] = int(getattr(row, "label"))
    np.savez(out_path, images=images, labels=labels)
    print(f"{out_path}: {n} images, label range [{labels.min()}, {labels.max()}]")


def main():
    convert_split(HF_DIR / "train.parquet", HF_DIR / "train.npz")
    convert_split(HF_DIR / "test.parquet", HF_DIR / "test.npz")


if __name__ == "__main__":
    main()
