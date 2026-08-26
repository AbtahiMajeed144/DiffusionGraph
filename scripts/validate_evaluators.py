"""
Sanity check the evaluator stack BEFORE trusting its softmax as a routing
signal (per the caveat in models/embeddings.py: CLIP is web/ImageNet-scale
trained, not CIFAR-native, and its behavior on 32x32-upsampled-to-224 images
needs to be checked empirically, not assumed).

Reports clean CIFAR-10 test accuracy for all three evaluators. Run this
after scripts/train_classifiers.py and before scripts/run_gate.py.

Usage:
    python scripts/validate_evaluators.py                       # rtx5090 defaults: resnet50, vit_base, clip_zeroshot
    python scripts/validate_evaluators.py --profile local_poc    # resnet18, vit_small, clip_zeroshot
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from diffusiongraph.config import get_profile
from diffusiongraph.data.cifar10 import CIFAR10Canonical
from diffusiongraph.eval.evaluators import load_evaluators


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="rtx5090", choices=["local_poc", "local_smoke", "rtx5090"])
    args = p.parse_args()
    evaluator_names = get_profile(args.profile).evaluator_names

    ds = CIFAR10Canonical(train=False, download=True)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    evaluators = load_evaluators(evaluator_names, device="cuda")

    for name, ev in evaluators.items():
        correct, total = 0, 0
        for x, y in loader:
            proba = ev.predict_proba(x)
            pred = proba.argmax(-1).cpu()
            correct += (pred == y).sum().item()
            total += x.size(0)
        acc = correct / total
        flag = ""
        if name == "clip_zeroshot" and acc < 0.5:
            flag = "  <-- LOW: CLIP zero-shot on upsampled 32px CIFAR may not be a trustworthy evaluator; consider dropping it or fine-tuning a linear probe instead of pure zero-shot."
        print(f"{name}: clean test accuracy = {acc:.4f}{flag}")


if __name__ == "__main__":
    main()
