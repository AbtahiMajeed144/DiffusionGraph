"""
Trains an independent-architecture evaluator (any name in
diffusiongraph.models.classifiers.ARCHITECTURES) on CIFAR-10, in canonical
[-1, 1] image format, for both the real labels and the label-permutation
control (SEED §3.4 / data/cifar10.py's PermutedLabelCIFAR10).

Usage:
    python scripts/train_classifiers.py --arch resnet18
    python scripts/train_classifiers.py --arch vit_small
    python scripts/train_classifiers.py --arch resnet50 --epochs 80
    python scripts/train_classifiers.py --arch vit_base --epochs 120 --permuted
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffusiongraph.config import CHECKPOINTS_DIR
from diffusiongraph.data.cifar10 import CIFAR10Canonical, PermutedLabelCIFAR10
from diffusiongraph.models.classifiers import ARCHITECTURES


def train(arch: str, permuted: bool, epochs: int, batch_size: int, lr: float, device: str):
    ds_cls = PermutedLabelCIFAR10 if permuted else CIFAR10Canonical
    train_ds = ds_cls(train=True, download=True, augment=True)
    test_ds = ds_cls(train=False, download=True, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    model = ARCHITECTURES[arch]().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(-1) == y).sum().item()
            total_n += x.size(0)
        sched.step()

        model.eval()
        test_correct, test_n = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                test_correct += (logits.argmax(-1) == y).sum().item()
                test_n += x.size(0)

        print(
            f"[{arch}{' perm' if permuted else ''}] epoch {epoch+1}/{epochs} "
            f"train_loss={total_loss/total_n:.4f} train_acc={total_correct/total_n:.4f} "
            f"test_acc={test_correct/test_n:.4f} ({time.time()-t0:.1f}s)"
        )

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_permuted" if permuted else ""
    dest = CHECKPOINTS_DIR / f"{arch}_cifar10{suffix}.pt"
    torch.save({"model_state_dict": model.state_dict(), "test_acc": test_correct / test_n}, dest)
    print(f"Saved: {dest}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=list(ARCHITECTURES), required=True)
    p.add_argument("--permuted", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    train(args.arch, args.permuted, args.epochs, args.batch_size, args.lr, args.device)


if __name__ == "__main__":
    main()
