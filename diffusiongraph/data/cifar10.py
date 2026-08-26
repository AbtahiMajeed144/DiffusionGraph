"""
CIFAR-10 loading, kept in our canonical image format: float tensors in
[-1, 1], shape [B, 3, 32, 32] -- the same range the EDM generator emits
(generate.py: `images*127.5+128`, i.e. net output is already [-1, 1]).
Classifiers and CLIP's preprocessing both consume this same format so path
samples and real images never need format-specific branching downstream.

Also provides per-class sample-pair retrieval for path endpoints: SEED §5
explicitly rejects class centroids (multimodal classes -> paths through
empty space) in favor of sample-to-sample paths, so what we need here is
"give me N real images of class c", not a mean/prototype.

Data source: prefers a local npz cache built by scripts/convert_hf_cifar10.py
(HuggingFace Hub mirror -- www.cs.toronto.edu, torchvision's default source,
was measured at <0.2 MB/s on this network, i.e. 5+ hours for 170MB; HF
Hub served the same data, verified same class order/count, at ~0.8 MB/s).
Falls back to torchvision's own download if the cache isn't present.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusiongraph.config import DATA_DIR, CIFAR10_CLASSES

HF_CACHE_DIR = DATA_DIR / "hf_cifar10"


class CIFAR10Canonical(Dataset):
    """images: uint8 [N,32,32,3], labels: int64 [N] -- from the HF npz cache
    if present, else torchvision's own (slow) download.

    augment=True applies standard CIFAR training augmentation (4px
    reflect-pad + random 32x32 crop, random horizontal flip) -- added after
    the first evaluator training run showed ResNet18 memorizing the training
    set (100% train acc by epoch 13, ~85% test acc: a real generalization
    gap) while ViT-small underfit badly (61% test acc) with zero
    augmentation. Both symptoms share one fix. Use augment=True for TRAINING
    only -- eval/gate code must see the deterministic, un-augmented canonical
    image, so this defaults to False and train_classifiers.py opts in
    explicitly for its train split alone."""

    def __init__(self, train: bool = True, download: bool = True, root: Path = DATA_DIR, augment: bool = False):
        self.augment = augment
        cache_path = HF_CACHE_DIR / ("train.npz" if train else "test.npz")
        if cache_path.exists():
            data = np.load(cache_path)
            self._images = data["images"]          # [N, 32, 32, 3] uint8
            self._labels = data["labels"].astype(np.int64)  # [N]
        else:
            import torchvision

            base = torchvision.datasets.CIFAR10(root=str(root), train=train, download=download)
            assert list(base.classes) == CIFAR10_CLASSES, (
                f"CIFAR10 class order mismatch: {base.classes} vs {CIFAR10_CLASSES}"
            )
            self._images = base.data                # [N, 32, 32, 3] uint8
            self._labels = np.array(base.targets, dtype=np.int64)

        self._by_class: Dict[int, List[int]] = {c: [] for c in range(10)}
        for idx, label in enumerate(self._labels.tolist()):
            self._by_class[label].append(idx)

    def __len__(self):
        return len(self._labels)

    def _to_canonical(self, img_uint8: np.ndarray) -> torch.Tensor:
        if self.augment:
            img_uint8 = self._augment(img_uint8)
        x = torch.from_numpy(img_uint8).float().permute(2, 0, 1)  # [3,32,32], 0..255
        return x / 127.5 - 1.0  # -> [-1, 1]

    @staticmethod
    def _augment(img_uint8: np.ndarray) -> np.ndarray:
        # 4px reflect-pad + random 32x32 crop
        padded = np.pad(img_uint8, ((4, 4), (4, 4), (0, 0)), mode="reflect")
        top = np.random.randint(0, 9)   # 0..8 inclusive, 32px window out of 40px
        left = np.random.randint(0, 9)
        cropped = padded[top:top + 32, left:left + 32, :]
        # random horizontal flip
        if np.random.rand() < 0.5:
            cropped = cropped[:, ::-1, :]
        return np.ascontiguousarray(cropped)

    def __getitem__(self, idx):
        return self._to_canonical(self._images[idx]), int(self._labels[idx])

    def indices_for_class(self, class_idx: int) -> List[int]:
        return self._by_class[class_idx]

    def sample_pairs(self, class_a: int, class_b: int, n: int, generator: Optional[torch.Generator] = None):
        """Return n real (x, y) sample pairs: x ~ class_a, y ~ class_b.
        Sample-pair endpoints per SEED §5, not centroids."""
        idx_a = self._by_class[class_a]
        idx_b = self._by_class[class_b]
        g = generator or torch.Generator().manual_seed(0)
        pick_a = torch.randperm(len(idx_a), generator=g)[:n]
        pick_b = torch.randperm(len(idx_b), generator=g)[:n]
        xs = torch.stack([self[idx_a[i]][0] for i in pick_a])
        ys = torch.stack([self[idx_b[i]][0] for i in pick_b])
        return xs, ys


class PermutedLabelCIFAR10(CIFAR10Canonical):
    """The identifiability negative control (SEED §3.4): same image
    distributions, class IDs randomly permuted. Any 'structure' that survives
    this is a metric/classifier artifact tied to arbitrary label identifiers,
    not the image geometry -- and kills the project per SEED §3.5."""

    def __init__(self, train: bool = True, download: bool = True, root: Path = DATA_DIR, permutation_seed: int = 1234, augment: bool = False):
        super().__init__(train=train, download=download, root=root, augment=augment)
        rng = np.random.RandomState(permutation_seed)
        self.permutation = rng.permutation(10)
        self._labels = self.permutation[self._labels].astype(np.int64)
        permuted_by_class: Dict[int, List[int]] = {c: [] for c in range(10)}
        for idx, label in enumerate(self._labels.tolist()):
            permuted_by_class[label].append(idx)
        self._by_class = permuted_by_class
        self._inverse_permutation = np.argsort(self.permutation)
