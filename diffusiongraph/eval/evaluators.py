"""
Loads the 3 independent evaluators (SEED §3.2 / §6) behind one uniform
interface: `.predict_proba(images) -> [B, num_classes] softmax`, images in
our canonical [-1, 1] format. trajectory.py sweeps a path and calls all
three identically -- routing must survive across ALL of them (SEED §3.4) to
count, so nothing downstream should special-case any one evaluator.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from diffusiongraph.config import CHECKPOINTS_DIR
from diffusiongraph.models.classifiers import ARCHITECTURES
from diffusiongraph.models.embeddings import ClipZeroShot


class TrainedClassifierEvaluator:
    """Wraps a trained checkpoint (any ARCHITECTURES entry) with the
    uniform predict_proba interface."""

    def __init__(self, model: torch.nn.Module, checkpoint_path: Path, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        state = torch.load(checkpoint_path, map_location=self.device)
        state_dict = state["model_state_dict"] if "model_state_dict" in state else state
        model.load_state_dict(state_dict)
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        logits = self.model(images)
        return F.softmax(logits, dim=-1)


def load_evaluators(names=("resnet50", "vit_base", "clip_zeroshot"), device: str = "cuda") -> Dict[str, object]:
    """Any name in ARCHITECTURES (models/classifiers.py) loads from
    checkpoints/{name}_cifar10.pt; "clip_zeroshot" is the one special case
    (no training, no checkpoint file -- see models/embeddings.py)."""
    evaluators = {}
    for name in names:
        if name == "clip_zeroshot":
            evaluators[name] = ClipZeroShot(device=device)
        elif name in ARCHITECTURES:
            ckpt = CHECKPOINTS_DIR / f"{name}_cifar10.pt"
            evaluators[name] = TrainedClassifierEvaluator(ARCHITECTURES[name](), ckpt, device=device)
        else:
            raise ValueError(f"Unknown evaluator '{name}'. Choices: clip_zeroshot, {list(ARCHITECTURES)}")
    return evaluators


def load_permuted_evaluators(names=("resnet50", "vit_base"), device: str = "cuda") -> Dict[str, object]:
    """The trained-from-scratch evaluators, retrained under the
    label-permutation control (SEED §3.4). CLIP is excluded -- it wasn't
    trained on CIFAR-10 labels at all, so there's nothing to permute; its
    zero-shot prompts are tied to real English class names and can't be
    meaningfully "permuted" without changing the experiment's meaning."""
    evaluators = {}
    for name in names:
        if name not in ARCHITECTURES:
            raise ValueError(f"Unknown evaluator '{name}'. Choices: {list(ARCHITECTURES)}")
        ckpt = CHECKPOINTS_DIR / f"{name}_cifar10_permuted.pt"
        evaluators[name] = TrainedClassifierEvaluator(ARCHITECTURES[name](), ckpt, device=device)
    return evaluators
