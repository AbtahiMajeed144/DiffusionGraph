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
from diffusiongraph.models.classifiers import resnet18, vit_small
from diffusiongraph.models.embeddings import ClipZeroShot


class TrainedClassifierEvaluator:
    """Wraps a trained resnet18/vit_small checkpoint with the uniform
    predict_proba interface."""

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


def load_evaluators(names=("resnet18", "vit_small", "clip_zeroshot"), device: str = "cuda") -> Dict[str, object]:
    evaluators = {}
    for name in names:
        if name == "resnet18":
            ckpt = CHECKPOINTS_DIR / "resnet18_cifar10.pt"
            evaluators[name] = TrainedClassifierEvaluator(resnet18(), ckpt, device=device)
        elif name == "vit_small":
            ckpt = CHECKPOINTS_DIR / "vit_small_cifar10.pt"
            evaluators[name] = TrainedClassifierEvaluator(vit_small(), ckpt, device=device)
        elif name == "clip_zeroshot":
            evaluators[name] = ClipZeroShot(device=device)
        else:
            raise ValueError(f"Unknown evaluator '{name}'")
    return evaluators


def load_permuted_evaluators(device: str = "cuda") -> Dict[str, object]:
    """The two trained-from-scratch evaluators, retrained under the
    label-permutation control (SEED §3.4). CLIP is excluded -- it wasn't
    trained on CIFAR-10 labels at all, so there's nothing to permute; its
    zero-shot prompts are tied to real English class names and can't be
    meaningfully "permuted" without changing the experiment's meaning."""
    ckpt_r = CHECKPOINTS_DIR / "resnet18_cifar10_permuted.pt"
    ckpt_v = CHECKPOINTS_DIR / "vit_small_cifar10_permuted.pt"
    return {
        "resnet18": TrainedClassifierEvaluator(resnet18(), ckpt_r, device=device),
        "vit_small": TrainedClassifierEvaluator(vit_small(), ckpt_v, device=device),
    }
