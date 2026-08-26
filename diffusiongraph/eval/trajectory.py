"""
Sweeps a PathResult through all evaluators, logging the full softmax
p(c | gamma(t)) per evaluator (SEED §3.3) -- not just argmax, so routing.py
can apply the C(A,B) definition and later work can look at full trajectories.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

import torch

from diffusiongraph.paths.base import PathResult


@dataclass
class TrajectoryResult:
    path_type: str
    class_a: int
    class_b: int
    sigma_tau: float
    seed: int
    t_values: torch.Tensor                      # [T]
    softmax_by_evaluator: Dict[str, torch.Tensor] = field(default_factory=dict)  # {name: [T, B, num_classes]}


@torch.no_grad()
def evaluate_path(path_result: PathResult, evaluators: Dict[str, object], class_a: int, class_b: int, seed: int) -> TrajectoryResult:
    """images: [T, B, C, H, W] -> for each evaluator, [T, B, num_classes]."""
    images = path_result.images
    softmax_by_evaluator = {}
    for name, evaluator in evaluators.items():
        per_t = []
        for t_idx in range(images.shape[0]):
            proba = evaluator.predict_proba(images[t_idx])  # [B, num_classes]
            per_t.append(proba.cpu())
        softmax_by_evaluator[name] = torch.stack(per_t, dim=0)  # [T, B, num_classes]

    return TrajectoryResult(
        path_type=path_result.path_type,
        class_a=class_a,
        class_b=class_b,
        sigma_tau=path_result.sigma_tau,
        seed=seed,
        t_values=path_result.t_values,
        softmax_by_evaluator=softmax_by_evaluator,
    )
