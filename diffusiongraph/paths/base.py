"""
Common interface for the four SEED §3.2 path types. Every constructor takes
a real sample pair (x_a ~ class A, x_b ~ class B) — never centroids, per
SEED §5 — and a working noise level sigma_tau (Strategic_Blind_Spots #1: the
graph is measured per-timestep, not integrated across the whole reverse
process), and returns a PathResult: a sequence of *realistic, decoded*
images along t in [0, 1] ready to hand to the evaluator stack.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class PathResult:
    path_type: str
    sigma_tau: float
    t_values: torch.Tensor          # [T]
    images: torch.Tensor            # [T, B, C, H, W], canonical [-1, 1]
    meta: dict = field(default_factory=dict)


def forward_diffuse(x0: torch.Tensor, sigma: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """EDM forward process: x_sigma = x_0 + sigma * eps, eps ~ N(0, I).
    This is exact (EDM's forward marginal is Gaussian around x_0 with std
    sigma, no schedule/alpha_bar bookkeeping needed) -- the noise level *is*
    the standard deviation in EDM's parameterization.
    """
    # Generate on CPU (generator is CPU-based for cross-run reproducibility
    # independent of which device is active) then move to match x0.
    eps = torch.randn(x0.shape, generator=generator, dtype=torch.float32).to(x0.device, x0.dtype)
    return x0 + sigma * eps


class PathConstructor:
    """Interface. Subclasses implement `construct`."""

    name: str = "base"

    def construct(
        self,
        denoiser,               # diffusiongraph.utils.edm_loader.EDMDenoiser
        x_a: torch.Tensor,      # [B, C, H, W] real samples of class A
        x_b: torch.Tensor,      # [B, C, H, W] real samples of class B
        class_a: int,
        class_b: int,
        sigma_tau: float,
        n_steps: int,
        seed: int = 0,
    ) -> PathResult:
        raise NotImplementedError
