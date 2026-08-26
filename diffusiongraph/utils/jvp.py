"""
JVP/VJP utilities for the score-Jacobian metric (SEED §5: "Never build full
Jacobians — use JVP/VJP, Hutchinson trace estimates, low-rank eigensolvers").

We need, for a score function s(x) = score(x, sigma, class_labels) and a
tangent vector v: J_s(x) @ v, where J_s = d s/d x. This is a forward-mode
Jacobian-vector product — never materialize the [B, C*H*W, C*H*W] Jacobian.
"""
from __future__ import annotations
from typing import Callable

import torch


def score_jvp(score_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute J_score(x) @ v -- never materializing J -- via the standard
    "double vjp" trick rather than torch.func.jvp:

        u = zeros_like(score_fn(x)), requires_grad
        g = grad(score_fn(x), x, grad_outputs=u, create_graph=True)   # = J^T u, fn of u
        Jv = grad(g, u, grad_outputs=v, create_graph=True)            # = J v

    We use this (pure reverse-mode, twice) instead of forward-mode jvp
    specifically because path-type 3 (tangential_geodesic.py) needs the
    result to stay differentiable w.r.t. x for an OUTER optimization loop
    (curve-energy minimization over control points) -- i.e. this is used
    inside a "differentiate through the JVP" composition, which reverse-mode
    double-backward supports unconditionally, without depending on every op
    in the pretrained U-Net having forward-mode AD coverage.

    score_fn: x -> score(x)  (sigma/class_labels already closed over)
    x, v: same shape [B, C, H, W]. Returns: J @ v, shape [B, C, H, W].
    """
    x = x.requires_grad_(True)
    out = score_fn(x)
    u = torch.zeros_like(out, requires_grad=True)
    g = torch.autograd.grad(out, x, grad_outputs=u, create_graph=True)[0]
    jv = torch.autograd.grad(g, u, grad_outputs=v, create_graph=True)[0]
    return jv


def score_metric_energy(score_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The Saito–Matsubara tangential-metric energy density at a curve point:
        g_x(v, v) = || J_score(x) @ v ||_2^2
    per-sample (batch dim preserved, spatial dims reduced).
    """
    jv = score_jvp(score_fn, x, v)
    return jv.flatten(1).pow(2).sum(dim=1)


def hutchinson_trace(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    num_probes: int = 4,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Hutchinson estimator of tr(J_score(x)^T J_score(x)) using random
    Rademacher probes and JVPs only (no full Jacobian). Not required for the
    Phase 1 gate's routing measurement, but kept here as the diagnostic tool
    RQ1 (thin-confidence-band / phase-boundary signature, deferred to
    post-gate work per SEED §7) will need.
    """
    b = x.shape[0]
    total = torch.zeros(b, device=x.device)
    for _ in range(num_probes):
        v = torch.randint(0, 2, x.shape, generator=generator, device=x.device, dtype=x.dtype) * 2 - 1
        jv = score_jvp(score_fn, x, v)
        total = total + jv.flatten(1).pow(2).sum(dim=1)
    return total / num_probes
