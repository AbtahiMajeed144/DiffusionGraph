"""
Path type 2 (SEED §3.2): slerp in noise space, "prior-respecting baseline".

Design [assumption], documented because SEED's one-line description
underdetermines the exact construction:
  1. Forward-diffuse the real endpoints to the working noise level sigma_tau:
     x_a_sigma = x_a + sigma_tau * eps_a,  x_b_sigma = x_b + sigma_tau * eps_b.
     (SEED §5: sample-pair endpoints, not centroids.)
  2. Spherically interpolate between these two noisy states (slerp keeps the
     norm on the sphere the Gaussian prior actually concentrates on, unlike a
     straight line, which shrinks toward the origin mid-path -- the actual
     baseline this path type is meant to test: does *just* respecting the
     prior's spherical geometry, with no manifold-tangential correction,
     already produce routing?).
  3. Finish the reverse ODE from sigma_tau to 0 at each interpolated point to
     get a realistic image (EDMDenoiser.denoise_to_clean).

Conditioning: this path runs entirely on the UNCONDITIONAL EDM/score_sde
CIFAR-10 checkpoint (see config.edm_checkpoint_uncond), not the
class-conditional one -- there is no class-label choice to make at all,
which is the structural fix for CFG/conditioning contamination
(Strategic_Blind_Spots #2), stronger than pinning a guidance weight on the
conditional model. class_a/class_b are accepted for interface symmetry
(used only to pick real endpoint images upstream) and logging.
"""
from __future__ import annotations
import torch

from diffusiongraph.paths.base import PathConstructor, PathResult, forward_diffuse


def slerp(a: torch.Tensor, b: torch.Tensor, t: float, eps: float = 1e-7) -> torch.Tensor:
    """Spherical linear interpolation, batched, flattening spatial dims to
    compute the angle then reshaping back."""
    shape = a.shape
    af = a.flatten(1)
    bf = b.flatten(1)
    a_n = af / af.norm(dim=1, keepdim=True).clamp_min(eps)
    b_n = bf / bf.norm(dim=1, keepdim=True).clamp_min(eps)
    dot = (a_n * b_n).sum(dim=1, keepdim=True).clamp(-1 + eps, 1 - eps)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta).clamp_min(eps)
    w_a = torch.sin((1 - t) * theta) / sin_theta
    w_b = torch.sin(t * theta) / sin_theta
    # Interpolate norms linearly (slerp direction, lerp magnitude) so the
    # result's scale tracks the two endpoints rather than being pinned to
    # whichever has larger norm.
    norm_a = af.norm(dim=1, keepdim=True)
    norm_b = bf.norm(dim=1, keepdim=True)
    out_dir = w_a * a_n + w_b * b_n
    out_norm = (1 - t) * norm_a + t * norm_b
    out = out_dir * out_norm
    return out.view(shape)


class SlerpNoisePath(PathConstructor):
    name = "slerp_noise"

    def construct(self, denoiser, x_a, x_b, class_a, class_b, sigma_tau, n_steps, seed=0):
        # `denoiser` here is expected to be the UNCONDITIONAL EDMDenoiser
        # (label_dim=0), passed in by the orchestrator (eval/trajectory.py).
        device = denoiser.device
        gen = torch.Generator(device="cpu").manual_seed(seed)

        x_a_sigma = forward_diffuse(x_a.to(device), sigma_tau, generator=gen)
        x_b_sigma = forward_diffuse(x_b.to(device), sigma_tau, generator=gen)

        t_values = torch.linspace(0, 1, n_steps)
        images = []
        for t in t_values:
            x_t = slerp(x_a_sigma, x_b_sigma, float(t))
            img = denoiser.denoise_to_clean(x_t, sigma_tau, class_labels=None, num_steps=18)
            images.append(img)

        return PathResult(
            path_type=self.name,
            sigma_tau=sigma_tau,
            t_values=t_values,
            images=torch.stack(images, dim=0),
            meta={"note": "slerp between forward-diffused real endpoints, unconditional decode"},
        )
