"""
Path type 1 (SEED §3.2): linear class-embedding interpolation.
c(t) = (1-t)*onehot(A) + t*onehot(B), full generation from pure noise at
each t. This is "Object 1" in SEED §2 -- moving the *condition*, not the
distribution geometry -- included only as the weak baseline it's expected to
be, per the project's explicit design discipline: a continuous condition
does not imply the interpolated path is semantically meaningful.

Scope note [assumption]: unlike path types 2/3/4, this baseline is *not*
evaluated per fixed noise level sigma_tau -- it's the classic "blend the
condition, run the whole sampler" baseline, so it produces one path per
class pair regardless of the routing_sigmas sweep. sigma_tau is accepted for
interface compatibility but unused; PathResult.sigma_tau is reported as
float('nan') to make that explicit downstream (see eval/trajectory.py).
"""
from __future__ import annotations
import torch

from diffusiongraph.paths.base import PathConstructor, PathResult


class LinearConditionPath(PathConstructor):
    name = "linear_condition"

    def construct(self, denoiser, x_a, x_b, class_a, class_b, sigma_tau, n_steps, seed=0):
        batch_size = x_a.shape[0]
        device = denoiser.device

        # Fixed initial noise, shared across all t -- the only thing that
        # varies along this path is the conditioning vector.
        gen = torch.Generator(device="cpu").manual_seed(seed)
        latents = torch.randn(
            batch_size, denoiser.img_channels, denoiser.img_resolution, denoiser.img_resolution,
            generator=gen,
        ).to(device)

        t_values = torch.linspace(0, 1, n_steps)
        onehot_a = denoiser.one_hot(class_a, batch_size)
        onehot_b = denoiser.one_hot(class_b, batch_size)

        images = []
        for t in t_values:
            c_t = (1 - t) * onehot_a + t * onehot_b
            img = denoiser.run_sampler(latents, c_t, num_steps=18, seed=seed)
            images.append(img)

        return PathResult(
            path_type=self.name,
            sigma_tau=float("nan"),
            t_values=t_values,
            images=torch.stack(images, dim=0),  # [T, B, C, H, W]
            meta={"note": "full generation per t, condition-only interpolation, fixed initial noise"},
        )
