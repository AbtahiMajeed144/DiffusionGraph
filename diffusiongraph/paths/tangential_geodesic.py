"""
Path type 3 (SEED §3.2): score-Jacobian manifold-tangential geodesic.

Original implementation of Saito & Matsubara's metric (arXiv:2510.05509) --
no code was released for that paper (confirmed by checking its arXiv page
directly; see THIRD_PARTY.md), so this is built from the published math on
top of stochman's curve-energy-minimization pattern, using our own JVP
utility (utils/jvp.py) for the "never build full Jacobians" requirement
(SEED §5).

Metric: g_x(v, v) = || J_s(x) v ||_2^2,  J_s(x) = d/dx score(x, sigma_tau).
Geodesic: minimize the discretized curve energy
    E(gamma) = sum_k || J_s(x_k) (x_{k+1} - x_k) ||_2^2
over interior control points, endpoints fixed at the forward-diffused real
samples. Low singular values of J_s correspond to tangent (manifold)
directions; minimizing this energy therefore pulls the curve to stay
*parallel* to the data manifold rather than cutting through low-density
space (unlike a pure density/likelihood objective -- see Moreau et al.'s
"likelihood-realism paradox", which is exactly why we don't just maximize
density here).

Runs on the UNCONDITIONAL EDM/score_sde checkpoint (Strategic_Blind_Spots
#2), same as path 2 -- see slerp_noise.py's docstring for why.
"""
from __future__ import annotations
import time

import torch

from diffusiongraph.paths.base import PathConstructor, PathResult, forward_diffuse
from diffusiongraph.paths.slerp_noise import slerp
from diffusiongraph.utils.jvp import score_jvp


class TangentialGeodesicPath(PathConstructor):
    name = "tangential_geodesic"

    def __init__(self, num_control_points: int = 16, optimizer_steps: int = 200, lr: float = 1e-2, jvp_chunk_size: int = 8):
        self.num_control_points = num_control_points
        self.optimizer_steps = optimizer_steps
        self.lr = lr
        self.jvp_chunk_size = jvp_chunk_size

    def _energy_backward_chunked(self, points: torch.Tensor, score_fn) -> float:
        """points: [B, K, C, H, W], requires_grad through `interior`.

        The JVP itself is computed via double reverse-mode (utils/jvp.py),
        so each segment's contribution needs create_graph=True on BOTH
        internal backward calls to stay differentiable w.r.t. the curve
        points for this OUTER optimization step -- that graph is roughly
        3x a forward pass in memory. A single flat batch of B*(K-1)
        segments blew past 4GB at just 10 segments in testing (see
        tangential_geodesic smoke test). So: call .backward() once per
        chunk of `jvp_chunk_size` segments instead of once for the whole
        batch -- gradients accumulate on `interior.grad` correctly by
        linearity (sum of chunk-backwards == backward of the sum), while
        each chunk's graph is freed immediately after its own backward().
        This bounds peak memory to one chunk regardless of B*(K-1).

        Returns the total energy (float, for logging only -- gradients are
        already accumulated into `interior.grad` as a side effect).
        """
        b, k = points.shape[0], points.shape[1]
        at = points[:, :-1].reshape(b * (k - 1), *points.shape[2:])
        diffs = (points[:, 1:] - points[:, :-1]).reshape(b * (k - 1), *points.shape[2:])
        n = at.shape[0]
        total_energy = 0.0
        for start in range(0, n, self.jvp_chunk_size):
            end = min(start + self.jvp_chunk_size, n)
            jv = score_jvp(score_fn, at[start:end], diffs[start:end])
            seg_energy = jv.flatten(1).pow(2).sum()
            seg_energy.backward()
            total_energy += seg_energy.item()
        return total_energy

    def construct(self, denoiser, x_a, x_b, class_a, class_b, sigma_tau, n_steps, seed=0):
        # `denoiser` is the UNCONDITIONAL EDMDenoiser (see module docstring).
        device = denoiser.device
        gen = torch.Generator(device="cpu").manual_seed(seed)

        x_a_sigma = forward_diffuse(x_a.to(device), sigma_tau, generator=gen)
        x_b_sigma = forward_diffuse(x_b.to(device), sigma_tau, generator=gen)
        b = x_a_sigma.shape[0]
        k = self.num_control_points

        # Initialize the curve by slerp (a better starting guess than linear,
        # per path 2's rationale) at K evenly spaced t values; only the
        # K-2 interior points are optimized, endpoints stay fixed.
        init_t = torch.linspace(0, 1, k)
        init_points = torch.stack([slerp(x_a_sigma, x_b_sigma, float(t)) for t in init_t], dim=1)  # [B,K,C,H,W]
        interior = init_points[:, 1:-1].clone().detach().requires_grad_(True)

        def score_fn(x):
            return denoiser.score(x, sigma_tau, class_labels=None)

        optimizer = torch.optim.Adam([interior], lr=self.lr)
        energy_history = []
        opt_t0 = time.time()
        log_every = max(1, self.optimizer_steps // 10)  # ~10 progress lines per combination
        for step in range(self.optimizer_steps):
            optimizer.zero_grad()
            points = torch.cat([x_a_sigma.unsqueeze(1), interior, x_b_sigma.unsqueeze(1)], dim=1)
            total_energy = self._energy_backward_chunked(points, score_fn)
            optimizer.step()
            energy_history.append(total_energy)
            if step == 0 or (step + 1) % log_every == 0 or step + 1 == self.optimizer_steps:
                elapsed = time.time() - opt_t0
                per_step = elapsed / (step + 1)
                eta = per_step * (self.optimizer_steps - step - 1)
                print(
                    f"    [tangential_geodesic] step {step+1}/{self.optimizer_steps} "
                    f"energy={total_energy:.2f} elapsed={elapsed:.1f}s "
                    f"({per_step:.3f}s/step, ETA {eta:.1f}s for this combo's optimization phase)",
                    flush=True,
                )

        print(f"    [tangential_geodesic] optimization done in {time.time()-opt_t0:.1f}s, decoding {k} control points...", flush=True)

        with torch.no_grad():
            final_points = torch.cat([x_a_sigma.unsqueeze(1), interior, x_b_sigma.unsqueeze(1)], dim=1)

        # Decode every control point to a realistic image for evaluation.
        decode_t0 = time.time()
        images = []
        with torch.no_grad():
            for i in range(k):
                img = denoiser.denoise_to_clean(final_points[:, i], sigma_tau, class_labels=None, num_steps=18)
                images.append(img)
        print(f"    [tangential_geodesic] decode done in {time.time()-decode_t0:.1f}s", flush=True)

        return PathResult(
            path_type=self.name,
            sigma_tau=sigma_tau,
            t_values=init_t,
            images=torch.stack(images, dim=0),  # [K, B, C, H, W]
            meta={
                "note": "score-Jacobian tangential geodesic, curve-energy minimized",
                "energy_history": energy_history,
                "num_control_points": k,
                "optimizer_steps": self.optimizer_steps,
            },
        )
