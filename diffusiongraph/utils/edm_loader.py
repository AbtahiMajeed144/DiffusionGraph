"""
Thin wrapper around a pretrained NVlabs/edm CIFAR-10 checkpoint.

Deliberately does NOT reimplement the denoiser/network — we load the exact
pretrained pickle from references/edm and just expose the pieces we need:
denoise(), score(), and the EDM sampler for generating start/end samples.

CFG note (Strategic_Blind_Spots #2): the EDM CIFAR-10 checkpoints are trained
class-conditional but do NOT bake in classifier-free guidance at inference —
guidance only enters if *we* mix conditional/unconditional scores at sample
time. We never do that here (guidance_weight is accepted for future ablation
but must stay 1.0 for the Phase 1 gate, per GateConfig).
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path
from typing import Optional

import torch

from diffusiongraph.config import REFERENCES_DIR

_EDM_REPO = REFERENCES_DIR / "edm"


def _ensure_edm_on_path() -> None:
    if not _EDM_REPO.exists():
        raise FileNotFoundError(
            f"references/edm not found at {_EDM_REPO}. "
            f"Run scripts/setup_references.sh first."
        )
    p = str(_EDM_REPO)
    if p not in sys.path:
        sys.path.insert(0, p)


class EDMDenoiser:
    """
    Exposes:
      - denoise(x, sigma, class_labels)  -> D_theta(x; sigma, c), an x0 estimate
      - score(x, sigma, class_labels)    -> grad_x log p_sigma(x | c)
                                             = (D_theta(x;sigma,c) - x) / sigma^2
      - one_hot(class_idx, batch_size)   -> conditioning vector for a class
      - sample(...)                      -> EDM Heun sampler (Algorithm 2)

    All shapes follow the underlying net's convention: x is [B, C, H, W],
    sigma is a scalar or [B] tensor, class_labels is [B, label_dim] one-hot
    (or None for the unconditional model).
    """

    def __init__(self, checkpoint_path: Path, device: str = "cuda", use_ema: bool = True):
        _ensure_edm_on_path()
        import dnnlib  # noqa: F401  -- required for unpickling the network
        from torch_utils import persistence  # noqa: F401

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"EDM checkpoint not found at {checkpoint_path}. "
                f"Run scripts/download_edm_checkpoint.py first."
            )
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)
        key = "ema" if (use_ema and "ema" in data) else "model"
        self.net = data[key].to(self.device).eval()

        self.label_dim = int(self.net.label_dim)
        self.img_resolution = int(self.net.img_resolution)
        self.img_channels = int(self.net.img_channels)
        self.sigma_min = float(self.net.sigma_min)
        self.sigma_max = float(self.net.sigma_max)

    def one_hot(self, class_idx: int, batch_size: int = 1) -> Optional[torch.Tensor]:
        if self.label_dim == 0:
            return None
        labels = torch.zeros(batch_size, self.label_dim, device=self.device)
        labels[:, class_idx] = 1
        return labels

    def _sigma_tensor(self, sigma, x: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(sigma):
            if sigma.dim() == 0:
                sigma = sigma.expand(x.shape[0])
            return sigma.to(x.device, x.dtype)
        return torch.full((x.shape[0],), float(sigma), device=x.device, dtype=x.dtype)

    def denoise(self, x: torch.Tensor, sigma, class_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        sigma_t = self._sigma_tensor(sigma, x)
        return self.net(x, sigma_t, class_labels)

    def score(self, x: torch.Tensor, sigma, class_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        sigma_t = self._sigma_tensor(sigma, x)
        denoised = self.net(x, sigma_t, class_labels)
        sigma_bc = sigma_t.view(-1, *([1] * (x.dim() - 1)))
        return (denoised - x) / (sigma_bc ** 2)

    @torch.no_grad()
    def run_sampler(
        self,
        latents: torch.Tensor,
        class_labels: Optional[torch.Tensor],
        num_steps: int = 18,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        rho: float = 7.0,
        seed: int = 0,
    ) -> torch.Tensor:
        """Run the EDM Heun sampler (Algorithm 2) from an explicit initial
        latent (already scaled by sigma_max, i.e. pure noise) and explicit
        (possibly blended/non-one-hot) class_labels. This is the low-level
        primitive every path constructor builds on -- unlike `sample()`, it
        never draws its own randomness for the *starting point*, only for
        the stochastic churn term (S_churn=0 by default -> deterministic).
        """
        _ensure_edm_on_path()
        from generate import edm_sampler, StackedRandomGenerator

        rnd = StackedRandomGenerator(self.device, list(range(seed, seed + latents.shape[0])))
        kwargs = dict(num_steps=num_steps, rho=rho)
        if sigma_min is not None:
            kwargs["sigma_min"] = sigma_min
        if sigma_max is not None:
            kwargs["sigma_max"] = sigma_max
        images = edm_sampler(self.net, latents, class_labels, randn_like=rnd.randn_like, **kwargs)
        return images.clamp(-1, 1).to(torch.float32)

    @torch.no_grad()
    def denoise_to_clean(
        self,
        x_sigma: torch.Tensor,
        sigma: float,
        class_labels: Optional[torch.Tensor],
        num_steps: int = 18,
        rho: float = 7.0,
        S_churn: float = 0.0,
    ) -> torch.Tensor:
        """Given a noisy point x_sigma *already at* noise level `sigma` (a
        point partway through the reverse process, as produced by a path
        constructor operating in noise space), finish the reverse ODE from
        sigma down to 0 to get an actual realistic image -- rather than a
        single-step Tweedie estimate, which is blurry at high sigma. This is
        what evaluators actually classify.

        NB: we do NOT call edm.generate.edm_sampler here -- that function
        always treats its `latents` argument as unit N(0,1) noise and
        rescales it by sigma_max internally (`x_next = latents * t_steps[0]`).
        We already have an actually-noised state, so this is a hand-rolled
        resume-from-x_sigma variant of the same Heun integrator (Algorithm 2,
        Karras et al. 2022), deterministic by default (S_churn=0).
        """
        # EDM's own sampler defaults sigma_min to 0.002 and clamps via
        # max(0.002, net.sigma_min) -- net.sigma_min is often exactly 0.0 for
        # VP-preconditioned checkpoints, which would put a literal 0 in the
        # t_steps schedule and divide-by-zero in d_cur below. Match their floor.
        sigma_min = max(0.002, self.sigma_min)
        sigma_max = min(float(sigma), self.sigma_max if self.sigma_max > 0 else float(sigma))
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=x_sigma.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
                   * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

        x_next = x_sigma.to(torch.float64)
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            gamma = min(S_churn / num_steps, 2 ** 0.5 - 1) if S_churn > 0 else 0.0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2) ** 0.5 * torch.randn_like(x_cur) if gamma > 0 else x_cur

            denoised = self.net(x_hat.to(torch.float32), t_hat.to(torch.float32), class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < num_steps - 1:
                denoised = self.net(x_next.to(torch.float32), t_next.to(torch.float32), class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next.clamp(-1, 1).to(torch.float32)

    @torch.no_grad()
    def sample(
        self,
        class_idx: Optional[int],
        batch_size: int = 1,
        num_steps: int = 18,
        seed: int = 0,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        rho: float = 7.0,
    ) -> torch.Tensor:
        """Draw `batch_size` fresh samples of `class_idx` using the EDM Heun
        sampler (Algorithm 2 from Karras et al.). Returns images in [-1, 1]."""
        _ensure_edm_on_path()
        from generate import StackedRandomGenerator  # references/edm/generate.py

        rnd = StackedRandomGenerator(self.device, list(range(seed, seed + batch_size)))
        latents = rnd.randn(
            [batch_size, self.img_channels, self.img_resolution, self.img_resolution],
            device=self.device,
        )
        class_labels = self.one_hot(class_idx, batch_size) if class_idx is not None else None
        return self.run_sampler(
            latents, class_labels, num_steps=num_steps,
            sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, seed=seed,
        )
