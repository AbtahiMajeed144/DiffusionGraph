"""
Stage-0 validation groups for GRAPH_BARRIER_EXPERIMENT.md §0.1.

  G1 real       : CIFAR-10 TEST images (EDM never saw the test split)
  G2 synthetic  : unconditional EDM samples (full reverse ODE)
  G3 interp     : slerp-in-noise midpoints (t=0.5, sigma), decoded -- manifold-aware
  G4 degraded   : (a) pixel-space linear blends at t=0.5 (double-exposure),
                  (b) Gaussian blur, (c) additive Gaussian noise SNR-matched to (b)

Plus a `graded_blur` set at sigma_blur in {0,1,2,3} for §0.3 criterion 4
(monotone response to graded damage). All images are canonical [-1,1]
[3,32,32]. Every image carries a provenance tag.

G3 uses slerp only (cheap). Tangential-geodesic midpoints would need the
150-step optimizer per pair (expensive) and are NOT required for the
decisive criterion 2 -- slerp midpoints are already manifold-aware (decoded
through the full reverse ODE), which is the property being tested against
pixel double-exposures. Add tangential later only if the gate is borderline.
"""
from __future__ import annotations
import math
from typing import Optional

import numpy as np
import torch

from diffusiongraph.data.cifar10 import CIFAR10Canonical
from diffusiongraph.paths.slerp_noise import slerp
from diffusiongraph.paths.base import forward_diffuse


def _balanced_real(dataset: CIFAR10Canonical, n: int, offset: int = 0) -> torch.Tensor:
    """EXACTLY n class-balanced real images, deterministic, starting at `offset`
    within each class's index list (offset carves out disjoint sets). Uses
    ceil(n/10) per class then round-robin interleaves and truncates to n, so the
    count is always exact -- `per_class = n//10` silently returned fewer than n
    when n was not a multiple of 10 (crashed the G4b loop at n=500)."""
    per_class = math.ceil(n / 10)
    cols = []
    for c in range(10):
        idxs = dataset.indices_for_class(c)[offset:offset + per_class]
        cols.append([dataset[i][0] for i in idxs])
    imgs = []
    for j in range(per_class):
        for c in range(10):
            if j < len(cols[c]):
                imgs.append(cols[c][j])
                if len(imgs) >= n:
                    break
        if len(imgs) >= n:
            break
    if len(imgs) < n:
        raise ValueError(f"_balanced_real: only {len(imgs)} images for n={n}, offset={offset}")
    return torch.stack(imgs[:n], dim=0)


def _gaussian_blur(x: torch.Tensor, sigma_blur: float) -> torch.Tensor:
    if sigma_blur <= 0:
        return x.clone()
    radius = max(1, int(math.ceil(3 * sigma_blur)))
    ks = 2 * radius + 1
    coords = torch.arange(ks, dtype=torch.float32, device=x.device) - radius
    g = torch.exp(-(coords ** 2) / (2 * sigma_blur ** 2))
    g = (g / g.sum()).to(x.dtype)
    kx = g.view(1, 1, 1, ks).repeat(3, 1, 1, 1)
    ky = g.view(1, 1, ks, 1).repeat(3, 1, 1, 1)
    x = torch.nn.functional.conv2d(x, kx, padding=(0, radius), groups=3)
    x = torch.nn.functional.conv2d(x, ky, padding=(radius, 0), groups=3)
    return x.clamp(-1, 1)


@torch.no_grad()
def condition_midpoints(cond_denoiser, class_pairs, seed=0, decode_steps=18, device="cuda"):
    """G5: linear class-condition midpoints. For each (class_a, class_b) pair,
    generate a single image under the blended one-hot condition
    c = 0.5*onehot(A) + 0.5*onehot(B) from a fresh latent, full reverse ODE.

    Unlike G3 (slerp of two real *images*, which can decode to an off-manifold
    smear) these are drawn from the model's own conditional manifold at a
    between-classes condition -- the genuinely-realistic class-mixed positive.
    Requires the CONDITIONAL checkpoint (label_dim=10). Returns [N,3,32,32] cpu."""
    assert cond_denoiser.label_dim > 0, "condition_midpoints needs the conditional checkpoint"
    imgs = []
    for i, (ca, cb) in enumerate(class_pairs):
        oa = cond_denoiser.one_hot(int(ca), 1)
        ob = cond_denoiser.one_hot(int(cb), 1)
        c_mid = 0.5 * oa + 0.5 * ob
        gen = torch.Generator(device="cpu").manual_seed(seed + i)
        latent = torch.randn(1, cond_denoiser.img_channels, cond_denoiser.img_resolution,
                             cond_denoiser.img_resolution, generator=gen).to(device)
        img = cond_denoiser.run_sampler(latent, c_mid, num_steps=decode_steps, seed=seed + i)
        imgs.append(img[0].cpu())
    return torch.stack(imgs, dim=0)


def random_cross_class_pairs(n, seed=0):
    """n deterministic (class_a, class_b) pairs with class_a != class_b."""
    g = torch.Generator().manual_seed(seed)
    pairs = []
    for _ in range(n):
        ca, cb = torch.randperm(10, generator=g)[:2].tolist()
        pairs.append((ca, cb))
    return pairs


@torch.no_grad()
def build_validation_groups(
    denoiser,
    n_per_group: int = 500,
    sigma_interp: float = 0.5,
    seed: int = 0,
    device: str = "cuda",
    decode_steps: int = 18,
) -> dict:
    """Returns {group_name: images[N,3,32,32]} plus 'provenance' (list of tags)
    and 'graded_blur' {sigma_blur: images}."""
    g = torch.Generator().manual_seed(seed)
    test = CIFAR10Canonical(train=False, download=True)
    train = CIFAR10Canonical(train=True, download=True)  # real reference bank for feature-kNN

    groups = {}

    # G1 real (test split)
    groups["G1_real"] = _balanced_real(test, n_per_group)

    # Reference bank for feature-kNN: real TRAIN images, disjoint from G1 (test).
    groups["_ref_bank"] = _balanced_real(train, min(5000, 10 * (n_per_group // 10 + 400)))

    # G2 synthetic (unconditional EDM samples)
    syn = []
    made = 0
    bs = 100
    s = seed
    while made < n_per_group:
        b = min(bs, n_per_group - made)
        syn.append(denoiser.sample(class_idx=None, batch_size=b, num_steps=decode_steps, seed=s))
        made += b
        s += b
    groups["G2_synth"] = torch.cat(syn, dim=0)[:n_per_group].cpu()

    # Random cross-class real endpoint pairs for G3 / G4a
    def _rand_pairs(m):
        a_imgs, b_imgs = [], []
        for _ in range(m):
            ca, cb = torch.randperm(10, generator=g)[:2].tolist()
            ia = test.indices_for_class(ca)[int(torch.randint(len(test.indices_for_class(ca)), (1,), generator=g))]
            ib = test.indices_for_class(cb)[int(torch.randint(len(test.indices_for_class(cb)), (1,), generator=g))]
            a_imgs.append(test[ia][0]); b_imgs.append(test[ib][0])
        return torch.stack(a_imgs), torch.stack(b_imgs)

    xa, xb = _rand_pairs(n_per_group)

    # G3 slerp midpoints (manifold-aware): forward-diffuse endpoints to sigma,
    # slerp at t=0.5, decode the full reverse ODE.
    mids = []
    for i in range(0, n_per_group, 64):
        a = xa[i:i + 64].to(device); b = xb[i:i + 64].to(device)
        a_s = forward_diffuse(a, sigma_interp, generator=g)
        b_s = forward_diffuse(b, sigma_interp, generator=g)
        mid = slerp(a_s, b_s, 0.5)
        dec = denoiser.denoise_to_clean(mid, sigma_interp, class_labels=None, num_steps=decode_steps)
        mids.append(dec.cpu())
    groups["G3_interp"] = torch.cat(mids, dim=0)

    # G4 degraded, ~1/3 each
    n_a = n_per_group // 3
    n_b = n_per_group // 3
    n_c = n_per_group - n_a - n_b
    xa2, xb2 = _rand_pairs(n_a)
    groups["G4a_blend"] = (0.5 * xa2 + 0.5 * xb2).clamp(-1, 1)                      # pixel double-exposure
    real_b = _balanced_real(test, n_b, offset=n_per_group // 10)                   # disjoint-ish reals
    m_b = real_b.shape[0]
    blur_sigmas = np.array([1, 2, 3])[np.arange(m_b) % 3]
    groups["G4b_blur"] = torch.stack([_gaussian_blur(real_b[i:i+1], float(blur_sigmas[i]))[0] for i in range(m_b)])
    real_c = _balanced_real(test, n_c, offset=2 * (n_per_group // 10))
    # SNR-match noise to blur sigma=2 damage (approx via matching MSE)
    ref_blur = _gaussian_blur(real_c, 2.0)
    target_mse = (real_c - ref_blur).pow(2).mean().item()
    noise_std = math.sqrt(max(target_mse, 1e-6))
    groups["G4c_noise"] = (real_c + noise_std * torch.randn(real_c.shape, generator=g)).clamp(-1, 1)

    # graded_blur for criterion 4 (same real images at each level)
    base = _balanced_real(test, min(150, n_per_group), offset=3 * (n_per_group // 10))
    groups["graded_blur"] = {sb: (base.clone() if sb == 0 else torch.stack(
        [_gaussian_blur(base[i:i+1], float(sb))[0] for i in range(base.shape[0])]
    )) for sb in [0, 1, 2, 3]}

    return groups
