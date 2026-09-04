"""
Realism-score candidates for GRAPH_BARRIER_EXPERIMENT.md Stage 0 (the HARD
GATE). Each candidate maps a batch of clean images x0 in [-1,1]^{3x32x32}
to a scalar R per image, higher = more realistic.

Graham reconstruction is deliberately NOT implemented (~10^7 NFE, dropped
for cost per the design's "cheapest that passes" gate). Candidates here are
all efficient:

  - scoped         : SCOPED (arXiv:2510.01456). T = sign(sum s)*||s||^2 /
                     (-tr(J_s) + eps), evaluated at low noise; R = -T.
  - score_norm     : raw score norm (SCOPED numerator alone). NEGATIVE
                     CONTROL, predicted to fail (vanishes at modes AND
                     saddles).
  - feat_knn_*     : Feature kNN (Sun et al. 2022): negative distance to the
                     k-th nearest real-image feature. Non-diffusion control.
  - eigenscore     : NOT yet implemented (arXiv:2510.07206). Held as the
                     designated fallback if SCOPED fails §0.3 criterion 2;
                     its Jacobian-free subspace iteration is the one heavier
                     routine, deferred until the gate says it's needed.

The SCOPED Jacobian trace tr(J_s) = divergence of the score is estimated
with Hutchinson's single-sided estimator E[v^T J v] (Rademacher v) -- NOTE
this is a DIFFERENT quantity from utils/jvp.hutchinson_trace, which returns
tr(J^T J) = ||Jv||^2 (Frobenius). We reuse utils.jvp.score_jvp for the JVP
and add the correct single-sided estimator here.
"""
from __future__ import annotations
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from diffusiongraph.utils.jvp import score_jvp


# ---------------------------------------------------------------------------
# Score-Jacobian divergence (SCOPED denominator)
# ---------------------------------------------------------------------------

def hutchinson_divergence(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    num_probes: int = 4,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Single-sided Hutchinson estimate of tr(J_score(x)) = div s(x):
        tr(J) ~= (1/P) sum_p  v_p . (J v_p),   v_p Rademacher.
    Returns [B]. Detached (Stage 0 needs no outer gradient)."""
    b = x.shape[0]
    total = torch.zeros(b, device=x.device)
    for _ in range(num_probes):
        v = torch.randint(0, 2, x.shape, generator=generator, device=x.device, dtype=x.dtype) * 2 - 1
        jv = score_jvp(score_fn, x, v)
        total = total + (v * jv).flatten(1).sum(dim=1).detach()
    return total / num_probes


# ---------------------------------------------------------------------------
# SCOPED and raw score norm (diffusion candidates)
# ---------------------------------------------------------------------------

def _forward_diffuse(x0: torch.Tensor, sigma: float, generator: Optional[torch.Generator]) -> torch.Tensor:
    eps = torch.randn(x0.shape, generator=generator, dtype=torch.float32).to(x0.device, x0.dtype)
    return x0 + sigma * eps


@torch.enable_grad()
def scoped_and_norm(
    denoiser,
    x0: torch.Tensor,
    sigma: float,
    num_noise: int = 4,
    num_probes: int = 4,
    eps: float = 1e-3,
    generator: Optional[torch.Generator] = None,
):
    """Compute SCOPED realism (-T) and raw-score-norm realism (-||s||^2) for
    a batch of clean images, averaged over `num_noise` forward-diffusion
    draws at noise level `sigma`. Runs on the UNCONDITIONAL model (score of
    the data distribution, class_labels=None).

    Returns dict of [B] numpy arrays:
      scoped        = -T                 (higher = more realistic)
      score_norm    = -mean ||s||^2      (negative control)
      _snorm2, _div = raw components, for diagnostics
    """
    def score_fn(x):
        return denoiser.score(x, sigma, class_labels=None)

    B = x0.shape[0]
    snorm2_acc = torch.zeros(B, device=x0.device)
    ssum_acc = torch.zeros(B, device=x0.device)
    div_acc = torch.zeros(B, device=x0.device)

    for _ in range(num_noise):
        x_sigma = _forward_diffuse(x0, sigma, generator)
        with torch.no_grad():
            s = denoiser.score(x_sigma, sigma, class_labels=None)     # [B,C,H,W]
            snorm2_acc += s.flatten(1).pow(2).sum(dim=1).detach()
            ssum_acc += s.flatten(1).sum(dim=1).detach()
        div_acc += hutchinson_divergence(score_fn, x_sigma, num_probes, generator)  # tr(J)

    # Average the COMPONENTS over noise draws (linear, stable), THEN form the
    # ratio ONCE. Averaging T itself is explosion-prone: any single draw whose
    # noisy tr(J) estimate flips positive drives denom -> eps and T -> ~1e7,
    # corrupting the mean. Component-averaging also tightens the tr(J) estimate
    # (num_noise x num_probes samples) before it hits the denominator.
    snorm2 = snorm2_acc / num_noise
    div = div_acc / num_noise
    sign = torch.sign(ssum_acc)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    # denom = -tr(J) + eps, floored positive: near a mode tr(J)<0 so -tr(J) is
    # large -> T small (realistic); OOD -> -tr(J) small, ||s||^2 large -> T large.
    denom = torch.clamp(-div, min=0.0) + eps
    T = (sign * snorm2 / denom).cpu().numpy()
    snorm2 = snorm2.cpu().numpy()
    div = div.cpu().numpy()
    return {
        "scoped": -T,
        "score_norm": -snorm2,
        "_snorm2": snorm2,
        "_div": div,
    }


# ---------------------------------------------------------------------------
# EigenScore (arXiv:2510.07206, ICLR 2026) -- the near-OOD specialist / fallback
# ---------------------------------------------------------------------------
# Score = sum of top-K eigenvalues of the posterior covariance
#   Sigma_t = Cov[x0 | x_t] = sigma^2 * dD/dx   (denoiser Jacobian, symmetric PSD),
# estimated by FORWARD-ONLY subspace iteration with central-difference JVPs:
#   (J_D v) ~= (D(x+cv) - D(x-cv)) / (2c).
# OOD inputs inflate the posterior covariance -> LARGER eigenvalues -> lower
# realism. Forward-only (no autograd graph) so it is memory-light vs SCOPED.

@torch.no_grad()
def eigenscore_mbar(
    denoiser,
    x0: torch.Tensor,
    sigma: float,
    K: int = 3,
    n_iter: int = 5,
    n_real: int = 2,
    c: float = 1e-2,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """m_bar = sum of the top-K eigenvalues of Sigma = sigma^2 * dD/dx, averaged
    over `n_real` forward-diffusion realizations. Returns [B]."""
    B = x0.shape[0]
    device = x0.device
    shape = x0.shape[1:]
    Ddim = int(np.prod([int(s) for s in shape]))

    def apply_JD(x_sigma, Q):
        # Q: [B, Ddim, K] orthonormal columns -> Y = J_D Q via central diff, [B, Ddim, K]
        V = Q.permute(0, 2, 1).reshape(B * K, *shape)
        x_rep = x_sigma.repeat_interleave(K, dim=0)
        dp = denoiser.denoise(x_rep + c * V, sigma)
        dm = denoiser.denoise(x_rep - c * V, sigma)
        return ((dp - dm) / (2 * c)).reshape(B, K, Ddim).permute(0, 2, 1)

    acc = torch.zeros(B, device=device)
    for _ in range(n_real):
        eps = torch.randn(x0.shape, generator=generator, dtype=torch.float32).to(device, x0.dtype)
        x_sigma = x0 + sigma * eps
        Q = torch.randn(B, Ddim, K, generator=generator, dtype=torch.float32).to(device)
        Q, _ = torch.linalg.qr(Q)                       # orthonormal init, per image
        for _ in range(n_iter):
            Y = apply_JD(x_sigma, Q)
            Q, _ = torch.linalg.qr(Y)                   # re-orthonormalize (subspace iteration)
        JQ = apply_JD(x_sigma, Q)                       # [B, Ddim, K]
        rayleigh = (Q * JQ).sum(dim=1)                  # [B, K] = v_k . (J_D v_k)
        lam = (sigma ** 2) * rayleigh                   # eigenvalues of Sigma
        acc += lam.sum(dim=1)                           # sum of top-K
    return acc / n_real


# ---------------------------------------------------------------------------
# Feature kNN (non-diffusion control)
# ---------------------------------------------------------------------------

class ResnetFeatureExtractor:
    """Penultimate (pre-linear) features of a trained CIFAR classifier, via a
    forward-pre-hook on the final Linear -- no change to classifiers.py."""

    def __init__(self, arch_name: str, checkpoint_path, device: str = "cuda"):
        from diffusiongraph.models.classifiers import ARCHITECTURES
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        model = ARCHITECTURES[arch_name]()
        state = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._feat = {}
        # ResNet's classifier head is `.linear`; ViTSmall's is `.fc` (a Sequential).
        head = getattr(self.model, "linear", None) or getattr(self.model, "fc", None)
        head.register_forward_pre_hook(lambda m, inp: self._feat.__setitem__("f", inp[0].detach()))

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> torch.Tensor:
        self.model(images.to(self.device))
        return self._feat["f"]


class ClipFeatureExtractor:
    def __init__(self, clip_zeroshot):
        self.clip = clip_zeroshot

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> torch.Tensor:
        x = self.clip._preprocess(images.to(self.clip.device))
        x = x.to(next(self.clip.model.parameters()).dtype)
        return self.clip.model.encode_image(x).float()


def build_feature_bank(extractor, images: torch.Tensor, batch: int = 256) -> torch.Tensor:
    feats = []
    for i in range(0, images.shape[0], batch):
        f = extractor.features(images[i:i + batch])
        feats.append(F.normalize(f, dim=-1).cpu())
    return torch.cat(feats, dim=0)  # [N_ref, D], L2-normalized, on CPU


def feature_knn_realism(extractor, bank: torch.Tensor, images: torch.Tensor, k: int = 50, batch: int = 256) -> np.ndarray:
    """R = -(distance to k-th nearest neighbour in the real-image feature
    bank). Cosine distance = 1 - cos-sim on L2-normalized features."""
    bank_dev = bank.to(images.device if torch.is_tensor(images) else "cpu")
    out = []
    for i in range(0, images.shape[0], batch):
        f = F.normalize(extractor.features(images[i:i + batch]), dim=-1)  # [b,D] on device
        sims = f @ bank_dev.to(f.device).T                                # [b, N_ref]
        # k-th largest similarity -> distance = 1 - that similarity
        kth_sim = sims.topk(k, dim=1).values[:, -1]
        out.append((1.0 - kth_sim).cpu().numpy())
    dist = np.concatenate(out)
    return -dist  # higher R = closer to real manifold


# ---------------------------------------------------------------------------
# AUROC (rank-based Mann-Whitney; higher score => positive class)
# ---------------------------------------------------------------------------

def auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(neg_scores, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    try:
        from scipy.stats import rankdata
        ranks = rankdata(np.concatenate([pos, neg]))  # average ranks for ties
    except Exception:
        order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
        ranks = np.empty(n_pos + n_neg, dtype=float)
        ranks[order] = np.arange(1, n_pos + n_neg + 1)
    sum_ranks_pos = ranks[:n_pos].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
