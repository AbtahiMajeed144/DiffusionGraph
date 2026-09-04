"""
P5 falsifier (GRAPH_BARRIER_EXPERIMENT.md 7 / 8): the forward-backward transition
matrix T_sigma (Sclocchi et al. PNAS 2025, arXiv:2402.16991) and its comparison
to tau*.

T_sigma[A,B] = P(class B | noise a class-A real image to sigma, denoise the full
reverse ODE with the UNCONDITIONAL model, classify the output). Below the phase
transition T~=I (class preserved); above it the class scrambles. Informative
structure lives in a band around the transition, so we sweep >=3 sigma (design 7).

P5: if tau* is merely a re-derivation of this cheap published quantity
(Spearman rho > 0.9 on the 45 off-diagonal entries), the barrier machinery adds
nothing. If rho < 0.8, tau* measures something T_sigma does not.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from diffusiongraph.config import get_profile, RESULTS_DIR, CHECKPOINTS_DIR
from diffusiongraph.utils.edm_loader import EDMDenoiser
from diffusiongraph.data.cifar10 import CIFAR10Canonical
from diffusiongraph.paths.base import forward_diffuse
from diffusiongraph.barrier.groups import _balanced_real

CLASSES = ["plane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def _load_classifier(arch, device):
    from diffusiongraph.models.classifiers import ARCHITECTURES
    m = ARCHITECTURES[arch]()
    st = torch.load(CHECKPOINTS_DIR / f"{arch}_cifar10.pt", map_location=device)
    m.load_state_dict(st["model_state_dict"] if "model_state_dict" in st else st)
    return m.to(device).eval()


@torch.no_grad()
def transition_matrix(den, clf, per_class_imgs, sigma, steps, device, seed=0):
    """T[A,B] = P(argmax classifier(denoise(A+sigma*eps)) == B), rows sum to 1."""
    g = torch.Generator().manual_seed(seed)
    T = np.zeros((10, 10))
    for A in range(10):
        x = per_class_imgs[A].to(device)
        xt = forward_diffuse(x, sigma, generator=g)
        xhat = den.denoise_to_clean(xt, sigma, class_labels=None, num_steps=steps)
        pred = clf(xhat).argmax(1).cpu().numpy()
        for b in pred:
            T[A, int(b)] += 1
        T[A] /= max(1, len(pred))
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="local_poc")
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--sigmas", default="0.5,1.0,2.0,4.0,8.0")
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--tau", default=None, help="path to tau.npy to compare against (P5)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = get_profile(args.profile)
    device = cfg.device
    arch = next((n for n in cfg.evaluator_names if n != "clip_zeroshot"), "resnet18")
    den = EDMDenoiser(cfg.edm_checkpoint_uncond, device=device)
    clf = _load_classifier(arch, device)
    test = CIFAR10Canonical(train=False, download=True)
    per_class = {A: torch.stack([test[i][0] for i in test.indices_for_class(A)[:args.per_class]])
                 for A in range(10)}
    sweep = [float(s) for s in args.sigmas.split(",")]

    tau = None
    tau_path = args.tau or (RESULTS_DIR / "barrier" / "tau" / args.profile / "tau.npy")
    if Path(tau_path).exists():
        tau = np.load(tau_path)
        print(f"Loaded tau* from {tau_path} for P5 comparison")
    else:
        print(f"(no tau.npy at {tau_path}; computing T_sigma only)")

    iu = np.triu_indices(10, 1)
    from scipy.stats import spearmanr
    out_dir = RESULTS_DIR / "barrier" / "tsigma" / args.profile
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for sig in sweep:
        T = transition_matrix(den, clf, per_class, sig, args.steps, device, seed=args.seed)
        Tsym = 0.5 * (T + T.T)
        diag_retention = float(np.mean(np.diag(T)))  # class-preservation (T~=I below transition)
        row = {"diag_retention": diag_retention}
        if tau is not None:
            # tau* HIGHER = easier to connect ~ MORE transition mass. Compare on off-diag.
            m = ~np.isnan(tau[iu])
            rho = spearmanr(tau[iu][m], Tsym[iu][m]).correlation if m.sum() > 2 else float("nan")
            row["spearman_tau_vs_Tsym"] = float(rho)
        results[f"s{sig:g}"] = row
        np.save(out_dir / f"Tsigma_s{sig:g}.npy", T)
        msg = f"  sigma={sig:g}: class-retention(diag)={diag_retention:.2f}"
        if tau is not None:
            msg += f"  Spearman(tau*, T_sym)={row['spearman_tau_vs_Tsym']:+.3f}"
        print(msg)

    (out_dir / "p5.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_dir}")
    if tau is not None:
        best = max((r.get("spearman_tau_vs_Tsym", float('nan')) for r in results.values()),
                   key=lambda x: -1 if np.isnan(x) else x)
        print(f"P5 read: max Spearman(tau*, T_sym) over sigmas = {best:+.3f}  "
              f"(>0.9 => tau* re-derives T_sigma, barrier adds nothing; <0.8 => tau* is distinct)")
        print("NOTE: if tau* is near-uniform (no structure), a low rho means 'nothing to compare',")
        print("not 'tau* adds signal' -- read alongside the shuffled-R null.")


if __name__ == "__main__":
    main()
