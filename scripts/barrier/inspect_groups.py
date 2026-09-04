"""
Stage-0 GATE REPAIR, step 1: visual inspection of the decisive-test groups.

The decisive criterion (GRAPH_BARRIER_EXPERIMENT.md 0.3 #2) is AUROC(G3 vs G4a).
If G3 (slerp-of-two-real-images midpoints, decoded) are off-manifold smears while
G4a (pixel double-exposures) retain real texture, then a realism score that ranks
G4a above G3 is CORRECT and the gate is testing a broken positive. This dumps a
labelled montage of, for the SAME cross-class pairs:

    row G1  real class-A / class-B endpoints (reference for "realistic")
    row G3  slerp-in-noise midpoint, decoded (uncond)         -- current positive
    row G4a pixel-space 0.5/0.5 blend of the two reals        -- the negative
    row G5  linear class-condition midpoint, sampled (cond)   -- proposed positive

Free (no scoring). Decides whether the repair (swap G3->G5) is warranted.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from diffusiongraph.config import get_profile, RESULTS_DIR
from diffusiongraph.utils.edm_loader import EDMDenoiser
from diffusiongraph.data.cifar10 import CIFAR10Canonical
from diffusiongraph.paths.slerp_noise import slerp
from diffusiongraph.paths.base import forward_diffuse
from diffusiongraph.barrier.groups import condition_midpoints, random_cross_class_pairs

CLASSES = ["plane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def _to_img(x):
    return ((x.detach().cpu().float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="local_poc")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = get_profile(args.profile)
    device = cfg.device
    uncond = EDMDenoiser(cfg.edm_checkpoint_uncond, device=device)
    cond = EDMDenoiser(cfg.edm_checkpoint_cond, device=device)
    test = CIFAR10Canonical(train=False, download=True)

    pairs = random_cross_class_pairs(args.n, seed=args.seed)
    g = torch.Generator().manual_seed(args.seed)

    a_imgs, b_imgs = [], []
    for (ca, cb) in pairs:
        ia = test.indices_for_class(ca)[int(torch.randint(len(test.indices_for_class(ca)), (1,), generator=g))]
        ib = test.indices_for_class(cb)[int(torch.randint(len(test.indices_for_class(cb)), (1,), generator=g))]
        a_imgs.append(test[ia][0]); b_imgs.append(test[ib][0])
    xa = torch.stack(a_imgs); xb = torch.stack(b_imgs)

    # G3 slerp-noise midpoint decoded (uncond) -- exactly build_validation_groups' recipe
    a_s = forward_diffuse(xa.to(device), args.sigma, generator=g)
    b_s = forward_diffuse(xb.to(device), args.sigma, generator=g)
    mid = slerp(a_s, b_s, 0.5)
    g3 = uncond.denoise_to_clean(mid, args.sigma, class_labels=None, num_steps=args.steps).cpu()

    # G4a pixel blend
    g4a = (0.5 * xa + 0.5 * xb).clamp(-1, 1)

    # G5 linear-condition midpoint (cond)
    g5 = condition_midpoints(cond, pairs, seed=args.seed, decode_steps=args.steps, device=device)

    rows = [("G1_A real", xa), ("G1_B real", xb), ("G3 slerp-mid", g3),
            ("G4a pix-blend", g4a), ("G5 cond-mid", g5)]
    fig, axes = plt.subplots(len(rows), args.n, figsize=(1.4 * args.n, 1.5 * len(rows)))
    for r, (label, imgs) in enumerate(rows):
        for c in range(args.n):
            ax = axes[r, c]
            ax.imshow(_to_img(imgs[c])); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=9, rotation=90, labelpad=2)
            if r == 0:
                ax.set_title(f"{CLASSES[pairs[c][0]]}/{CLASSES[pairs[c][1]]}", fontsize=8)
    fig.suptitle("Stage-0 group inspection: G3 (current positive) vs G4a (negative) vs G5 (proposed positive)", fontsize=11)
    fig.tight_layout()
    out = RESULTS_DIR / "barrier" / "stage0" / args.profile / "group_inspection.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Saved montage: {out}")
    print("Look at rows G3 vs G4a vs G5. If G3 is smeared/off-manifold and G5 is a clean")
    print("single coherent image, G3 is a broken positive and the gate should use G5.")


if __name__ == "__main__":
    main()
