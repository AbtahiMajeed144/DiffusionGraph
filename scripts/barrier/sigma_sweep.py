"""
Stage-0 gate repair, step 1b: does a genuinely-realistic *mixed* positive exist?

The user's manual ranking showed neither G3 (slerp@0.5) nor G5 (blended-condition)
reliably beats G4a pixel blends in realism. G3 smears because sigma=0.5 is too low
(both images' pixel structure survives). This sweeps the slerp noise level: at
higher sigma the reverse ODE has freedom to commit to one coherent mode. We want
the sigma (if any) where midpoints look clearly more realistic than G4a.

Rows: G1_A, G1_B (real refs); slerp-decoded midpoints at each sigma; G4a pixel blend.
Same 10 cross-class pairs as inspect_groups.py (seed 0).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from diffusiongraph.config import get_profile, RESULTS_DIR
from diffusiongraph.utils.edm_loader import EDMDenoiser
from diffusiongraph.data.cifar10 import CIFAR10Canonical
from diffusiongraph.paths.slerp_noise import slerp
from diffusiongraph.paths.base import forward_diffuse
from diffusiongraph.barrier.groups import random_cross_class_pairs

CLASSES = ["plane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def _img(x):
    return ((x.detach().cpu().float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="local_poc")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--sigmas", default="0.5,1.0,2.0,4.0,8.0")
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = get_profile(args.profile)
    device = cfg.device
    den = EDMDenoiser(cfg.edm_checkpoint_uncond, device=device)
    test = CIFAR10Canonical(train=False, download=True)
    sweep = [float(s) for s in args.sigmas.split(",")]

    pairs = random_cross_class_pairs(args.n, seed=args.seed)
    g = torch.Generator().manual_seed(args.seed)
    a_imgs, b_imgs = [], []
    for (ca, cb) in pairs:
        ia = test.indices_for_class(ca)[int(torch.randint(len(test.indices_for_class(ca)), (1,), generator=g))]
        ib = test.indices_for_class(cb)[int(torch.randint(len(test.indices_for_class(cb)), (1,), generator=g))]
        a_imgs.append(test[ia][0]); b_imgs.append(test[ib][0])
    xa = torch.stack(a_imgs).to(device); xb = torch.stack(b_imgs).to(device)

    rows = [("G1_A", xa.cpu()), ("G1_B", xb.cpu())]
    for sig in sweep:
        a_s = forward_diffuse(xa, sig, generator=g)
        b_s = forward_diffuse(xb, sig, generator=g)
        mid = slerp(a_s, b_s, 0.5)
        dec = den.denoise_to_clean(mid, sig, class_labels=None, num_steps=args.steps).cpu()
        rows.append((f"slerp s={sig:g}", dec))
    rows.append(("G4a blend", (0.5 * xa + 0.5 * xb).clamp(-1, 1).cpu()))

    fig, axes = plt.subplots(len(rows), args.n, figsize=(1.4 * args.n, 1.5 * len(rows)))
    for r, (label, imgs) in enumerate(rows):
        for c in range(args.n):
            ax = axes[r, c]
            ax.imshow(_img(imgs[c])); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=9, rotation=90, labelpad=2)
            if r == 0:
                ax.set_title(f"{CLASSES[pairs[c][0]]}/{CLASSES[pairs[c][1]]}", fontsize=8)
    fig.suptitle("slerp midpoint realism vs noise level sigma (want a row that clearly beats G4a)", fontsize=11)
    fig.tight_layout()
    out = RESULTS_DIR / "barrier" / "stage0" / args.profile / "sigma_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
