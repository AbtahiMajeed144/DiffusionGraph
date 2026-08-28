"""
Generate and SAVE the actual images along a specific path, for visual
inspection. The resume cache (utils/checkpoint_cache.py) only ever stored
evaluator outputs (softmax), never raw images -- by design, to keep the
cache small across a 45x3x3 sweep -- so there is no way to look at what a
flagged combo's midpoints actually look like without regenerating it. This
script does exactly that for one (or a few) specific, deliberately chosen
combos, matching the real sweep's settings exactly (defaults to the
rtx5090 profile) so what you see is representative of what's being
measured, not some different config.

Motivation: after several rounds of purely classifier-softmax-based
inference (C(A,B), argmax-flip, p(A,B) realism proxies -- see
EXPERIMENT_REPORT.md §5.6-5.7), several open questions can only be
resolved by actually looking:
  - is a low-confidence midpoint a realistic chimera classifiers just
    aren't trained to recognize, or genuinely degraded/unrealistic?
  - does the one reproducible finding (automobile<->ship -> "airplane"
    consensus at both sigma=0.5 and 2.0) correspond to anything visible?
  - does confidence dropping at higher sigma correspond to a visible
    quality drop, or is that purely a classifier-side effect?
  - does tangential_geodesic actually look more realistic than the
    slerp_noise baseline for the same pair, the whole premise for using it?

Usage:
    python scripts/inspect_path_images.py --pair automobile,ship --sigma 0.5 --path tangential_geodesic --seed 0
    python scripts/inspect_path_images.py --pair automobile,deer --sigma 2.0 --path tangential_geodesic --seed 0
    python scripts/inspect_path_images.py --pair automobile,ship --sigma 0.5 --path slerp_noise --seed 0
    python scripts/inspect_path_images.py --pair automobile,truck --sigma 0.5 --path tangential_geodesic --seed 0 --profile local_poc  # for a quick check on weaker hardware

Output: results/inspect/<pair>_sigma<sigma>_<path>_seed<seed>.png -- a grid,
rows = sample-pairs (up to --n-rows), columns = t (path progress A->B),
each cell resized up for visibility. Also prints the per-t, per-evaluator
predictions if a matching cache combo exists, so the image lines up with
the numbers already analyzed.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image, ImageDraw

from diffusiongraph.config import get_profile, CIFAR10_CLASSES, RESULTS_DIR
from diffusiongraph.data.cifar10 import CIFAR10Canonical
from diffusiongraph.utils.edm_loader import EDMDenoiser
from diffusiongraph.paths import build_path, PATH_USES_CONDITIONAL_MODEL
from diffusiongraph.utils.checkpoint_cache import make_combo_key


def canonical_to_pil(x: torch.Tensor, size: int = 128) -> Image.Image:
    """x: [3,32,32] float in [-1,1] -> upsized PIL image for visibility."""
    arr = ((x.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, "RGB").resize((size, size), Image.NEAREST)


def build_grid(images: torch.Tensor, t_values: torch.Tensor, n_rows: int, cell_size: int = 128, label_h: int = 24) -> Image.Image:
    """images: [T, B, 3, 32, 32]. Rows = sample-pairs (up to n_rows), cols = T."""
    T = images.shape[0]
    n_rows = min(n_rows, images.shape[1])
    grid = Image.new("RGB", (T * cell_size, n_rows * cell_size + label_h), "white")
    draw = ImageDraw.Draw(grid)
    for row in range(n_rows):
        for t_idx in range(T):
            cell = canonical_to_pil(images[t_idx, row], cell_size)
            grid.paste(cell, (t_idx * cell_size, label_h + row * cell_size))
    for t_idx in range(T):
        t_val = float(t_values[t_idx])
        draw.text((t_idx * cell_size + 4, 4), f"t={t_val:.2f}", fill="black")
    return grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", required=True, help="e.g. automobile,ship (CIFAR10 class names, comma-separated)")
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--path", default="tangential_geodesic", choices=["linear_condition", "slerp_noise", "tangential_geodesic"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile", default="rtx5090", help="settings (samples_per_class, control_points, optimizer_steps, etc.) to match -- use the SAME profile as the run you're inspecting")
    p.add_argument("--n-rows", type=int, default=4, help="how many sample-pairs' full trajectories to show")
    p.add_argument("--run-name", default=None, help="if set, also print this combo's cached numeric results (evaluator_c) alongside the image, for cross-reference")
    args = p.parse_args()

    class_a_name, class_b_name = [s.strip() for s in args.pair.split(",")]
    class_a, class_b = CIFAR10_CLASSES.index(class_a_name), CIFAR10_CLASSES.index(class_b_name)

    cfg = get_profile(args.profile)
    print(f"=== Inspecting: {class_a_name}<->{class_b_name}, sigma={args.sigma}, path={args.path}, seed={args.seed} (profile={args.profile}) ===")

    uses_cond = PATH_USES_CONDITIONAL_MODEL[args.path]
    ckpt = cfg.edm_checkpoint_cond if uses_cond else cfg.edm_checkpoint_uncond
    denoiser = EDMDenoiser(ckpt, device=cfg.device)

    dataset = CIFAR10Canonical(train=True, download=True)
    gen = torch.Generator().manual_seed(args.seed)
    x_a, x_b = dataset.sample_pairs(class_a, class_b, cfg.samples_per_class, generator=gen)

    path_ctor = build_path(
        args.path,
        **({"num_control_points": cfg.geodesic_num_control_points,
            "optimizer_steps": cfg.geodesic_optimizer_steps,
            "lr": cfg.geodesic_lr,
            "jvp_chunk_size": cfg.geodesic_jvp_chunk_size}
           if args.path == "tangential_geodesic" else {})
    )
    result = path_ctor.construct(
        denoiser, x_a, x_b, class_a=class_a, class_b=class_b,
        sigma_tau=args.sigma, n_steps=cfg.path_t_steps, seed=args.seed,
    )
    print(f"Generated images: {result.images.shape} (T, B, C, H, W)")

    grid = build_grid(result.images, result.t_values, args.n_rows)
    out_dir = RESULTS_DIR / "inspect"
    out_dir.mkdir(parents=True, exist_ok=True)
    sigma_str = str(args.sigma).replace(".", "p")
    out_path = out_dir / f"{class_a_name}-{class_b_name}_sigma{sigma_str}_{args.path}_seed{args.seed}.png"
    grid.save(out_path)
    print(f"Saved: {out_path}")

    if args.run_name:
        cache_dir = RESULTS_DIR / "gate" / args.run_name / "cache"
        key = make_combo_key("real", args.path, args.sigma, class_a, class_b, args.seed)
        json_path = cache_dir / f"{key}.json"
        if json_path.exists():
            payload = json.loads(json_path.read_text())
            print(f"Cross-reference (cached numeric result): evaluator_c={payload['evaluator_c']}")
            print(f"  evaluator_argmax_class={ {k: CIFAR10_CLASSES[v] for k, v in payload['evaluator_argmax_class'].items()} }")
        else:
            print(f"(No cached result found at {json_path} for cross-reference)")


if __name__ == "__main__":
    main()
