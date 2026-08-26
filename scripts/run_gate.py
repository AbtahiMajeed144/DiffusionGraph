"""
The Phase 1 gate orchestrator (SEED §3, §10). Ties together: both EDM
checkpoints, the 3 evaluators, CIFAR-10 sample pairs, the enabled path
types, the sigma_tau sweep, and the label-permutation control -- produces
the routing matrix figure(s), trajectory plots, and a decision memo
(GO / PIVOT / KILL per SEED §3.5).

Usage:
    python scripts/run_gate.py --profile local_poc
    python scripts/run_gate.py --profile rtx5090
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from diffusiongraph.config import get_profile, CIFAR10_CLASSES, RESULTS_DIR
from diffusiongraph.data.cifar10 import CIFAR10Canonical, PermutedLabelCIFAR10
from diffusiongraph.utils.edm_loader import EDMDenoiser
from diffusiongraph.eval.evaluators import load_evaluators, load_permuted_evaluators
from diffusiongraph.eval.trajectory import evaluate_path
from diffusiongraph.eval.routing import compute_C, build_routing_matrix, is_routing_event
from diffusiongraph.paths import build_path, PATH_USES_CONDITIONAL_MODEL
from diffusiongraph.analysis.controls import evaluate_controls
from diffusiongraph.analysis.figures import plot_routing_matrix, plot_trajectory


def get_class_pairs(cfg):
    if cfg.class_pair_mode == "poc_subset":
        return list(cfg.poc_pairs)
    return [(a, b) for a in range(10) for b in range(a + 1, 10)]


def run_sweep(cfg, dataset, denoiser_cond, denoiser_uncond, evaluators, class_pairs, out_dir: Path, tag: str):
    """Runs every (path_type, sigma_tau-or-N/A, seed, class_pair) combination
    for one dataset/evaluator-set (real labels or permuted). Returns:
      routing_matrices: {(path_type, sigma_tau_key): {'strength':.., 'routing_event':..}}
      all_pair_c_results: {(path_type, sigma_tau_key): {(a,b): [PairCResult,...]}}
    """
    routing_matrices = {}
    all_pair_c_results = {}
    trajectories_for_plots = []

    for path_name in cfg.enabled_paths:
        uses_cond = PATH_USES_CONDITIONAL_MODEL[path_name]
        denoiser = denoiser_cond if uses_cond else denoiser_uncond
        sigma_list = [None] if path_name == "linear_condition" else cfg.routing_sigmas

        for sigma_tau in sigma_list:
            key = (path_name, sigma_tau if sigma_tau is not None else "final")
            results_by_pair = {}

            for (a, b) in class_pairs:
                per_seed_results = []
                for seed in cfg.routing_seeds:
                    gen = torch.Generator().manual_seed(seed)
                    x_a, x_b = dataset.sample_pairs(a, b, cfg.samples_per_class, generator=gen)

                    path_ctor = build_path(
                        path_name,
                        **({"num_control_points": cfg.geodesic_num_control_points,
                            "optimizer_steps": cfg.geodesic_optimizer_steps,
                            "lr": cfg.geodesic_lr,
                            "jvp_chunk_size": cfg.geodesic_jvp_chunk_size}
                           if path_name == "tangential_geodesic" else {})
                    )
                    t0 = time.time()
                    path_result = path_ctor.construct(
                        denoiser, x_a, x_b, class_a=a, class_b=b,
                        sigma_tau=sigma_tau if sigma_tau is not None else 0.0,
                        n_steps=cfg.path_t_steps, seed=seed,
                    )
                    dt = time.time() - t0

                    traj = evaluate_path(path_result, evaluators, class_a=a, class_b=b, seed=seed)
                    c_result = compute_C(traj)
                    per_seed_results.append(c_result)
                    trajectories_for_plots.append((key, a, b, seed, traj))

                    print(
                        f"[{tag}] {path_name} sigma={sigma_tau} pair=({CIFAR10_CLASSES[a]},{CIFAR10_CLASSES[b]}) "
                        f"seed={seed} C={c_result.evaluator_c} ({dt:.1f}s)"
                    )

                results_by_pair[(a, b)] = per_seed_results

            all_pair_c_results[key] = results_by_pair
            routing_matrices[key] = build_routing_matrix(results_by_pair, tau=cfg.routing_threshold_tau)

    return routing_matrices, all_pair_c_results, trajectories_for_plots


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=["local_poc", "local_smoke", "rtx5090"], default="local_poc")
    p.add_argument("--skip-permutation", action="store_true", help="skip the label-permutation control (faster iteration only -- never for a real gate report)")
    args = p.parse_args()

    cfg = get_profile(args.profile)
    out_dir = RESULTS_DIR / "gate" / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Phase 1 gate: profile={args.profile} run_name={cfg.run_name} ===")

    denoiser_cond = EDMDenoiser(cfg.edm_checkpoint_cond, device=cfg.device)
    denoiser_uncond = EDMDenoiser(cfg.edm_checkpoint_uncond, device=cfg.device)
    evaluators = load_evaluators(cfg.evaluator_names, device=cfg.device)
    dataset = CIFAR10Canonical(train=True, download=True)
    class_pairs = get_class_pairs(cfg)
    print(f"Class pairs: {class_pairs}")

    routing_matrices, pair_c_results, trajectories = run_sweep(
        cfg, dataset, denoiser_cond, denoiser_uncond, evaluators, class_pairs, out_dir, tag="real"
    )

    # --- figures: one routing-matrix heatmap per (path_type, sigma_tau) ---
    strength_by_path = {}
    for key, mats in routing_matrices.items():
        path_name, sigma_key = key
        title = f"{path_name} (sigma_tau={sigma_key})"
        fname = out_dir / "figures" / f"routing_matrix_{path_name}_{sigma_key}.png"
        plot_routing_matrix(mats["strength"], mats["routing_event"], title, fname)
        strength_by_path.setdefault(path_name, []).append(mats["strength"])

    import numpy as np
    strength_avg_by_path = {name: np.nanmean(np.stack(mats), axis=0) for name, mats in strength_by_path.items()}

    # --- trajectory plots for the most-routed pairs (top 3 by conservative strength) ---
    from diffusiongraph.eval.routing import conservative_routing_strength
    scored = []
    for key, results_by_pair in pair_c_results.items():
        for (a, b), per_seed in results_by_pair.items():
            scored.append((conservative_routing_strength(per_seed), key, a, b))
    scored.sort(key=lambda x: -x[0] if x[0] == x[0] else -1)  # NaN-safe descending sort
    for strength, key, a, b in scored[:3]:
        matches = [t for (k, ta, tb, seed, t) in trajectories if k == key and ta == a and tb == b]
        if matches:
            fname = out_dir / "figures" / f"trajectory_{key[0]}_{key[1]}_{CIFAR10_CLASSES[a]}_{CIFAR10_CLASSES[b]}.png"
            plot_trajectory(matches[0], fname)

    # --- label-permutation control ---
    if not args.skip_permutation:
        print("=== Running label-permutation control ===")
        perm_dataset = PermutedLabelCIFAR10(train=True, download=True, permutation_seed=cfg.permutation_seed)
        # NB: permuted evaluators are trained separately -- see
        # scripts/train_classifiers.py --permuted. CLIP is excluded (see
        # eval/evaluators.load_permuted_evaluators docstring).
        perm_names = tuple(n for n in cfg.evaluator_names if n != "clip_zeroshot")
        perm_evaluators = load_permuted_evaluators(names=perm_names, device=cfg.device)
        perm_routing_matrices, perm_pair_c_results, _ = run_sweep(
            cfg, perm_dataset, denoiser_cond, denoiser_uncond, perm_evaluators, class_pairs, out_dir, tag="permuted"
        )
        geo_key = [k for k in perm_routing_matrices if k[0] == "tangential_geodesic"]
        if geo_key:
            perm_events = perm_routing_matrices[geo_key[0]]["routing_event"]
            perm_total = len(class_pairs)
        else:
            perm_events = np.zeros((10, 10), dtype=bool)
            perm_total = 0
    else:
        import numpy as np
        perm_events = np.zeros((10, 10), dtype=bool)
        perm_total = 0
        print("=== SKIPPED label-permutation control (--skip-permutation) -- NOT a valid gate result ===")

    report = evaluate_controls(strength_avg_by_path, perm_events, perm_total)

    memo = {
        "profile": args.profile,
        "run_name": cfg.run_name,
        "class_pairs": class_pairs,
        "gate_decision": report.gate_decision,
        "notes": report.notes,
        "straight_line_contrast": report.straight_line_contrast,
        "geometry_beats_baselines": report.geometry_beats_baselines,
        "permutation_routing_rate": report.permutation_routing_rate,
        "permutation_survives": report.permutation_survives,
        "routing_threshold_tau": cfg.routing_threshold_tau,
    }
    memo_path = out_dir / "decision_memo.json"
    memo_path.write_text(json.dumps(memo, indent=2))
    print(f"\n=== GATE DECISION: {report.gate_decision} ===\n{report.notes}\n")
    print(f"Decision memo: {memo_path}")
    print(f"Figures: {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
