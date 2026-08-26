"""
The gate deliverable's figure (SEED §3.5): routing matrix R + example
classifier-trajectory plots for 2-3 routed pairs, saved into results/figures/.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusiongraph.config import CIFAR10_CLASSES, RESULTS_DIR
from diffusiongraph.eval.trajectory import TrajectoryResult


def plot_routing_matrix(strength: np.ndarray, routing_event: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    masked = np.ma.masked_invalid(strength)
    im = ax.imshow(masked, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CIFAR10_CLASSES)
    for i in range(10):
        for j in range(10):
            if not np.isnan(strength[i, j]):
                marker = "*" if routing_event[i, j] else ""
                ax.text(j, i, f"{strength[i, j]:.2f}{marker}", ha="center", va="center",
                         color="white" if strength[i, j] < 0.5 else "black", fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="C(A,B) routing strength (min over evaluators & seeds)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trajectory(trajectory: TrajectoryResult, out_path: Path, top_k: int = 4):
    """One subplot per evaluator: p(c|gamma(t)) for the top-k most active
    classes, so a routing peak (a real bump in class C's curve) is visually
    obvious next to the two endpoint classes."""
    evaluators = list(trajectory.softmax_by_evaluator.keys())
    fig, axes = plt.subplots(1, len(evaluators), figsize=(5 * len(evaluators), 4), squeeze=False)
    t = trajectory.t_values.numpy()

    for ax, name in zip(axes[0], evaluators):
        softmax = trajectory.softmax_by_evaluator[name].mean(dim=1).numpy()  # [T, num_classes]
        # Always show endpoints; add the top-k other classes by peak value.
        show = {trajectory.class_a, trajectory.class_b}
        other_peaks = [(c, softmax[:, c].max()) for c in range(softmax.shape[1]) if c not in show]
        other_peaks.sort(key=lambda x: -x[1])
        for c, _ in other_peaks[:top_k]:
            show.add(c)
        for c in sorted(show):
            style = "-" if c in (trajectory.class_a, trajectory.class_b) else "--"
            ax.plot(t, softmax[:, c], style, label=CIFAR10_CLASSES[c])
        ax.set_title(f"{name}\n{CIFAR10_CLASSES[trajectory.class_a]} -> {CIFAR10_CLASSES[trajectory.class_b]}")
        ax.set_xlabel("t")
        ax.set_ylabel("p(c | gamma(t))")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7)

    fig.suptitle(f"{trajectory.path_type}  sigma_tau={trajectory.sigma_tau}  seed={trajectory.seed}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
