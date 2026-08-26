"""
SEED §3.4 controls -- the part that makes a routing claim credible rather
than a metric artifact.

1. Straight-line contrast: routing must be stronger/more frequent on the
   geometry-aware paths (tangential_geodesic) than on the baselines
   (linear_condition, slerp_noise). If baselines route just as much, the
   geometry claim is empty (SEED §3.4).
2. Label-permutation negative control: same images, permuted class IDs.
   Any "structure" that survives is a metric/classifier artifact tied to
   arbitrary identifiers, not the image geometry -- kills the project per
   SEED §3.5 if it does NOT vanish.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ControlReport:
    straight_line_contrast: Dict[str, float]   # {path_type: mean routing strength}
    geometry_beats_baselines: bool
    permutation_routing_rate: float             # fraction of permuted pairs that register as routing events
    permutation_survives: bool                  # True is BAD (artifact) -- see SEED §3.5
    gate_decision: str                          # "GO" | "PIVOT" | "KILL"
    notes: str


def straight_line_contrast(routing_matrices: Dict[str, np.ndarray]) -> Dict[str, float]:
    """routing_matrices: {path_type: strength_matrix [10,10] with NaN off-measured}.
    Returns mean routing strength per path type (nan-safe)."""
    return {name: float(np.nanmean(mat)) for name, mat in routing_matrices.items()}


def evaluate_controls(
    routing_matrices: Dict[str, np.ndarray],
    permutation_routing_events: np.ndarray,
    permutation_total_pairs: int,
    geometry_path_name: str = "tangential_geodesic",
    baseline_path_names=("linear_condition", "slerp_noise"),
) -> ControlReport:
    contrast = straight_line_contrast(routing_matrices)
    geometry_score = contrast.get(geometry_path_name, float("nan"))
    baseline_scores = [contrast[b] for b in baseline_path_names if b in contrast]
    geometry_beats_baselines = bool(baseline_scores) and geometry_score > max(baseline_scores)

    # routing_event matrices are filled symmetrically (R[a,b] == R[b,a]) by
    # build_routing_matrix -- count each unordered pair once via the upper
    # triangle, or .sum() silently double-counts every event.
    n_perm_events = int(np.triu(permutation_routing_events, k=1).sum()) if permutation_total_pairs else 0
    perm_rate = n_perm_events / permutation_total_pairs if permutation_total_pairs else float("nan")
    permutation_survives = perm_rate > 0.05  # any material rate under permutation is disqualifying

    if permutation_survives:
        decision = "KILL"
        notes = (
            f"Routing survives label permutation ({perm_rate:.1%} of permuted pairs still "
            f"register as routing events) -- this is a metric/classifier artifact per SEED "
            f"§3.5, not genuine structure. Do not proceed to Phase 2 regardless of the "
            f"geometry-vs-baseline contrast below."
        )
    elif geometry_beats_baselines:
        decision = "GO"
        notes = (
            f"Geometry-aware path ({geometry_path_name}, mean strength={geometry_score:.3f}) "
            f"beats baseline paths ({dict(zip(baseline_path_names, baseline_scores))}), and "
            f"permutation control is clean ({perm_rate:.1%} false-positive rate). Proceed to Phase 2."
        )
    else:
        decision = "PIVOT"
        notes = (
            f"Permutation control is clean, but the geometry-aware path does not clearly "
            f"beat baselines (geometry={geometry_score:.3f} vs baselines={baseline_scores}). "
            f"Per SEED §3.5: if path objectives differ clearly in realism/chimera-rate even "
            f"without a routing-strength gap, the project becomes RQ4-first."
        )

    return ControlReport(
        straight_line_contrast=contrast,
        geometry_beats_baselines=geometry_beats_baselines,
        permutation_routing_rate=perm_rate,
        permutation_survives=permutation_survives,
        gate_decision=decision,
        notes=notes,
    )
