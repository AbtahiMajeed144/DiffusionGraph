"""
The core measurement (SEED §3.3-§3.4):

    C(A,B) = max_t [ max_{c not in {A,B}} p(c | gamma(t)) ]

A routing event = C(A,B) > tau, required to persist across ALL evaluators
and ALL seeds (SEED §3.3) before it counts -- a single evaluator or seed
spiking is not evidence, it's noise or an artifact.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from diffusiongraph.config import CIFAR10_CLASSES
from diffusiongraph.eval.trajectory import TrajectoryResult


@dataclass
class PairCResult:
    """Per-evaluator C(A,B) and argmax intermediate class, for one trajectory
    (one path type, one sigma_tau, one seed)."""
    evaluator_c: Dict[str, float]
    evaluator_argmax_class: Dict[str, int]


def compute_C(trajectory: TrajectoryResult) -> PairCResult:
    evaluator_c: Dict[str, float] = {}
    evaluator_argmax_class: Dict[str, int] = {}
    for name, softmax in trajectory.softmax_by_evaluator.items():
        # softmax: [T, B, num_classes]. Aggregate the sample-pair (B) axis by
        # mean -- multiple real (x~A, y~B) pairs per SEED §5, not centroids;
        # averaging their per-t class distributions is the sample-set
        # analogue of "the path's" trajectory, robust to any single noisy pair.
        mean_softmax = softmax.mean(dim=1)  # [T, num_classes]
        num_classes = mean_softmax.shape[-1]
        mask = torch.ones(num_classes, dtype=torch.bool)
        mask[trajectory.class_a] = False
        mask[trajectory.class_b] = False
        other_probs = mean_softmax[:, mask]           # [T, num_classes-2]
        other_class_ids = torch.arange(num_classes)[mask]
        flat_idx = other_probs.argmax()
        t_idx, c_idx = divmod(flat_idx.item(), other_probs.shape[1])
        evaluator_c[name] = other_probs[t_idx, c_idx].item()
        evaluator_argmax_class[name] = other_class_ids[c_idx].item()
    return PairCResult(evaluator_c=evaluator_c, evaluator_argmax_class=evaluator_argmax_class)


def is_routing_event(per_seed_results: List[PairCResult], tau: float) -> bool:
    """Requires C(A,B) > tau for EVERY evaluator, in EVERY seed (SEED §3.3)."""
    for r in per_seed_results:
        if any(v <= tau for v in r.evaluator_c.values()):
            return False
    return True


def conservative_routing_strength(per_seed_results: List[PairCResult]) -> float:
    """R[A,B] as reported in the routing-matrix figure: the min C(A,B) over
    all evaluators and seeds -- the most conservative single number that
    still has to clear tau for every independent check to pass (mirrors
    is_routing_event's AND-of-all-checks logic in one scalar)."""
    all_vals = [v for r in per_seed_results for v in r.evaluator_c.values()]
    return min(all_vals) if all_vals else float("nan")


def consensus_intermediate_class(per_seed_results: List[PairCResult]) -> Optional[int]:
    """Majority-vote intermediate class across the evaluator x seed argmaxes,
    for reporting/plotting ("this pair routes through class X"). Returns
    None if there's no majority (i.e. evaluators disagree on WHICH class,
    even if they agree THAT some routing occurred) -- itself worth flagging."""
    votes = [c for r in per_seed_results for c in r.evaluator_argmax_class.values()]
    if not votes:
        return None
    values, counts = np.unique(votes, return_counts=True)
    top = values[np.argmax(counts)]
    if counts.max() <= len(votes) / 2:
        return None
    return int(top)


def build_routing_matrix(
    results_by_pair: Dict[tuple, List[PairCResult]],
    num_classes: int = 10,
    tau: float = 0.5,
) -> Dict[str, np.ndarray]:
    """results_by_pair: {(class_a, class_b): [PairCResult, ...] across seeds}
    for ONE (path_type, sigma_tau) combination -- call once per combination
    you want a separate matrix for (SEED / Strategic_Blind_Spots #1: the
    graph is G(tau), plural, not one object).

    Returns dict with:
      'strength'      [num_classes, num_classes] conservative C(A,B), NaN where unmeasured
      'routing_event' [num_classes, num_classes] bool, tau-thresholded AND-of-all check
    """
    strength = np.full((num_classes, num_classes), np.nan)
    routing_event = np.zeros((num_classes, num_classes), dtype=bool)
    for (a, b), per_seed in results_by_pair.items():
        s = conservative_routing_strength(per_seed)
        ev = is_routing_event(per_seed, tau)
        strength[a, b] = s
        strength[b, a] = s
        routing_event[a, b] = ev
        routing_event[b, a] = ev
    return {"strength": strength, "routing_event": routing_event}


def tau_sensitivity(results_by_pair: Dict[tuple, List[PairCResult]], taus=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) -> Dict[float, int]:
    """SEED §6: 'Report tau-sensitivity ... for every routing claim.' Counts
    how many pairs register as a routing event at each threshold."""
    counts = {}
    for tau in taus:
        n = sum(is_routing_event(per_seed, tau) for per_seed in results_by_pair.values())
        counts[tau] = n
    return counts
