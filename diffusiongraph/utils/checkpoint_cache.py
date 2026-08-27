"""
Checkpoint/resume cache for the Phase 1 gate sweep (scripts/run_gate.py).

A full rtx5090 sweep is a multi-day unattended run (see config.rtx5090()'s
docstring for the timing math -- ~42-51 hours). Until this module existed,
run_gate.py held every result in memory and only wrote anything to disk
AFTER THE ENTIRE SWEEP FINISHED -- a crash, disconnect, or power blip at
hour 40 would lose all progress with nothing recoverable. This module
makes each (tag, path_type, sigma_tau, class_pair, seed) combination's
result durable IMMEDIATELY after it's computed, and skips recomputing
anything already on disk when the script is re-run with the same
run_name.

Layout: results/gate/<run_name>/cache/
  run_config.json      -- fingerprint of the settings that affect computed
                           results. A mismatch on resume means the config
                           changed since the cache was written; resuming
                           would silently mix incompatible results, so we
                           refuse and raise instead.
  <combo_key>.json      -- PairCResult + trajectory metadata (small)
  <combo_key>.npz       -- trajectory softmax arrays (compact, compressed)

Resume is automatic: just re-run the same
`python scripts/run_gate.py --profile <name>` command. Anything already
cached is skipped (loaded, not recomputed); anything not yet computed
picks up where it left off. To force a clean restart, delete
results/gate/<run_name>/cache/ (or use a different run_name).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from diffusiongraph.eval.routing import PairCResult
from diffusiongraph.eval.trajectory import TrajectoryResult


def make_combo_key(tag: str, path_name: str, sigma_key, a: int, b: int, seed: int) -> str:
    sigma_str = str(sigma_key).replace(".", "p")
    return f"{tag}__{path_name}__sigma{sigma_str}__pair{a}-{b}__seed{seed}"


class ComboCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _json_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _npz_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npz"

    def exists(self, key: str) -> bool:
        # Require BOTH files -- a process killed mid-save (between the npz
        # and json writes) must be treated as incomplete and recomputed,
        # never loaded as if it were a valid, finished result.
        return self._json_path(key).exists() and self._npz_path(key).exists()

    def save(self, key: str, pair_result: PairCResult, trajectory: TrajectoryResult) -> None:
        payload = {
            "evaluator_c": pair_result.evaluator_c,
            "evaluator_argmax_class": pair_result.evaluator_argmax_class,
            "path_type": trajectory.path_type,
            "class_a": trajectory.class_a,
            "class_b": trajectory.class_b,
            "sigma_tau": trajectory.sigma_tau,
            "seed": trajectory.seed,
            "t_values": trajectory.t_values.tolist(),
        }
        # Write npz FIRST, json SECOND -- exists() requires both, so a kill
        # between these two writes correctly looks incomplete on resume,
        # not silently corrupt.
        np.savez_compressed(
            self._npz_path(key),
            **{name: t.numpy() for name, t in trajectory.softmax_by_evaluator.items()},
        )
        self._json_path(key).write_text(json.dumps(payload))

    def load(self, key: str) -> Tuple[PairCResult, TrajectoryResult]:
        payload = json.loads(self._json_path(key).read_text())
        pair_result = PairCResult(
            evaluator_c=payload["evaluator_c"],
            evaluator_argmax_class=payload["evaluator_argmax_class"],
        )
        npz = np.load(self._npz_path(key))
        softmax_by_evaluator = {name: torch.from_numpy(npz[name]) for name in npz.files}
        trajectory = TrajectoryResult(
            path_type=payload["path_type"],
            class_a=payload["class_a"],
            class_b=payload["class_b"],
            sigma_tau=payload["sigma_tau"],
            seed=payload["seed"],
            t_values=torch.tensor(payload["t_values"]),
            softmax_by_evaluator=softmax_by_evaluator,
        )
        return pair_result, trajectory


def _fingerprint_fields(cfg) -> dict:
    """The subset of GateConfig fields that affect COMPUTED RESULTS --
    mismatching any of these on resume means previously-cached combos are
    not valid for the current settings and must not be silently reused."""
    return {
        "class_pair_mode": cfg.class_pair_mode,
        "poc_pairs": [list(p) for p in cfg.poc_pairs],
        "samples_per_class": cfg.samples_per_class,
        "routing_sigmas": list(cfg.routing_sigmas),
        "path_t_steps": cfg.path_t_steps,
        "enabled_paths": list(cfg.enabled_paths),
        "geodesic_optimizer_steps": cfg.geodesic_optimizer_steps,
        "geodesic_num_control_points": cfg.geodesic_num_control_points,
        "geodesic_lr": cfg.geodesic_lr,
        "routing_seeds": list(cfg.routing_seeds),
        "evaluator_names": list(cfg.evaluator_names),
        "permutation_seed": cfg.permutation_seed,
        "edm_checkpoint_cond": str(cfg.edm_checkpoint_cond),
        "edm_checkpoint_uncond": str(cfg.edm_checkpoint_uncond),
        # NB: geodesic_jvp_chunk_size is deliberately EXCLUDED -- it only
        # controls how the computation is chunked for memory, not the
        # (deterministic, up to float-associativity) mathematical result.
        # This means you CAN safely resume after retuning chunk size for
        # different hardware without invalidating the whole cache.
    }


def verify_or_write_run_config(cache_dir: Path, cfg) -> None:
    """First run: writes cache_dir/run_config.json as the fingerprint.
    Resume: verifies the current cfg's result-affecting fields match what
    was cached, raising a clear error on mismatch -- silently mixing
    cached results from a different configuration would produce a
    corrupted, misleading routing matrix with no indication anything was
    wrong, which is worse than just refusing to proceed."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    config_path = cache_dir / "run_config.json"
    current = _fingerprint_fields(cfg)

    if not config_path.exists():
        config_path.write_text(json.dumps(current, indent=2))
        return

    saved = json.loads(config_path.read_text())
    mismatches = {k: (saved.get(k), current[k]) for k in current if saved.get(k) != current[k]}
    if mismatches:
        lines = "\n".join(f"  {k}: cached={old!r} vs current={new!r}" for k, (old, new) in mismatches.items())
        raise RuntimeError(
            f"Cache at {cache_dir} was written with different settings than this "
            f"run -- resuming would silently mix incompatible results:\n{lines}\n\n"
            f"Either delete {cache_dir} to start fresh, or use a different run_name."
        )
