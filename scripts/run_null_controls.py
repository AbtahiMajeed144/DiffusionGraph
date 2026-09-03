"""
Null controls for the tangential_geodesic instrument (audit §3.3 + the
corrected identifiability control from the review of PHASE0_AUDIT.md).

Why this exists: C(A,B) = max_t max_{c not in {A,B}} p(c) is a DOUBLE
maximum (over ~16 control points x 8 classes), so it is upward-biased by
selection alone. The real sweep's 0.2577 mean is uninterpretable in
absolute terms without a null computed with the SAME double-max machinery.
The permutation control as originally built cannot provide this (it is a
relabeled copy of the real experiment for unconditional paths, and buggy
for the conditional path -- see PHASE0_AUDIT.md). Two proper nulls:

  --control same_class   (A<->A):  geodesic between two DISTINCT real
      samples of the SAME class. There is no third-class route to find, so
      whatever C emerges is the pure floor from midpoint ambiguity + the
      double-max selection. THE most valuable missing number (per review):
      if A<->A also gives ~0.26, then the real 0.2577 carries no routing
      information at all. Masking: compute_C masks {class_a, class_b}={c},
      i.e. max over the other 9 classes (a mild upper bound vs the real
      case's max over 8 -- noted, and conservative: if even this generous
      floor matches the real mean, the null verdict is only stronger).

  --control decoupled    (corrected identifiability control): geodesic
      between two real samples from a RANDOMLY chosen class pair (p,q),
      but C computed masking a DIFFERENT, independently-random nominal
      pair (A,B) (with p,q,A,B all distinct so the endpoints' own classes
      stay UNmasked only if they coincide with neither A nor B -- see
      note). Keeps the evaluator fully real/competent (unlike the review's
      correctly-rejected "random per-image labels" idea, which would cripple
      the evaluator and pass mechanically), and scrambles ONLY the
      pair-identity<->geometry correspondence. If real (coupled) C
      systematically exceeds decoupled C, genuine class-pairing elevates C;
      if equal, pair identity does not matter.

      DESIGN NOTE (flagged for confirmation): "mask a nominal {A,B}" is
      genuinely ambiguous. This implements: sampling classes (p,q) random;
      masking classes (A,B) an independent random pair chosen DISJOINT from
      {p,q} where possible, so the endpoints' own classes p,q remain in the
      'other' set and the measurement is 'does some third class beat a
      geodesic whose endpoints we are NOT crediting'. An alternative reading
      (mask {p,q} but shuffle which nominal pair's RESULT this counts as) is
      not implemented; confirm intent before citing decoupled numbers.

Writes to results/gate/<run-name>/cache/ under distinct run-names
(phase1_null_same_class / phase1_null_decoupled by default) so
analyze_cache.py and audit_cache.py work on the output unchanged.

Usage (on the 5090, autoeval env):
    python scripts/run_null_controls.py --control same_class --profile rtx5090
    python scripts/run_null_controls.py --control decoupled  --profile rtx5090
Defaults to tangential_geodesic only (the instrument under test); add
--paths tangential_geodesic,slerp_noise to include the baseline.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from diffusiongraph.config import get_profile, CIFAR10_CLASSES, RESULTS_DIR
from diffusiongraph.data.cifar10 import CIFAR10Canonical
from diffusiongraph.utils.edm_loader import EDMDenoiser
from diffusiongraph.eval.evaluators import load_evaluators
from diffusiongraph.eval.trajectory import evaluate_path
from diffusiongraph.eval.routing import compute_C
from diffusiongraph.paths import build_path, PATH_USES_CONDITIONAL_MODEL
from diffusiongraph.utils.checkpoint_cache import ComboCache, make_combo_key, verify_or_write_run_config


def build_control_units(control: str, cfg, rng: np.random.RandomState):
    """Returns a list of dicts: sample_a, sample_b (classes to DRAW endpoints
    from) and mask_a, mask_b (classes to pass to compute_C for masking).
    Each also carries a 'slot' (a,b) used only for the cache key."""
    units = []
    if control == "same_class":
        # sample BOTH endpoints from class c (no third class to route to).
        # MASK-MATCHING (review fix 1): mask c AND a second random class, so
        # C = max over 8 -- identical selection budget to the real sweep
        # (which masks the 2 endpoint classes). The second mask class is
        # drawn ONCE here per unit, so it is the SAME at every sigma (review:
        # "same class set at both sigma"). Without this, same_class masked
        # only 1 class -> max over 9 -> an upward bias that made the null a
        # loose (conservative) upper bound rather than a matched comparison.
        for c in range(10):
            others = [k for k in range(10) if k != c]
            m2 = int(rng.choice(others))
            units.append(dict(slot=(c, c), sample_a=c, sample_b=c, mask_a=c, mask_b=m2))
    elif control == "decoupled":
        # One decoupled unit per real unordered pair, to match the real
        # sweep's 45-unit budget. Geometry classes (p,q) and mask classes
        # (A,B) are independently drawn, kept disjoint where possible.
        n_units = 45
        for i in range(n_units):
            p, q = rng.choice(10, size=2, replace=False)
            remaining = [c for c in range(10) if c not in (p, q)]
            if len(remaining) >= 2:
                a, b = rng.choice(remaining, size=2, replace=False)
            else:
                a, b = rng.choice(10, size=2, replace=False)
            units.append(dict(slot=(i, i), sample_a=int(p), sample_b=int(q), mask_a=int(a), mask_b=int(b)))
    else:
        raise ValueError(f"unknown control '{control}'")
    return units


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", required=True, choices=["same_class", "decoupled"])
    p.add_argument("--profile", default="rtx5090")
    p.add_argument("--paths", default="tangential_geodesic",
                   help="comma-separated path types (unconditional only). Default: tangential_geodesic")
    p.add_argument("--run-name", default=None, help="cache run-name (default: phase1_null_<control>)")
    p.add_argument("--sigmas", default=None,
                   help="comma-separated sigma levels to run (default: the profile's routing_sigmas). "
                        "Use e.g. --sigmas 2.0 to target only the sigma the real data you want to "
                        "compare against actually has -- makes a decisive first read cheap.")
    p.add_argument("--seeds", default=None,
                   help="comma-separated seeds to run (default: the profile's routing_seeds). "
                        "Use e.g. --seeds 0 for a fast first read.")
    args = p.parse_args()

    cfg = get_profile(args.profile)
    run_name = args.run_name or f"phase1_null_{args.control}"
    paths = [s.strip() for s in args.paths.split(",")]
    # All path types allowed (review fix 3: A<->A for slerp AND linear, to
    # decide PIVOT vs KILL on the soft geometry-vs-baseline bar by giving each
    # path type its OWN artifact floor to subtract). linear_condition uses the
    # conditional model and ignores sigma -- handled per-path below.

    sigmas = [float(s) for s in args.sigmas.split(",")] if args.sigmas else list(cfg.routing_sigmas)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(cfg.routing_seeds)
    # NB: overriding sigmas/seeds changes the run_config fingerprint's
    # effective coverage but NOT the per-combo math, and combos are cached
    # by (control,path,sigma,slot,seed) -- so a later fuller run resumes and
    # adds combos rather than conflicting. The fingerprint guard still keeps
    # geometry-affecting settings (steps, control points, etc.) consistent.

    out_dir = RESULTS_DIR / "gate" / run_name
    cache_dir = out_dir / "cache"
    # NB: this cache is a DIFFERENT experiment from the real sweep; its own
    # run_config fingerprint guards resume, same as run_gate.py.
    verify_or_write_run_config(cache_dir, cfg)
    # Control-scheme guard: the combo cache keys by (control,path,sigma,slot,
    # seed) and do NOT encode the masking scheme. The mask-matching fix (fix 1)
    # changed same_class from mask-1 to mask-2, so an OLD cache's combos are
    # now numerically wrong and would be silently reused as [cached]. Stamp the
    # scheme and refuse to mix.
    scheme = {"control": args.control, "same_class_mask_count": 2, "scheme_version": 2}
    scheme_path = cache_dir / "control_scheme.json"
    if scheme_path.exists():
        old = json.loads(scheme_path.read_text())
        if old != scheme:
            raise RuntimeError(
                f"Cache {cache_dir} was written under a different control scheme "
                f"({old}) than this run ({scheme}) -- its cached combos are numerically "
                f"incompatible. Delete {cache_dir} or use a new --run-name.")
    else:
        scheme_path.write_text(json.dumps(scheme, indent=2))

    cache = ComboCache(cache_dir)
    print(f"=== Null control: {args.control} | profile={args.profile} | paths={paths} | "
          f"sigmas={sigmas} | seeds={seeds} | run_name={run_name} ===")

    denoiser_uncond = EDMDenoiser(cfg.edm_checkpoint_uncond, device=cfg.device)
    denoiser_cond = None  # lazy-loaded only if a conditional path (linear_condition) is requested
    evaluators = load_evaluators(cfg.evaluator_names, device=cfg.device)
    dataset = CIFAR10Canonical(train=True, download=True)
    rng = np.random.RandomState(cfg.permutation_seed)  # reproducible control draws
    units = build_control_units(args.control, cfg, rng)

    for path_name in paths:
        uses_cond = PATH_USES_CONDITIONAL_MODEL[path_name]
        if uses_cond and denoiser_cond is None:
            denoiser_cond = EDMDenoiser(cfg.edm_checkpoint_cond, device=cfg.device)
        denoiser = denoiser_cond if uses_cond else denoiser_uncond
        # linear_condition ignores sigma (generates from noise per t); run it
        # once (sigma_key "final"), like run_gate.py does.
        sigma_list = [None] if path_name == "linear_condition" else sigmas
        for sigma_tau in sigma_list:
            sigma_key = sigma_tau if sigma_tau is not None else "final"
            for u in units:
                for seed in seeds:
                    a, b = u["slot"]
                    combo_key = make_combo_key(args.control, path_name, sigma_key, a, b, seed)
                    if cache.exists(combo_key):
                        c_result, _ = cache.load(combo_key)
                        print(f"[{args.control}] {path_name} sigma={sigma_key} slot=({a},{b}) "
                              f"sample=({CIFAR10_CLASSES[u['sample_a']]},{CIFAR10_CLASSES[u['sample_b']]}) "
                              f"mask=({CIFAR10_CLASSES[u['mask_a']]},{CIFAR10_CLASSES[u['mask_b']]}) seed={seed} [cached]")
                        continue

                    gen = torch.Generator().manual_seed(seed)
                    x_a, x_b = dataset.sample_pairs(u["sample_a"], u["sample_b"], cfg.samples_per_class, generator=gen)

                    ctor = build_path(
                        path_name,
                        **({"num_control_points": cfg.geodesic_num_control_points,
                            "optimizer_steps": cfg.geodesic_optimizer_steps,
                            "lr": cfg.geodesic_lr,
                            "jvp_chunk_size": cfg.geodesic_jvp_chunk_size}
                           if path_name == "tangential_geodesic" else {})
                    )
                    t0 = time.time()
                    # construct() gets the SAMPLE/geometry classes (matters only
                    # for conditional linear_condition, which conditions on them;
                    # unconditional paths ignore these args and use x_a/x_b pixels).
                    # evaluate_path/compute_C get the MASK classes.
                    path_result = ctor.construct(
                        denoiser, x_a, x_b, class_a=u["sample_a"], class_b=u["sample_b"],
                        sigma_tau=sigma_tau if sigma_tau is not None else 0.0, n_steps=cfg.path_t_steps, seed=seed,
                    )
                    traj = evaluate_path(path_result, evaluators, class_a=u["mask_a"], class_b=u["mask_b"], seed=seed)
                    c_result = compute_C(traj)
                    cache.save(combo_key, c_result, traj)
                    print(f"[{args.control}] {path_name} sigma={sigma_key} slot=({a},{b}) "
                          f"sample=({CIFAR10_CLASSES[u['sample_a']]},{CIFAR10_CLASSES[u['sample_b']]}) "
                          f"mask=({CIFAR10_CLASSES[u['mask_a']]},{CIFAR10_CLASSES[u['mask_b']]}) seed={seed} "
                          f"C={c_result.evaluator_c} ({time.time()-t0:.1f}s)")

    print(f"\nDone. Analyze with:\n  python scripts/audit_cache.py --run-name {run_name}")


if __name__ == "__main__":
    main()
