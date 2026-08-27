"""
Central config for the Phase 1 gate (SEED_semantic_class_graph.md §3), amended
per Analysis_of_gpt.md and Strategic_Blind_Spots_Analysis.md.

Deliberately dataclass-based (not a giant YAML) so the two live profiles —
`local_poc()` (this machine, RTX 3050 4GB) and `rtx5090()` (scale-up target) —
are explicit, diffable, and the *only* thing that changes between "prove it
works" and "run the real gate" is which profile a script loads.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCES_DIR = REPO_ROOT / "references"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@dataclass
class GateConfig:
    # --- identity / bookkeeping ---
    run_name: str = "phase1_gate_poc"
    device: str = "cuda"
    seed: int = 0

    # --- generator (SEED §3.2, amended per Strategic_Blind_Spots #2: CFG/
    # conditioning contamination is avoided structurally, not just by pinning
    # w=1.0 — path type 1 (Object 1, conditioning geometry) uses the
    # class-conditional checkpoint; path types 2/3/4 (Object 2, distribution
    # geometry) use the UNCONDITIONAL checkpoint, so no class-label choice
    # of ours ever biases which mode a geometry-aware path is pulled toward.
    # See scripts/download_edm_checkpoint.py docstring. ) ---
    edm_checkpoint_cond: Path = CHECKPOINTS_DIR / "edm-cifar10-32x32-cond-vp.pkl"
    edm_checkpoint_uncond: Path = CHECKPOINTS_DIR / "baseline-cifar10-32x32-uncond-vp.pkl"
    guidance_weight: float = 1.0          # unused while paths 2-4 run on the uncond model; kept for a future CFG ablation.
    sampler_steps: int = 18               # EDM default (Karras et al.)

    # --- which class pairs to run (SEED §3.2: all 45 unordered pairs at full
    # scale; the PoC profile below restricts this to a handful for a fast
    # correctness pass) ---
    class_pair_mode: str = "poc_subset"   # "poc_subset" | "all_pairs"
    poc_pairs: tuple = (
        (3, 5),   # cat  <-> dog     (plausible route: shared "small mammal" mode)
        (5, 7),   # dog  <-> horse   (plausible route: quadruped shape)
        (1, 9),   # auto <-> truck   (plausible route: vehicle silhouette)
        (3, 0),   # cat  <-> airplane (implausible route: tests barrier case C)
    )
    samples_per_class: int = 4            # x~A, y~B sample-pair sets (SEED §5: not centroids)

    # --- timestep filtration (Strategic_Blind_Spots #1: the graph is G(tau),
    # not a single object). We measure routing at fixed noise levels rather
    # than integrating across the whole reverse process. ---
    routing_sigmas: tuple = (0.5, 2.0, 8.0)   # low / mid / high noise, in EDM sigma units
    path_t_steps: int = 24                     # points sampled along each t in [0,1] path

    # --- path types (SEED §3.2) ---
    # Phase 1 build order: 1 & 2 are cheap baselines, wired first end-to-end.
    # 3 (tangential geodesic) is the primary geometry-aware path.
    # 4 (string method) is DEFERRED to confirmatory use on flagged pairs only
    # (Strategic_Blind_Spots #4: full finite-temp string method is too
    # expensive to run exhaustively, especially on a 4GB GPU).
    enabled_paths: tuple = ("linear_condition", "slerp_noise", "tangential_geodesic")
    geodesic_optimizer_steps: int = 200   # curve-energy minimization iterations (path 3)
    geodesic_num_control_points: int = 16 # discretized curve resolution (path 3)
    geodesic_lr: float = 1e-2
    geodesic_jvp_chunk_size: int = 8      # segments per double-backward JVP call (path 3) --
                                           # bounds peak memory (see tangential_geodesic.py),
                                           # calibrated at 8 for a 4GB card (~3.2GB peak).
                                           # MUST be scaled up for bigger GPUs or it silently
                                           # caps throughput regardless of available VRAM.

    # --- routing measurement (SEED §3.3) ---
    routing_threshold_tau: float = 0.5
    routing_seeds: tuple = (0, 1, 2)

    # --- evaluators (SEED §3.2: >=3 independent, different architectures) ---
    evaluator_names: tuple = ("resnet18", "vit_small", "clip_zeroshot")

    # --- controls (SEED §3.4) ---
    run_label_permutation_control: bool = True
    permutation_seed: int = 1234

    # --- compute footprint knobs (the actual local/5090 delta) ---
    batch_size: int = 4
    amp_dtype: str = "float16"            # "float16" on 4GB GPU, "bfloat16"/"float32" on 5090
    grad_checkpointing: bool = True


def local_poc() -> GateConfig:
    """This machine: RTX 3050, 4GB VRAM. Small subset of pairs, small batches,
    fp16, gradient checkpointing on. Purpose: prove the pipeline is correct
    end-to-end AND get a real (if narrower) first read on routing, before
    committing to the full exhaustive sweep anywhere.

    Calibrated from an actual benchmark on this GPU (see PR discussion /
    session log): at optimizer_steps=200, control_points=16, samples=4, ONE
    (class-pair, sigma_tau) combination for path 3 took 1472s (~24.5 min),
    peak 3.26GB. That's ~14.7 hours for the original 4-pair x 3-sigma x
    3-seed local_poc scope -- too slow even for a patient overnight run. The
    numbers below cut the *optimization budget* (steps, control points,
    sigma/seed sweep breadth) rather than sample-pair count (kept at 4,
    since averaging over real sample pairs is what SEED §5 relies on to
    avoid centroid artifacts) -- projected to ~2 min/combination, ~30 min
    for the full path-3 sweep, ~1hr total including the permutation control.
    """
    return GateConfig(
        run_name="phase1_gate_poc_local",
        class_pair_mode="poc_subset",
        samples_per_class=4,
        batch_size=4,
        amp_dtype="float16",
        grad_checkpointing=True,
        geodesic_optimizer_steps=60,
        geodesic_num_control_points=8,
        routing_sigmas=(2.0,),   # single mid-noise level for the first pass; widen once this is confirmed to work
        routing_seeds=(0, 1),
    )


def local_smoke() -> GateConfig:
    """Fast iteration/debugging profile -- NOT a real gate result, just
    'does the pipeline run end-to-end without erroring, on something small
    enough to finish in a couple minutes'. Benchmarked on this machine's
    RTX 3050 4GB: the tangential-geodesic curve optimizer's JVP-via-double-
    backward is the bottleneck (segment-chunked to bound memory, at a real
    time cost -- see paths/tangential_geodesic.py). local_poc()'s settings
    (200 opt steps x 16 control points x 4 sample-pairs) take on the order
    of tens of minutes per (class-pair, sigma_tau) combination here; this
    profile cuts that to under a minute for correctness-checking code
    changes before committing to a real (if patient) local_poc run."""
    return GateConfig(
        run_name="phase1_gate_smoke",
        class_pair_mode="poc_subset",
        poc_pairs=((3, 5),),
        samples_per_class=2,
        batch_size=2,
        amp_dtype="float16",
        grad_checkpointing=True,
        geodesic_optimizer_steps=30,
        geodesic_num_control_points=8,
        routing_sigmas=(2.0,),
        routing_seeds=(0,),
        path_t_steps=8,
    )


def convergence_check() -> GateConfig:
    """ONE-OFF DIAGNOSTIC, not a gate profile -- exists to answer a single
    question with real data instead of a guess: where does the
    tangential_geodesic curve-energy optimization actually plateau?

    Same small scale as local_smoke (batch=2, 8 control points,
    jvp_chunk_size=8 -- so per-step cost matches the already-measured
    ~0.31s/step on the 5090) but with optimizer_steps pushed well past any
    plausible convergence point (300, vs the 30 local_smoke uses) so the
    live per-step energy log (see tangential_geodesic.py's progress
    printing) shows the full decay curve, not just its early/steep part.
    Only 1 pair, 1 seed, no permutation control -- this run is purely about
    reading the printed energy values, not producing a routing verdict.
    Expected wall-clock: ~300 steps x 0.31s/step (2 chunks/step) =~ 1.5-2 min.

    Usage: python scripts/run_gate.py --profile convergence_check --evaluators resnet50,vit_base,clip_zeroshot --skip-permutation
    Then read the printed step/energy lines to decide the REAL
    geodesic_optimizer_steps value for rtx5090(), instead of guessing."""
    return GateConfig(
        run_name="convergence_check",
        class_pair_mode="poc_subset",
        poc_pairs=((3, 5),),
        samples_per_class=2,
        batch_size=2,
        amp_dtype="float16",
        grad_checkpointing=True,
        geodesic_optimizer_steps=300,
        geodesic_num_control_points=8,
        geodesic_jvp_chunk_size=8,
        routing_sigmas=(2.0,),
        routing_seeds=(0,),
        path_t_steps=8,
        enabled_paths=("tangential_geodesic",),  # skip paths 1/2, not relevant to this check
    )


def rtx5090() -> GateConfig:
    """Scale-up target: full exhaustive 45-pair CIFAR-10 sweep, larger
    evaluator architectures (resnet50/vit_base instead of resnet18/
    vit_small -- see models/classifiers.py), larger batches, no gradient
    checkpointing needed, all 3 routing_sigmas levels x 3 seeds. Same code
    path as local_poc() — only these numbers change. See
    experiment/run_rtx5090_poc.sh for the end-to-end script that uses this
    profile.

    Seeds: 3, not 5 -- this is not a corner cut, it's an EXACT match to
    SEED_semantic_class_graph.md §3.3's own stated bar: "persists across
    all >=3 evaluators and >=3 seeds." 5 was padding beyond spec, not a
    requirement; 3 seeds is what the gate's own definition asks for.

    Convergence settings (optimizer_steps, num_control_points) were
    CALIBRATED FROM A REAL MEASUREMENT on the actual 5090, not guessed: a
    local_smoke run (batch=2, 8 control points, jvp_chunk_size=8) measured
    ~0.31s/step; scaling that to this profile's higher segment count
    (samples_per_class=16 x control_points -> more chunks/step at
    jvp_chunk_size=64) gave ~12-15 min/combo at the original 500-step/
    32-point settings -- 675 combos x 2 (real+permuted) x that rate would
    be ~270-340 hours, not tenable. Trimmed to 150/16 (~4x fewer
    optimizer-step-chunks); combined with the 5->3 seed cut, estimated
    ~42-51 hours total across the full 45-pair x 3-sigma x 3-seed sweep
    (real + permutation control) -- long but no longer multi-day-plus,
    full exhaustive pair/sigma coverage preserved.

    optimizer_steps=150 itself is STILL A REASONED ESTIMATE, not a
    verified plateau point -- an earlier version of this docstring claimed
    local runs were "already well-converged before 200 steps," but that
    was written without actually checking real energy_history data, which
    was wrong to assert as fact. Use config.convergence_check() to get
    real per-step energy data on the actual target hardware before
    trusting this number for a multi-day run; adjust
    geodesic_optimizer_steps here once that's checked against real data."""
    return GateConfig(
        run_name="phase1_gate_full",
        class_pair_mode="all_pairs",
        samples_per_class=16,
        batch_size=64,
        amp_dtype="bfloat16",
        grad_checkpointing=False,
        geodesic_jvp_chunk_size=64,  # ~8x the 4GB-card value; a 32GB card has
                                     # roughly 8x the headroom (measured: 8
                                     # segments -> ~3.2GB peak on the 4GB card).
                                     # Raise further if VRAM usage still looks
                                     # low -- this was the actual bottleneck
                                     # behind low utilization on first run.
        geodesic_optimizer_steps=150,   # trimmed from 500 -- PENDING real verification, see docstring above
        geodesic_num_control_points=16, # trimmed from 32 -- see docstring above
        routing_seeds=(0, 1, 2),        # 3 seeds = SEED §3.3's own stated minimum, not a cut -- see docstring above
        evaluator_names=("resnet50", "vit_base", "clip_zeroshot"),
    )


PROFILES = {
    "local_poc": local_poc,
    "local_smoke": local_smoke,
    "convergence_check": convergence_check,
    "rtx5090": rtx5090,
}


def get_profile(name: str) -> GateConfig:
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Choices: {list(PROFILES)}")
    return PROFILES[name]()
