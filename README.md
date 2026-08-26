# DiffusionGraph — Semantic Class Graph & Class-Routing in Generative Models

Phase 1 (the go/no-go gate, `SEED_semantic_class_graph.md` §3) implementation.
See `SEED_semantic_class_graph.md`, `Analysis_of_gpt.md`, and
`Strategic_Blind_Spots_Analysis.md` for the research design this code
implements.

## Setup

```bash
# Reference repos (gitignored — mixed licenses, see THIRD_PARTY.md)
bash scripts/setup_references.sh

# Python env: this project uses the pre-existing `loope` conda env
# (C:\REGULATA\conda\envs\loope), which already has torch 2.8+cu126 and
# most scientific deps. Only ftfy (CLIP's tokenizer dep) had to be added.
C:/REGULATA/conda/envs/loope/python.exe -m pip install ftfy

# Pretrained EDM checkpoints (~450MB total, two files — see below)
python scripts/download_edm_checkpoint.py
```

### Why two EDM checkpoints

- `edm-cifar10-32x32-cond-vp.pkl` — class-conditional. Used ONLY by path
  type 1 (`linear_condition`), which is explicitly about conditioning
  interpolation (SEED §2, "Object 1").
- `baseline-cifar10-32x32-uncond-vp.pkl` — unconditional (re-imported by
  NVIDIA from Yang Song's `score_sde_pytorch`, VP). Used by path types 2/3/4
  (`slerp_noise`, `tangential_geodesic`, `string_method`), which study
  *distribution* geometry (SEED §2, "Object 2") and must not have any class
  label of ours biasing which mode a path is pulled toward — the structural
  fix for CFG/conditioning contamination flagged in
  `Strategic_Blind_Spots_Analysis.md` #2.

## Pipeline

```bash
# 1. Train the two independent-architecture evaluators (real + permuted labels)
python scripts/train_classifiers.py --arch resnet18
python scripts/train_classifiers.py --arch vit_small
python scripts/train_classifiers.py --arch resnet18 --permuted
python scripts/train_classifiers.py --arch vit_small --permuted

# 2. Sanity-check all 3 evaluators on clean CIFAR-10 before trusting them
python scripts/validate_evaluators.py

# 3. Run the gate
python scripts/run_gate.py --profile local_smoke  # ~1 pair, ~1 min -- pipeline correctness check only
python scripts/run_gate.py --profile local_poc    # this machine, RTX 3050 4GB, real (patient) PoC run
python scripts/run_gate.py --profile rtx5090      # full exhaustive 45-pair sweep
```

Output: `results/gate/<run_name>/decision_memo.json` (GO / PIVOT / KILL per
SEED §3.5) and `results/gate/<run_name>/figures/` (routing matrix heatmaps
per path-type/sigma_tau, trajectory plots for the top-routed pairs).

## Package layout

```
diffusiongraph/
  config.py              GateConfig + local_poc()/rtx5090() profiles
  utils/
    edm_loader.py         EDMDenoiser: denoise/score/sample, resume-from-noisy-state
    jvp.py                 JVP via double-backward (never build full Jacobians)
  data/cifar10.py          canonical [-1,1] CIFAR-10, sample-pair retrieval, permuted-label control
  models/
    classifiers.py         ResNet18 + small ViT (evaluators #1, #2)
    embeddings.py           CLIP zero-shot (evaluator #3)
  paths/                   the 4 SEED §3.2 path constructors
    linear_condition.py     path 1 (conditional model)
    slerp_noise.py           path 2 (unconditional model)
    tangential_geodesic.py    path 3 (unconditional model, score-Jacobian metric)
    string_method.py          path 4 -- DEFERRED, see its docstring
  eval/
    evaluators.py            loads the 3-evaluator stack
    trajectory.py             sweeps a path through all evaluators
    routing.py                 C(A,B), routing matrix, tau-sensitivity
  analysis/
    controls.py               straight-line contrast + label-permutation control -> GO/PIVOT/KILL
    figures.py                  routing matrix heatmap + trajectory plots
scripts/
  setup_references.sh, download_edm_checkpoint.py
  train_classifiers.py, validate_evaluators.py, run_gate.py
references/     gitignored — external repos we study/build on, see THIRD_PARTY.md
checkpoints/    gitignored — trained evaluators + downloaded EDM checkpoints
results/        gitignored — gate outputs (figures, decision memos)
```

## Known compute constraint (as of 2026-08-26)

This machine has a laptop RTX 3050, 4GB VRAM — not the RTX 5090 the SEED doc
assumed as "ample". The tangential-geodesic path (type 3) computes JVPs via
a double-backward trick that needs `create_graph=True` on both internal
backward passes, which is memory-heavy; it's implemented with
segment-chunked backward (`jvp_chunk_size`) to bound peak memory regardless
of batch size, at a real wall-clock cost — expect single-digit minutes per
(class-pair, sigma_tau) combination at `local_poc` settings on this GPU.
`local_poc` is scoped to a handful of class pairs for a first correctness
pass; the full exhaustive 45-pair sweep is the `rtx5090` profile's job.
