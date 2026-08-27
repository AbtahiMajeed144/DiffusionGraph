# RTX 5090 Phase 1 gate — end-to-end script

`run_rtx5090_poc.sh` is the one thing to run. It does everything: env
setup, reference repos, EDM checkpoints, CIFAR-10, evaluator training, and
the full exhaustive 45-pair gate sweep (`config.rtx5090()` profile).

## Usage

```bash
bash experiment/run_rtx5090_poc.sh
```

Assumes `conda` is on PATH and a conda env named `autoeval` **already has
everything installed** — torch+CUDA (RTX 5090/Blackwell needs CUDA >=12.8
wheels), numpy, scipy, pillow, matplotlib, ftfy, regex, omegaconf, click,
tqdm, pandas, pyarrow (see `requirements.txt`). **This script does not
install or modify any packages** — it only checks the env has what it
needs and fails fast with a clear list of anything missing, so you can
install exactly that yourself.

## What it does, in order

1. Activates `autoeval`, checks (does not install) that torch+CUDA and the
   rest of the dependencies are present — exits with a clear missing-package
   list if not.
2. `scripts/setup_references.sh` — clones the 7 pinned reference repos.
3. `scripts/download_edm_checkpoint.py` — both EDM checkpoints (conditional
   + unconditional; see that script's docstring for why we need both).
4. Downloads CIFAR-10 from the HuggingFace Hub mirror via
   `scripts/parallel_download.py` (multi-connection — single-stream
   downloads to both torchvision's default source and naive HF downloads
   throttled hard on the dev laptop's network; parallel Range requests
   fixed it there, kept as the default here too), then converts to the npz
   cache via `scripts/convert_hf_cifar10.py`.
5. Trains the **larger** evaluator architectures — `resnet50` and
   `vit_base` (see `models/classifiers.py`; ~5x and ~7x the params of the
   `resnet18`/`vit_small` used in the local 4GB-GPU PoC), for **both** real
   and permutation-control labels (SEED §3.4), with augmentation
   (`CIFAR10Canonical(augment=True)`) and more epochs than the local run
   (80/120 by default — override via `RESNET_EPOCHS`/`VIT_EPOCHS`).
6. `scripts/validate_evaluators.py --profile rtx5090` — clean-CIFAR-10
   accuracy sanity check before trusting any of this for routing.
7. `scripts/run_gate.py --profile rtx5090` — the actual Phase 1 gate: all
   45 CIFAR-10 pairs, 3 noise levels (`routing_sigmas`), 3 seeds (matches
   SEED §3.3's own stated minimum), full label-permutation control.
   Produces `results/gate/phase1_gate_full/decision_memo.json` and
   `figures/`. **This step is resumable** — see below.

## Resuming after an interruption

The gate sweep (step 7) is a long, multi-hour run and checkpoints itself:
every `(real/permuted, path_type, sigma_tau, class_pair, seed)` combination
is saved to `results/gate/<run_name>/cache/` immediately after it's
computed. If the process dies, gets disconnected, or you kill it on
purpose, **just re-run the exact same command** —
`python scripts/run_gate.py --profile rtx5090` — and it will skip every
combo already on disk (`[cached]` in the log) and continue from where it
left off. At most the single in-flight combo is lost, never more.

If you change any setting that affects results (pairs, seeds, sigmas,
optimizer steps, control points, evaluator names, etc.) between runs
while reusing the same `run_name`, the script will refuse to resume and
raise a clear error rather than silently mixing incompatible cached
results — delete `results/gate/<run_name>/cache/` to start fresh in that
case, or use a different profile/run_name.

## Tunables (env vars, all optional)

| Var | Default | |
|---|---|---|
| `CONDA_ENV_NAME` | `autoeval` | |
| `RESNET_EPOCHS` | `80` | |
| `VIT_EPOCHS` | `120` | ViT needs more epochs than ResNet to converge from scratch |
| `BATCH_SIZE` | `256` | bump further if VRAM allows |
| `GATE_PROFILE` | `rtx5090` | can point at `local_poc`/`local_smoke` too, e.g. to re-verify the pipeline before committing to the full run |

## Notes

- Every download step checks for the target file/checkpoint first and
  skips if already present — safe to re-run after an interruption.
- The evaluator-training step does **not** skip on re-run (always
  retrains) — comment out lines in the script if you need to resume
  mid-way through the 4 training runs. (The gate sweep in step 7 *is*
  resumable, per above — this only applies to steps 1-6.)
- Timing, measured on the actual 5090 (not estimated): `tangential_geodesic`
  runs at ~0.31s per optimizer step at this profile's chunk settings.
  `geodesic_optimizer_steps=150` was chosen from a real 300-step
  convergence check on that hardware — it captures 92% of the achievable
  curve-energy decrease (see `config.rtx5090()`'s docstring for the full
  curve data). Combined with 3 seeds, the full 45-pair × 3-sigma × 3-seed
  sweep (real + permutation control) is estimated at **~42-51 hours** —
  a long unattended run; use the resume capability above rather than
  trying to babysit it in one sitting.
