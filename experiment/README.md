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
   45 CIFAR-10 pairs, 3 noise levels (`routing_sigmas`), 5 seeds, full
   label-permutation control. Produces
   `results/gate/phase1_gate_full/decision_memo.json` and `figures/`.

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
  mid-way through the 4 training runs.
- Sizing this run's actual wall-clock time on a 5090 wasn't possible from
  the dev laptop (RTX 3050 4GB) this was built on — the closest real data
  point: on that card, one (class-pair, sigma, seed) combination for the
  geodesic path took ~25 min at the heavier settings this profile restores
  (200+ optimizer steps, 16+ control points). A 5090 should be dramatically
  faster, but budget time to let this run unattended rather than assuming
  a specific number.
