#!/usr/bin/env bash
# ============================================================================
# Phase 1 gate -- full RTX 5090 PoC run. ONE script, run once, does everything:
#   env setup -> reference repos -> checkpoints -> CIFAR-10 -> train larger
#   evaluators (real + permuted labels) -> validate -> full 45-pair gate sweep.
#
# Usage (from repo root, or anywhere -- the script locates itself):
#   bash experiment/run_rtx5090_poc.sh
#
# Assumes: conda is on PATH, and a conda env named `autoeval` already exists
# with everything it needs installed (torch+CUDA for RTX 5090/Blackwell,
# numpy, scipy, pillow, matplotlib, ftfy, regex, omegaconf, click, tqdm,
# pandas, pyarrow -- see requirements.txt). This script does NOT install or
# modify packages -- it only CHECKS that env for the required imports and
# fails fast with a clear list of what's missing, so you can install exactly
# that yourself. Everything else below is idempotent: re-running after an
# interruption skips whatever's already done (downloads check file
# existence; only the evaluator training step always re-runs, by design).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- tunables -----------------------------------------------------------
CONDA_ENV_NAME="${CONDA_ENV_NAME:-autoeval}"
RESNET_EPOCHS="${RESNET_EPOCHS:-80}"
VIT_EPOCHS="${VIT_EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GATE_PROFILE="${GATE_PROFILE:-rtx5090}"

echo "=== repo root: $REPO_ROOT ==="
echo "=== conda env:  $CONDA_ENV_NAME ==="

# --- 1. activate conda env (portable -- works even if conda isn't already
#        initialized in this shell) --------------------------------------
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
PYTHON="python"
echo "=== python: $($PYTHON --version) at $(command -v $PYTHON) ==="

# --- 2. dependency check ONLY -- no installs. Fails fast with a clear list
#        of missing packages if anything's absent. -----------------------
echo "=== [1/7] checking dependencies (no installs) ==="
$PYTHON - <<'PYEOF'
import importlib, sys
required = ["torch", "torchvision", "numpy", "scipy", "PIL", "matplotlib",
            "ftfy", "regex", "omegaconf", "click", "tqdm", "pandas", "pyarrow"]
missing = []
for mod in required:
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print(f"MISSING PACKAGES: {missing}")
    print("Install these yourself in the autoeval env before re-running.")
    sys.exit(1)
import torch
if not torch.cuda.is_available():
    print("torch is installed but torch.cuda.is_available() is False -- "
          "check the CUDA build matches the driver (RTX 5090/Blackwell "
          "needs CUDA >=12.8 wheels).")
    sys.exit(1)
print(f"OK -- torch {torch.__version__}, CUDA available, device: {torch.cuda.get_device_name(0)}")
PYEOF

# --- 3. reference repos (pinned commits, see THIRD_PARTY.md) -------------
echo "=== [2/7] setting up reference repos ==="
bash scripts/setup_references.sh

# --- 4. EDM checkpoints (conditional + unconditional, see
#        scripts/download_edm_checkpoint.py docstring for why both) ------
echo "=== [3/7] downloading EDM checkpoints ==="
$PYTHON scripts/download_edm_checkpoint.py

# --- 5. CIFAR-10 (HF Hub mirror, parallel-connection download -- see
#        scripts/parallel_download.py: single-stream downloads to both
#        torchvision's default source AND naive HF downloads throttled
#        hard on the dev network; parallel Range requests fixed it) ------
echo "=== [4/7] downloading + converting CIFAR-10 ==="
mkdir -p data/hf_cifar10
HF_BASE="https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text"
if [ ! -f data/hf_cifar10/train.parquet ]; then
    $PYTHON scripts/parallel_download.py "$HF_BASE/train-00000-of-00001.parquet" data/hf_cifar10/train.parquet --connections 8
fi
if [ ! -f data/hf_cifar10/test.parquet ]; then
    $PYTHON scripts/parallel_download.py "$HF_BASE/test-00000-of-00001.parquet" data/hf_cifar10/test.parquet --connections 8
fi
if [ ! -f data/hf_cifar10/train.npz ]; then
    $PYTHON scripts/convert_hf_cifar10.py
fi

# --- 6. evaluators: LARGER variants (resnet50, vit_base -- see
#        models/classifiers.py), more epochs, real + permuted labels.
#        Augmentation (random crop + flip) is baked into
#        CIFAR10Canonical(augment=True), used automatically by
#        train_classifiers.py's training split. ---------------------------
echo "=== [5/7] training evaluators: resnet50 x$RESNET_EPOCHS, vit_base x$VIT_EPOCHS (real + permuted) ==="
$PYTHON -u scripts/train_classifiers.py --arch resnet50 --epochs "$RESNET_EPOCHS" --batch-size "$BATCH_SIZE"
$PYTHON -u scripts/train_classifiers.py --arch vit_base  --epochs "$VIT_EPOCHS"    --batch-size "$BATCH_SIZE"
$PYTHON -u scripts/train_classifiers.py --arch resnet50 --epochs "$RESNET_EPOCHS" --batch-size "$BATCH_SIZE" --permuted
$PYTHON -u scripts/train_classifiers.py --arch vit_base  --epochs "$VIT_EPOCHS"    --batch-size "$BATCH_SIZE" --permuted

echo "=== [6/7] validating evaluators on clean CIFAR-10 ==="
$PYTHON -u scripts/validate_evaluators.py --profile "$GATE_PROFILE"

# --- 7. the actual Phase 1 gate: full exhaustive 45-pair sweep -----------
echo "=== [7/7] running the gate (profile=$GATE_PROFILE) ==="
$PYTHON -u scripts/run_gate.py --profile "$GATE_PROFILE"

echo "=== DONE. See results/gate/<run_name>/decision_memo.json and figures/ ==="
