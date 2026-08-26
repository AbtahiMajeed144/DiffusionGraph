#!/usr/bin/env bash
# ============================================================================
# Phase 1 gate -- full RTX 5090 PoC run. ONE script, run once, does everything:
#   env setup -> reference repos -> checkpoints -> CIFAR-10 -> train larger
#   evaluators (real + permuted labels) -> validate -> full 45-pair gate sweep.
#
# Usage (from repo root, or anywhere -- the script locates itself):
#   bash experiment/run_rtx5090_poc.sh
#
# Assumes: conda is on PATH, and a conda env named `autoeval` exists (create
# it first with `conda create -n autoeval python=3.11` if it doesn't -- this
# script installs everything else into it). Everything below is idempotent:
# re-running after an interruption skips whatever's already done (downloads
# check file existence; only the evaluator training step always re-runs, by
# design -- see TRAIN_* vars below to skip stages manually).
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
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
# RTX 5090 is Blackwell (compute capability sm_120) -- needs a torch build
# with CUDA >=12.8 wheels. If this index is wrong for whatever torch/CUDA
# is current when this actually runs, override: TORCH_INDEX_URL=... bash ...

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

# --- 2. dependencies (idempotent -- pip skips what's already satisfied) --
echo "=== [1/7] installing dependencies ==="
if ! $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "installing torch (CUDA, $TORCH_INDEX_URL)..."
    $PYTHON -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
else
    echo "torch + CUDA already available, skipping torch install"
fi
$PYTHON -m pip install numpy scipy pyyaml tqdm pillow matplotlib ftfy regex omegaconf click psutil requests pandas pyarrow

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
