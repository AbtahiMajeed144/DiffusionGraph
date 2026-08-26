#!/usr/bin/env bash
# Reproducibly (re-)clone the external reference repositories this project studies
# and builds on. `references/` is gitignored (mixed licenses, some non-commercial,
# not ours to redistribute) — this script is the source of truth for exactly which
# commit of each we used. Re-run any time to restore references/ from scratch.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p references
cd references

clone_pinned() {
  local url=$1 dir=$2 commit=$3
  if [ -d "$dir/.git" ]; then
    echo "=== $dir already present, skipping ==="
    return
  fi
  echo "=== cloning $dir @ $commit ==="
  git clone "$url" "$dir"
  git -C "$dir" checkout -q "$commit"
}

# repo                                          dir                   pinned commit                              role
clone_pinned https://github.com/NVlabs/edm                          edm                 008a4e5316c8e3bfe61a62f874bddba254295afb   # generator (CIFAR-10 cond. diffusion)
clone_pinned https://github.com/enkeejunior1/Diffusion-Pullback     Diffusion-Pullback  859c0122b5cb1c8e6488dede29959c785af9aed1   # pullback-metric reference (Park et al.)
clone_pinned https://github.com/MachineLearningLifeScience/stochman stochman            b0acd1e1d3aae4cdf36dc4b399a48ed4d530ba85   # geodesic-curve solver core
clone_pinned https://github.com/yang-song/score_sde_pytorch         score_sde_pytorch   cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44   # score-SDE reference / backup generator
clone_pinned https://github.com/kuangliu/pytorch-cifar              pytorch-cifar       49b7aa97b0c12fe0d4054e670403a16b6b834ddd   # classifier evaluator #1 (ResNet)
clone_pinned https://github.com/omihub777/ViT-CIFAR                 ViT-CIFAR           ab9043ea44cfe8311d30295fc0e0467d676187f1   # classifier evaluator #2 (ViT)
clone_pinned https://github.com/openai/CLIP                         CLIP                d05afc436d78f1c48dc0dbf8e5980a9d471f35f6   # SSL/embedding evaluator #3

echo "=== done ==="
