"""
Strong pretrained evaluators -- NOT currently wired into the default
pipeline (eval/evaluators.py uses the from-scratch resnet50/vit_base in
models/classifiers.py instead, trained via experiment/run_rtx5090_poc.sh).
Kept here as a validated, working alternative:

- PretrainedResNetEvaluator (chenyaofo/pytorch-cifar-models, cifar10_resnet56):
  UNTESTED end-to-end -- its GitHub Releases checkpoint host
  (objects.githubusercontent.com) timed out completely on the dev laptop's
  network (DNS resolved, TLS connected, then no data -- not a throttling
  pattern our parallel_download.py could route around). May well work fine
  from a different network; not relied on by default given that risk.
- PretrainedViTEvaluator (nateraw/vit-base-patch16-224-cifar10): DOES work
  -- loaded successfully and confirmed correct label ordering via HuggingFace
  Hub on the dev network. Not used by default anyway, per the project
  decision to train larger from-scratch resnet50/vit_base instead (bigger
  GPU, more epochs, no external-checkpoint dependency risk at all) --  but
  if you want a transfer-learned ViT evaluator instead of from-scratch,
  this is ready to use: swap "vit_base" for a call to PretrainedViTEvaluator
  in eval/evaluators.py's load_evaluators(). Its permuted-label counterpart
  would need scripts/finetune_vit_cifar.py (not written -- would fine-tune
  backbone_for_scratch="google/vit-base-patch16-224-in21k" on permuted
  labels, same recipe as train_classifiers.py but via the transformers
  Trainer API instead of a from-scratch training loop).

Original motivation, still valid background: our from-scratch training
showed ResNet18 overfitting hard (100% train acc by epoch 13, ~85% test
acc) and ViT-small badly underfit (61% test acc, no augmentation, tiny
model) -- both real reliability concerns for evaluators that have to
classify out-of-distribution synthetic path images, not just clean
CIFAR-10 photos. Fixed instead by: (1) adding augmentation
(CIFAR10Canonical(augment=True)), (2) training larger architectures
(resnet50/vit_base) for more epochs on the 5090.

- ResNet: chenyaofo/pytorch-cifar-models, cifar10_resnet56, 94.37% top-1
  (BSD-3-Clause). A genuine CIFAR-native ResNet (He et al.'s original
  3-stage CIFAR topology, not an ImageNet-style resnet18 squeezed down).
- ViT: nateraw/vit-base-patch16-224-cifar10, ViT-B/16 pretrained on
  ImageNet-21k then fine-tuned on CIFAR-10 (Apache-2.0). ~97%+ reported
  elsewhere for this recipe -- verify empirically, don't trust the number,
  see scripts/validate_evaluators.py.

IMPORTANT (SEED §3.4 label-permutation control): neither of these has a
"trained on permuted CIFAR-10 labels" counterpart anywhere on the internet
-- that's not a task anyone publishes. The permutation control still
requires training a matched-architecture model ourselves on permuted
labels: resnet56_cifar(pretrained=False) trained from scratch (cheap, same
recipe as before), and vit-base-patch16-224-in21k (the PRE-fine-tuning
backbone) fine-tuned on permuted labels ourselves (see
scripts/finetune_vit_cifar.py). Swapping to pretrained real evaluators does
NOT remove the need for scripts/train_classifiers.py's permuted runs.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard CIFAR-10 per-channel normalization stats (used by the
# akamaster-lineage CIFAR ResNet training recipes chenyaofo's repo follows).
_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


def resnet56_cifar(pretrained: bool = True) -> nn.Module:
    """chenyaofo/pytorch-cifar-models' cifar10_resnet56. pretrained=False
    returns the same architecture with random init, for the permuted-label
    control (matched architecture, fair real-vs-permuted comparison)."""
    return torch.hub.load(
        "chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=pretrained
    )


class PretrainedResNetEvaluator:
    """Canonical [-1,1] images in -> [B, 10] softmax out."""

    def __init__(self, device: str = "cuda", checkpoint_path=None):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        if checkpoint_path is not None:
            # permuted-control variant: architecture only, our own trained weights
            self.model = resnet56_cifar(pretrained=False)
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        else:
            self.model = resnet56_cifar(pretrained=True)
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x01 = (x.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
        mean = torch.tensor(_CIFAR_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(_CIFAR_STD, device=x.device).view(1, 3, 1, 1)
        return (x01 - mean) / std

    @torch.no_grad()
    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        logits = self.model(self._preprocess(images))
        return F.softmax(logits, dim=-1)


class PretrainedViTEvaluator:
    """Canonical [-1,1] images in -> [B, 10] softmax out. Resizes to 224;
    no further normalization needed -- the checkpoint's own preprocessor
    config uses mean=std=0.5, i.e. exactly our [-1,1] canonical range."""

    def __init__(
        self,
        device: str = "cuda",
        checkpoint: str = "nateraw/vit-base-patch16-224-cifar10",
        backbone_for_scratch: Optional[str] = None,
    ):
        from transformers import ViTForImageClassification

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        source = backbone_for_scratch or checkpoint
        self.model = ViTForImageClassification.from_pretrained(
            source,
            num_labels=10,
            ignore_mismatched_sizes=backbone_for_scratch is not None,
        ).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x.clamp(-1, 1), size=224, mode="bicubic", align_corners=False, antialias=True)

    @torch.no_grad()
    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        logits = self.model(pixel_values=self._preprocess(images)).logits
        return F.softmax(logits, dim=-1)
