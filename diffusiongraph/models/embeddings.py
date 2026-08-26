"""
Evaluator #3: CLIP zero-shot nearest-class, per SEED §3.2 ("two classifiers of
different architectures... plus one self-supervised embedding"). This is the
evaluator that is architecturally and *training-distribution* independent of
the other two (web-scale contrastive pretraining vs. CIFAR-10-only supervised
training) — the strongest check against "the intermediate class is just a
classifier artifact" (Strategic_Blind_Spots #5, GPT analysis §13).

Caveat we must actually check empirically, not assume (flagged in the SEED
handoff): CLIP is ImageNet/web-scale trained on real-resolution photos, not
32x32 CIFAR. We upsample to CLIP's native 224 with bicubic interpolation, but
its zero-shot behavior on upsampled-from-32px images needs a sanity pass
(clean CIFAR test accuracy) before we trust its softmax as ground truth --
see scripts/validate_evaluators.py.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from diffusiongraph.config import REFERENCES_DIR, CIFAR10_CLASSES

_CLIP_REPO = REFERENCES_DIR / "CLIP"


def _ensure_clip_on_path() -> None:
    if not _CLIP_REPO.exists():
        raise FileNotFoundError(
            f"references/CLIP not found at {_CLIP_REPO}. Run scripts/setup_references.sh first."
        )
    p = str(_CLIP_REPO)
    if p not in sys.path:
        sys.path.insert(0, p)


# CLIP's published preprocessing constants (ViT-B/32 checkpoint).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipZeroShot:
    """Canonical image format in, per-class probability vector out -- same
    interface as the trained classifiers, so trajectory.py can treat all
    three evaluators uniformly.

    Input images: float tensor [B, 3, 32, 32] in [-1, 1] (our canonical
    format, matching the EDM generator's I/O range).
    Output: [B, num_classes] softmax over CIFAR10_CLASSES.
    """

    def __init__(self, device: str = "cuda", model_name: str = "ViT-B/32", class_names=None):
        _ensure_clip_on_path()
        import clip  # references/CLIP/clip

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model, _ = clip.load(model_name, device=self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        class_names = class_names or CIFAR10_CLASSES
        prompts = [f"a photo of a {name}" for name in class_names]
        tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            self.text_features = F.normalize(text_features, dim=-1)
        self.logit_scale = self.model.logit_scale.exp().item()

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # [-1, 1] -> [0, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        x = F.interpolate(x, size=224, mode="bicubic", align_corners=False, antialias=True)
        mean = torch.tensor(_CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_STD, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device, self.model.dtype if hasattr(self.model, "dtype") else torch.float32)
        x = self._preprocess(images).to(next(self.model.parameters()).dtype)
        image_features = F.normalize(self.model.encode_image(x), dim=-1)
        logits = self.logit_scale * image_features @ self.text_features.T
        return F.softmax(logits, dim=-1)
