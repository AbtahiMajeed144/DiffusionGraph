"""
Evaluator architectures #1 and #2 (SEED §3.2: >=3 independent evaluators of
*different* architecture families, so a routing peak can't be dismissed as
one classifier's artifact).

ResNet18 is adapted from kuangliu/pytorch-cifar (MIT) and ViTSmall from
omihub777/ViT-CIFAR (MIT) — see THIRD_PARTY.md. Inlined (not imported from
references/) because we retrain both from scratch anyway (including the
label-permutation control, which needs a second training run with shuffled
labels — see analysis/controls.py), so we own and version the exact code
that produced each checkpoint.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Evaluator #1: ResNet18 (adapted from kuangliu/pytorch-cifar, MIT)
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return F.relu(out)


def resnet18(num_classes: int = 10) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet50(num_classes: int = 10) -> ResNet:
    """Larger evaluator variant for the rtx5090 profile -- same CIFAR-adapted
    stem (3x3 stride-1 conv1, no maxpool) as resnet18, just deeper/wider
    Bottleneck blocks. Not the pretrained torch.hub cifar10_resnet56 we
    tried first (that GitHub-releases host was unreachable on the dev
    network -- see THIRD_PARTY.md) -- this trains from scratch, same as
    resnet18, just bigger + more epochs on faster hardware."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


# ---------------------------------------------------------------------------
# Evaluator #2: small ViT trained from scratch on 32x32 (adapted from
# omihub777/ViT-CIFAR, MIT) -- a genuinely different inductive bias (global
# attention vs. ResNet's local conv/texture bias), per Strategic_Blind_Spots
# #5 (texture-biased CNNs can hallucinate routing off chimera texture alone).
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats: int, head: int = 8, dropout: float = 0.0):
        super().__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = feats ** 0.5
        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)
        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, f // self.head).transpose(1, 2)
        k = self.k(x).view(b, n, self.head, f // self.head).transpose(1, 2)
        v = self.v(x).view(b, n, self.head, f // self.head).transpose(1, 2)
        score = F.softmax(torch.einsum("bhif,bhjf->bhij", q, k) / self.sqrt_d, dim=-1)
        attn = torch.einsum("bhij,bhjf->bihf", score, v)
        return self.dropout(self.o(attn.flatten(2)))


class TransformerEncoder(nn.Module):
    def __init__(self, feats: int, mlp_hidden: int, head: int = 8, dropout: float = 0.0):
        super().__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        return self.mlp(self.la2(out)) + out


class ViTSmall(nn.Module):
    """~2-3M params at hidden=192/6 layers -- kept small deliberately for the
    4GB-GPU proof-of-concept; bump hidden/num_layers for the rtx5090 profile."""

    def __init__(
        self, in_c: int = 3, num_classes: int = 10, img_size: int = 32, patch: int = 4,
        dropout: float = 0.1, num_layers: int = 6, hidden: int = 192, mlp_hidden: int = 384,
        head: int = 6, is_cls_token: bool = True,
    ):
        super().__init__()
        self.patch = patch
        self.is_cls_token = is_cls_token
        self.patch_size = img_size // patch
        f = self.patch_size ** 2 * in_c
        num_tokens = patch ** 2 + 1 if is_cls_token else patch ** 2

        self.emb = nn.Linear(f, hidden)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden))
        self.enc = nn.Sequential(*[
            TransformerEncoder(hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head)
            for _ in range(num_layers)
        ])
        self.fc = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, num_classes))

    def _to_words(self, x):
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        out = out.permute(0, 2, 3, 4, 5, 1)
        return out.reshape(x.size(0), self.patch ** 2, -1)

    def forward(self, x):
        out = self.emb(self._to_words(x))
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        out = out[:, 0] if self.is_cls_token else out.mean(1)
        return self.fc(out)


def vit_small(num_classes: int = 10) -> ViTSmall:
    return ViTSmall(num_classes=num_classes)


def vit_base(num_classes: int = 10) -> ViTSmall:
    """Larger evaluator variant for the rtx5090 profile -- ~21M params
    (hidden=384, 12 layers, 12 heads) vs vit_small's ~2-3M. Still a
    from-scratch CIFAR-native ViT (not a fine-tuned ImageNet backbone),
    trained longer with augmentation on faster hardware."""
    return ViTSmall(num_classes=num_classes, hidden=384, mlp_hidden=384 * 4, num_layers=12, head=12)


ARCHITECTURES = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "vit_small": vit_small,
    "vit_base": vit_base,
}
