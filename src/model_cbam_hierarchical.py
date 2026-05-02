"""InceptionV3 + CBAM with 3 hierarchical branching heads (PyTorch).

Architecture:

    Image (3, 299, 299)
        |
    segment_a  ->  Mixed_5d output  (B, 288, 35, 35)  --+
        |                                                |--> Head 1: crack/no_crack (2)
        v                                                |
    segment_b  ->  Mixed_6e output  (B, 768, 17, 17)  --+--> Head 2: single/multi (2)
        |                                                |
        v                                                |
    segment_c  ->  Mixed_7c output  (B, 2048, 8, 8)   --+--> Head 3: subtype (4)

Each head is: CBAM -> AdaptiveAvgPool -> Dropout -> Linear -> Linear.
The CBAM module is imported verbatim from src.model_cbam (no duplication).
"""

import os

import torch
import torch.nn as nn
from torchvision import models

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model_cbam import CBAM
from src.hierarchical import STAGE1_CLASSES, STAGE2_CLASSES, STAGE3_CLASSES


# Layer indices in the existing src.model_cbam backbone Sequential
# (kept identical so freeze utilities and indices stay aligned with src/model.py)
#  0..2  Conv2d_1a..2b
#  3     MaxPool
#  4..5  Conv2d_3b, Conv2d_4a
#  6     MaxPool
#  7..9  Mixed_5b..5d            <- end of segment A
#  10..14 Mixed_6a..6e           <- end of segment B
#  15..17 Mixed_7a..7c           <- end of segment C
SEG_A_END = 10   # exclusive
SEG_B_END = 15
SEG_C_END = 18


def _build_inception_segments() -> tuple[nn.Sequential, nn.Sequential, nn.Sequential]:
    """Split a fresh InceptionV3 backbone into 3 sequential segments."""
    inception = models.inception_v3(weights="IMAGENET1K_V1")
    inception.aux_logits = False

    layers = [
        inception.Conv2d_1a_3x3,
        inception.Conv2d_2a_3x3,
        inception.Conv2d_2b_3x3,
        nn.MaxPool2d(kernel_size=3, stride=2),
        inception.Conv2d_3b_1x1,
        inception.Conv2d_4a_3x3,
        nn.MaxPool2d(kernel_size=3, stride=2),
        inception.Mixed_5b,
        inception.Mixed_5c,
        inception.Mixed_5d,    # idx 9 -> seg A end
        inception.Mixed_6a,
        inception.Mixed_6b,
        inception.Mixed_6c,
        inception.Mixed_6d,
        inception.Mixed_6e,    # idx 14 -> seg B end
        inception.Mixed_7a,
        inception.Mixed_7b,
        inception.Mixed_7c,    # idx 17 -> seg C end
    ]
    seg_a = nn.Sequential(*layers[:SEG_A_END])           # -> (B, 288, 35, 35)
    seg_b = nn.Sequential(*layers[SEG_A_END:SEG_B_END])  # -> (B, 768, 17, 17)
    seg_c = nn.Sequential(*layers[SEG_B_END:SEG_C_END])  # -> (B, 2048, 8, 8)
    return seg_a, seg_b, seg_c


class _HierHead(nn.Module):
    """CBAM -> GAP -> Dropout -> Linear(hidden) -> ReLU -> Linear(num_classes)."""

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        num_classes: int,
        dropout: float,
        cbam_ratio: int = config.CBAM_REDUCTION_RATIO,
        cbam_kernel: int = config.CBAM_KERNEL_SIZE,
    ):
        super().__init__()
        self.cbam = CBAM(in_channels, ratio=cbam_ratio, kernel_size=cbam_kernel)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, feat):
        x = self.cbam(feat)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class InceptionV3CBAMHierarchical(nn.Module):
    """Single InceptionV3 backbone with 3 CBAM-equipped heads.

    forward(x) -> (logits_stage1, logits_stage2, logits_stage3)
        logits_stage1 : (B, 2)
        logits_stage2 : (B, 2)
        logits_stage3 : (B, 4)
    """

    def __init__(
        self,
        dropout_rate: float = config.DROPOUT_RATE,
        cbam_ratio: int = config.CBAM_REDUCTION_RATIO,
        cbam_kernel: int = config.CBAM_KERNEL_SIZE,
    ):
        super().__init__()
        self.segment_a, self.segment_b, self.segment_c = _build_inception_segments()

        # Stage-1 head: easiest task, smallest hidden, lower dropout
        self.head1 = _HierHead(
            in_channels=288, hidden=128, num_classes=len(STAGE1_CLASSES),
            dropout=max(dropout_rate - 0.1, 0.1),
            cbam_ratio=cbam_ratio, cbam_kernel=cbam_kernel,
        )
        # Stage-2 head: medium difficulty
        self.head2 = _HierHead(
            in_channels=768, hidden=256, num_classes=len(STAGE2_CLASSES),
            dropout=dropout_rate,
            cbam_ratio=cbam_ratio, cbam_kernel=cbam_kernel,
        )
        # Stage-3 head: hardest, biggest hidden, highest dropout
        self.head3 = _HierHead(
            in_channels=2048, hidden=512, num_classes=len(STAGE3_CLASSES),
            dropout=min(dropout_rate + 0.1, 0.6),
            cbam_ratio=cbam_ratio, cbam_kernel=cbam_kernel,
        )

    def forward(self, x):
        feat_a = self.segment_a(x)        # (B, 288, 35, 35)
        feat_b = self.segment_b(feat_a)   # (B, 768, 17, 17)
        feat_c = self.segment_c(feat_b)   # (B, 2048, 8, 8)

        out1 = self.head1(feat_a)
        out2 = self.head2(feat_b)
        out3 = self.head3(feat_c)
        return out1, out2, out3


# ============================================================
# FREEZE / UNFREEZE UTILITIES
# ============================================================


def freeze_backbone(model: InceptionV3CBAMHierarchical) -> None:
    """Freeze all 3 backbone segments. Heads (incl. their CBAMs) stay trainable."""
    for seg in (model.segment_a, model.segment_b, model.segment_c):
        for p in seg.parameters():
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Backbone frozen.  Trainable: {trainable:,} / {total:,}")


def unfreeze_segment_b(model: InceptionV3CBAMHierarchical) -> None:
    """Unfreeze segments b and c (mid-network onward). Segment a stays frozen."""
    for p in model.segment_a.parameters():
        p.requires_grad = False
    for seg in (model.segment_b, model.segment_c):
        for p in seg.parameters():
            p.requires_grad = True
    for head in (model.head1, model.head2, model.head3):
        for p in head.parameters():
            p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Unfroze segments b+c.  Trainable: {trainable:,} / {total:,}")


def unfreeze_all(model: InceptionV3CBAMHierarchical) -> None:
    for p in model.parameters():
        p.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    print(f"All {total:,} params unfrozen")


def freeze_for_crt(model: InceptionV3CBAMHierarchical) -> None:
    """Freeze everything except head fc layers (for decoupled classifier retraining).

    Freezes: segment_a, segment_b, segment_c, head{1,2,3}.cbam, head{1,2,3}.pool.
    Trainable: head{1,2,3}.fc (Dropout-Linear-ReLU-Dropout-Linear per head).
    """
    # Freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze only the fc Sequential in each head
    for head in (model.head1, model.head2, model.head3):
        for p in head.fc.parameters():
            p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Frozen for cRT.  Trainable (fc only): {trainable:,} / {total:,}")
