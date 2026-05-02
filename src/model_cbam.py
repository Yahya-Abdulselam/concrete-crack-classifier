"""InceptionV3 + CBAM (Convolutional Block Attention Module) model (PyTorch).

CBAM adds lightweight channel and spatial attention between the InceptionV3
backbone and the classification head. This focuses the model on crack-relevant
features while suppressing background noise, adding <0.1% extra parameters.

Reference: Woo et al. (2018), "CBAM: Convolutional Block Attention Module", ECCV.
"""

import os

import torch
import torch.nn as nn
from torchvision import models

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class ChannelAttention(nn.Module):
    """Channel attention: which feature channels are most important?"""

    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        avg_pool = x.mean(dim=[2, 3])       # (B, C)
        max_pool = x.amax(dim=[2, 3])       # (B, C)
        att = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))  # (B, C)
        return x * att.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """Spatial attention: which locations contain the crack?"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        avg = x.mean(dim=1, keepdim=True)    # (B, 1, H, W)
        mx = x.amax(dim=1, keepdim=True)     # (B, 1, H, W)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))  # (B, 1, H, W)
        return x * att


class CBAM(nn.Module):
    """Full CBAM: channel attention followed by spatial attention."""

    def __init__(self, channels: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class InceptionV3CBAM(nn.Module):
    """InceptionV3 backbone -> CBAM -> AdaptiveAvgPool -> classification head.

    Same architecture as InceptionV3Classifier but with CBAM inserted
    between the backbone output and the pooling layer.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        dropout_rate: float = config.DROPOUT_RATE,
        reduction_ratio: int = config.CBAM_REDUCTION_RATIO,
        kernel_size: int = config.CBAM_KERNEL_SIZE,
    ):
        super().__init__()

        inception = models.inception_v3(weights="IMAGENET1K_V1")
        inception.aux_logits = False

        self.backbone = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
        )

        self.cbam = CBAM(2048, ratio=reduction_ratio, kernel_size=kernel_size)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Linear(2048, config.DENSE_UNITS_1),
            nn.ReLU(),
            nn.BatchNorm1d(config.DENSE_UNITS_1),
            nn.Dropout(dropout_rate),
            nn.Linear(config.DENSE_UNITS_1, config.DENSE_UNITS_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(config.DENSE_UNITS_2, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)         # (B, 2048, H, W)
        features = self.cbam(features)      # CBAM attention
        features = self.pool(features)      # (B, 2048, 1, 1)
        features = features.flatten(1)      # (B, 2048)
        return self.head(features)
