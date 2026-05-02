"""4-block CNN from scratch for crack classification (PyTorch).

A sequential CNN with four convolutional blocks following the standard
doubling filter pattern (32 -> 64 -> 128 -> 256), followed by a
classification head. No pretrained weights — trains entirely from scratch.

Reference: Bukaita et al. (2025, American Journal of Civil Engineering),
Flah et al. (2020, Cement and Concrete Composites).
"""

import os

import torch.nn as nn

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class ConvBlock(nn.Module):
    """Single convolutional block: Conv->BN->ReLU->Conv->BN->ReLU->MaxPool->Dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class CrackCNN(nn.Module):
    """4-block CNN from scratch (~500K parameters).

    Architecture per block: Conv(3x3)->BN->ReLU->Conv(3x3)->BN->ReLU->MaxPool->Dropout
    Followed by: GAP -> Dense(256) -> BN -> Dropout -> Dense(128) -> Dropout -> Dense(6)
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        filters: list = None,
        dropout_conv: float = config.CNN_DROPOUT_CONV,
        dropout_dense: float = config.CNN_DROPOUT_DENSE,
        dense_units: list = None,
    ):
        super().__init__()
        if filters is None:
            filters = config.CNN_FILTERS
        if dense_units is None:
            dense_units = config.CNN_DENSE_UNITS

        # Build conv blocks
        blocks = []
        in_ch = 3
        for out_ch in filters:
            blocks.append(ConvBlock(in_ch, out_ch, dropout_conv))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Linear(filters[-1], dense_units[0]),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(dense_units[0]),
            nn.Dropout(dropout_dense),
            nn.Linear(dense_units[0], dense_units[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_dense * 0.6),
            nn.Linear(dense_units[1], num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.head(x)
