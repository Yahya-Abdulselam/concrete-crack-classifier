"""InceptionV3 model architecture and training utilities (PyTorch)."""

import os

import torch
import torch.nn as nn
from torchvision import models

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class InceptionV3Classifier(nn.Module):
    """InceptionV3 backbone + classification head.

    Architecture:
        InceptionV3 (ImageNet) -> AdaptiveAvgPool -> Dense(256)+BN+Dropout
        -> Dense(128)+Dropout -> Dense(6)
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        dropout_rate: float = config.DROPOUT_RATE,
    ):
        super().__init__()

        # Load pretrained InceptionV3 and extract feature layers
        inception = models.inception_v3(weights="IMAGENET1K_V1")
        inception.aux_logits = False

        # Everything except the final FC layer becomes the backbone
        # InceptionV3 outputs 2048 features from its last conv block
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
        features = self.pool(features)      # (B, 2048, 1, 1)
        features = features.flatten(1)      # (B, 2048)
        return self.head(features)


# Named layer boundaries for freeze/unfreeze
# Maps stage names to backbone sequential indices
_LAYER_MAP = {
    "Mixed_5b": 7,
    "Mixed_5c": 8,
    "Mixed_5d": 9,
    "Mixed_6a": 10,
    "Mixed_6b": 11,
    "Mixed_6c": 12,
    "Mixed_6d": 13,
    "Mixed_6e": 14,
    "Mixed_7a": 15,
    "Mixed_7b": 16,
    "Mixed_7c": 17,
}


def freeze_backbone(model: InceptionV3Classifier) -> None:
    """Freeze all backbone layers."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone frozen. Trainable params: {trainable:,}")


def unfreeze_from(model: InceptionV3Classifier, layer_name: str = "Mixed_7a") -> None:
    """Unfreeze backbone layers from the specified layer onwards.

    Args:
        model: The InceptionV3Classifier model.
        layer_name: Name of the layer from which to start unfreezing.
    """
    if layer_name not in _LAYER_MAP:
        raise ValueError(f"Unknown layer: {layer_name}. Options: {list(_LAYER_MAP.keys())}")

    start_idx = _LAYER_MAP[layer_name]
    for idx, child in enumerate(model.backbone):
        for param in child.parameters():
            param.requires_grad = idx >= start_idx

    # Head is always trainable
    for param in model.head.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Unfroze from '{layer_name}': {trainable:,}/{total:,} params trainable")


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all layers."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters())
    print(f"All {trainable:,} params unfrozen")
