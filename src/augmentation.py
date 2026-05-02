"""Safe augmentation configuration for crack classification (PyTorch).

Training augmentations include geometric and photometric transforms.
Validation/test use preprocessing only (no augmentation).

Note: Bilateral denoise + CLAHE are applied in the Dataset __getitem__
before these transforms (they operate on PIL/numpy, not tensors).
"""

from torchvision import transforms

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.preprocessing import IMAGENET_MEAN, IMAGENET_STD


def _get_normalize(normalize: str):
    """Get normalization transform based on model type."""
    if normalize == "imagenet":
        return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    elif normalize == "rescale":
        # ToTensor already scales to [0, 1], no further normalization needed
        return transforms.Lambda(lambda x: x)
    else:
        raise ValueError(f"Unknown normalize mode: {normalize}")


def get_train_transforms(img_size: int = config.IMG_SIZE, normalize: str = "imagenet"):
    """Returns training transforms: augmentation + ToTensor + normalize.

    Augmentations applied (safe for crack classification):
    - Horizontal flip (cracks are orientation-independent horizontally)
    - Rotation up to 15 degrees
    - Affine shift up to 10%, zoom 85-115%
    - Brightness/contrast variation
    - No vertical flip (preserves gravity-dependent crack patterns)
    """
    norm = _get_normalize(normalize)
    # brightness: config is (0.80, 1.20) — torchvision wants a single float
    # where 0.2 means jitter by factor in [1-0.2, 1+0.2] = [0.8, 1.2]
    brightness_jitter = config.BRIGHTNESS_RANGE[1] - 1.0  # 0.2

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        # Rotation + shift + zoom in a single RandomAffine (avoids double rotation)
        transforms.RandomAffine(
            degrees=config.ROTATION_RANGE,
            translate=(config.WIDTH_SHIFT, config.HEIGHT_SHIFT),
            scale=(1.0 - config.ZOOM_RANGE, 1.0 + config.ZOOM_RANGE),
            fill=128,
        ),
        transforms.ColorJitter(brightness=brightness_jitter),
        transforms.ToTensor(),
        norm,
    ])


def get_val_test_transforms(img_size: int = config.IMG_SIZE, normalize: str = "imagenet"):
    """Returns val/test transforms: resize + ToTensor + normalize (no augmentation)."""
    norm = _get_normalize(normalize)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm,
    ])
