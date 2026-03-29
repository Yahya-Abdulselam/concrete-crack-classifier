"""Safe augmentation configuration for crack classification.

Training augmentations include geometric and photometric transforms.
Validation/test use preprocessing only (no augmentation).
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.preprocessing import preprocessing_function_for_datagen


def get_train_datagen() -> ImageDataGenerator:
    """Returns an ImageDataGenerator with safe augmentations for training.

    Augmentations applied:
    - Horizontal flip (cracks are orientation-independent horizontally)
    - Rotation up to 15 degrees
    - Width/height shift up to 10%
    - Brightness variation 80-120%
    - Zoom up to 15%
    - No vertical flip (preserves gravity-dependent crack patterns)
    """
    return ImageDataGenerator(
        preprocessing_function=preprocessing_function_for_datagen,
        horizontal_flip=config.HORIZONTAL_FLIP,
        vertical_flip=config.VERTICAL_FLIP,
        rotation_range=config.ROTATION_RANGE,
        width_shift_range=config.WIDTH_SHIFT,
        height_shift_range=config.HEIGHT_SHIFT,
        brightness_range=config.BRIGHTNESS_RANGE,
        zoom_range=config.ZOOM_RANGE,
        fill_mode=config.FILL_MODE,
        channel_shift_range=config.CHANNEL_SHIFT_RANGE,
        shear_range=config.SHEAR_RANGE,
    )


def get_val_test_datagen() -> ImageDataGenerator:
    """Returns an ImageDataGenerator with preprocessing only (no augmentation)."""
    return ImageDataGenerator(
        preprocessing_function=preprocessing_function_for_datagen,
    )
