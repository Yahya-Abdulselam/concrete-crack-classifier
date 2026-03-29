"""5-step image preprocessing pipeline for crack classification.

Steps:
1. Load & force RGB
2. Resize to 299x299 with LANCZOS
3. Morphological filtering (3x3 elliptical kernel, open then close)
4. CLAHE on L channel in LAB space
5. InceptionV3 normalization to [-1, 1]
"""

import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.applications.inception_v3 import preprocess_input

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def load_and_convert_rgb(image_path: str) -> Image.Image:
    """Step 1: Load image and force RGB conversion."""
    img = Image.open(image_path).convert("RGB")
    return img


def resize_image(img: Image.Image, size: int = config.IMG_SIZE) -> Image.Image:
    """Step 2: Resize to target size with LANCZOS resampling."""
    return img.resize((size, size), Image.LANCZOS)


def morphological_filter(img_array: np.ndarray) -> np.ndarray:
    """Step 3: Morphological filtering (open then close) with elliptical kernel."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE),
    )
    filtered = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
    return filtered


def apply_clahe(img_array: np.ndarray) -> np.ndarray:
    """Step 4: Apply CLAHE on L channel in LAB color space."""
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID,
    )
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return rgb_enhanced


def normalize_inception(img_array: np.ndarray) -> np.ndarray:
    """Step 5: Normalize to [-1, 1] using InceptionV3 preprocessing."""
    img_float = img_array.astype(np.float32)
    return preprocess_input(img_float)


def preprocess_image(image_path: str) -> np.ndarray:
    """Full 5-step preprocessing pipeline.

    Args:
        image_path: Path to the input image.

    Returns:
        Preprocessed image array of shape (299, 299, 3) in [-1, 1] range.
    """
    img = load_and_convert_rgb(image_path)
    img = resize_image(img)
    img_array = np.array(img)
    img_array = morphological_filter(img_array)
    img_array = apply_clahe(img_array)
    img_array = normalize_inception(img_array)
    return img_array


def preprocess_for_inference(image_path: str) -> np.ndarray:
    """Preprocess a single image for model inference.

    Args:
        image_path: Path to the input image.

    Returns:
        Preprocessed image array of shape (1, 299, 299, 3).
    """
    img = preprocess_image(image_path)
    return np.expand_dims(img, axis=0)


def preprocessing_function_for_datagen(img_array: np.ndarray) -> np.ndarray:
    """Preprocessing function compatible with ImageDataGenerator.

    Applies steps 3-5 (morphology, CLAHE, normalization).
    Steps 1-2 (load, resize) are handled by flow_from_directory.

    Args:
        img_array: Image array of shape (299, 299, 3), uint8.

    Returns:
        Preprocessed image array in [-1, 1] range.
    """
    img_array = img_array.astype(np.uint8)
    img_array = morphological_filter(img_array)
    img_array = apply_clahe(img_array)
    img_array = normalize_inception(img_array)
    return img_array
