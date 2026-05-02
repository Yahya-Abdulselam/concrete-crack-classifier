"""Image preprocessing pipeline for crack classification (PyTorch).

Steps:
1. Load & force RGB
2. Resize with LANCZOS
3. Bilateral denoising (before CLAHE to avoid noise amplification)
4. CLAHE on L channel in LAB space
5. Normalization (ImageNet or [0,1] depending on model)
"""

import numpy as np
from PIL import Image
import cv2

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_and_convert_rgb(image_path: str) -> Image.Image:
    """Step 1: Load image and force RGB conversion."""
    img = Image.open(image_path).convert("RGB")
    return img


def resize_image(img: Image.Image, size: int = config.IMG_SIZE) -> Image.Image:
    """Step 2: Resize to target size with LANCZOS resampling."""
    return img.resize((size, size), Image.LANCZOS)


def bilateral_denoise(img_array: np.ndarray) -> np.ndarray:
    """Step 3: Bilateral filter denoising before CLAHE.

    Smooths homogeneous concrete regions while preserving crack edges.
    Prevents CLAHE from amplifying sensor/texture noise.
    """
    return cv2.bilateralFilter(
        img_array,
        d=config.BILATERAL_DIAMETER,
        sigmaColor=config.BILATERAL_SIGMA_COLOR,
        sigmaSpace=config.BILATERAL_SIGMA_SPACE,
    )


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


def preprocess_numpy(image_path: str, size: int = config.IMG_SIZE) -> np.ndarray:
    """Full numpy preprocessing: Load -> Resize -> Denoise -> CLAHE.

    Returns uint8 RGB array ready for torchvision transforms.
    """
    img = load_and_convert_rgb(image_path)
    img = resize_image(img, size)
    img_array = np.array(img)
    img_array = bilateral_denoise(img_array)
    img_array = apply_clahe(img_array)
    return img_array


def preprocess_pil(image_path: str, size: int = config.IMG_SIZE) -> Image.Image:
    """Full PIL preprocessing: Load -> Resize -> Denoise -> CLAHE.

    Returns PIL Image ready for torchvision transforms.
    """
    img_array = preprocess_numpy(image_path, size)
    return Image.fromarray(img_array)


def preprocess_for_inference(
    image_path: str,
    size: int = config.IMG_SIZE,
    normalize: str = "imagenet",
):
    """Preprocess a single image for model inference.

    Args:
        image_path: Path to the input image.
        size: Target size.
        normalize: "imagenet" for pretrained models, "rescale" for CNN from scratch.

    Returns:
        Tensor of shape (1, 3, size, size).
    """
    import torch
    from torchvision import transforms

    img = preprocess_pil(image_path, size)
    transform_list = [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ]
    if normalize == "imagenet":
        transform_list.append(
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )
    transform = transforms.Compose(transform_list)
    tensor = transform(img)
    return tensor.unsqueeze(0)


def preprocess_for_svm(image_path: str, size: int = None) -> np.ndarray:
    """Preprocess a single image for SVM feature extraction.

    Pipeline: load grayscale -> resize -> CLAHE on grayscale directly.

    Args:
        image_path: Path to the input image.
        size: Target size (default from config.SVM_IMG_SIZE).

    Returns:
        Preprocessed grayscale image array of shape (size, size), uint8.
    """
    if size is None:
        size = config.SVM_IMG_SIZE
    img = Image.open(image_path).convert("L")
    img = img.resize((size, size), Image.LANCZOS)
    img_array = np.array(img)
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID,
    )
    img_array = clahe.apply(img_array)
    return img_array
