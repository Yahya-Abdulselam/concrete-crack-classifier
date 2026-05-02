"""Handcrafted feature extraction for traditional ML (SVM, RF, kNN).

Extracts four feature families from grayscale images:
1. HOG (Histogram of Oriented Gradients) — crack orientation
2. LBP (Local Binary Patterns) — micro-texture
3. GLCM (Gray-Level Co-occurrence Matrix) — spatial texture relationships
4. Edge Density — fraction of edge pixels via Canny

Features are cached to disk as .npz to avoid repeated extraction
(~0.3-0.5s per image, total ~3-4 hours for the training set).
"""

import os

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.preprocessing import preprocess_for_svm


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def extract_hog(gray_img: np.ndarray) -> np.ndarray:
    """HOG features capturing crack orientation.

    Args:
        gray_img: Grayscale image array (uint8).

    Returns:
        1D feature vector (~8,100 dims for 128x128 with default params).
    """
    features = hog(
        gray_img,
        orientations=config.SVM_HOG_ORIENTATIONS,
        pixels_per_cell=config.SVM_HOG_PIXELS_PER_CELL,
        cells_per_block=config.SVM_HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features


def extract_lbp(gray_img: np.ndarray) -> np.ndarray:
    """LBP histogram capturing micro-texture.

    Args:
        gray_img: Grayscale image array (uint8).

    Returns:
        1D histogram vector (n_points + 2 dims).
    """
    n_points = config.SVM_LBP_N_POINTS
    radius = config.SVM_LBP_RADIUS
    lbp = local_binary_pattern(gray_img, n_points, radius, method="uniform")
    n_bins = n_points + 2  # uniform LBP has P+2 bins
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_glcm(gray_img: np.ndarray) -> np.ndarray:
    """GLCM texture properties at multiple scales and orientations.

    Args:
        gray_img: Grayscale image array (uint8).

    Returns:
        1D feature vector (5 properties x n_distances x 4 angles).
    """
    # Quantize to 64 gray levels for efficiency
    gray_quantized = (gray_img // 4).astype(np.uint8)
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    glcm = graycomatrix(
        gray_quantized,
        distances=config.SVM_GLCM_DISTANCES,
        angles=angles,
        levels=64,
        symmetric=True,
        normed=True,
    )

    props = []
    for prop_name in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        props.append(graycoprops(glcm, prop_name).ravel())
    return np.concatenate(props)


def extract_edge_density(gray_img: np.ndarray) -> np.ndarray:
    """Edge density via Canny detector.

    Args:
        gray_img: Grayscale image array (uint8).

    Returns:
        1D array with single value: fraction of edge pixels.
    """
    edges = cv2.Canny(gray_img, 50, 150)
    density = np.sum(edges > 0) / edges.size
    return np.array([density])


def extract_all_features(gray_img: np.ndarray) -> np.ndarray:
    """Concatenate all four feature families.

    Args:
        gray_img: Grayscale image array (uint8).

    Returns:
        1D feature vector (~8,187 dims).
    """
    return np.concatenate([
        extract_hog(gray_img),
        extract_lbp(gray_img),
        extract_glcm(gray_img),
        extract_edge_density(gray_img),
    ])


def extract_features_from_directory(
    split_dir: str = config.SPLIT_DIR,
    subset: str = "train",
    img_size: int = config.SVM_IMG_SIZE,
    cache_dir: str = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for all images in a split subset.

    Features are cached to .npz files. If the cache exists, it is loaded
    directly without re-extracting.

    Args:
        split_dir: Root split directory containing train/val/test subfolders.
        subset: Which subset to extract ("train", "val", or "test").
        img_size: Target image size for SVM preprocessing.
        cache_dir: Directory for cached features (default: outputs/svm/).

    Returns:
        Tuple of (X, y, paths) where X is (n_samples, n_features).
    """
    if cache_dir is None:
        cache_dir = os.path.join(config.OUTPUT_DIR, "svm")
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"features_{subset}.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        print(f"Loaded cached features from {cache_path}")
        print(f"  Shape: {data['X'].shape}, Classes: {len(np.unique(data['y']))}")
        return data["X"], data["y"], data["paths"]

    subset_dir = os.path.join(split_dir, subset)
    X_list, y_list, paths_list = [], [], []

    total_files = 0
    for cls_name in config.CLASS_NAMES:
        cls_dir = os.path.join(subset_dir, cls_name)
        if os.path.isdir(cls_dir):
            total_files += len([
                f for f in os.listdir(cls_dir)
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
            ])

    print(f"\nExtracting features for {subset} set ({total_files:,} images)...")
    print(f"  Image size: {img_size}x{img_size} grayscale")
    print(f"  Features: HOG + LBP + GLCM + Edge Density")

    with tqdm(total=total_files, desc=f"{subset}") as pbar:
        for cls_idx, cls_name in enumerate(config.CLASS_NAMES):
            cls_dir = os.path.join(subset_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            files = sorted([
                f for f in os.listdir(cls_dir)
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
            ])

            for fname in files:
                fpath = os.path.join(cls_dir, fname)
                try:
                    gray = preprocess_for_svm(fpath, size=img_size)
                    features = extract_all_features(gray)
                    X_list.append(features)
                    y_list.append(cls_idx)
                    paths_list.append(fpath)
                except Exception as e:
                    print(f"\n  WARNING: Failed to process {fpath}: {e}")
                pbar.update(1)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    paths = np.array(paths_list)

    np.savez_compressed(cache_path, X=X, y=y, paths=paths)
    print(f"\nFeatures cached to {cache_path}")
    print(f"  Shape: {X.shape} ({X.nbytes / 1024 / 1024:.1f} MB)")

    return X, y, paths
