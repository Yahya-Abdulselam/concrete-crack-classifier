"""Central configuration for the crack classification system."""

import os

# === Paths ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# Split directory: auto-detect from possible locations
_SPLIT_CANDIDATES = [
    os.path.join(DATA_DIR, "split"),
    os.path.join(DATA_DIR, "qu1stdata", "split"),
    "/data/dataset/QU1stdata/split",
]
SPLIT_DIR = next(
    (p for p in _SPLIT_CANDIDATES if os.path.isdir(p)),
    os.path.join(DATA_DIR, "split"),  # fallback default
)

# === Synthetic Data ===
_SYNTHETIC_CANDIDATES = [
    os.path.join(os.path.dirname(PROJECT_ROOT), "Synthetic cracks", "Synthetic cracks"),
    os.path.join(DATA_DIR, "synthetic_cracks"),
    "/data/dataset/autocaddata",
]
SYNTHETIC_DIR = next(
    (p for p in _SYNTHETIC_CANDIDATES if os.path.isdir(p)),
    os.path.join(DATA_DIR, "synthetic_cracks"),  # fallback default
)

# Mapping: synthetic subfolder pairs -> target class name
# corrosion = debonding (renamed); Hair Crack + Wide Crack merged per type
SYNTHETIC_MAPPING = {
    "debonding": [
        ("Synthetic corrosion cracks by AutoCAD (0-5)", "Hair Crack"),
        ("Synthetic corrosion cracks by AutoCAD (0-5)", "Wide Crack"),
    ],
    "flexural": [
        ("Synthetic flexural cracks by AutoCAD (75-90)", "Hair Crack"),
        ("Synthetic flexural cracks by AutoCAD (75-90)", "Wide Crack"),
    ],
    "shear": [
        ("Synthetic shear cracks by AutoCAD (30-50)", "Hair crack"),
        ("Synthetic shear cracks by AutoCAD (30-50)", "Wide crack"),
    ],
}

# Split directory with synthetic data injected into training only
SPLIT_SYNTHETIC_DIR = os.path.join(DATA_DIR, "split_synthetic")

# === QU Dataset Mapping ===
QU_DATASET_DIR = os.path.join(DATA_DIR, "Dataset edited by QU")
DATASET_MAPPING = {
    "debonding": os.path.join(QU_DATASET_DIR, "Positive crack", "01 Single crack", "Debonding crack"),
    "flexural": os.path.join(QU_DATASET_DIR, "Positive crack", "01 Single crack", "Flexural crack"),
    "shear": os.path.join(QU_DATASET_DIR, "Positive crack", "01 Single crack", "Shear crack"),
    "multi_crack": os.path.join(QU_DATASET_DIR, "Positive crack", "02 Multi cracks"),
    "no_crack": os.path.join(QU_DATASET_DIR, "Negative - no crack"),
    "others": os.path.join(QU_DATASET_DIR, "Positive crack", "03 Others"),
}

# === Image Settings ===
IMG_SIZE = 299
NUM_CLASSES = 6
CLASS_NAMES = ["debonding", "flexural", "shear", "multi_crack", "no_crack", "others"]

# === Dataset Splits ===
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# === Training Hyperparameters ===
# Per-stage batch sizes (optimized for RTX 3050 4GB VRAM)
STAGE1_BATCH_SIZE = 16  # Frozen backbone — low memory
STAGE2_BATCH_SIZE = 16  # Partial unfreeze — moderate memory
STAGE3_BATCH_SIZE = 8   # Full unfreeze — high memory, OOM risk at 16

# Stage 1: Feature extraction (frozen backbone)
STAGE1_LR = 1e-3
STAGE1_EPOCHS = 15

# Stage 2: Partial fine-tuning (unfreeze from mixed7)
STAGE2_LR = 1e-4
STAGE2_EPOCHS = 20

# Stage 3: Full fine-tuning (all layers)
STAGE3_LR = 1e-5
STAGE3_EPOCHS = 30

# === Device / Workers ===
NUM_WORKERS = 4
USE_MULTIPROCESSING = False  # Must be False on Windows with TF

# === Model Architecture ===
DROPOUT_RATE = 0.4
L2_REG = 1e-4
DENSE_UNITS_1 = 256
DENSE_UNITS_2 = 128

# === Augmentation Cap ===
MAX_AUG_FACTOR = 6  # Max times any single image is seen per epoch (×6 cap rule)
MULTI_AUG_FACTOR = 3  # Multiplier for multi-crack oversampling in hierarchical stage 2

# === Augmentation Parameters ===
ROTATION_RANGE = 15
WIDTH_SHIFT = 0.10
HEIGHT_SHIFT = 0.10
BRIGHTNESS_RANGE = (0.80, 1.20)
ZOOM_RANGE = 0.15
HORIZONTAL_FLIP = True
VERTICAL_FLIP = False
FILL_MODE = "reflect"
CHANNEL_SHIFT_RANGE = 10.0
SHEAR_RANGE = 0.0  # Disabled — stacks angular noise on rotation

# === Preprocessing Parameters ===
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
# Bilateral filter (denoising before CLAHE — avoids noise amplification)
BILATERAL_DIAMETER = 7
BILATERAL_SIGMA_COLOR = 35
BILATERAL_SIGMA_SPACE = 5

# === Callbacks ===
EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

# === Inference ===
CONFIDENCE_THRESHOLD = 0.70

# ============================================================
# PER-MODEL CONFIGURATIONS
# ============================================================

# --- InceptionV3 + CBAM ---
CBAM_REDUCTION_RATIO = 16
CBAM_KERNEL_SIZE = 7

# --- CNN from Scratch ---
CNN_IMG_SIZE = 224
CNN_FILTERS = [32, 64, 128, 256]
CNN_DROPOUT_CONV = 0.25
CNN_DROPOUT_DENSE = 0.5
CNN_DENSE_UNITS = [256, 128]
CNN_LR = 1e-3
CNN_EPOCHS = 100
CNN_BATCH_SIZE = 32
CNN_EARLY_STOPPING_PATIENCE = 15
CNN_LABEL_SMOOTHING = 0.1

# --- SVM / Traditional ML ---
SVM_IMG_SIZE = 128
SVM_HOG_ORIENTATIONS = 9
SVM_HOG_PIXELS_PER_CELL = (8, 8)
SVM_HOG_CELLS_PER_BLOCK = (2, 2)
SVM_LBP_RADIUS = 3
SVM_LBP_N_POINTS = 24
SVM_GLCM_DISTANCES = [1, 3, 5]
SVM_PCA_VARIANCE = 0.95

# --- YOLOv8 ---
YOLO_IMG_SIZE = 224
YOLO_MODEL = "yolov8s-cls.pt"
YOLO_EPOCHS = 100
YOLO_PATIENCE = 15
YOLO_LR0 = 0.001
YOLO_LRF = 0.01
YOLO_DROPOUT = 0.3
