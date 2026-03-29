"""Central configuration for the crack classification system."""

import os

# === Paths ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SPLIT_DIR = os.path.join(DATA_DIR, "split")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

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
MORPH_KERNEL_SIZE = 3

# === Callbacks ===
EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

# === Inference ===
CONFIDENCE_THRESHOLD = 0.70
