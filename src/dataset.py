"""Dataset loading, splitting, and generator creation."""

import os
import shutil
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.augmentation import get_train_datagen, get_val_test_datagen


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def prepare_dataset(
    mapping: dict = None,
    target_dir: str = None,
) -> dict[str, int]:
    """Populate flat class folders from the QU nested dataset structure.

    Creates symlinks from the QU dataset directories to flat class folders
    under target_dir. Falls back to file copying if symlinks fail (common
    on Windows without developer mode).

    Args:
        mapping: Dict mapping class_name -> source directory path.
                 Defaults to config.DATASET_MAPPING.
        target_dir: Target directory for flat class folders.
                    Defaults to config.DATA_DIR.

    Returns:
        Dict mapping class_name -> image count.
    """
    if mapping is None:
        mapping = config.DATASET_MAPPING
    if target_dir is None:
        target_dir = config.DATA_DIR

    counts = {}

    for class_name, source_dir in mapping.items():
        dest_dir = os.path.join(target_dir, class_name)

        if not os.path.isdir(source_dir):
            print(f"WARNING: Source not found for '{class_name}': {source_dir}")
            counts[class_name] = 0
            continue

        # Skip if destination already exists and has files
        if os.path.isdir(dest_dir):
            existing = [f for f in os.listdir(dest_dir)
                        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
            if existing:
                counts[class_name] = len(existing)
                print(f"  {class_name:<12}: {len(existing):>6} images (already exists, skipping)")
                continue

        os.makedirs(dest_dir, exist_ok=True)

        # Collect source files
        src_files = [f for f in os.listdir(source_dir)
                     if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]

        if not src_files:
            print(f"WARNING: No images found in source for '{class_name}': {source_dir}")
            counts[class_name] = 0
            continue

        # Try symlinks first, fall back to copying
        use_copy = False
        try:
            test_src = os.path.join(source_dir, src_files[0])
            test_dst = os.path.join(dest_dir, "__symlink_test__")
            os.symlink(test_src, test_dst)
            os.remove(test_dst)
        except (OSError, NotImplementedError):
            use_copy = True

        method = "copying" if use_copy else "symlinking"
        print(f"  {class_name:<12}: {len(src_files):>6} images ({method})")

        for fname in src_files:
            src_path = os.path.join(source_dir, fname)
            dst_path = os.path.join(dest_dir, fname)
            if os.path.exists(dst_path):
                continue
            if use_copy:
                shutil.copy2(src_path, dst_path)
            else:
                os.symlink(src_path, dst_path)

        counts[class_name] = len(src_files)

    # Print summary
    total = sum(counts.values())
    print(f"\nDataset preparation complete: {total:,} images across {len(counts)} classes")
    for name in config.CLASS_NAMES:
        if name in counts:
            print(f"  {name:<12}: {counts[name]:>6}")

    return counts


def collect_file_paths(data_dir: str) -> tuple[list[str], list[str]]:
    """Collect all image file paths and their labels from the dataset directory.

    Args:
        data_dir: Root directory with class subfolders.

    Returns:
        Tuple of (file_paths, labels).
    """
    file_paths = []
    labels = []

    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: class directory not found: {class_dir}")
            continue

        for fname in os.listdir(class_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(class_name)

    return file_paths, labels


def split_dataset(
    data_dir: str = config.DATA_DIR,
    split_dir: str = config.SPLIT_DIR,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """Split dataset into train/val/test with stratification and copy files.

    Args:
        data_dir: Root directory containing class subfolders.
        split_dir: Output directory for the split dataset.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with split statistics.
    """
    file_paths, labels = collect_file_paths(data_dir)

    if len(file_paths) == 0:
        raise ValueError(f"No images found in {data_dir}. "
                         f"Expected subfolders: {config.CLASS_NAMES}")

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    X_train, X_valtest, y_train, y_valtest = train_test_split(
        file_paths, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / val_test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_valtest, y_valtest,
        test_size=relative_test_ratio,
        stratify=y_valtest,
        random_state=seed,
    )

    # Copy files into split directory structure
    stats = defaultdict(lambda: defaultdict(int))
    splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}

    for split_name, (paths, split_labels) in splits.items():
        for src_path, label in zip(paths, split_labels):
            dest_dir = os.path.join(split_dir, split_name, label)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dest_path)
            stats[split_name][label] += 1

    # Print summary
    print("\nDataset split summary:")
    print(f"{'Split':<8} " + " ".join(f"{c:<12}" for c in config.CLASS_NAMES) + f" {'Total':<8}")
    print("-" * 70)
    for split_name in ["train", "val", "test"]:
        counts = [stats[split_name][c] for c in config.CLASS_NAMES]
        total = sum(counts)
        print(f"{split_name:<8} " + " ".join(f"{c:<12}" for c in counts) + f" {total:<8}")

    return dict(stats)


def get_generators(
    split_dir: str = config.SPLIT_DIR,
    batch_size: int = config.STAGE1_BATCH_SIZE,
) -> tuple:
    """Create data generators for train, validation, and test sets.

    Args:
        split_dir: Directory containing train/val/test subfolders.
        batch_size: Batch size for generators.

    Returns:
        Tuple of (train_generator, val_generator, test_generator).
    """
    train_datagen = get_train_datagen()
    val_test_datagen = get_val_test_datagen()

    train_gen = train_datagen.flow_from_directory(
        os.path.join(split_dir, "train"),
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        classes=config.CLASS_NAMES,
        shuffle=True,
        seed=config.RANDOM_SEED,
    )

    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(split_dir, "val"),
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        classes=config.CLASS_NAMES,
        shuffle=False,
    )

    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(split_dir, "test"),
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        classes=config.CLASS_NAMES,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def compute_class_weights(split_dir: str = config.SPLIT_DIR) -> dict:
    """Compute class weights to handle imbalanced datasets.

    Prints detailed class distribution analysis including per-class counts,
    weights, and effective weighted contribution.

    Args:
        split_dir: Directory containing the split dataset.

    Returns:
        Dictionary mapping class index to weight.
    """
    train_dir = os.path.join(split_dir, "train")
    labels = []
    class_counts = {}

    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            count = len([f for f in os.listdir(class_dir)
                        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS])
            labels.extend([class_idx] * count)
            class_counts[class_idx] = count

    labels = np.array(labels)
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    weight_dict = dict(enumerate(weights))

    # Print detailed class distribution analysis
    total_train = sum(class_counts.values())
    max_count = max(class_counts.values())
    print("\n" + "=" * 75)
    print("CLASS DISTRIBUTION & BALANCING ANALYSIS (Training Set)")
    print("=" * 75)
    print(f"{'Class':<14} {'Files':>7} {'%':>7} {'Weight':>8} {'Effective':>10} {'Augment x':>10}")
    print("-" * 75)
    for idx, class_name in enumerate(config.CLASS_NAMES):
        count = class_counts.get(idx, 0)
        pct = 100.0 * count / total_train if total_train else 0
        w = weight_dict.get(idx, 1.0)
        effective = count * w
        aug_factor = max_count / count if count > 0 else 0
        print(f"  {class_name:<12} {count:>7,} {pct:>6.1f}% {w:>8.3f} {effective:>10,.1f} {aug_factor:>9.1f}x")
    print("-" * 75)
    print(f"  {'TOTAL':<12} {total_train:>7,}")
    print()
    print("How balancing works:")
    print("  - On-the-fly augmentation (flip, rotate, shift, zoom, brightness)")
    print("    is applied UNIFORMLY to all classes each epoch.")
    print("  - class_weight scales the LOSS: minority classes contribute more")
    print("    gradient signal per sample, so the model learns them equally.")
    print(f"  - Effective = Files x Weight (balanced target: ~{total_train / len(class_counts):,.0f} per class)")
    print("=" * 75)

    return weight_dict
