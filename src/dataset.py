"""Dataset loading, splitting, balanced generator, and class weight computation."""

import math
import os
import shutil
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence

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


class BalancedGenerator(Sequence):
    """Training generator with x6-capped per-class balanced sampling.

    Implements the rule:
        TARGET_PER_CLASS = smallest_class_count x MAX_AUG_FACTOR

    - The smallest class is seen exactly x6 per epoch (the safe cap)
    - Larger classes are UNDERSAMPLED to the same target (seen <x6)
    - On-the-fly augmentation (flip, rotate, shift, zoom, brightness)
      is applied to every sample, so each pass is a different image
    - Indices are reshuffled every epoch for maximum diversity
    - Only ORIGINAL files are used (any 'aug_' prefixed files are ignored)
    """

    def __init__(
        self,
        split_dir: str = config.SPLIT_DIR,
        batch_size: int = config.STAGE1_BATCH_SIZE,
        seed: int = config.RANDOM_SEED,
    ):
        train_dir = os.path.join(split_dir, "train")
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

        # Load ORIGINAL file paths per class (exclude any offline aug_ files)
        self.class_files = {}
        self.original_counts = {}
        for cls_name in config.CLASS_NAMES:
            cls_dir = os.path.join(train_dir, cls_name)
            files = sorted([
                os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
                and not f.startswith("aug_")
            ])
            self.class_files[cls_name] = files
            self.original_counts[cls_name] = len(files)

        # x6 cap rule: target = smallest_class x MAX_AUG_FACTOR
        self.smallest_count = min(self.original_counts.values())
        self.target_per_class = self.smallest_count * config.MAX_AUG_FACTOR

        # Augmentation pipeline (same transforms as before, applied on-the-fly)
        self.datagen = get_train_datagen()

        # Build initial epoch indices and print config
        self._build_epoch_indices()
        self._print_config()

    def _print_config(self):
        """Print the balanced sampling configuration."""
        total_original = sum(self.original_counts.values())
        total_balanced = self.target_per_class * len(config.CLASS_NAMES)

        print(f"\n{'='*70}")
        print(f"BALANCED GENERATOR — x{config.MAX_AUG_FACTOR} CAP RULE")
        print(f"{'='*70}")
        print(f"  Smallest class:     {self.smallest_count:,} files")
        print(f"  Target per class:   {self.target_per_class:,}"
              f"  ({self.smallest_count:,} x {config.MAX_AUG_FACTOR})")
        print(f"  Original files:     {total_original:,}")
        print(f"  Balanced per epoch: {total_balanced:,}")
        print()
        print(f"  {'Class':<14} {'Original':>9} {'Per Epoch':>10} {'Multiplier':>11}")
        print(f"  {'-'*48}")
        for cls_name in config.CLASS_NAMES:
            orig = self.original_counts[cls_name]
            target = self.target_per_class
            mult = target / orig if orig > 0 else 0
            direction = "oversample" if mult > 1.0 else "undersample"
            print(f"  {cls_name:<14} {orig:>9,} {target:>10,}"
                  f"    x{mult:.1f} ({direction})")
        print(f"  {'-'*48}")
        print(f"  {'TOTAL':<14} {total_original:>9,} {total_balanced:>10,}")
        print()
        print(f"  No image is seen more than x{config.MAX_AUG_FACTOR} per epoch.")
        print(f"  Each pass applies random augmentation (new image every time).")
        print(f"  Val/test sets: NO augmentation, original distribution.")
        print(f"{'='*70}")

    def _build_epoch_indices(self):
        """Build balanced list of (file_path, class_index) for one epoch.

        - Classes with more files than target: randomly subsample (no replacement)
        - Classes with fewer files than target: repeat up to x6 and truncate
        """
        indices = []
        for cls_idx, cls_name in enumerate(config.CLASS_NAMES):
            files = self.class_files[cls_name]
            n = len(files)
            target = self.target_per_class

            if n >= target:
                # Undersample: randomly pick 'target' files, no repeats
                selected = list(self.rng.choice(files, size=target, replace=False))
            else:
                # Oversample: repeat file list and truncate to target
                # ceil(target/n) <= MAX_AUG_FACTOR, so no file exceeds the cap
                repeats = math.ceil(target / n)
                pool = list(files) * repeats
                self.rng.shuffle(pool)
                selected = pool[:target]

            indices.extend([(f, cls_idx) for f in selected])

        self.rng.shuffle(indices)
        self.indices = indices

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx):
        batch = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.zeros((len(batch), config.IMG_SIZE, config.IMG_SIZE, 3),
                     dtype=np.float32)
        y = np.zeros((len(batch), config.NUM_CLASSES), dtype=np.float32)

        for i, (fpath, cls_idx) in enumerate(batch):
            img = load_img(fpath, target_size=(config.IMG_SIZE, config.IMG_SIZE))
            x = img_to_array(img)
            x = self.datagen.random_transform(x)   # on-the-fly augmentation
            x = self.datagen.standardize(x)         # preprocessing + normalization
            X[i] = x
            y[i, cls_idx] = 1.0

        return X, y

    def on_epoch_end(self):
        """Reshuffle indices at the end of every epoch."""
        self._build_epoch_indices()

    @property
    def samples(self):
        """Total samples per epoch (target_per_class x num_classes)."""
        return len(self.indices)

    @property
    def class_indices(self):
        return {c: i for i, c in enumerate(config.CLASS_NAMES)}


def get_generators(
    split_dir: str = config.SPLIT_DIR,
    batch_size: int = config.STAGE1_BATCH_SIZE,
) -> tuple:
    """Create data generators for train, validation, and test sets.

    Training uses BalancedGenerator (x6 cap rule, online augmentation).
    Validation and test use flow_from_directory (no augmentation).

    Args:
        split_dir: Directory containing train/val/test subfolders.
        batch_size: Batch size for generators.

    Returns:
        Tuple of (train_generator, val_generator, test_generator).
    """
    train_gen = BalancedGenerator(split_dir, batch_size)

    val_test_datagen = get_val_test_datagen()

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
            # Count only ORIGINAL files (exclude any offline aug_ files)
            count = len([f for f in os.listdir(class_dir)
                        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
                        and not f.startswith("aug_")])
            labels.extend([class_idx] * count)
            class_counts[class_idx] = count

    labels = np.array(labels)
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    weight_dict = dict(enumerate(weights))

    # Print class weights analysis
    total_real = sum(class_counts.values())
    target_per_class = min(class_counts.values()) * config.MAX_AUG_FACTOR
    print(f"\n{'='*75}")
    print("CLASS WEIGHTS (computed from ORIGINAL file counts)")
    print(f"{'='*75}")
    print(f"  {'Class':<14} {'Real Files':>10} {'%':>7} {'Weight':>8} {'Per Epoch':>10}")
    print(f"  {'-'*55}")
    for idx, class_name in enumerate(config.CLASS_NAMES):
        count = class_counts.get(idx, 0)
        pct = 100.0 * count / total_real if total_real else 0
        w = weight_dict.get(idx, 1.0)
        print(f"  {class_name:<12} {count:>10,} {pct:>6.1f}% {w:>8.3f} {target_per_class:>10,}")
    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<14} {total_real:>10,}{'':>16} {target_per_class * len(class_counts):>10,}")
    print()
    print("  Weights scale the LOSS: a misclassified minority image costs more")
    print("  than a misclassified majority image. This works on top of the")
    print(f"  x{config.MAX_AUG_FACTOR}-capped balanced sampling — both are always used together.")
    print(f"{'='*75}")

    return weight_dict
