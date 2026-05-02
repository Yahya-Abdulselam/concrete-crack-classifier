"""Dataset loading, splitting, balanced sampling, and class weight computation (PyTorch)."""

import math
import os
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.augmentation import get_train_transforms, get_val_test_transforms
from src.preprocessing import bilateral_denoise, apply_clahe


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
    """Collect all image file paths and their labels from a directory.

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
    """Split dataset into train/val/test with stratification and copy files."""
    file_paths, labels = collect_file_paths(data_dir)

    if len(file_paths) == 0:
        raise ValueError(f"No images found in {data_dir}. "
                         f"Expected subfolders: {config.CLASS_NAMES}")

    val_test_ratio = val_ratio + test_ratio
    X_train, X_valtest, y_train, y_valtest = train_test_split(
        file_paths, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed,
    )

    relative_test_ratio = test_ratio / val_test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_valtest, y_valtest,
        test_size=relative_test_ratio,
        stratify=y_valtest,
        random_state=seed,
    )

    stats = defaultdict(lambda: defaultdict(int))
    splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}

    for split_name, (paths, split_labels) in splits.items():
        for src_path, label in zip(paths, split_labels):
            dest_dir = os.path.join(split_dir, split_name, label)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dest_path)
            stats[split_name][label] += 1

    print("\nDataset split summary:")
    print(f"{'Split':<8} " + " ".join(f"{c:<12}" for c in config.CLASS_NAMES) + f" {'Total':<8}")
    print("-" * 70)
    for split_name in ["train", "val", "test"]:
        counts = [stats[split_name][c] for c in config.CLASS_NAMES]
        total = sum(counts)
        print(f"{split_name:<8} " + " ".join(f"{c:<12}" for c in counts) + f" {total:<8}")

    return dict(stats)


def prepare_synthetic_split(
    original_split_dir: str = None,
    synthetic_dir: str = None,
    synthetic_mapping: dict = None,
    target_dir: str = None,
) -> dict[str, int]:
    """Create an augmented split directory with synthetic images in training only.

    Materializes target_dir with:
        train/  — original train files + synthetic images (symlinked or copied)
        val/    — symlink (or copy) of original val directory
        test/   — symlink (or copy) of original test directory

    Synthetic filenames are prefixed with 'synth_' for provenance tracking.
    Idempotent: skips if target_dir already contains synthetic files.

    Args:
        original_split_dir: Base split directory (default: config.SPLIT_DIR).
        synthetic_dir: Root of synthetic crack images (default: config.SYNTHETIC_DIR).
        synthetic_mapping: Dict mapping class_name -> list of (subfolder, sub_subfolder)
                          tuples (default: config.SYNTHETIC_MAPPING).
        target_dir: Output directory (default: config.SPLIT_SYNTHETIC_DIR).

    Returns:
        Dict mapping class_name -> total training image count.
    """
    if original_split_dir is None:
        original_split_dir = config.SPLIT_DIR
    if synthetic_dir is None:
        synthetic_dir = config.SYNTHETIC_DIR
    if synthetic_mapping is None:
        synthetic_mapping = config.SYNTHETIC_MAPPING
    if target_dir is None:
        target_dir = config.SPLIT_SYNTHETIC_DIR

    # Idempotency: check if already materialized
    train_dir = os.path.join(target_dir, "train")
    if os.path.isdir(train_dir):
        has_synth = False
        for cls_name in config.CLASS_NAMES:
            cls_dir = os.path.join(train_dir, cls_name)
            if os.path.isdir(cls_dir):
                if any(f.startswith("synth_") for f in os.listdir(cls_dir)):
                    has_synth = True
                    break
        if has_synth:
            print(f"Synthetic split already exists: {target_dir}")
            counts = {}
            for cls_name in config.CLASS_NAMES:
                cls_dir = os.path.join(train_dir, cls_name)
                if os.path.isdir(cls_dir):
                    n = len([f for f in os.listdir(cls_dir)
                             if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS])
                    counts[cls_name] = n
                else:
                    counts[cls_name] = 0
            total = sum(counts.values())
            print(f"  Training images: {total:,}")
            for cls_name in config.CLASS_NAMES:
                print(f"    {cls_name:<14}: {counts[cls_name]:>6,}")
            return counts

    os.makedirs(target_dir, exist_ok=True)

    # Detect symlink capability
    use_copy = False
    probe_src = os.path.join(original_split_dir, "val")
    if os.path.isdir(probe_src):
        test_dst = os.path.join(target_dir, "__symlink_test__")
        try:
            os.symlink(probe_src, test_dst, target_is_directory=True)
            os.remove(test_dst)
        except (OSError, NotImplementedError):
            use_copy = True

    method = "copying" if use_copy else "symlinking"
    print(f"\nPreparing synthetic split ({method})...")
    print(f"  Source split: {original_split_dir}")
    print(f"  Synthetic:    {synthetic_dir}")
    print(f"  Target:       {target_dir}")

    # --- val/ and test/: link or copy from original ---
    for subset in ("val", "test"):
        src = os.path.join(original_split_dir, subset)
        dst = os.path.join(target_dir, subset)
        if os.path.exists(dst):
            continue
        if use_copy:
            shutil.copytree(src, dst)
        else:
            try:
                os.symlink(os.path.abspath(src), dst, target_is_directory=True)
            except (OSError, NotImplementedError):
                shutil.copytree(src, dst)
        print(f"  {subset}/: linked from original")

    # --- train/: original files + synthetic ---
    orig_counts = {}
    synth_counts = {}

    for cls_name in config.CLASS_NAMES:
        src_cls = os.path.join(original_split_dir, "train", cls_name)
        dst_cls = os.path.join(train_dir, cls_name)
        os.makedirs(dst_cls, exist_ok=True)

        # Copy/link original training files
        n_orig = 0
        if os.path.isdir(src_cls):
            for fname in os.listdir(src_cls):
                if os.path.splitext(fname)[1].lower() not in SUPPORTED_EXTENSIONS:
                    continue
                src_path = os.path.join(src_cls, fname)
                dst_path = os.path.join(dst_cls, fname)
                if os.path.exists(dst_path):
                    n_orig += 1
                    continue
                if use_copy:
                    shutil.copy2(src_path, dst_path)
                else:
                    try:
                        os.symlink(os.path.abspath(src_path), dst_path)
                    except (OSError, NotImplementedError):
                        shutil.copy2(src_path, dst_path)
                n_orig += 1
        orig_counts[cls_name] = n_orig

        # Add synthetic images for mapped classes
        n_synth = 0
        if cls_name in synthetic_mapping:
            for subfolder, sub_subfolder in synthetic_mapping[cls_name]:
                src_dir = os.path.join(synthetic_dir, subfolder, sub_subfolder)
                if not os.path.isdir(src_dir):
                    print(f"  WARNING: Synthetic source not found: {src_dir}")
                    continue
                for fname in os.listdir(src_dir):
                    if os.path.splitext(fname)[1].lower() not in SUPPORTED_EXTENSIONS:
                        continue
                    dst_name = f"synth_{fname}"
                    dst_path = os.path.join(dst_cls, dst_name)
                    if os.path.exists(dst_path):
                        n_synth += 1
                        continue
                    src_path = os.path.join(src_dir, fname)
                    if use_copy:
                        shutil.copy2(src_path, dst_path)
                    else:
                        try:
                            os.symlink(os.path.abspath(src_path), dst_path)
                        except (OSError, NotImplementedError):
                            shutil.copy2(src_path, dst_path)
                    n_synth += 1
        synth_counts[cls_name] = n_synth

    # Print summary
    total_orig = sum(orig_counts.values())
    total_synth = sum(synth_counts.values())
    total_all = total_orig + total_synth

    print(f"\n{'='*70}")
    print("SYNTHETIC SPLIT PREPARED (training only)")
    print(f"{'='*70}")
    print(f"  {'Class':<14} {'Original':>9} {'Synthetic':>10} {'Total':>8}")
    print(f"  {'-'*45}")
    for cls_name in config.CLASS_NAMES:
        o = orig_counts[cls_name]
        s = synth_counts[cls_name]
        print(f"  {cls_name:<14} {o:>9,} {s:>10,} {o+s:>8,}")
    print(f"  {'-'*45}")
    print(f"  {'TOTAL':<14} {total_orig:>9,} {total_synth:>10,} {total_all:>8,}")
    print(f"{'='*70}")
    print(f"  Val/test: unchanged (from {original_split_dir})")

    return {cls: orig_counts[cls] + synth_counts[cls] for cls in config.CLASS_NAMES}


class CrackDataset(Dataset):
    """PyTorch Dataset for crack classification images.

    Loads images from split directory, applies bilateral denoise + CLAHE
    preprocessing, then torchvision transforms (augmentation + normalization).
    """

    def __init__(
        self,
        split_dir: str,
        subset: str,
        transform=None,
        img_size: int = config.IMG_SIZE,
        class_names: list[str] = None,
        mirror_synthetic: bool = False,
    ):
        """
        Args:
            split_dir: Root split directory containing train/val/test subfolders.
            subset: Which subset ("train", "val", or "test").
            transform: Torchvision transform pipeline.
            img_size: Target image size for preprocessing.
            class_names: Restrict to a subset of classes (default: config.CLASS_NAMES).
            mirror_synthetic: If True, every file with basename starting with
                "synth_" appears twice in the dataset — once as-is, once
                horizontally flipped. The flip is applied in __getitem__ before
                the transform pipeline (so the standard random aug still runs
                on top). Honors the civil-engineer requirement that all AutoCAD
                synthetic crack images are seen in both orientations every epoch.
        """
        self.img_size = img_size
        self.transform = transform
        self.class_names = list(class_names) if class_names is not None else list(config.CLASS_NAMES)

        subset_dir = os.path.join(split_dir, subset)
        self.file_paths: list[str] = []
        self.labels: list[int] = []
        self.flips: list[bool] = []
        self.class_counts: dict[str, int] = {}

        for cls_idx, cls_name in enumerate(self.class_names):
            cls_dir = os.path.join(subset_dir, cls_name)
            if not os.path.isdir(cls_dir):
                self.class_counts[cls_name] = 0
                continue

            files = sorted([
                os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
                and not f.startswith("aug_")
            ])

            # Add original entries
            self.file_paths.extend(files)
            self.labels.extend([cls_idx] * len(files))
            self.flips.extend([False] * len(files))

            # Mirror synthetic files: each synth_* path appended again with flip=True
            n_mirrored = 0
            if mirror_synthetic:
                synth_files = [p for p in files if os.path.basename(p).startswith("synth_")]
                self.file_paths.extend(synth_files)
                self.labels.extend([cls_idx] * len(synth_files))
                self.flips.extend([True] * len(synth_files))
                n_mirrored = len(synth_files)

            self.class_counts[cls_name] = len(files) + n_mirrored

        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)

        # Forced horizontal mirror for synthetic-mirror entries (deterministic)
        if self.flips[idx]:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Apply bilateral denoise + CLAHE on numpy array
        img_array = np.array(img)
        img_array = bilateral_denoise(img_array)
        img_array = apply_clahe(img_array)
        img = Image.fromarray(img_array)

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label


def get_balanced_sampler(
    dataset: CrackDataset,
    max_aug_factor: int = config.MAX_AUG_FACTOR,
) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler implementing the x6 cap rule.

    Target per class = smallest_class_count x max_aug_factor.
    Minority classes oversampled, majority classes undersampled.

    Args:
        dataset: CrackDataset with labels and class_counts.
        max_aug_factor: Maximum oversampling factor.

    Returns:
        WeightedRandomSampler for use with DataLoader.
    """
    counts = dataset.class_counts
    class_names = dataset.class_names
    smallest_count = min(c for c in counts.values() if c > 0)
    target_per_class = smallest_count * max_aug_factor
    total_samples = target_per_class * len(class_names)

    # Compute per-sample weight: inverse of class size, scaled to hit target
    class_weights = {}
    for cls_idx, cls_name in enumerate(class_names):
        n = counts.get(cls_name, 0)
        if n > 0:
            class_weights[cls_idx] = target_per_class / n
        else:
            class_weights[cls_idx] = 0.0

    sample_weights = np.array([class_weights[label] for label in dataset.labels],
                              dtype=np.float64)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,
        replacement=True,
    )

    # Print config
    total_original = sum(counts.values())
    print(f"\n{'='*70}")
    print(f"BALANCED SAMPLER — x{max_aug_factor} CAP RULE")
    print(f"{'='*70}")
    print(f"  Classes:            {len(class_names)}  ({', '.join(class_names)})")
    print(f"  Smallest class:     {smallest_count:,} files")
    print(f"  Target per class:   {target_per_class:,}"
          f"  ({smallest_count:,} x {max_aug_factor})")
    print(f"  Original files:     {total_original:,}")
    print(f"  Balanced per epoch: {total_samples:,}")
    print()
    print(f"  {'Class':<14} {'Original':>9} {'Per Epoch':>10} {'Multiplier':>11}")
    print(f"  {'-'*48}")
    for cls_name in class_names:
        orig = counts.get(cls_name, 0)
        mult = target_per_class / orig if orig > 0 else 0
        direction = "oversample" if mult > 1.0 else "undersample"
        print(f"  {cls_name:<14} {orig:>9,} {target_per_class:>10,}"
              f"    x{mult:.1f} ({direction})")
    print(f"  {'-'*48}")
    print(f"  {'TOTAL':<14} {total_original:>9,} {total_samples:>10,}")
    print(f"{'='*70}")

    return sampler


def get_dataloaders(
    split_dir: str = config.SPLIT_DIR,
    batch_size: int = config.STAGE1_BATCH_SIZE,
    img_size: int = config.IMG_SIZE,
    normalize: str = "imagenet",
    num_workers: int = 4,
    class_names: list[str] = None,
    mirror_synthetic: bool = False,
    test_split_dir: str = None,
) -> tuple:
    """Create train, validation, and test DataLoaders.

    Training uses WeightedRandomSampler (x6 cap rule, online augmentation).
    Validation and test use sequential loading (no augmentation).

    Args:
        split_dir: Directory containing train/val/test subfolders.
        batch_size: Batch size for all loaders.
        img_size: Target image size.
        normalize: Normalization mode ("imagenet" or "rescale").
        num_workers: Number of data loading workers.
        class_names: Restrict to a subset of classes (default: all 6).
        mirror_synthetic: Forced horizontal mirror for every "synth_*" file
            in the train set (val/test never mirror). See CrackDataset docstring.
        test_split_dir: Optional override for val+test source. Useful when
            training uses split_synthetic/ but test must come from the clean
            split/ for fair comparison. Defaults to split_dir.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_transform = get_train_transforms(img_size, normalize)
    val_test_transform = get_val_test_transforms(img_size, normalize)
    val_test_dir = test_split_dir if test_split_dir is not None else split_dir

    train_dataset = CrackDataset(
        split_dir, "train", train_transform, img_size,
        class_names=class_names, mirror_synthetic=mirror_synthetic,
    )
    val_dataset = CrackDataset(
        val_test_dir, "val", val_test_transform, img_size,
        class_names=class_names, mirror_synthetic=False,
    )
    test_dataset = CrackDataset(
        val_test_dir, "test", val_test_transform, img_size,
        class_names=class_names, mirror_synthetic=False,
    )

    sampler = get_balanced_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_dataset):,} images, {len(train_loader):,} batches (balanced)")
    print(f"  Val:   {len(val_dataset):,} images, {len(val_loader):,} batches")
    print(f"  Test:  {len(test_dataset):,} images, {len(test_loader):,} batches")

    return train_loader, val_loader, test_loader


def compute_class_weights(
    split_dir: str = config.SPLIT_DIR,
    class_names: list[str] = None,
    count_synth_mirror: bool = False,
) -> dict:
    """Compute class weights to handle imbalanced datasets.

    Args:
        split_dir: Directory containing the split dataset.
        class_names: Restrict to a subset of classes (default: all 6 from config).
        count_synth_mirror: If True, every "synth_*" file is counted twice
            (matching the doubled view CrackDataset(mirror_synthetic=True) uses).
            Keeps the loss-weight tensor consistent with what the sampler sees.

    Returns:
        Dictionary mapping class index to weight.
    """
    if class_names is None:
        class_names = list(config.CLASS_NAMES)

    train_dir = os.path.join(split_dir, "train")
    labels = []
    class_counts = {}

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            files = [f for f in os.listdir(class_dir)
                     if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
                     and not f.startswith("aug_")]
            count = len(files)
            if count_synth_mirror:
                count += sum(1 for f in files if f.startswith("synth_"))
            labels.extend([class_idx] * count)
            class_counts[class_idx] = count

    labels = np.array(labels)
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    weight_dict = dict(enumerate(weights))

    total_real = sum(class_counts.values())
    smallest = min(class_counts.values()) if class_counts else 0
    target_per_class = smallest * config.MAX_AUG_FACTOR
    label = "ORIGINAL + MIRRORED" if count_synth_mirror else "ORIGINAL"
    print(f"\n{'='*75}")
    print(f"CLASS WEIGHTS (computed from {label} file counts)")
    print(f"{'='*75}")
    print(f"  Classes: {len(class_names)} ({', '.join(class_names)})")
    print(f"  {'Class':<14} {'Files':>10} {'%':>7} {'Weight':>8} {'Per Epoch':>10}")
    print(f"  {'-'*55}")
    for idx, class_name in enumerate(class_names):
        count = class_counts.get(idx, 0)
        pct = 100.0 * count / total_real if total_real else 0
        w = weight_dict.get(idx, 1.0)
        print(f"  {class_name:<12} {count:>10,} {pct:>6.1f}% {w:>8.3f} {target_per_class:>10,}")
    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<14} {total_real:>10,}{'':>16} {target_per_class * len(class_counts):>10,}")
    print(f"{'='*75}")

    return weight_dict


def prepare_3class_yolo_split(
    with_synthetic: bool,
    class_names: tuple = ("debonding", "flexural", "shear"),
    original_split_dir: str = None,
    synthetic_dir: str = None,
    synthetic_mapping: dict = None,
    target_dir: str = None,
) -> dict[str, int]:
    """Materialize a 3-class split directory for YOLO (Ultralytics).

    YOLO's dataloader cannot accept a custom flip flag, so for the synthetic
    variant we bake horizontal mirrors of every synthetic image to disk
    (`synth_mirror_*.jpg`). Real images are symlinked unchanged. Val and test
    are always taken from the clean original split (no synth, no mirror).

    Args:
        with_synthetic: If False -> data/split_3class/ (QU only).
                        If True  -> data/split_3class_synth/ (QU + synth + mirror).
        class_names: Tuple of class names to include.
        original_split_dir: Source split (default: config.SPLIT_DIR).
        synthetic_dir: Synthetic crack root (default: config.SYNTHETIC_DIR).
        synthetic_mapping: Class -> synth subfolders (default: config.SYNTHETIC_MAPPING).
        target_dir: Output dir (default: data/split_3class[_synth]/).

    Returns:
        Dict mapping class_name -> total training image count (post-mirror).
    """
    if original_split_dir is None:
        original_split_dir = config.SPLIT_DIR
    if synthetic_dir is None:
        synthetic_dir = config.SYNTHETIC_DIR
    if synthetic_mapping is None:
        synthetic_mapping = config.SYNTHETIC_MAPPING
    if target_dir is None:
        suffix = "_synth" if with_synthetic else ""
        target_dir = os.path.join(config.DATA_DIR, f"split_3class{suffix}")

    os.makedirs(target_dir, exist_ok=True)

    # Detect symlink capability (mirrors are always real files; this is for the rest)
    use_copy = False
    probe_src = os.path.join(original_split_dir, "val")
    if os.path.isdir(probe_src):
        test_dst = os.path.join(target_dir, "__symlink_test__")
        try:
            os.symlink(probe_src, test_dst, target_is_directory=True)
            os.remove(test_dst)
        except (OSError, NotImplementedError):
            use_copy = True

    method = "copying" if use_copy else "symlinking"
    label = "QU + synth + mirror" if with_synthetic else "QU only"
    print(f"\nPreparing 3-class YOLO split [{label}] ({method})...")
    print(f"  Source split: {original_split_dir}")
    print(f"  Target:       {target_dir}")
    print(f"  Classes:      {', '.join(class_names)}")

    def _link_or_copy_file(src_path: str, dst_path: str) -> None:
        if os.path.exists(dst_path):
            return
        if use_copy:
            shutil.copy2(src_path, dst_path)
        else:
            try:
                os.symlink(os.path.abspath(src_path), dst_path)
            except (OSError, NotImplementedError):
                shutil.copy2(src_path, dst_path)

    counts = {}

    for subset in ("train", "val", "test"):
        for cls_name in class_names:
            src_cls = os.path.join(original_split_dir, subset, cls_name)
            dst_cls = os.path.join(target_dir, subset, cls_name)
            os.makedirs(dst_cls, exist_ok=True)

            n_real = 0
            if os.path.isdir(src_cls):
                for fname in os.listdir(src_cls):
                    if os.path.splitext(fname)[1].lower() not in SUPPORTED_EXTENSIONS:
                        continue
                    if fname.startswith("aug_"):
                        continue
                    _link_or_copy_file(
                        os.path.join(src_cls, fname),
                        os.path.join(dst_cls, fname),
                    )
                    n_real += 1

            # Synth + mirror only injected into TRAIN
            n_synth = 0
            n_mirror = 0
            if subset == "train" and with_synthetic and cls_name in synthetic_mapping:
                for subfolder, sub_subfolder in synthetic_mapping[cls_name]:
                    src_dir = os.path.join(synthetic_dir, subfolder, sub_subfolder)
                    if not os.path.isdir(src_dir):
                        print(f"  WARNING: Synthetic source not found: {src_dir}")
                        continue
                    for fname in os.listdir(src_dir):
                        if os.path.splitext(fname)[1].lower() not in SUPPORTED_EXTENSIONS:
                            continue
                        # Original synthetic
                        synth_dst_name = f"synth_{fname}"
                        synth_dst_path = os.path.join(dst_cls, synth_dst_name)
                        _link_or_copy_file(
                            os.path.join(src_dir, fname),
                            synth_dst_path,
                        )
                        n_synth += 1

                        # Hard-coded horizontal mirror (real file, not symlink).
                        # Preserve the source extension in the stem so that
                        # e.g. "0001.png" and "0001.jpg" don't both map to
                        # "synth_mirror_0001.jpg" (would silently drop one).
                        mirror_stem = fname.replace(".", "_")
                        mirror_dst_name = f"synth_mirror_{mirror_stem}.jpg"
                        mirror_dst_path = os.path.join(dst_cls, mirror_dst_name)
                        if not os.path.exists(mirror_dst_path):
                            with Image.open(os.path.join(src_dir, fname)) as im:
                                im_rgb = im.convert("RGB")
                                im_flipped = im_rgb.transpose(Image.FLIP_LEFT_RIGHT)
                                im_flipped.save(mirror_dst_path, format="JPEG", quality=95)
                        n_mirror += 1

            total_for_class = n_real + n_synth + n_mirror
            if subset == "train":
                counts[cls_name] = total_for_class
            print(f"  {subset}/{cls_name:<12}  real={n_real:>5,}  "
                  f"synth={n_synth:>4,}  mirror={n_mirror:>4,}  total={total_for_class:>6,}")

    total = sum(counts.values())
    print(f"\n{'='*70}")
    print(f"3-CLASS YOLO SPLIT READY [{label}]")
    print(f"{'='*70}")
    for cls_name in class_names:
        print(f"  {cls_name:<14} train total: {counts[cls_name]:>6,}")
    print(f"  {'TOTAL':<14} train total: {total:>6,}")
    print(f"  Val/test: clean from {original_split_dir} (no synth, no mirror)")
    print(f"{'='*70}")

    return counts
