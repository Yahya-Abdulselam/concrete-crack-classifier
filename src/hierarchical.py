"""Hierarchical classification infrastructure (PyTorch + YOLO).

Implements a 3-stage hierarchy on top of the existing 6-class flat dataset:

    Stage 1: crack vs no_crack
    Stage 2: single vs multi   (only on 'crack' images)
    Stage 3: debonding/flexural/shear/others   (only on 'single' images)

Routing of original 6 classes:
    no_crack    -> (no_crack, -, -)
    multi_crack -> (crack, multi, -)
    debonding   -> (crack, single, debonding)
    flexural    -> (crack, single, flexural)
    shear       -> (crack, single, shear)
    others      -> (crack, single, others)

This module provides:
    * label routing constants and helpers
    * a HierarchicalCrackDataset that returns masked stage labels
    * balanced samplers, including the special "multi x3" rule for stage 2
    * YOLO folder builder/rebalancer (for the cascaded YOLO experiment)
    * cascaded inference helpers (cascade_predict)
    * hierarchical precision/recall/F1 metrics
    * train_hierarchical_model (multi-task variant of src.trainer.train_model)
"""

import csv
import math
import os
import random
import shutil
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.augmentation import get_train_transforms, get_val_test_transforms
from src.preprocessing import bilateral_denoise, apply_clahe


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================
# LABEL ROUTING
# ============================================================

STAGE1_CLASSES = ["no_crack", "crack"]                              # 2
STAGE2_CLASSES = ["multi", "single"]                                # 2
STAGE3_CLASSES = ["debonding", "flexural", "shear", "others"]       # 4

STAGE1_TO_IDX = {c: i for i, c in enumerate(STAGE1_CLASSES)}
STAGE2_TO_IDX = {c: i for i, c in enumerate(STAGE2_CLASSES)}
STAGE3_TO_IDX = {c: i for i, c in enumerate(STAGE3_CLASSES)}

# Original 6-class -> hierarchical path
ORIG_TO_HIER = {
    "no_crack":    ("no_crack", None,     None),
    "multi_crack": ("crack",    "multi",  None),
    "debonding":   ("crack",    "single", "debonding"),
    "flexural":    ("crack",    "single", "flexural"),
    "shear":       ("crack",    "single", "shear"),
    "others":      ("crack",    "single", "others"),
}


def hier_path_for_orig(orig_class: str) -> tuple:
    """Return (stage1, stage2, stage3) tuple for an original class name."""
    return ORIG_TO_HIER[orig_class]


def cascaded_to_orig(stage1: str, stage2: str | None, stage3: str | None) -> str:
    """Collapse a hierarchical prediction back to one of the 6 flat classes.

    Used to make the hierarchical pipeline directly comparable to the flat
    baseline via the existing evaluate_predictions() utility.
    """
    if stage1 == "no_crack":
        return "no_crack"
    if stage2 == "multi":
        return "multi_crack"
    # crack + single -> stage3 must be one of the 4 subtype classes
    return stage3


# ============================================================
# PYTORCH DATASET (CBAM EXPERIMENT)
# ============================================================


class HierarchicalCrackDataset(Dataset):
    """PyTorch dataset that yields masked hierarchical labels.

    Reuses the existing bilateral + CLAHE preprocessing and the standard
    torchvision transform pipeline (no augmentation changes).

    __getitem__ returns:
        image       : tensor (3, H, W)
        labels      : dict with keys 'stage1', 'stage2', 'stage3' (long tensors)
        masks       : dict with keys 'stage1', 'stage2', 'stage3' (float tensors,
                      0.0 if the label is undefined for this image, else 1.0)

    The mask scheme:
        stage1 mask = 1.0 always
        stage2 mask = 1.0 iff stage1 == 'crack'
        stage3 mask = 1.0 iff stage2 == 'single'
    """

    def __init__(
        self,
        split_dir: str,
        subset: str,
        transform=None,
        img_size: int = config.IMG_SIZE,
    ):
        self.img_size = img_size
        self.transform = transform

        subset_dir = os.path.join(split_dir, subset)
        self.file_paths: list[str] = []
        self.orig_labels: list[str] = []   # original 6-class string per sample
        self.class_counts: dict[str, int] = {}

        for cls_name in config.CLASS_NAMES:
            cls_dir = os.path.join(subset_dir, cls_name)
            if not os.path.isdir(cls_dir):
                self.class_counts[cls_name] = 0
                continue

            files = sorted(
                os.path.join(cls_dir, f)
                for f in os.listdir(cls_dir)
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
                and not f.startswith("aug_")
            )
            self.file_paths.extend(files)
            self.orig_labels.extend([cls_name] * len(files))
            self.class_counts[cls_name] = len(files)

        # Pre-compute hierarchical labels and masks per sample
        self.s1 = np.zeros(len(self.file_paths), dtype=np.int64)
        self.s2 = np.zeros(len(self.file_paths), dtype=np.int64)
        self.s3 = np.zeros(len(self.file_paths), dtype=np.int64)
        self.m2 = np.zeros(len(self.file_paths), dtype=np.float32)
        self.m3 = np.zeros(len(self.file_paths), dtype=np.float32)

        for i, orig in enumerate(self.orig_labels):
            s1, s2, s3 = ORIG_TO_HIER[orig]
            self.s1[i] = STAGE1_TO_IDX[s1]
            if s2 is not None:
                self.s2[i] = STAGE2_TO_IDX[s2]
                self.m2[i] = 1.0
            if s3 is not None:
                self.s3[i] = STAGE3_TO_IDX[s3]
                self.m3[i] = 1.0

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)

        arr = np.array(img)
        arr = bilateral_denoise(arr)
        arr = apply_clahe(arr)
        img = Image.fromarray(arr)

        if self.transform:
            img = self.transform(img)

        labels = {
            "stage1": int(self.s1[idx]),
            "stage2": int(self.s2[idx]),
            "stage3": int(self.s3[idx]),
        }
        masks = {
            "stage1": 1.0,
            "stage2": float(self.m2[idx]),
            "stage3": float(self.m3[idx]),
        }
        return img, labels, masks


# ============================================================
# BALANCED SAMPLERS
# ============================================================


def _per_class_counts(orig_labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for c in orig_labels:
        counts[c] += 1
    return dict(counts)


def get_stage1_sampler(
    dataset: HierarchicalCrackDataset,
    max_aug_factor: int = config.MAX_AUG_FACTOR,
) -> WeightedRandomSampler:
    """Stage-1 balancing using the standard x6 cap rule on {crack, no_crack}."""
    n_crack = int((dataset.s1 == STAGE1_TO_IDX["crack"]).sum())
    n_no    = int((dataset.s1 == STAGE1_TO_IDX["no_crack"]).sum())
    smallest = min(n_crack, n_no)
    target = smallest * max_aug_factor
    total = target * 2

    w_crack = target / n_crack if n_crack else 0.0
    w_no    = target / n_no    if n_no    else 0.0
    weights = np.where(dataset.s1 == STAGE1_TO_IDX["crack"], w_crack, w_no).astype(np.float64)

    print(f"\n[Stage 1 sampler] crack={n_crack}  no_crack={n_no}")
    print(f"  target/class={target}, samples/epoch={total}")
    return WeightedRandomSampler(weights=weights, num_samples=total, replacement=True)


def get_stage2_sampler(
    dataset: HierarchicalCrackDataset,
    multi_factor: int = config.MULTI_AUG_FACTOR,
) -> WeightedRandomSampler:
    """Stage-2 sampler implementing the multi-factor rule:
        target_multi  = multi_factor * N_multi    (oversample)
        target_single = multi_factor * N_multi    (downsample to match)

    Only crack images participate. no_crack images get weight 0.
    """
    is_crack = dataset.m2 > 0
    is_single = is_crack & (dataset.s2 == STAGE2_TO_IDX["single"])
    is_multi  = is_crack & (dataset.s2 == STAGE2_TO_IDX["multi"])

    n_single = int(is_single.sum())
    n_multi  = int(is_multi.sum())
    if n_multi == 0:
        raise ValueError("No 'multi' samples found for stage 2 sampler.")

    target_multi  = multi_factor * n_multi
    target_single = multi_factor * n_multi
    total = target_multi + target_single

    w_single = target_single / n_single if n_single else 0.0
    w_multi  = target_multi  / n_multi

    weights = np.zeros(len(dataset), dtype=np.float64)
    weights[is_single] = w_single
    weights[is_multi]  = w_multi

    print(f"\n[Stage 2 sampler] single={n_single}  multi={n_multi}")
    print(f"  target_single={target_single}  target_multi={target_multi}  "
          f"(multi x{multi_factor} oversample, single downsampled)")
    print(f"  samples/epoch={total}")
    return WeightedRandomSampler(weights=weights, num_samples=total, replacement=True)


def get_stage3_sampler(
    dataset: HierarchicalCrackDataset,
    max_aug_factor: int = config.MAX_AUG_FACTOR,
) -> WeightedRandomSampler:
    """Stage-3 balancing across the 4 subtype classes via the x6 cap rule."""
    is_single = dataset.m3 > 0
    counts = {
        c: int(((dataset.s3 == STAGE3_TO_IDX[c]) & is_single).sum())
        for c in STAGE3_CLASSES
    }
    nonzero = [v for v in counts.values() if v > 0]
    if not nonzero:
        raise ValueError("No 'single' samples found for stage 3 sampler.")
    smallest = min(nonzero)
    target = smallest * max_aug_factor
    total = target * len(STAGE3_CLASSES)

    weights = np.zeros(len(dataset), dtype=np.float64)
    for c, n in counts.items():
        if n == 0:
            continue
        mask = is_single & (dataset.s3 == STAGE3_TO_IDX[c])
        weights[mask] = target / n

    print(f"\n[Stage 3 sampler] " + "  ".join(f"{c}={n}" for c, n in counts.items()))
    print(f"  target/class={target}  samples/epoch={total}")
    return WeightedRandomSampler(weights=weights, num_samples=total, replacement=True)


def get_hierarchical_sampler(
    dataset: HierarchicalCrackDataset,
    no_crack_mix: float = 0.25,
    max_aug_factor: int = config.MAX_AUG_FACTOR,
) -> WeightedRandomSampler:
    """Combined sampler used during the joint Phase 3 of CBAM hierarchical training.

    Strategy:
        - Within crack images, balance via the x6 cap on the 4 stage-3 subtypes
          (and apportion 'multi' images so they remain visible).
        - Mix in no_crack images at a fixed ratio (default 25%) so head 1
          does not catastrophically forget.
    """
    is_no = dataset.s1 == STAGE1_TO_IDX["no_crack"]
    is_crack = ~is_no
    is_multi = is_crack & (dataset.s2 == STAGE2_TO_IDX["multi"])
    is_single = is_crack & (dataset.s2 == STAGE2_TO_IDX["single"])

    # Per-subtype counts (single branch)
    sub_counts = {
        c: int(((dataset.s3 == STAGE3_TO_IDX[c]) & is_single).sum())
        for c in STAGE3_CLASSES
    }
    nonzero = [v for v in sub_counts.values() if v > 0]
    smallest = min(nonzero) if nonzero else 1
    target = smallest * max_aug_factor   # per-subtype target inside the crack pool

    n_multi = int(is_multi.sum())
    n_no = int(is_no.sum())

    # Crack pool size: 4 subtypes + multi all sized to `target`
    crack_pool = target * (len(STAGE3_CLASSES) + 1)
    no_pool = int(round(crack_pool * no_crack_mix / max(1.0 - no_crack_mix, 1e-6)))
    total = crack_pool + no_pool

    weights = np.zeros(len(dataset), dtype=np.float64)
    for c, n in sub_counts.items():
        if n == 0:
            continue
        mask = is_single & (dataset.s3 == STAGE3_TO_IDX[c])
        weights[mask] = target / n
    if n_multi > 0:
        weights[is_multi] = target / n_multi
    if n_no > 0:
        weights[is_no] = no_pool / n_no

    print(f"\n[Hierarchical sampler]  no_crack_mix={no_crack_mix:.0%}")
    print(f"  Crack subtype targets={target} each   multi target={target}")
    print(f"  no_crack pool={no_pool}   total samples/epoch={total}")
    return WeightedRandomSampler(weights=weights, num_samples=total, replacement=True)


# ============================================================
# YOLO STAGE FOLDER BUILDER (EXPERIMENT A)
# ============================================================


def _link_or_copy(src: str, dst: str, use_copy: bool):
    if os.path.exists(dst):
        return
    if use_copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src, dst)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)


def _detect_link_strategy(probe_src: str, probe_dir: str) -> bool:
    """Return True if we must fall back to copying."""
    test_dst = os.path.join(probe_dir, "__symlink_test__")
    try:
        os.symlink(probe_src, test_dst)
        os.remove(test_dst)
        return False
    except (OSError, NotImplementedError):
        return True


def build_yolo_stage_dirs(
    split_dir: str,
    out_root: str,
) -> dict:
    """Materialize 3 stage-specific folder trees from the existing flat split.

    Output layout (per stage):
        out_root/stage1/{train,val,test}/{crack,no_crack}/
        out_root/stage2/{train,val,test}/{single,multi}/
        out_root/stage3/{train,val,test}/{debonding,flexural,shear,others}/

    Reuses the existing 70/15/15 stratified split (no resplitting).
    Symlinks where possible, falls back to copy on Windows w/o dev mode.
    """
    os.makedirs(out_root, exist_ok=True)

    stage_classes = {
        "stage1": STAGE1_CLASSES,
        "stage2": STAGE2_CLASSES,
        "stage3": STAGE3_CLASSES,
    }

    counts: dict = {s: defaultdict(lambda: defaultdict(int)) for s in stage_classes}

    # Detect link strategy once
    use_copy = False
    probe_done = False

    for subset in ("train", "val", "test"):
        for orig_cls in config.CLASS_NAMES:
            src_dir = os.path.join(split_dir, subset, orig_cls)
            if not os.path.isdir(src_dir):
                continue

            files = [
                f for f in os.listdir(src_dir)
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
                and not f.startswith("aug_")
            ]
            if not files:
                continue

            s1, s2, s3 = ORIG_TO_HIER[orig_cls]

            # ---- Stage 1: every image ----
            stage1_dst = os.path.join(out_root, "stage1", subset, s1)
            os.makedirs(stage1_dst, exist_ok=True)
            if not probe_done:
                use_copy = _detect_link_strategy(os.path.join(src_dir, files[0]), stage1_dst)
                probe_done = True
            for f in files:
                _link_or_copy(os.path.join(src_dir, f),
                              os.path.join(stage1_dst, f), use_copy)
            counts["stage1"][subset][s1] += len(files)

            # ---- Stage 2: only crack images ----
            if s2 is not None:
                stage2_dst = os.path.join(out_root, "stage2", subset, s2)
                os.makedirs(stage2_dst, exist_ok=True)
                for f in files:
                    _link_or_copy(os.path.join(src_dir, f),
                                  os.path.join(stage2_dst, f), use_copy)
                counts["stage2"][subset][s2] += len(files)

            # ---- Stage 3: only single-crack images ----
            if s3 is not None:
                stage3_dst = os.path.join(out_root, "stage3", subset, s3)
                os.makedirs(stage3_dst, exist_ok=True)
                for f in files:
                    _link_or_copy(os.path.join(src_dir, f),
                                  os.path.join(stage3_dst, f), use_copy)
                counts["stage3"][subset][s3] += len(files)

    # Pretty print
    print(f"\n{'='*70}")
    print(f"YOLO STAGE FOLDERS BUILT  ({'copy' if use_copy else 'symlink'} mode)")
    print(f"  root: {out_root}")
    print(f"{'='*70}")
    for stage, classes in stage_classes.items():
        print(f"\n  {stage}:  classes={classes}")
        for subset in ("train", "val", "test"):
            row = "    " + f"{subset:<6}"
            for c in classes:
                row += f"  {c}={counts[stage][subset].get(c, 0):>5}"
            row += f"  total={sum(counts[stage][subset].values()):>5}"
            print(row)

    return counts


def materialize_balanced_yolo_train(
    yolo_stage_root: str,
    stage: str,
    rule: str,
    max_aug_factor: int = config.MAX_AUG_FACTOR,
    seed: int = config.RANDOM_SEED,
    multi_factor: int = config.MULTI_AUG_FACTOR,
) -> dict:
    """Rebalance the train/ folder of a stage in-place via duplication / sampling.

    Args:
        yolo_stage_root: e.g. <out_root>/stage1
        stage: 'stage1' | 'stage2' | 'stage3'
        rule:
            'cap6'    -> per-class target = smallest_class * max_aug_factor (x6 cap)
            'multix3' -> stage 2 special: target = multi_factor * N_multi per class
        multi_factor: multiplier for 'multix3' rule (default from config).

    Returns:
        Dict mapping class -> resulting file count after rebalancing.
    """
    rng = random.Random(seed)
    train_dir = os.path.join(yolo_stage_root, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(train_dir)

    classes = sorted(d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d)))

    def _list_originals(cls):
        cdir = os.path.join(train_dir, cls)
        return [
            f for f in os.listdir(cdir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
            and "_dup" not in f
        ]

    # First strip any prior _dup files so re-runs are idempotent
    for cls in classes:
        cdir = os.path.join(train_dir, cls)
        for f in os.listdir(cdir):
            if "_dup" in f:
                try:
                    os.remove(os.path.join(cdir, f))
                except OSError:
                    pass

    originals = {cls: _list_originals(cls) for cls in classes}
    counts = {cls: len(v) for cls, v in originals.items()}

    if rule == "cap6":
        smallest = min(c for c in counts.values() if c > 0)
        targets = {cls: smallest * max_aug_factor for cls in classes}
    elif rule == "multix3":
        if "multi" not in counts or counts["multi"] == 0:
            raise ValueError("'multi' class missing for multix3 rule")
        n_multi = counts["multi"]
        targets = {cls: multi_factor * n_multi for cls in classes}
    else:
        raise ValueError(f"Unknown rule: {rule}")

    # Detect link strategy
    probe_src = os.path.join(train_dir, classes[0], originals[classes[0]][0])
    use_copy = _detect_link_strategy(probe_src, os.path.join(train_dir, classes[0]))

    final = {}
    for cls in classes:
        cdir = os.path.join(train_dir, cls)
        files = originals[cls]
        target = targets[cls]
        n = len(files)

        if n == 0:
            final[cls] = 0
            continue

        if target <= n:
            # Undersample: keep a deterministic random subset, delete the rest
            keep = set(rng.sample(files, target))
            for f in files:
                if f not in keep:
                    try:
                        os.remove(os.path.join(cdir, f))
                    except OSError:
                        pass
            final[cls] = target
        else:
            # Oversample: keep all originals, add (target - n) symlinks/copies
            extra = target - n
            for i in range(extra):
                src_name = files[i % n]
                src_path = os.path.join(cdir, src_name)
                stem, ext = os.path.splitext(src_name)
                dst_name = f"{stem}_dup{i}{ext}"
                dst_path = os.path.join(cdir, dst_name)
                if use_copy:
                    shutil.copy2(src_path, dst_path)
                else:
                    try:
                        os.symlink(src_path, dst_path)
                    except (OSError, NotImplementedError):
                        shutil.copy2(src_path, dst_path)
            final[cls] = target

    print(f"\n[{stage} rebalance:{rule}]  " +
          "  ".join(f"{c}: {counts[c]}->{final[c]}" for c in classes))
    return final


# ============================================================
# CASCADED INFERENCE & METRICS
# ============================================================


def cascade_predict(
    p_stage1: np.ndarray,
    p_stage2: np.ndarray,
    p_stage3: np.ndarray,
    t1: float = 0.5,
    t2: float = 0.5,
) -> tuple[str, str | None, str | None]:
    """Apply hierarchical routing to a single sample's per-stage class probabilities.

    Args:
        p_stage1: shape (2,) over STAGE1_CLASSES
        p_stage2: shape (2,) over STAGE2_CLASSES (only consulted if crack)
        p_stage3: shape (4,) over STAGE3_CLASSES (only consulted if single)
        t1: probability threshold for routing into the 'crack' branch
            (using P(crack) >= t1).
        t2: probability threshold for routing into the 'single' branch
            (using P(single) >= t2).

    Returns:
        (stage1_label, stage2_label_or_None, stage3_label_or_None)
    """
    p_crack = float(p_stage1[STAGE1_TO_IDX["crack"]])
    if p_crack < t1:
        return ("no_crack", None, None)

    p_single = float(p_stage2[STAGE2_TO_IDX["single"]])
    if p_single < t2:
        return ("crack", "multi", None)

    s3_idx = int(np.argmax(p_stage3))
    return ("crack", "single", STAGE3_CLASSES[s3_idx])


def hierarchical_pr_f1(
    y_true_paths: list[tuple],
    y_pred_paths: list[tuple],
) -> dict:
    """Compute hierarchical precision / recall / F1 (Kiritchenko 2005 style).

    A hierarchical "path" is a tuple like ('crack', 'single', 'flexural'),
    with None entries removed.  Predictions get partial credit for matching
    ancestors of the true class.

        hP = sum_i |P_i ∩ T_i| / sum_i |P_i|
        hR = sum_i |P_i ∩ T_i| / sum_i |T_i|
        hF = 2 hP hR / (hP + hR)
    """
    inter = pred_total = true_total = 0
    for tp, pp in zip(y_true_paths, y_pred_paths):
        ts = set(x for x in tp if x is not None)
        ps = set(x for x in pp if x is not None)
        inter += len(ts & ps)
        pred_total += len(ps)
        true_total += len(ts)

    hp = inter / pred_total if pred_total else 0.0
    hr = inter / true_total if true_total else 0.0
    hf = (2 * hp * hr / (hp + hr)) if (hp + hr) else 0.0
    return {"hP": hp, "hR": hr, "hF": hf}


def per_stage_confusion_matrices(
    y_true_paths: list[tuple],
    y_pred_paths: list[tuple],
) -> dict:
    """Compute per-stage confusion matrices, restricted to samples that
    actually have a label / prediction at that stage.

    Stage-1 CM is over all samples.
    Stage-2 CM is over samples whose TRUE stage1 == 'crack' (regardless of pred).
    Stage-3 CM is over samples whose TRUE stage2 == 'single'.
    """
    out: dict = {}

    # Stage 1
    yt1 = [STAGE1_TO_IDX[t[0]] for t in y_true_paths]
    yp1 = [STAGE1_TO_IDX[p[0]] for p in y_pred_paths]
    out["stage1"] = {
        "y_true": np.array(yt1),
        "y_pred": np.array(yp1),
        "classes": STAGE1_CLASSES,
        "cm": confusion_matrix(yt1, yp1, labels=list(range(len(STAGE1_CLASSES)))),
    }

    # Stage 2 (only where true stage1 == crack)
    yt2, yp2 = [], []
    for t, p in zip(y_true_paths, y_pred_paths):
        if t[0] == "crack" and t[1] is not None:
            # Pred stage2 may be None if cascade routed to no_crack -> count as wrong
            pred_s2 = p[1] if p[1] is not None else "multi"  # default miss bucket
            # Better: use a sentinel index = the opposite of true.  We instead
            # treat None as "multi" only when true is "single", otherwise "single".
            if p[1] is None:
                pred_s2 = "multi" if t[1] == "single" else "single"
            yt2.append(STAGE2_TO_IDX[t[1]])
            yp2.append(STAGE2_TO_IDX[pred_s2])
    if yt2:
        out["stage2"] = {
            "y_true": np.array(yt2),
            "y_pred": np.array(yp2),
            "classes": STAGE2_CLASSES,
            "cm": confusion_matrix(yt2, yp2, labels=list(range(len(STAGE2_CLASSES)))),
        }

    # Stage 3 (only where true stage2 == single)
    yt3, yp3 = [], []
    for t, p in zip(y_true_paths, y_pred_paths):
        if t[1] == "single" and t[2] is not None:
            if p[2] is None:
                # Cascade stopped early -> pick a deterministic wrong class
                pred_s3 = "others" if t[2] != "others" else "debonding"
            else:
                pred_s3 = p[2]
            yt3.append(STAGE3_TO_IDX[t[2]])
            yp3.append(STAGE3_TO_IDX[pred_s3])
    if yt3:
        out["stage3"] = {
            "y_true": np.array(yt3),
            "y_pred": np.array(yp3),
            "classes": STAGE3_CLASSES,
            "cm": confusion_matrix(yt3, yp3, labels=list(range(len(STAGE3_CLASSES)))),
        }

    return out


def error_attribution(
    y_true_paths: list[tuple],
    y_pred_paths: list[tuple],
) -> dict:
    """For each end-to-end mistake, identify which stage caused it."""
    by_stage = {"stage1": 0, "stage2": 0, "stage3": 0}
    correct = 0
    for t, p in zip(y_true_paths, y_pred_paths):
        if t == p:
            correct += 1
            continue
        if t[0] != p[0]:
            by_stage["stage1"] += 1
        elif (t[1] or "") != (p[1] or ""):
            by_stage["stage2"] += 1
        else:
            by_stage["stage3"] += 1
    return {"correct": correct, "errors_by_stage": by_stage,
            "total": len(y_true_paths)}


# ============================================================
# MULTI-TASK TRAINER (CBAM EXPERIMENT)
# ============================================================


def hierarchical_collate(batch):
    """Stack a batch of (img, labels_dict, masks_dict) into tensors."""
    imgs = torch.stack([b[0] for b in batch], dim=0)
    labels = {
        "stage1": torch.tensor([b[1]["stage1"] for b in batch], dtype=torch.long),
        "stage2": torch.tensor([b[1]["stage2"] for b in batch], dtype=torch.long),
        "stage3": torch.tensor([b[1]["stage3"] for b in batch], dtype=torch.long),
    }
    masks = {
        "stage1": torch.tensor([b[2]["stage1"] for b in batch], dtype=torch.float32),
        "stage2": torch.tensor([b[2]["stage2"] for b in batch], dtype=torch.float32),
        "stage3": torch.tensor([b[2]["stage3"] for b in batch], dtype=torch.float32),
    }
    return imgs, labels, masks


def get_hierarchical_dataloaders(
    split_dir: str = config.SPLIT_DIR,
    batch_size: int = 16,
    img_size: int = config.IMG_SIZE,
    num_workers: int = 0,
    sampler_kind: str = "stage1",     # 'stage1' | 'stage2' | 'stage3' | 'joint'
    no_crack_mix: float = 0.25,
    multi_factor: int = config.MULTI_AUG_FACTOR,
):
    train_tf = get_train_transforms(img_size, "imagenet")
    val_tf = get_val_test_transforms(img_size, "imagenet")

    train_ds = HierarchicalCrackDataset(split_dir, "train", train_tf, img_size)
    val_ds   = HierarchicalCrackDataset(split_dir, "val",   val_tf,   img_size)
    test_ds  = HierarchicalCrackDataset(split_dir, "test",  val_tf,   img_size)

    if sampler_kind == "stage1":
        sampler = get_stage1_sampler(train_ds)
    elif sampler_kind == "stage2":
        sampler = get_stage2_sampler(train_ds, multi_factor=multi_factor)
    elif sampler_kind == "stage3":
        sampler = get_stage3_sampler(train_ds)
    elif sampler_kind == "joint":
        sampler = get_hierarchical_sampler(train_ds, no_crack_mix=no_crack_mix)
    else:
        raise ValueError(sampler_kind)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=hierarchical_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=hierarchical_collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=hierarchical_collate,
    )

    print(f"\nHierarchical loaders ({sampler_kind}):")
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    return train_loader, val_loader, test_loader


def _masked_ce(logits, targets, mask, weight=None):
    """Cross-entropy with per-sample masking. Returns scalar (mean over active)."""
    losses = nn.functional.cross_entropy(
        logits, targets, weight=weight, reduction="none",
    )
    denom = mask.sum().clamp(min=1.0)
    return (losses * mask).sum() / denom


def _stage_accuracy(logits, targets, mask):
    """Accuracy among samples where mask == 1."""
    if mask.sum() == 0:
        return float("nan")
    preds = logits.argmax(dim=1)
    correct = ((preds == targets).float() * mask).sum()
    return float(correct / mask.sum())


def train_hierarchical_model(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    device: torch.device,
    epochs: int,
    output_dir: str,
    stage: int,
    loss_weights: tuple[float, float, float] = (1.0, 0.0, 0.0),
    class_weights: dict | None = None,    # {'stage1': tensor, 'stage2': tensor, 'stage3': tensor}
    patience: int = config.EARLY_STOPPING_PATIENCE,
    lr_patience: int = config.REDUCE_LR_PATIENCE,
    lr_factor: float = config.REDUCE_LR_FACTOR,
    model_name: str = "",
):
    """Multi-task training loop with masked CE and per-stage metrics.

    Mirrors src.trainer.train_model: ReduceLROnPlateau, early stopping on
    val_loss, checkpointing on best val_loss, CSV logging.
    """
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    cw = class_weights or {}
    w1 = cw.get("stage1", None)
    w2 = cw.get("stage2", None)
    w3 = cw.get("stage3", None)
    if w1 is not None: w1 = w1.to(device)
    if w2 is not None: w2 = w2.to(device)
    if w3 is not None: w3 = w3.to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=1e-7,
    )

    prefix = f"{model_name}_" if model_name else ""
    csv_path = os.path.join(output_dir, "logs", f"{prefix}hier_stage{stage}_metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    fieldnames = [
        "epoch", "lr", "loss", "loss_s1", "loss_s2", "loss_s3",
        "acc_s1", "acc_s2", "acc_s3",
        "val_loss", "val_loss_s1", "val_loss_s2", "val_loss_s3",
        "val_acc_s1", "val_acc_s2", "val_acc_s3",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    best_val = float("inf")
    bad_epochs = 0
    ckpt = os.path.join(output_dir, "models", f"best_{prefix}hier_phase{stage}.pt")

    history = {k: [] for k in fieldnames if k != "epoch"}
    history["epoch"] = []

    print(f"\nPhase {stage}: epochs={epochs}  loss_weights={loss_weights}")
    print(f"  LR={optimizer.param_groups[0]['lr']:.1e}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        cur_lr = optimizer.param_groups[0]["lr"]

        # ---- TRAIN ----
        model.train()
        agg = {k: 0.0 for k in ["loss", "loss_s1", "loss_s2", "loss_s3"]}
        cnt = 0
        s1_correct = s1_n = 0
        s2_correct = s2_n = 0.0
        s3_correct = s3_n = 0.0

        for imgs, labels, masks in tqdm(train_loader, desc="  train", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            t1 = labels["stage1"].to(device); m1 = masks["stage1"].to(device)
            t2 = labels["stage2"].to(device); m2 = masks["stage2"].to(device)
            t3 = labels["stage3"].to(device); m3 = masks["stage3"].to(device)

            optimizer.zero_grad()
            l1, l2, l3 = model(imgs)

            ls1 = _masked_ce(l1, t1, m1, w1) if loss_weights[0] > 0 else torch.tensor(0.0, device=device)
            ls2 = _masked_ce(l2, t2, m2, w2) if loss_weights[1] > 0 else torch.tensor(0.0, device=device)
            ls3 = _masked_ce(l3, t3, m3, w3) if loss_weights[2] > 0 else torch.tensor(0.0, device=device)
            loss = (loss_weights[0] * ls1 + loss_weights[1] * ls2 + loss_weights[2] * ls3)

            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            cnt += bs
            agg["loss"]    += loss.item() * bs
            agg["loss_s1"] += ls1.item() * bs
            agg["loss_s2"] += ls2.item() * bs
            agg["loss_s3"] += ls3.item() * bs

            with torch.no_grad():
                p1 = l1.argmax(1); p2 = l2.argmax(1); p3 = l3.argmax(1)
                s1_correct += int(((p1 == t1).float() * m1).sum())
                s1_n       += int(m1.sum())
                s2_correct += int(((p2 == t2).float() * m2).sum())
                s2_n       += int(m2.sum())
                s3_correct += int(((p3 == t3).float() * m3).sum())
                s3_n       += int(m3.sum())

        train_metrics = {
            "loss":    agg["loss"]    / max(cnt, 1),
            "loss_s1": agg["loss_s1"] / max(cnt, 1),
            "loss_s2": agg["loss_s2"] / max(cnt, 1),
            "loss_s3": agg["loss_s3"] / max(cnt, 1),
            "acc_s1":  s1_correct / max(s1_n, 1),
            "acc_s2":  s2_correct / max(s2_n, 1),
            "acc_s3":  s3_correct / max(s3_n, 1),
        }

        # ---- VALIDATE ----
        model.eval()
        vagg = {k: 0.0 for k in ["loss", "loss_s1", "loss_s2", "loss_s3"]}
        vcnt = 0
        v1_c = v1_n = v2_c = v2_n = v3_c = v3_n = 0
        with torch.no_grad():
            for imgs, labels, masks in tqdm(val_loader, desc="  val", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                t1 = labels["stage1"].to(device); m1 = masks["stage1"].to(device)
                t2 = labels["stage2"].to(device); m2 = masks["stage2"].to(device)
                t3 = labels["stage3"].to(device); m3 = masks["stage3"].to(device)

                l1, l2, l3 = model(imgs)
                ls1 = _masked_ce(l1, t1, m1, w1)
                ls2 = _masked_ce(l2, t2, m2, w2)
                ls3 = _masked_ce(l3, t3, m3, w3)
                # Validation loss uses the SAME loss weights so it's comparable to train
                vloss = (loss_weights[0] * ls1 + loss_weights[1] * ls2 + loss_weights[2] * ls3)

                bs = imgs.size(0)
                vcnt += bs
                vagg["loss"]    += vloss.item() * bs
                vagg["loss_s1"] += ls1.item() * bs
                vagg["loss_s2"] += ls2.item() * bs
                vagg["loss_s3"] += ls3.item() * bs

                p1 = l1.argmax(1); p2 = l2.argmax(1); p3 = l3.argmax(1)
                v1_c += int(((p1 == t1).float() * m1).sum()); v1_n += int(m1.sum())
                v2_c += int(((p2 == t2).float() * m2).sum()); v2_n += int(m2.sum())
                v3_c += int(((p3 == t3).float() * m3).sum()); v3_n += int(m3.sum())

        val_metrics = {
            "val_loss":    vagg["loss"]    / max(vcnt, 1),
            "val_loss_s1": vagg["loss_s1"] / max(vcnt, 1),
            "val_loss_s2": vagg["loss_s2"] / max(vcnt, 1),
            "val_loss_s3": vagg["loss_s3"] / max(vcnt, 1),
            "val_acc_s1":  v1_c / max(v1_n, 1),
            "val_acc_s2":  v2_c / max(v2_n, 1),
            "val_acc_s3":  v3_c / max(v3_n, 1),
        }

        elapsed = time.time() - t0
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["val_loss"])
        new_lr = optimizer.param_groups[0]["lr"]
        lr_msg = f" | LR {old_lr:.1e}->{new_lr:.1e}" if new_lr != old_lr else ""

        ckpt_msg = ""
        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            torch.save(model.state_dict(), ckpt)
            ckpt_msg = " | SAVED"
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(f"  E{epoch:>3}/{epochs} [{elapsed:>5.1f}s] "
              f"loss={train_metrics['loss']:.4f} "
              f"acc(s1/s2/s3)={train_metrics['acc_s1']:.3f}/"
              f"{train_metrics['acc_s2']:.3f}/{train_metrics['acc_s3']:.3f} | "
              f"val_loss={val_metrics['val_loss']:.4f} "
              f"val_acc={val_metrics['val_acc_s1']:.3f}/"
              f"{val_metrics['val_acc_s2']:.3f}/{val_metrics['val_acc_s3']:.3f}"
              f"{lr_msg}{ckpt_msg}")

        row = {"epoch": epoch, "lr": cur_lr, **train_metrics, **val_metrics}
        writer.writerow(row); csv_file.flush()
        for k, v in row.items():
            history[k].append(v)

        if bad_epochs >= patience:
            print(f"\n  Early stopping at epoch {epoch} (no val_loss improvement for {patience} epochs)")
            break

    csv_file.close()

    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"  Restored best weights from {ckpt} (val_loss={best_val:.4f})")

    return history


# ============================================================
# CASCADED EVALUATION HELPER
# ============================================================


@torch.no_grad()
def evaluate_hierarchical_model(
    model: nn.Module,
    test_loader,
    device: torch.device,
    t1: float = 0.5,
    t2: float = 0.5,
):
    """Run a HierarchicalCrackDataset test loader through a 3-head model and
    collect both per-stage softmax outputs and final cascaded predictions.

    Returns:
        dict with keys:
            'y_true_paths': list of (s1,s2,s3) tuples (Nones for absent levels)
            'y_pred_paths': list of (s1,s2,s3) tuples
            'y_true_flat':  list of original 6-class labels
            'y_pred_flat':  list of original 6-class labels (cascaded)
    """
    model.eval()
    y_true_paths, y_pred_paths = [], []
    y_true_flat, y_pred_flat = [], []

    for imgs, labels, masks in tqdm(test_loader, desc="Cascade eval"):
        imgs = imgs.to(device, non_blocking=True)
        l1, l2, l3 = model(imgs)
        p1 = nn.functional.softmax(l1, dim=1).cpu().numpy()
        p2 = nn.functional.softmax(l2, dim=1).cpu().numpy()
        p3 = nn.functional.softmax(l3, dim=1).cpu().numpy()

        s1_t = labels["stage1"].numpy()
        s2_t = labels["stage2"].numpy(); m2 = masks["stage2"].numpy()
        s3_t = labels["stage3"].numpy(); m3 = masks["stage3"].numpy()

        for i in range(imgs.size(0)):
            true_path = (
                STAGE1_CLASSES[int(s1_t[i])],
                STAGE2_CLASSES[int(s2_t[i])] if m2[i] > 0 else None,
                STAGE3_CLASSES[int(s3_t[i])] if m3[i] > 0 else None,
            )
            pred_path = cascade_predict(p1[i], p2[i], p3[i], t1=t1, t2=t2)

            y_true_paths.append(true_path)
            y_pred_paths.append(pred_path)
            y_true_flat.append(cascaded_to_orig(*true_path))
            y_pred_flat.append(cascaded_to_orig(*pred_path))

    return {
        "y_true_paths": y_true_paths,
        "y_pred_paths": y_pred_paths,
        "y_true_flat":  y_true_flat,
        "y_pred_flat":  y_pred_flat,
    }
