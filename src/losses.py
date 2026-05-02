"""Loss functions and post-hoc adjustment utilities for imbalance experiments.

Provides:
    * Post-hoc logit adjustment (Menon et al. 2021)
    * Tau-normalization of classifier weights (Kang et al. ICLR 2020)
    * Class-Balanced Focal Loss (Cui et al. CVPR 2019)
    * Pairwise Confusion regularization (Dubey et al. ECCV 2018)
    * Confusion-weighted oversampling sampler
    * train_hierarchical_model_v2 — extended multi-task trainer
"""

import copy
import csv
import os
import time
from collections import defaultdict
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.hierarchical import (
    STAGE1_CLASSES, STAGE2_CLASSES, STAGE3_CLASSES,
    STAGE1_TO_IDX, STAGE2_TO_IDX, STAGE3_TO_IDX,
    HierarchicalCrackDataset,
    cascade_predict, cascaded_to_orig,
    hierarchical_collate,
    get_hierarchical_sampler,
)


# ============================================================
# CLASS FREQUENCY COMPUTATION
# ============================================================


def compute_class_frequencies(
    dataset: HierarchicalCrackDataset,
) -> dict[str, np.ndarray]:
    """Compute per-stage class frequency vectors from training set.

    Stage 1: frequencies over ALL samples.
    Stage 2: frequencies among crack-only samples (m2 > 0).
    Stage 3: frequencies among single-crack-only samples (m3 > 0).

    Returns:
        {'stage1': array([freq_nocrack, freq_crack]),
         'stage2': array([freq_multi, freq_single]),
         'stage3': array([freq_deb, freq_flex, freq_shear, freq_others])}
    """
    n = len(dataset)

    # Stage 1: all samples
    s1_counts = np.bincount(dataset.s1, minlength=len(STAGE1_CLASSES)).astype(np.float64)
    s1_freq = s1_counts / s1_counts.sum()

    # Stage 2: crack-only
    mask2 = dataset.m2 > 0
    s2_labels = dataset.s2[mask2]
    s2_counts = np.bincount(s2_labels, minlength=len(STAGE2_CLASSES)).astype(np.float64)
    s2_freq = s2_counts / max(s2_counts.sum(), 1)

    # Stage 3: single-only
    mask3 = dataset.m3 > 0
    s3_labels = dataset.s3[mask3]
    s3_counts = np.bincount(s3_labels, minlength=len(STAGE3_CLASSES)).astype(np.float64)
    s3_freq = s3_counts / max(s3_counts.sum(), 1)

    return {"stage1": s1_freq, "stage2": s2_freq, "stage3": s3_freq}


def compute_class_counts(
    dataset: HierarchicalCrackDataset,
) -> dict[str, np.ndarray]:
    """Compute per-stage raw class counts (for CB focal loss weights).

    Same masking logic as compute_class_frequencies but returns integer counts.
    """
    s1_counts = np.bincount(dataset.s1, minlength=len(STAGE1_CLASSES))

    mask2 = dataset.m2 > 0
    s2_counts = np.bincount(dataset.s2[mask2], minlength=len(STAGE2_CLASSES))

    mask3 = dataset.m3 > 0
    s3_counts = np.bincount(dataset.s3[mask3], minlength=len(STAGE3_CLASSES))

    return {"stage1": s1_counts, "stage2": s2_counts, "stage3": s3_counts}


# ============================================================
# POST-HOC LOGIT ADJUSTMENT (Exp 1)
# ============================================================


def apply_logit_adjustment(
    logits: torch.Tensor,
    log_frequencies: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Subtract tau * log(pi_y) from logits (Menon et al. 2021).

    Args:
        logits: (B, C) raw model logits.
        log_frequencies: (C,) log of per-class training frequencies.
        tau: adjustment strength.

    Returns:
        Adjusted logits of same shape.
    """
    return logits - tau * log_frequencies.to(logits.device)


@torch.no_grad()
def evaluate_hierarchical_with_logit_adj(
    model: nn.Module,
    loader,
    device: torch.device,
    log_freqs: dict[str, torch.Tensor],
    tau: dict[str, float] | float,
    t1: float = 0.5,
    t2: float = 0.5,
) -> dict:
    """Evaluate a 3-head hierarchical model with per-head logit adjustment.

    Args:
        model: InceptionV3CBAMHierarchical.
        loader: DataLoader yielding (imgs, labels_dict, masks_dict).
        device: torch.device.
        log_freqs: {'stage1': tensor(2,), 'stage2': tensor(2,), 'stage3': tensor(4,)}.
        tau: float (shared across heads) or dict {'stage1': t, 'stage2': t, 'stage3': t}.
        t1, t2: cascade thresholds.

    Returns:
        Same dict format as hierarchical.evaluate_hierarchical_model.
    """
    if isinstance(tau, (int, float)):
        tau = {"stage1": float(tau), "stage2": float(tau), "stage3": float(tau)}

    model.eval()
    y_true_paths, y_pred_paths = [], []
    y_true_flat, y_pred_flat = [], []

    for imgs, labels, masks in tqdm(loader, desc="Logit-adj eval"):
        imgs = imgs.to(device, non_blocking=True)
        l1, l2, l3 = model(imgs)

        # Apply logit adjustment per head
        l1_adj = apply_logit_adjustment(l1, log_freqs["stage1"], tau["stage1"])
        l2_adj = apply_logit_adjustment(l2, log_freqs["stage2"], tau["stage2"])
        l3_adj = apply_logit_adjustment(l3, log_freqs["stage3"], tau["stage3"])

        p1 = F.softmax(l1_adj, dim=1).cpu().numpy()
        p2 = F.softmax(l2_adj, dim=1).cpu().numpy()
        p3 = F.softmax(l3_adj, dim=1).cpu().numpy()

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
        "y_true_flat": y_true_flat,
        "y_pred_flat": y_pred_flat,
    }


# ============================================================
# TAU-NORMALIZATION (Exp 2)
# ============================================================


def tau_normalize_linear(layer: nn.Linear, tau: float) -> None:
    """In-place tau-normalization: w_y = w_y / ||w_y||^tau.

    Kang et al. (ICLR 2020) showed classifier weight norms grow with
    class frequency. This corrects the bias without retraining.
    """
    with torch.no_grad():
        norms = layer.weight.data.norm(dim=1, keepdim=True)  # (num_classes, 1)
        norms = norms.clamp(min=1e-8)
        layer.weight.data = layer.weight.data / (norms ** tau)


def tau_normalize_hierarchical(model, tau1: float, tau2: float, tau3: float) -> None:
    """Apply tau-normalization to each head's final Linear layer.

    Targets model.head{1,2,3}.fc[-1] (the last nn.Linear in each head).
    """
    tau_normalize_linear(model.head1.fc[-1], tau1)
    tau_normalize_linear(model.head2.fc[-1], tau2)
    tau_normalize_linear(model.head3.fc[-1], tau3)


# ============================================================
# CLASS-BALANCED FOCAL LOSS (Exp 3)
# ============================================================


class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss (Cui et al., CVPR 2019).

    Per-class weight: w_y = (1 - beta) / (1 - beta^n_y)
    Focal modulation: (1 - p_t)^gamma

    Args:
        samples_per_class: list or array of sample counts per class.
        beta: effective number parameter, default 0.999.
        gamma: focal modulation exponent, default 2.0.
    """

    def __init__(
        self,
        samples_per_class,
        beta: float = 0.999,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.gamma = gamma

        samples = np.array(samples_per_class, dtype=np.float64)
        effective_num = 1.0 - np.power(beta, samples)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        # Normalize so weights sum to num_classes (like balanced class weights)
        weights = weights / weights.sum() * len(samples)

        self.register_buffer("cb_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute per-sample CB focal loss.

        Args:
            logits: (B, C) raw logits.
            targets: (B,) class indices.

        Returns:
            (B,) per-sample losses (no reduction).
        """
        # Standard cross-entropy per sample
        log_probs = F.log_softmax(logits, dim=1)  # (B, C)
        probs = torch.exp(log_probs)               # (B, C)

        # Gather the target class probability
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        p_t = (probs * targets_one_hot).sum(dim=1)  # (B,)

        # Focal modulation
        focal_weight = (1.0 - p_t) ** self.gamma  # (B,)

        # Class-balanced weight per sample
        cb_weight = self.cb_weights[targets]  # (B,)

        # CE loss per sample
        ce_loss = -(log_probs * targets_one_hot).sum(dim=1)  # (B,)

        return cb_weight * focal_weight * ce_loss


def masked_cb_focal(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    cb_focal: ClassBalancedFocalLoss,
) -> torch.Tensor:
    """Drop-in replacement for _masked_ce using CB focal loss.

    Same masking logic: loss = sum(loss_i * mask_i) / sum(mask_i).
    """
    losses = cb_focal(logits, targets)  # (B,)
    denom = mask.sum().clamp(min=1.0)
    return (losses * mask).sum() / denom


# ============================================================
# PAIRWISE CONFUSION LOSS (Exp 5)
# ============================================================


def pairwise_confusion_loss(logits: torch.Tensor) -> torch.Tensor:
    """Pairwise Confusion regularizer (Dubey et al., ECCV 2018).

    L_PC = (1/B^2) * sum_{i!=j} ||p_i - p_j||_2

    Pushes probability distributions toward each other, forcing the
    network to learn more discriminative features to maintain CE accuracy.

    Args:
        logits: (B, C) raw logits.

    Returns:
        Scalar loss.
    """
    probs = F.softmax(logits, dim=1)  # (B, C)
    B = probs.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=logits.device)

    # Pairwise L2 distances: ||p_i - p_j||_2 for all i != j
    # Using expanded form: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    dists = torch.cdist(probs, probs, p=2)  # (B, B)

    # Exclude diagonal (i == j) and average
    mask = 1.0 - torch.eye(B, device=logits.device)
    return (dists * mask).sum() / (B * B)


# ============================================================
# CONFUSION-WEIGHTED OVERSAMPLING (Exp 6)
# ============================================================


@torch.no_grad()
def identify_misclassified(
    model: nn.Module,
    dataset: HierarchicalCrackDataset,
    device: torch.device,
    batch_size: int = 16,
    t1: float = 0.5,
    t2: float = 0.5,
) -> list[int]:
    """Forward pass on dataset to find misclassified sample indices.

    Uses cascaded prediction to determine if the final 6-class label
    matches ground truth.

    Returns:
        List of dataset indices where prediction != ground truth.
    """
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
        collate_fn=hierarchical_collate,
    )

    misclassified = []
    idx = 0

    for imgs, labels, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        l1, l2, l3 = model(imgs)
        p1 = F.softmax(l1, dim=1).cpu().numpy()
        p2 = F.softmax(l2, dim=1).cpu().numpy()
        p3 = F.softmax(l3, dim=1).cpu().numpy()

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
            true_flat = cascaded_to_orig(*true_path)
            pred_flat = cascaded_to_orig(*pred_path)

            if true_flat != pred_flat:
                misclassified.append(idx)
            idx += 1

    return misclassified


def build_confusion_weighted_sampler(
    dataset: HierarchicalCrackDataset,
    misclassified_indices: list[int],
    oversample_factor: int = 3,
    no_crack_mix: float = 0.25,
    max_aug_factor: int = config.MAX_AUG_FACTOR,
) -> WeightedRandomSampler:
    """Build a sampler that boosts weights for misclassified samples.

    Starts from the base hierarchical sampler weights, then multiplies
    weights for misclassified samples by oversample_factor.
    """
    # Start with the hierarchical sampler's weight computation
    is_no = dataset.s1 == STAGE1_TO_IDX["no_crack"]
    is_crack = ~is_no
    is_multi = is_crack & (dataset.s2 == STAGE2_TO_IDX["multi"])
    is_single = is_crack & (dataset.s2 == STAGE2_TO_IDX["single"])

    sub_counts = {
        c: int(((dataset.s3 == STAGE3_TO_IDX[c]) & is_single).sum())
        for c in STAGE3_CLASSES
    }
    nonzero = [v for v in sub_counts.values() if v > 0]
    smallest = min(nonzero) if nonzero else 1
    target = smallest * max_aug_factor

    n_multi = int(is_multi.sum())
    n_no = int(is_no.sum())

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

    # Boost misclassified samples
    for idx in misclassified_indices:
        if 0 <= idx < len(weights):
            weights[idx] *= oversample_factor

    # Recompute total to account for increased weights
    total = int(round(weights.sum()))

    print(f"\n[Confusion-weighted sampler] {len(misclassified_indices)} misclassified, "
          f"boost={oversample_factor}x, samples/epoch={total}")
    return WeightedRandomSampler(weights=weights, num_samples=total, replacement=True)


# ============================================================
# MASKED CE (local copy for train_hierarchical_model_v2)
# ============================================================


def _masked_ce(logits, targets, mask, weight=None):
    """Cross-entropy with per-sample masking. Returns scalar (mean over active)."""
    losses = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    denom = mask.sum().clamp(min=1.0)
    return (losses * mask).sum() / denom


# ============================================================
# EXTENDED HIERARCHICAL TRAINER (v2)
# ============================================================


def train_hierarchical_model_v2(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    device: torch.device,
    epochs: int,
    output_dir: str,
    stage: int,
    loss_weights: tuple[float, float, float] = (1.0, 0.0, 0.0),
    class_weights: dict | None = None,
    loss_fn_per_stage: dict[str, Callable] | None = None,
    reg_fn: Callable | None = None,
    reg_lambda: float = 0.0,
    sampler_update_fn: Callable | None = None,
    train_dataset: HierarchicalCrackDataset | None = None,
    batch_size: int = 16,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    lr_patience: int = config.REDUCE_LR_PATIENCE,
    lr_factor: float = config.REDUCE_LR_FACTOR,
    model_name: str = "",
):
    """Extended multi-task training loop with pluggable loss functions.

    Mirrors src.hierarchical.train_hierarchical_model with additional features:
        - loss_fn_per_stage: custom loss per stage (e.g. CB focal).
          Each callable has signature: (logits, targets, mask) -> scalar loss.
          When None, falls back to standard masked CE.
        - reg_fn: regularization applied to each active head's logits.
          Signature: (logits) -> scalar loss. Weighted by reg_lambda.
        - sampler_update_fn: called after each val epoch to rebuild the sampler.
          Signature: (model, dataset, device) -> WeightedRandomSampler.
        - train_dataset + batch_size: needed if sampler_update_fn is used.
    """
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    # Class weights for standard CE fallback
    cw = class_weights or {}
    w1 = cw.get("stage1", None)
    w2 = cw.get("stage2", None)
    w3 = cw.get("stage3", None)
    if w1 is not None: w1 = w1.to(device)
    if w2 is not None: w2 = w2.to(device)
    if w3 is not None: w3 = w3.to(device)

    # Loss functions per stage
    if loss_fn_per_stage is None:
        fn_s1 = lambda logits, targets, mask: _masked_ce(logits, targets, mask, w1)
        fn_s2 = lambda logits, targets, mask: _masked_ce(logits, targets, mask, w2)
        fn_s3 = lambda logits, targets, mask: _masked_ce(logits, targets, mask, w3)
    else:
        fn_s1 = loss_fn_per_stage.get("stage1", lambda l, t, m: _masked_ce(l, t, m, w1))
        fn_s2 = loss_fn_per_stage.get("stage2", lambda l, t, m: _masked_ce(l, t, m, w2))
        fn_s3 = loss_fn_per_stage.get("stage3", lambda l, t, m: _masked_ce(l, t, m, w3))

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
    if loss_fn_per_stage:
        print(f"  Custom loss functions active")
    if reg_fn:
        print(f"  Regularization: lambda={reg_lambda}")

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
            t1_lbl = labels["stage1"].to(device); m1 = masks["stage1"].to(device)
            t2_lbl = labels["stage2"].to(device); m2 = masks["stage2"].to(device)
            t3_lbl = labels["stage3"].to(device); m3 = masks["stage3"].to(device)

            optimizer.zero_grad()
            l1, l2, l3 = model(imgs)

            ls1 = fn_s1(l1, t1_lbl, m1) if loss_weights[0] > 0 else torch.tensor(0.0, device=device)
            ls2 = fn_s2(l2, t2_lbl, m2) if loss_weights[1] > 0 else torch.tensor(0.0, device=device)
            ls3 = fn_s3(l3, t3_lbl, m3) if loss_weights[2] > 0 else torch.tensor(0.0, device=device)
            loss = loss_weights[0] * ls1 + loss_weights[1] * ls2 + loss_weights[2] * ls3

            # Regularization (e.g. pairwise confusion)
            if reg_fn is not None and reg_lambda > 0:
                reg_total = torch.tensor(0.0, device=device)
                if loss_weights[0] > 0:
                    reg_total = reg_total + reg_fn(l1)
                if loss_weights[1] > 0:
                    reg_total = reg_total + reg_fn(l2)
                if loss_weights[2] > 0:
                    reg_total = reg_total + reg_fn(l3)
                loss = loss + reg_lambda * reg_total

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
                s1_correct += int(((p1 == t1_lbl).float() * m1).sum())
                s1_n       += int(m1.sum())
                s2_correct += int(((p2 == t2_lbl).float() * m2).sum())
                s2_n       += int(m2.sum())
                s3_correct += int(((p3 == t3_lbl).float() * m3).sum())
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
                t1_lbl = labels["stage1"].to(device); m1 = masks["stage1"].to(device)
                t2_lbl = labels["stage2"].to(device); m2 = masks["stage2"].to(device)
                t3_lbl = labels["stage3"].to(device); m3 = masks["stage3"].to(device)

                l1, l2, l3 = model(imgs)
                ls1 = fn_s1(l1, t1_lbl, m1)
                ls2 = fn_s2(l2, t2_lbl, m2)
                ls3 = fn_s3(l3, t3_lbl, m3)
                vloss = loss_weights[0] * ls1 + loss_weights[1] * ls2 + loss_weights[2] * ls3

                bs = imgs.size(0)
                vcnt += bs
                vagg["loss"]    += vloss.item() * bs
                vagg["loss_s1"] += ls1.item() * bs
                vagg["loss_s2"] += ls2.item() * bs
                vagg["loss_s3"] += ls3.item() * bs

                p1 = l1.argmax(1); p2 = l2.argmax(1); p3 = l3.argmax(1)
                v1_c += int(((p1 == t1_lbl).float() * m1).sum()); v1_n += int(m1.sum())
                v2_c += int(((p2 == t2_lbl).float() * m2).sum()); v2_n += int(m2.sum())
                v3_c += int(((p3 == t3_lbl).float() * m3).sum()); v3_n += int(m3.sum())

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

        # Per-epoch sampler update (Exp 6)
        if sampler_update_fn is not None and train_dataset is not None:
            new_sampler = sampler_update_fn(model, train_dataset, device)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=new_sampler,
                num_workers=0, pin_memory=True, drop_last=True,
                collate_fn=hierarchical_collate,
            )

        if bad_epochs >= patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no val_loss improvement for {patience} epochs)")
            break

    csv_file.close()

    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"  Restored best weights from {ckpt} (val_loss={best_val:.4f})")

    return history
