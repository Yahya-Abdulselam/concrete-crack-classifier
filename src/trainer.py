"""Shared PyTorch training engine for all models.

Provides train_one_epoch, validate, EarlyStopping, CSVLogger, and a
top-level train_model function that replaces Keras model.fit().
"""

import csv
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch.

    Returns:
        Dict with loss, accuracy, precision, recall.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        "loss": running_loss / n,
        "accuracy": (all_preds == all_labels).mean(),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
    }


def validate(model, loader, criterion, device):
    """Validate on a dataset.

    Returns:
        Dict with loss, accuracy, precision, recall.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  val", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        "loss": running_loss / n,
        "accuracy": (all_preds == all_labels).mean(),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
    }


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def step(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class CSVLogger:
    """Log per-epoch metrics to a CSV file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.writer = None
        self.file = None

    def open(self, fieldnames: list[str]):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.file = open(self.filepath, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, row: dict):
        if self.writer is None:
            self.open(list(row.keys()))
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


@dataclass
class TrainingHistory:
    """Stores per-epoch metrics (equivalent to Keras History object)."""
    history: dict = field(default_factory=lambda: {
        "loss": [], "accuracy": [], "precision": [], "recall": [],
        "val_loss": [], "val_accuracy": [], "val_precision": [], "val_recall": [],
    })

    def append(self, train_metrics: dict, val_metrics: dict):
        self.history["loss"].append(train_metrics["loss"])
        self.history["accuracy"].append(train_metrics["accuracy"])
        self.history["precision"].append(train_metrics["precision"])
        self.history["recall"].append(train_metrics["recall"])
        self.history["val_loss"].append(val_metrics["loss"])
        self.history["val_accuracy"].append(val_metrics["accuracy"])
        self.history["val_precision"].append(val_metrics["precision"])
        self.history["val_recall"].append(val_metrics["recall"])


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device: torch.device,
    epochs: int,
    output_dir: str,
    stage: int = 1,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    lr_patience: int = config.REDUCE_LR_PATIENCE,
    lr_factor: float = config.REDUCE_LR_FACTOR,
    model_name: str = "",
) -> TrainingHistory:
    """Full training loop with early stopping, LR scheduling, checkpointing, CSV logging.

    Args:
        model: PyTorch model (already on device).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: PyTorch optimizer.
        criterion: Loss function.
        device: torch.device.
        epochs: Maximum number of epochs.
        output_dir: Base directory for saving outputs.
        stage: Training stage number (for file naming).
        patience: Early stopping patience.
        lr_patience: ReduceLROnPlateau patience.
        lr_factor: ReduceLROnPlateau factor.

    Returns:
        TrainingHistory with per-epoch metrics.
    """
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=1e-7,
    )
    early_stopping = EarlyStopping(patience=patience, mode="min")
    prefix = f"{model_name}_" if model_name else ""
    csv_logger = CSVLogger(os.path.join(output_dir, "logs", f"{prefix}stage{stage}_metrics.csv"))
    history = TrainingHistory()

    best_val_acc = 0.0
    checkpoint_path = os.path.join(output_dir, "models", f"best_{prefix}stage{stage}.pt")

    print(f"\nStarting training: {epochs} epochs, patience={patience}")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.1e}, "
          f"LR patience: {lr_patience}, LR factor: {lr_factor}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        history.append(train_metrics, val_metrics)

        # LR scheduling
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["loss"])
        new_lr = optimizer.param_groups[0]["lr"]
        lr_msg = f" | LR {old_lr:.1e}->{new_lr:.1e}" if new_lr != old_lr else ""

        # Checkpoint best model
        ckpt_msg = ""
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), checkpoint_path)
            ckpt_msg = " | SAVED"

        print(f"  Epoch {epoch:>3}/{epochs} [{elapsed:>5.1f}s] "
              f"loss: {train_metrics['loss']:.4f} "
              f"acc: {train_metrics['accuracy']:.4f} | "
              f"val_loss: {val_metrics['loss']:.4f} "
              f"val_acc: {val_metrics['accuracy']:.4f}"
              f"{lr_msg}{ckpt_msg}")

        # CSV logging
        csv_logger.log({
            "epoch": epoch,
            "lr": current_lr,
            "loss": train_metrics["loss"],
            "accuracy": train_metrics["accuracy"],
            "precision": train_metrics["precision"],
            "recall": train_metrics["recall"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        })

        # Early stopping
        if early_stopping.step(val_metrics["loss"]):
            print(f"\n  Early stopping triggered at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    csv_logger.close()

    # Restore best weights
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"  Restored best weights from {checkpoint_path} (val_acc={best_val_acc:.4f})")

    print(f"\nStage {stage} complete: {epoch}/{epochs} epochs, best val_acc={best_val_acc:.4f}")
    return history
