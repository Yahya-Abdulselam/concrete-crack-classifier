"""Model evaluation: metrics, confusion matrix, IoU, and classification reports (PyTorch)."""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def compute_iou_from_cm(
    cm: np.ndarray,
    class_names: list[str],
) -> tuple[dict[str, float], float]:
    """Compute per-class IoU from a confusion matrix.

    IoU_i = TP_i / (TP_i + FP_i + FN_i)
    """
    iou_per_class = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = tp + fp + fn
        iou_per_class[name] = float(tp / denom) if denom > 0 else 0.0
    mean_iou = float(np.mean(list(iou_per_class.values())))
    return iou_per_class, mean_iou


def _compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute the full metrics suite from true/predicted label arrays."""
    accuracy = accuracy_score(y_true, y_pred)
    precision_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    precision_m = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_m = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    iou_per_class, mean_iou = compute_iou_from_cm(cm, class_names)

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True,
    )

    return {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision_w),
        "precision_macro": float(precision_m),
        "recall_weighted": float(recall_w),
        "recall_macro": float(recall_m),
        "f1_weighted": float(f1_w),
        "f1_macro": float(f1_m),
        "mean_iou": float(mean_iou),
        "iou_per_class": iou_per_class,
        "classification_report": report_dict,
    }


def _save_metrics(
    metrics: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_dir: str,
    model_name: str,
) -> None:
    """Save classification report text, confusion matrix plot, and metrics JSON."""
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    suffix = f"_{model_name}" if model_name else ""

    # Classification report (text)
    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report_text)

    report_path = os.path.join(output_dir, f"classification_report{suffix}.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report — {model_name or 'model'}\n")
        f.write("=" * 60 + "\n")
        f.write(report_text)
        f.write("\n\nPer-class IoU:\n")
        for name, iou_val in metrics["iou_per_class"].items():
            f.write(f"  {name:<14}: {iou_val:.4f}\n")
        f.write(f"  {'Mean IoU':<14}: {metrics['mean_iou']:.4f}\n")
    print(f"Report saved to {report_path}")

    # Metrics JSON
    json_path = os.path.join(output_dir, f"metrics{suffix}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics JSON saved to {json_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(cm, class_names, output_dir, model_name)

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY — {model_name or 'model'}")
    print(f"{'='*60}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):     {metrics['f1_weighted']:.4f}")
    print(f"  Mean IoU:          {metrics['mean_iou']:.4f}")
    print(f"{'='*60}")


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    class_names: list[str] = config.CLASS_NAMES,
    output_dir: str = config.OUTPUT_DIR,
    model_name: str = "",
) -> dict:
    """Evaluate a PyTorch model on test set and generate full reports.

    Args:
        model: Trained PyTorch model.
        test_loader: Test DataLoader.
        device: torch.device.
        class_names: List of class names.
        output_dir: Directory to save plots and reports.
        model_name: Model identifier for output file naming.

    Returns:
        Dictionary with all evaluation metrics.
    """
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = _compute_all_metrics(y_true, y_pred, class_names)
    _save_metrics(metrics, y_true, y_pred, class_names, output_dir, model_name)
    return metrics


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = config.CLASS_NAMES,
    output_dir: str = config.OUTPUT_DIR,
    model_name: str = "",
) -> dict:
    """Evaluate from raw prediction arrays (for sklearn models, YOLOv8, etc.)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = _compute_all_metrics(y_true, y_pred, class_names)
    _save_metrics(metrics, y_true, y_pred, class_names, output_dir, model_name)
    return metrics


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_dir: str,
    model_name: str = "",
) -> None:
    """Plot and save confusion matrix heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    title_suffix = f" — {model_name}" if model_name else ""

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"Confusion Matrix (Counts){title_suffix}")

    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title(f"Confusion Matrix (Normalized){title_suffix}")

    plt.tight_layout()
    suffix = f"_{model_name}" if model_name else ""
    save_path = os.path.join(output_dir, "plots", f"confusion_matrix{suffix}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_history(
    histories: list,
    output_dir: str = config.OUTPUT_DIR,
    stage_names: list[str] = None,
    model_name: str = "",
) -> None:
    """Plot training metrics across all stages.

    Args:
        histories: List of TrainingHistory objects (one per stage).
        output_dir: Directory to save the plot.
        stage_names: Optional list of stage names for the legend.
        model_name: Model identifier for output file naming.
    """
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    if stage_names is None:
        stage_names = [f"Stage {i+1}" for i in range(len(histories))]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    title_suffix = f" — {model_name}" if model_name else ""
    epoch_offset = 0
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, (history, name) in enumerate(zip(histories, stage_names)):
        h = history.history
        epochs = range(epoch_offset, epoch_offset + len(h["loss"]))
        color = colors[i % len(colors)]

        axes[0, 0].plot(epochs, h["loss"], color=color, linestyle="-", label=f"{name} (train)")
        axes[0, 0].plot(epochs, h["val_loss"], color=color, linestyle="--", label=f"{name} (val)")

        axes[0, 1].plot(epochs, h["accuracy"], color=color, linestyle="-", label=f"{name} (train)")
        axes[0, 1].plot(epochs, h["val_accuracy"], color=color, linestyle="--", label=f"{name} (val)")

        axes[1, 0].plot(epochs, h["precision"], color=color, linestyle="-", label=f"{name} (train)")
        axes[1, 0].plot(epochs, h["val_precision"], color=color, linestyle="--", label=f"{name} (val)")

        axes[1, 1].plot(epochs, h["recall"], color=color, linestyle="-", label=f"{name} (train)")
        axes[1, 1].plot(epochs, h["val_recall"], color=color, linestyle="--", label=f"{name} (val)")

        if i > 0:
            for ax_row in axes:
                for ax in ax_row:
                    ax.axvline(x=epoch_offset, color="gray", linestyle=":", alpha=0.5)
        epoch_offset += len(h["loss"])

    for ax, ylabel, title in [
        (axes[0, 0], "Loss", f"Training & Validation Loss{title_suffix}"),
        (axes[0, 1], "Accuracy", f"Training & Validation Accuracy{title_suffix}"),
        (axes[1, 0], "Precision", f"Training & Validation Precision{title_suffix}"),
        (axes[1, 1], "Recall", f"Training & Validation Recall{title_suffix}"),
    ]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = f"_{model_name}" if model_name else ""
    save_path = os.path.join(output_dir, "plots", f"training_history{suffix}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Training history plot saved to {save_path}")
