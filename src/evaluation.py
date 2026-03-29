"""Model evaluation: metrics, confusion matrix, and classification reports."""

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def evaluate_model(
    model,
    test_generator,
    class_names: list[str] = config.CLASS_NAMES,
    output_dir: str = config.OUTPUT_DIR,
) -> dict:
    """Evaluate model on test set and generate reports.

    Args:
        model: Trained Keras model.
        test_generator: Test data generator.
        class_names: List of class names.
        output_dir: Directory to save plots and reports.

    Returns:
        Dictionary with evaluation metrics.
    """
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # Get predictions
    test_generator.reset()
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    metrics = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    }

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report)

    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print(f"Report saved to {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(cm, class_names, output_dir)

    return metrics


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_dir: str,
) -> None:
    """Plot and save confusion matrix heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Confusion Matrix (Counts)")

    # Normalized (percentages)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix (Normalized)")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "plots", "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_history(
    histories: list,
    output_dir: str = config.OUTPUT_DIR,
    stage_names: list[str] = None,
) -> None:
    """Plot training loss and accuracy across all stages.

    Args:
        histories: List of Keras History objects (one per stage).
        output_dir: Directory to save the plot.
        stage_names: Optional list of stage names for the legend.
    """
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    if stage_names is None:
        stage_names = [f"Stage {i+1}" for i in range(len(histories))]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    epoch_offset = 0
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, (history, name) in enumerate(zip(histories, stage_names)):
        epochs = range(epoch_offset, epoch_offset + len(history.history["loss"]))
        color = colors[i % len(colors)]

        # Loss
        axes[0, 0].plot(epochs, history.history["loss"], color=color,
                        linestyle="-", label=f"{name} (train)")
        axes[0, 0].plot(epochs, history.history["val_loss"], color=color,
                        linestyle="--", label=f"{name} (val)")

        # Accuracy
        axes[0, 1].plot(epochs, history.history["accuracy"], color=color,
                        linestyle="-", label=f"{name} (train)")
        axes[0, 1].plot(epochs, history.history["val_accuracy"], color=color,
                        linestyle="--", label=f"{name} (val)")

        # Precision
        if "precision" in history.history:
            axes[1, 0].plot(epochs, history.history["precision"], color=color,
                            linestyle="-", label=f"{name} (train)")
            axes[1, 0].plot(epochs, history.history["val_precision"], color=color,
                            linestyle="--", label=f"{name} (val)")

        # Recall
        if "recall" in history.history:
            axes[1, 1].plot(epochs, history.history["recall"], color=color,
                            linestyle="-", label=f"{name} (train)")
            axes[1, 1].plot(epochs, history.history["val_recall"], color=color,
                            linestyle="--", label=f"{name} (val)")

        # Add vertical separator between stages
        if i > 0:
            for ax_row in axes:
                for ax in ax_row:
                    ax.axvline(x=epoch_offset, color="gray", linestyle=":", alpha=0.5)

        epoch_offset += len(history.history["loss"])

    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training & Validation Loss")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Training & Validation Accuracy")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Training & Validation Precision")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_title("Training & Validation Recall")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "plots", "training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Training history plot saved to {save_path}")
