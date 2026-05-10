from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix


try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def model_parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy_by_snr(y_true: np.ndarray, y_pred: np.ndarray, snrs: np.ndarray) -> Dict[str, float]:
    scores = {}
    for snr in sorted(np.unique(snrs)):
        mask = snrs == snr
        scores[str(int(snr))] = float(np.mean(y_true[mask] == y_pred[mask]))
    return scores


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[str], save_dir: Path) -> list:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    csv_path = save_dir / "confusion_matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *classes])
        for class_name, row in zip(classes, cm):
            writer.writerow([class_name, *row.tolist()])

    if plt is not None:
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(save_dir / "confusion_matrix.png", dpi=160)
        plt.close(fig)
    return cm.tolist()


def plot_history(history: dict, save_dir: Path) -> None:
    if plt is None:
        print("matplotlib is not installed; skipping history.png.")
        return

    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    if len(epochs) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_dir / "history.png", dpi=160)
    plt.close(fig)


def plot_snr_accuracy(snr_scores: Dict[str, float], save_dir: Path) -> None:
    if plt is None:
        print("matplotlib is not installed; skipping accuracy_vs_snr.png.")
        return

    snrs = sorted((int(k), v) for k, v in snr_scores.items())
    x = [item[0] for item in snrs]
    y = [item[1] for item in snrs]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs SNR")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "accuracy_vs_snr.png", dpi=160)
    plt.close(fig)
