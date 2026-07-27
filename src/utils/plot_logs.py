"""
Plot Training Logs for the Plant Disease Classification project
---------------------------------------------------------------
Reads CSV training logs and visualizes:

1. Loss curves (Train / Val)
2. Accuracy curves (Train / Val)

Highlights potential overfitting points.

Input:
LOG_DIR/train_log.csv

Output:
PLOT_DIR/training_loss.png
PLOT_DIR/training_accuracy.png
"""

import csv, sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# ==============================
# CONFIG
# ==============================
from config.config import (
    LOG_DIR, PLOT_DIR
)

LOG_FILE = (
    LOG_DIR / "train_log.csv"
)

# ----------------------------------
# Matplotlib settings for publication
# ----------------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ==============================
# PLOT FUNCTION
# ==============================
def plot_metric(
    ax,
    epochs,
    train,
    val,
    ylabel,
    title=None,
    gap_threshold=None,
    legend_loc="best"
):
    """
    Plot train/validation metrics.

    Parameters
    ----------
    ax:
        Matplotlib axis.

    epochs:
        List of epoch indices.

    train:
        Train metric values.

    val:
        Validation metric values.

    ylabel:
        Y-axis label.

    title:
        Optional subplot title.

    gap_threshold:
        Used for overfitting detection.
        If None:
            Detect divergence in loss.
        Else:
            Detect train-validation gap.
    """

    ax.plot(
        epochs,
        train,
        label="Train"
    )

    ax.plot(
        epochs,
        val,
        label="Validation"
    )

    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    ax.legend(loc=legend_loc)
    ax.grid(True)

    # --------------------------
    # Overfitting detection
    # --------------------------
    for i in range(1, len(epochs)):

        # Loss divergence
        if gap_threshold is None:

            if (
                val[i] > val[i - 1]
                and
                train[i] < train[i - 1]
            ):
                ax.axvline(
                    epochs[i],
                    linestyle="--",
                    alpha=0.15
                )

        # Accuracy gap
        else:

            if (
                train[i] - val[i]
                > gap_threshold
            ):
                ax.axvline(
                    epochs[i],
                    linestyle="--",
                    alpha=0.15
                )

# ==============================
# MAIN VISUALIZATION FUNCTION
# ==============================
def plot_training_curves() -> None:
    """
    Load CSV logs and generate training plots.
    """

    if not LOG_FILE.exists():
        raise FileNotFoundError(
            f"CSV log file not found: {LOG_FILE}"
        )

    # --------------------------
    # Read CSV
    # --------------------------
    epochs = []

    train_loss = []
    val_loss = []

    train_acc = []
    val_acc = []

    with open(LOG_FILE, "r") as f:

        # Skip metadata lines
        lines = f.readlines()[3:]

        reader = csv.DictReader(lines)

        for row in reader:

            epochs.append(
                int(row["epoch"])
            )

            train_loss.append(
                float(row["train_loss"])
            )

            val_loss.append(
                float(row["val_loss"])
            )

            train_acc.append(
                float(row["train_acc"])
            )

            val_acc.append(
                float(row["val_acc"])
            )

    # ==========================
    # LOSS CURVE (SEPARATE FIGURE)
    # ==========================
    fig1, ax1 = plt.subplots(
        figsize=(7, 5)
    )

    plot_metric(
        ax=ax1,
        epochs=epochs,
        train=train_loss,
        val=val_loss,
        ylabel="Loss",
        title="Training Loss",
        legend_loc="upper right"
    )

    ax1.set_xlabel("Epoch")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "training_loss.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig1)

    # ==========================
    # ACCURACY CURVE (SEPARATE FIGURE)
    # ==========================
    fig2, ax2 = plt.subplots(
        figsize=(7, 5)
    )

    plot_metric(
        ax=ax2,
        epochs=epochs,
        train=train_acc,
        val=val_acc,
        ylabel="Accuracy",
        title="Training Accuracy",
        gap_threshold=0.1,
        legend_loc="lower right"
    )

    ax2.set_xlabel("Epoch")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "training_accuracy.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig2)

    print(
        f"✅ Training curves saved to: {PLOT_DIR}"
    )

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    plot_training_curves()