"""
Train Plant Disease Classification Model
----------------------------------------
Training pipeline for PlantVillage classification
using ResNet50 + DCBAM backbone.

Features:
- Reproducible training
- Pretrained backbone
- Early stopping
- Best loss / accuracy checkpoints
- CSV logging
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# project root
sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from datasets.plant_dataset import PlantDataset, get_transform
from utils.plot_logs import plot_training_curves

from models.plant_model import PlantModel

# ==========================================================
# CONFIG
# ==========================================================
from config.config import (
    SEED,

    DATA_ROOT,
    TRAIN_SPLIT,

    FEATURE_LEVEL,
    PRETRAINED,

    BATCH_SIZE,

    DEVICE,

    LR,
    WEIGHT_DECAY,

    EPOCHS,
    PATIENCE,

    NUM_WORKERS,
    PIN_MEMORY,

    LOG_DIR,
    CHECKPOINT_DIR,

    BEST_LOSS_PATH,
    BEST_ACC_PATH,
)

# ==========================================================
# Reproducibility
# ==========================================================
random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==========================================================
# Create Stratified Train/Validation Split
# ==========================================================
def create_train_val_split(train_split, seed, val_ratio=0.1):
    """
    Create deterministic, stratified train/validation splits
    from the official training split.
    """

    train_split = Path(train_split)

    paths = [
        line.strip()
        for line in train_split.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    labels = [
        Path(image_path).parent.name
        for image_path in paths
    ]

    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate image paths found in training split.")

    train_paths, val_paths = train_test_split(
        paths,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )

    output_dir = train_split.parent

    train_output = output_dir / "color_train_internal.txt"
    val_output = output_dir / "color_val.txt"

    train_output.write_text(
        "\n".join(train_paths) + "\n",
        encoding="utf-8",
    )

    val_output.write_text(
        "\n".join(val_paths) + "\n",
        encoding="utf-8",
    )

    return train_output, val_output

# ==========================================================
# MAIN
# ==========================================================
def main():
    
    # ==========================================================
    # Load Dataset
    # ==========================================================
    train_split_file, val_split_file = create_train_val_split(
        train_split=TRAIN_SPLIT,
        seed=SEED,
        val_ratio=0.1,
    )

    train_dataset = PlantDataset(
        split_file=train_split_file,
        data_root=DATA_ROOT,
        transform=get_transform(augment=True),
        preflight_check=True,
        fail_on_error=True,
    )

    torch.save(
        train_dataset.class_to_idx,
        CHECKPOINT_DIR / "class_to_idx.pth",
    )

    val_dataset = PlantDataset(
        split_file=val_split_file,
        data_root=DATA_ROOT,
        class_to_idx=train_dataset.class_to_idx,
        transform=get_transform(augment=False),
        preflight_check=True,
        fail_on_error=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    num_classes = len(train_dataset.classes)

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    print(
        f"Classes: {num_classes}"
    )

    # ==========================================================
    # CSV Logger Setup
    # ==========================================================
    log_file = LOG_DIR / "train_log.csv"

    with open(
        log_file,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "train_samples",
                len(train_dataset)
            ]
        )

        writer.writerow(
            [
                "val_samples",
                len(val_dataset)
            ]
        )

        writer.writerow([])

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "train_acc",
                "val_acc",
            ]
        )

    def log_csv(
        epoch,
        train_loss,
        val_loss,
        train_acc,
        val_acc
    ):

        with open(
            log_file,
            "a",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    epoch,
                    train_loss,
                    val_loss,
                    train_acc,
                    val_acc,
                ]
            )

    # ==========================================================
    # Model, loss, optimizer
    # ==========================================================
    model = PlantModel(
        num_classes=num_classes,
        feature_level=FEATURE_LEVEL,
        pretrained=PRETRAINED,
    ).to(DEVICE)

    criterion = CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0

    counter = 0

    # ==========================================================
    # Learning rate scheduler
    # ==========================================================
    def lr_scheduler(epoch, lr):

        if epoch < 15:
            return lr

        return max(
            lr - 1e-5,
            4e-5
        )

    # ==========================================================
    # Training loop
    # ==========================================================
    for epoch in range(EPOCHS):

        # -------------------------
        # Train
        # -------------------------
        model.train()

        train_loss = 0.0
        train_correct = 0

        for x, y in train_loader:

            x = x.to(
                DEVICE,
                non_blocking=True
            )

            y = y.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            output = model(x)

            loss = criterion(
                output,
                y
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                *
                x.size(0)
            )

            train_correct += (
                output.argmax(1) == y
            ).sum().item()

        train_loss /= len(train_loader.dataset)

        train_acc = (
            train_correct
            /
            len(train_loader.dataset)
        )

        # -------------------------
        # Validation
        # -------------------------
        model.eval()

        val_loss = 0.0
        val_correct = 0

        with torch.inference_mode():

            for x, y in val_loader:

                x = x.to(
                    DEVICE,
                    non_blocking=True
                )

                y = y.to(
                    DEVICE,
                    non_blocking=True
                )

                output = model(x)

                loss = criterion(
                    output,
                    y
                )

                val_loss += (
                    loss.item()
                    *
                    x.size(0)
                )

                val_correct += (
                    output.argmax(1)==y
                ).sum().item()

        val_loss /= len(val_loader.dataset)

        val_acc = (
            val_correct
            /
            len(val_loader.dataset)
        )

        # -------------------------
        # CSV Logging
        # -------------------------
        log_csv(
            epoch + 1,
            train_loss,
            val_loss,
            train_acc,
            val_acc
        )

        # -------------------------
        # Save Best Checkpoints
        # -------------------------
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            counter = 0

            torch.save(
                model.state_dict(),
                BEST_LOSS_PATH
            )

        else:
            counter += 1

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                BEST_ACC_PATH
            )

        # -------------------------
        # LR Update
        # -------------------------
        for group in optimizer.param_groups:

            group["lr"] = lr_scheduler(
                epoch,
                group["lr"]
            )

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Acc {val_acc:.4f}"
        )

        # -------------------------
        # Early Stopping
        # -------------------------
        if counter >= PATIENCE:

            print(
                f"Early stopping at epoch {epoch+1}"
            )

            break

    print(
        "🔥 Training finished."
    )

    # -------------------------
    # Plot training curves
    # -------------------------
    plot_training_curves()

# ==================================================
# RUN MAIN
# ==================================================
if __name__ == "__main__":
    main()