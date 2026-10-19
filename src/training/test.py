"""
Test Plant Disease Classification Model
---------------------------------------
Evaluation script for PlantModel.

Features:
- Loads trained checkpoints
- Evaluates test split
- Classification report
- Confusion matrix
- Saves results
"""

import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from datasets.plant_dataset import (
    PlantDataset,
    get_transform
)

from models.plant_model import PlantModel

# ==================================================
# CONFIG
# ==================================================
from config.config import (
    TEST_SPLIT,
    DATA_ROOT,
    BATCH_SIZE,
    DEVICE,
    BEST_LOSS_PATH,
    BEST_ACC_PATH,
    PLOT_DIR,
    RESULT_DIR,
    CHECKPOINT_DIR,
    FEATURE_LEVEL,
    PRETRAINED,
    PIN_MEMORY
)

# ==================================================
# Matplotlib settings for publication
# ==================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ==================================================
# DATASET
# ==================================================
CLASS_MAPPING_PATH = (
    CHECKPOINT_DIR /
    "class_to_idx.pth"
)

class_to_idx = torch.load(
    CLASS_MAPPING_PATH,
    weights_only=False
)

test_dataset = PlantDataset(
    split_file=TEST_SPLIT,
    data_root=DATA_ROOT,
    class_to_idx=class_to_idx,
    transform=get_transform(
        augment=False
    ),
    preflight_check=True,
    fail_on_error=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=PIN_MEMORY
)

NUM_CLASSES = len(
    test_dataset.classes
)

print(
    f"Test samples: {len(test_dataset)}"
)

print(
    f"Classes: {NUM_CLASSES}"
)

# ==================================================
# MODEL
# ==================================================
model = PlantModel(
    num_classes=NUM_CLASSES,
    feature_level=FEATURE_LEVEL,
    pretrained=PRETRAINED
).to(DEVICE)

# ==================================================
# CONFUSION MATRIX
# ==================================================
def plot_confusion_matrix(
    cm,
    classes,
    title,
    save_path
):

    N = len(classes)

    plt.figure(
        figsize=(
            N * 0.6,
            N * 0.6
        )
    )

    plt.imshow(
        cm,
        interpolation="nearest",
        cmap=plt.cm.Blues
    )

    plt.xticks(
        np.arange(N),
        classes,
        rotation=90
    )

    plt.yticks(
        np.arange(N),
        classes
    )

    for i in range(N):

        for j in range(N):

            value = cm[i,j] * 100

            plt.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7
            )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "True"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        format="pdf",
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

# ==================================================
# RESULT LOGGER
# ==================================================
RESULT_FILE = (
    RESULT_DIR /
    "test_results.txt"
)

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Plant Disease Classification Results\n"
    )

    f.write(
        "=" * 50 + "\n\n"
    )

def save_result(text):

    with open(
        RESULT_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(text)
        f.write("\n")

# ==================================================
# EVALUATION
# ==================================================
def evaluate(
    checkpoint,
    name
):

    print(
        f"\n===== {name.title()} ====="
    )

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    y_true = []
    y_pred = []

    with torch.inference_mode():

        for x,y in test_loader:

            x = x.to(
                DEVICE,
                non_blocking=True
            )

            y = y.to(
                DEVICE,
                non_blocking=True
            )

            output = model(x)

            pred = output.argmax(
                dim=1
            )

            y_true.extend(
                y.cpu().numpy()
            )

            y_pred.extend(
                pred.cpu().numpy()
            )

    report = classification_report(
        y_true,
        y_pred,
        target_names=test_dataset.classes,
        digits=4
    )

    print(report)

    save_result(
        f"\n===== {name.title()} =====\n"
        f"{report}\n"
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        normalize="true"
    )

    plot_path = (
        PLOT_DIR /
        f"{name}_confusion_matrix.pdf"
    )

    plot_confusion_matrix(
        cm,
        test_dataset.classes,
        name,
        plot_path
    )

# ==================================================
# RUN
# ==================================================
evaluate(
    BEST_LOSS_PATH,
    "best_validation_loss"
)

evaluate(
    BEST_ACC_PATH,
    "best_validation_accuracy"
)

print(
    f"\nTesting Results saved at: {RESULT_FILE}"
)
