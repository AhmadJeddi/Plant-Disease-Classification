"""
Central configuration file
--------------------------
This module defines dataset paths, model configuration, training settings,
experiment directories, and runtime options used throughout the project.
"""

import torch
from pathlib import Path

# ==========================================================
# RANDOMNESS
# ==========================================================
SEED = 42

# ==========================================================
# DATASET
# ==========================================================
DATA_ROOT = Path("data/PlantVillage")

SPLIT_DIR = DATA_ROOT / "splits"

TRAIN_SPLIT = SPLIT_DIR / "color_train.txt"
TEST_SPLIT = SPLIT_DIR / "color_test.txt"

# ==========================================================
# IMAGE
# ==========================================================
IMAGE_SIZE = 224

# ==========================================================
# MODEL
# ==========================================================
# Feature extraction level used from the backbone network
FEATURE_LEVEL = "c5"

# Use ImageNet pretrained backbone weights
PRETRAINED = True

# Channel reduction ratio used in DCBAM attention module
DCBAM_REDUCTION = 16

# ==========================================================
# TRAINING
# ==========================================================
BATCH_SIZE = 32

EPOCHS = 100

LR = 1e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 10

# ==========================================================
# DATALOADER
# ==========================================================
NUM_WORKERS = 4
PIN_MEMORY = True

# ==========================================================
# DEVICE
# ==========================================================
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ==========================================================
# EXPERIMENT PATHS
# ==========================================================
EXPERIMENT_DIR = Path("experiments")

CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
LOG_DIR = EXPERIMENT_DIR / "logs"
PLOT_DIR = EXPERIMENT_DIR / "plots"
RESULT_DIR = EXPERIMENT_DIR / "results"

# ==========================================================
# CHECKPOINTS
# ==========================================================
BEST_LOSS_PATH = CHECKPOINT_DIR / "best_loss.pth"
BEST_ACC_PATH = CHECKPOINT_DIR / "best_acc.pth"

# ==========================================================
# CREATE DIRECTORIES
# ==========================================================
for directory in (
    CHECKPOINT_DIR,
    LOG_DIR,
    PLOT_DIR,
    RESULT_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )
