# Plant-Disease-Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-PlantVillage-green.svg)](https://huggingface.co/datasets/mohanty/PlantVillage)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A ResNet50-DCBAM Framework for Plant Disease Classification on the PlantVillage Dataset**

------------------------------------------------------------------------

## Project Overview

This repository provides a reproducible implementation of a deep
learning framework for plant disease classification.

The project investigates an architecture based on a pretrained
**ResNet50 backbone enhanced with DCBAM (Dual Channel & Spatial
Attention Module)** for image-based plant disease recognition.

The main objective is to improve discriminative feature learning by
combining:

-   ImageNet pretrained feature extraction
-   ResNet50 hierarchical feature representation
-   Channel and spatial attention refinement using DCBAM
-   Flexible multi-level feature selection
-   Efficient classification using global average pooling

The framework is designed for controlled experiments and reproducible
research.

<p align="center">
<img src="docs/figures/architecture_overview.png" width="100%">
</p>

<p align="center">
<b>Overview of the proposed ResNet50-DCBAM framework for plant disease classification.</b>
</p>

------------------------------------------------------------------------

## Key Features

-   ResNet50 backbone with pretrained ImageNet weights
-   DCBAM attention integration
    -   Channel attention
    -   Spatial attention
-   Flexible feature-level selection (`c2`, `c3`, `c4`, `c5`)
-   PlantVillage dataset support
-   Official train/test split support
-   Data augmentation during training
-   Automatic checkpoint management
-   Validation-based model selection:
    -   Best Validation Loss
    -   Best Validation Accuracy
-   Classification report generation
-   Normalized confusion matrix visualization
-   Experiment directory organization

------------------------------------------------------------------------

## Project Structure

``` text
Plant-Disease-Classification/
│
├── data/
│   └── PlantVillage/
│       ├── raw/
│       └── splits/
│
├── docs/
│   ├── figures/
│   │   └── architecture_overview.png
│   │
│   └── project_structure.md
│
├── experiments/
│   ├── checkpoints/
│   ├── logs/
│   ├── plots/
│   └── results/
│
├── src/
│   ├── config/
│   │   └── config.py
│   │
│   ├── datasets/
│   │   └── plant_dataset.py
│   │
│   ├── models/
│   │   ├── attention/
│   │   │   └── dcbam.py
│   │   │
│   │   ├── backbone/
│   │   │   └── resnet_dcbam.py
│   │   │
│   │   ├── classifier.py
│   │   └── plant_model.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   └── test.py
│   │
│   └── utils/
│       ├── plot_logs.py
│       └── export_requirements.py
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

------------------------------------------------------------------------

## Dataset

The experiments use the public **PlantVillage** dataset.

Dataset access: [PlantVillage Dataset](https://huggingface.co/datasets/mohanty/PlantVillage)

The dataset contains plant leaf images from multiple crop species and
disease categories.

The project uses official split files:

    data/
    └── PlantVillage/
        ├── raw/
        │   └── color/
        │
        └── splits/
            ├── color_train.txt
            └── color_test.txt

Each split file contains relative image paths:

    raw/color/Tomato___Late_blight/image_name.JPG

------------------------------------------------------------------------

## Installation

Clone the repository:

``` bash
git clone https://github.com/AhmadJeddi/Plant-Disease-Classification.git
cd Plant-Disease-Classification
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Recommended environment:

-   Python \>= 3.10
-   PyTorch \>= 2.5

------------------------------------------------------------------------

## Configuration

All experiments are controlled through:

    src/config/config.py

Main configurable parameters:

-   Dataset paths
-   Image size
-   Batch size
-   Learning rate
-   Training epochs
-   Weight decay
-   Early stopping patience
-   Feature extraction level
-   Pretrained backbone option
-   Experiment directories

Example:

``` python
IMAGE_SIZE = 224

FEATURE_LEVEL = "c5"

PRETRAINED = True

BATCH_SIZE = 32
```

------------------------------------------------------------------------

## Model Architecture

The proposed model consists of three main components:

    Input Image
         |
         v
    ResNet50 Backbone
         |
         +--> DCBAM Attention Refinement
         |
         v
    Feature Selection (c2/c3/c4/c5)
         |
         v
    Global Average Pooling
         |
         v
    Linear Classifier
         |
         v
    Disease Class Prediction

### ResNet50 Backbone

The backbone extracts hierarchical visual representations:

-   c2
-   c3
-   c4
-   c5

The selected feature map is passed to the classifier.

### DCBAM Attention

DCBAM improves feature representation through:

-   Channel attention:
    -   Adaptive average pooling
    -   Adaptive max pooling
    -   Shared MLP transformation
-   Spatial attention:
    -   Channel-wise average pooling
    -   Channel-wise max pooling
    -   Convolution-based spatial refinement

------------------------------------------------------------------------

## Training

Run training:

``` bash
python src/training/train.py
```

Training includes:

-   Cross Entropy Loss
-   Adam optimizer
-   Data augmentation
-   Validation monitoring
-   Early stopping
-   Automatic checkpoint saving
-   Training curve visualization

Generated checkpoints:

    experiments/
    └── checkpoints/
        ├── best_loss.pth
        └── best_acc.pth

------------------------------------------------------------------------

## Evaluation

Run evaluation:

``` bash
python src/training/test.py
```

Evaluation provides:

-   Classification report
-   Precision
-   Recall
-   F1-score
-   Normalized confusion matrix
-   Saved experiment results

Results are stored in:

    experiments/
    └── results/

Visualizations:

    experiments/
    └── plots/

------------------------------------------------------------------------

## Experiments

Two checkpoints are evaluated:

### Best Validation Loss

Selected based on minimum validation loss.

### Best Validation Accuracy

Selected based on maximum validation accuracy.

Both checkpoints are evaluated using the identical test protocol.

------------------------------------------------------------------------

## Results

The final model was evaluated on the PlantVillage test split containing
**10,709 images across 38 plant disease classes**.

| Metric | Score |
|--------|-------|
| Accuracy | **99.79%** |
| Macro F1-score | **99.69%** |
| Weighted F1-score | **99.79%** |

Both best-loss and best-accuracy checkpoints achieved identical test performance.

------------------------------------------------------------------------

## Reproducibility

The project provides:

-   Fixed random seed
-   Centralized configuration
-   Deterministic experiment structure
-   Saved checkpoints
-   Automated result logging

------------------------------------------------------------------------

## Acknowledgements

This project builds upon the ResNet50-DCBAM implementation from:

**[ResNet-DCBAM](https://github.com/AhmadJeddi/Resnet-DCBAM)**

The original implementation was integrated into this framework with improved
documentation, project-specific organization, and adaptation for the Plant
Disease Classification task.

------------------------------------------------------------------------

## License

This project is licensed under the **MIT License**.

© Ahmad J. Zahed — Please keep the copyright notice and attribution when using or modifying this code.
For full details, see the [LICENSE](./LICENSE) file.
