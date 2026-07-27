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