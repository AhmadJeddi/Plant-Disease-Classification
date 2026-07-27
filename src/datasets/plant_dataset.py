"""
Plant Disease Classification Dataset
------------------------------------
PyTorch Dataset for image-based plant disease classification.
The dataset uses official split files provided by PlantVillage:

Example split file:
raw/color/Raspberry___healthy/image_name.JPG

Expected structure:
data/
└── PlantVillage/
    ├── raw/
    │   └── color/
    │       ├── Apple___Apple_scab/
    │       ├── Tomato___Late_blight/
    │       └── ...
    │
    └── splits/
        ├── color_train.txt
        └── color_test.txt

Output:
    image tensor:
        Shape -> (3, H, W)
    label:
        Integer class index
"""


from pathlib import Path
from typing import List, Tuple, Dict

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

from config.config import IMAGE_SIZE

def get_transform(augment: bool = True):
    """
    Returns image preprocessing pipeline.
    Includes resizing, optional data augmentation,
    tensor conversion, and ImageNet normalization.
    """

    transform_list: List = [
        transforms.Resize(
            (IMAGE_SIZE,IMAGE_SIZE)
        )
    ]

    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(
                p=0.5
            ),
            transforms.RandomRotation(
                15
            ),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    return transforms.Compose(transform_list)

class PlantDataset(Dataset):
    """
    Generic image classification dataset.
    Reads image paths from a split file and automatically
    creates class labels from directory names.
    """

    def __init__(
        self,
        split_file: Path,
        data_root: Path,
        transform=None,
        class_to_idx: Dict[str, int] = None,
    ):
        """
        Args:
            split_file:
                Path to train/test split txt file.

            data_root:
                Root directory containing raw images.

            transform:
                torchvision transformations.

            class_to_idx:
                Optional predefined class mapping.
                Useful to keep train/test labels identical.
        """

        self.data_root = Path(data_root)
        self.split_file = Path(split_file)
        self.transform = transform

        # ---------------------------------------
        # Load image paths
        # ---------------------------------------
        self.samples = self._load_split()

        # ---------------------------------------
        # Build class mapping
        # ---------------------------------------
        if class_to_idx is None:

            classes = sorted(
                {
                    label
                    for _, label in self.samples
                }
            )

            self.class_to_idx = {
                cls_name: idx
                for idx, cls_name in enumerate(classes)
            }

        else:
            self.class_to_idx = class_to_idx

        # ---------------------------------------
        # Convert labels to integers
        # ---------------------------------------
        self.samples = [
            (
                image_path,
                self.class_to_idx[label]
            )
            for image_path, label in self.samples
        ]

        self.classes = [
            cls
            for cls, _ in sorted(
                self.class_to_idx.items(),
                key=lambda x: x[1]
            )
        ]

    # ==================================================
    # Read split file
    # ==================================================
    def _load_split(self) -> List[Tuple[Path, str]]:
        """
        Reads image paths from split file.

        Returns:
            List of:
            (
                image_path,
                class_name
            )
        """

        samples = []

        with open(
            self.split_file,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                relative_path = line.strip()

                if not relative_path:
                    continue

                image_path = (
                    self.data_root
                    /
                    relative_path
                )

                class_name = (
                    image_path
                    .parent
                    .name
                )

                samples.append(
                    (
                        image_path,
                        class_name
                    )
                )

        return samples

    # ==================================================
    # Dataset length
    # ==================================================
    def __len__(self):
        return len(self.samples)

    # ==================================================
    #  Load and preprocess image
    # ==================================================
    def __getitem__(self, index):

        image_path, label = self.samples[index]

        try:
            image = Image.open(image_path).convert("RGB")

        except Exception as e:
            raise RuntimeError(
                f"Cannot load image: {image_path}"
            ) from e        

        if self.transform:
            image = self.transform(image)

        return image, label
