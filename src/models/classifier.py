"""
Classification head for image feature maps
------------------------------------------
Applies global average pooling followed by
a linear classifier.
"""

import torch
import torch.nn as nn

class ImageClassifier(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int
    ):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(
            in_channels,
            num_classes
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        x = self.pool(x)

        x = torch.flatten(
            x,
            1
        )

        return self.classifier(x)
