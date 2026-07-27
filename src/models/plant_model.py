"""
Plant Classification Model
--------------------------
Combines:
- ResNet50 + DCBAM backbone
- Feature selection
- Classification head

Designed for flexible experiments with
multi-scale feature maps.
"""

import torch.nn as nn

from models.backbone.resnet_dcbam import ResNet50DCBAMBackbone
from models.classifier import ImageClassifier

class PlantModel(nn.Module):

    """
    End-to-end image classification model for plant disease recognition.
    The model extracts feature maps using the backbone, selects one
    feature level, and predicts class logits using the classifier head.
    """
    
    def __init__(
        self,
        num_classes: int,
        feature_level: str = "c5",
        pretrained: bool = True,
        reduction: int = 16
    ):
        super().__init__()

        self.feature_level = feature_level

        # Backbone
        self.backbone = ResNet50DCBAMBackbone(
            pretrained=pretrained,
            reduction=reduction
        )

        # Automatically get channels
        if feature_level not in self.backbone.out_channels:
            raise ValueError(
                f"Unknown feature level: {feature_level}"
            )

        in_channels = self.backbone.out_channels[
            feature_level
        ]

        # Classification head
        self.classifier = ImageClassifier(
            in_channels=in_channels,
            num_classes=num_classes
        )

    def forward(self, x):

        features = self.backbone(x)

        # Select feature map
        x = features[
            self.feature_level
        ]

        out = self.classifier(x)

        return out
