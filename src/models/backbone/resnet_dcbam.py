"""
ResNet50 backbone with DCBAM attention integration
--------------------------------------------------
This module implements a ResNet50-style bottleneck backbone where the
DCBAM module can be attached to the last block of chosen stages.

The implementation loads pretrained ResNet50 weights when available and
copies matching parameters.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision.models.resnet import conv1x1

from models.attention.dcbam import DCBAM


class BottleneckDCBAM(nn.Module):
    """
    Bottleneck block compatible with ResNet50 expansion=4.
    DCBAM is optionally applied on the block's output.
    """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        use_dcbam: bool = False,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.use_dcbam = use_dcbam
        if use_dcbam:
            self.dcbam = DCBAM(planes * self.expansion, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.use_dcbam:
            out = self.dcbam(out)

        out = self.relu(out)
        return out


class ResNet50DCBAMBackbone(nn.Module):
    """
    ResNet50 backbone with DCBAM attention integration.

    Args:
        pretrained: whether to attempt loading ImageNet pretrained ResNet50 weights.
        reduction: reduction ratio for DCBAM.
        use_dcbam: global switch; actual insertion controlled per-stage.
        freeze_layers: dict mapping module-name prefixes to booleans indicating
                       whether to freeze those parameters.
    """

    def __init__(
        self,
        pretrained: bool = True,
        reduction: int = 16,
        use_dcbam: bool = True,
        freeze_layers: Optional[Dict[str, bool]] = None,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.use_dcbam = use_dcbam
        self.freeze_layers = freeze_layers or {
            "conv1": True,
            "bn1": True,
            "layer1": True,
            "layer2": True,
            "layer3": False,
            "layer4": False,
        }

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_layer(64, 3, reduction=reduction, use_dcbam=False)
        self.layer2 = self._make_layer(128, 4, stride=2, reduction=reduction, use_dcbam=False)
        self.layer3 = self._make_layer(256, 6, stride=2, reduction=reduction, use_dcbam=True)
        self.layer4 = self._make_layer(512, 3, stride=2, reduction=reduction, use_dcbam=True)

        # Output channels of feature maps
        self.out_channels = {
            "c2": 64 * BottleneckDCBAM.expansion,
            "c3": 128 * BottleneckDCBAM.expansion,
            "c4": 256 * BottleneckDCBAM.expansion,
            "c5": 512 * BottleneckDCBAM.expansion,
        }

        if pretrained:
            self._load_pretrained_and_freeze()

    def _make_layer(self, planes: int, blocks: int, stride: int = 1, reduction: int = 16, use_dcbam: bool = False) -> nn.Sequential:
        downsample: Optional[nn.Module] = None
        if stride != 1 or self.inplanes != planes * BottleneckDCBAM.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BottleneckDCBAM.expansion, stride),
                nn.BatchNorm2d(planes * BottleneckDCBAM.expansion),
            )

        layers = []
        for i in range(blocks):
            layers.append(
                BottleneckDCBAM(
                    self.inplanes,
                    planes,
                    stride=stride if i == 0 else 1,
                    downsample=downsample if i == 0 else None,
                    use_dcbam=(use_dcbam and i == blocks - 1),
                    reduction=reduction,
                )
            )
            self.inplanes = planes * BottleneckDCBAM.expansion
        return nn.Sequential(*layers)

    def _load_pretrained_and_freeze(self) -> None:
        """
        Try to load torchvision's ResNet50 weights and freeze parameters according to `freeze_layers`.

        The method copies matching parameters from torchvision's resnet50.
        If weights aren't available, it falls back to random init and warns.
        """
        
        from torchvision.models import resnet50, ResNet50_Weights

        try:
            base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[WARN] Could not load pretrained weights: {e}")
            base = resnet50(weights=None)

        base_dict = base.state_dict()
        my_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in base_dict.items() if k in my_dict and v.size() == my_dict[k].size()}
        my_dict.update(pretrained_dict)
        self.load_state_dict(my_dict)

        # Freeze parameters by prefix (except DCBAM params)
        for name, param in self.named_parameters():
            freeze = False
            for layer_name, do_freeze in self.freeze_layers.items():
                if name.startswith(layer_name) and do_freeze:
                    freeze = True
            if freeze and "dcbam" not in name:
                param.requires_grad = False

        # Put BN layers of frozen parts into eval mode
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) and hasattr(m, "weight") and not m.weight.requires_grad:
                m.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return intermediate feature maps similar to many detection/segmentation backbones.
        Returns dict with keys: 'c2', 'c3', 'c4', 'c5'.
        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}
