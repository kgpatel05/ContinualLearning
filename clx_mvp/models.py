# clx_mvp/models.py
from __future__ import annotations
from typing import Optional
import torch.nn as nn
from torchvision import models

def build_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    Create a torchvision ResNet18 with final layer replaced by a Linear head.

    Inputs:
        num_classes: size of the classifier output
        pretrained: if True, load ImageNet weights

    Returns:
        nn.Module: model with attribute .fc = nn.Linear(in_features, num_classes)

    Notes:
        The MVP uses CrossEntropyLoss, so your labels should be Long dtype.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m
