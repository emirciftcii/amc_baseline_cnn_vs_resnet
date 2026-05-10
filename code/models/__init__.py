from __future__ import annotations

from .baseline_cnn import BaselineCNN
from .resnet1d import ResNet1D


def build_model(model_name: str, num_classes: int, base_channels: int = 64, dropout: float = 0.2):
    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes, dropout=dropout)
    if model_name == "resnet1d":
        return ResNet1D(num_classes=num_classes, base_channels=base_channels, dropout=dropout)
    raise ValueError(f"Unknown model_name: {model_name}")
