import torch
import torch.nn as nn


class DeltaClassifier(nn.Module):
    """
    Small CNN classifier for delta maps.
    Input: delta tensor in [0, 1], shape [B, 3, H, W]
    Output: logits [B, 1]
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        z = self.backbone(x).flatten(1)
        return self.head(z)


class CLIPModelRectifyDiscrepancyAttention(nn.Module):
    """
    Compatibility name used in D3-style code.
    In this repo, we provide a lightweight delta classifier implementation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        _ = (args, kwargs)
        self.model = DeltaClassifier(in_channels=3)

    def forward(self, x, return_feature=False):
        logits = self.model(x)
        if return_feature:
            return logits, x
        return logits

