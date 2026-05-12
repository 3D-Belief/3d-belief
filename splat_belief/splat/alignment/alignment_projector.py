import torch
from torch import nn


class ConvProjector(nn.Module):
    """REPA-style projector: maps U-ViT latents to VGGT patch-token features.

    Default output shape (2048, 37, 37) matches VGGT-1B's last-layer patch grid for
    518x518 input with patch_size 14: 37x37 = 1369 patches, each 2048-dim.
    """

    def __init__(
        self,
        in_channels=128,
        in_h=128,
        in_w=128,
        out_h=37,
        out_w=37,
        mid_channels=512,
        out_channels=2048,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((out_h, out_w))
        self.channel_reducer = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.channel_reducer(x)
        return x
