"""Small shared neural-network building blocks.

Reusable modules for the Part A landmark models (and later Part B segmentation):
a conv-bn-relu unit, a double-convolution block, a downsampling stage
(maxpool + double conv), and an upsampling stage (upsample + skip concat +
double conv).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution -> BatchNorm -> ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    """Two stacked :class:`ConvBNReLU` layers (the classic U-Net block)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        mid = mid_channels if mid_channels is not None else out_channels
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, mid),
            ConvBNReLU(mid, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downsampling stage: 2x2 max-pool followed by a :class:`DoubleConv`."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsampling stage: upsample the input, concat the skip, then double-conv.

    Parameters
    ----------
    in_channels:
        Channels of the (lower-resolution) feature map being upsampled.
    skip_channels:
        Channels of the encoder skip connection concatenated after upsampling.
    out_channels:
        Output channels of the fused :class:`DoubleConv`.
    bilinear:
        If ``True`` use bilinear upsampling (parameter-free); otherwise use a
        learned ``ConvTranspose2d``.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        if bilinear:
            self.up: nn.Module = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            )
            conv_in = in_channels + skip_channels
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            conv_in = in_channels // 2 + skip_channels
        self.conv = DoubleConv(conv_in, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Align spatial dims to the skip tensor (handles odd sizes).
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
