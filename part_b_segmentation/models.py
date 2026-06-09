"""Model architectures for Part B cranium segmentation.

Defines a baseline U-Net and an attention-gated variant, both mapping a
``(B, 1, 256, 256)`` ultrasound batch to a single-channel per-pixel cranium
probability map ``(B, 1, 256, 256)`` (final sigmoid):

- :class:`SegUNet` -- plain U-Net (Part B hypothesis H2).
- :class:`SegUNetAttn` -- U-Net with attention gates on the skip connections
  (Part B hypothesis H3 variant).

A :func:`get_seg_model` factory returns either by key.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.blocks import ConvBNReLU, DoubleConv, Down, Up


class SegUNet(nn.Module):
    """H2: U-Net producing a single sigmoid cranium-probability channel."""

    def __init__(self, in_channels: int = 1, base: int = 64, bilinear: bool = True) -> None:
        super().__init__()
        b = base
        self.inc = DoubleConv(in_channels, b)         # 256, b
        self.down1 = Down(b, b * 2)                    # 128, 2b
        self.down2 = Down(b * 2, b * 4)               # 64,  4b
        self.down3 = Down(b * 4, b * 8)               # 32,  8b
        self.down4 = Down(b * 8, b * 8)               # 16,  8b (bottleneck)

        self.up1 = Up(b * 8, b * 8, b * 4, bilinear)  # 32,  4b
        self.up2 = Up(b * 4, b * 4, b * 2, bilinear)  # 64,  2b
        self.up3 = Up(b * 2, b * 2, b, bilinear)      # 128, b
        self.up4 = Up(b, b, b, bilinear)              # 256, b
        self.outc = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 1, 256, 256)`` per-pixel cranium probability."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return torch.sigmoid(self.outc(x))


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018).

    Gates a skip-connection feature ``x`` using a coarser gating signal ``g``
    (the upsampled decoder feature at the same resolution). Returns ``x``
    re-weighted by a learned spatial attention coefficient in ``[0, 1]``.

    Parameters
    ----------
    gate_channels:
        Channels of the gating signal ``g``.
    skip_channels:
        Channels of the skip feature ``x``.
    inter_channels:
        Channels of the shared intermediate representation.
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return the gated skip feature ``x * attention(g, x)``."""
        attention = self.psi(self.relu(self.w_g(g) + self.w_x(x)))
        return x * attention


class AttnUp(nn.Module):
    """Upsampling stage with an attention gate applied to the skip connection."""

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
            gate_channels = in_channels
            conv_in = in_channels + skip_channels
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            gate_channels = in_channels // 2
            conv_in = in_channels // 2 + skip_channels
        self.att = AttentionGate(
            gate_channels=gate_channels,
            skip_channels=skip_channels,
            inter_channels=max(skip_channels // 2, 1),
        )
        self.conv = DoubleConv(conv_in, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        skip = self.att(g=x, x=skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SegUNetAttn(nn.Module):
    """H3 variant: U-Net with attention gates on every skip connection."""

    def __init__(self, in_channels: int = 1, base: int = 64, bilinear: bool = True) -> None:
        super().__init__()
        b = base
        self.inc = DoubleConv(in_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        self.down4 = Down(b * 8, b * 8)

        self.up1 = AttnUp(b * 8, b * 8, b * 4, bilinear)
        self.up2 = AttnUp(b * 4, b * 4, b * 2, bilinear)
        self.up3 = AttnUp(b * 2, b * 2, b, bilinear)
        self.up4 = AttnUp(b, b, b, bilinear)
        self.outc = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 1, 256, 256)`` per-pixel cranium probability."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return torch.sigmoid(self.outc(x))


def get_seg_model(name: str, **kwargs: object) -> nn.Module:
    """Factory returning a Part B segmentation model by key.

    Parameters
    ----------
    name:
        ``"unet"`` (H2) or ``"unet_attn"`` (H3 variant).
    **kwargs:
        Forwarded to the selected model's constructor.
    """
    key = name.lower()
    builders = {
        "unet": SegUNet,
        "unet_attn": SegUNetAttn,
    }
    if key not in builders:
        raise ValueError(f"unknown seg model {name!r}; choose from {sorted(builders)}")
    return builders[key](**kwargs)  # type: ignore[arg-type]
