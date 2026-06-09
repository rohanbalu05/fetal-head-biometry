"""Model architectures for Part A landmark detection.

Defines three hypotheses, each an ``nn.Module`` mapping a ``(B, 1, 256, 256)``
ultrasound batch to predictions for the four landmarks, in CSV order
``ofd_1, ofd_2, bpd_1, bpd_2`` (never reordered):

- H1 :class:`CoordRegressionNet` -- direct coordinate regression baseline.
- H2 :class:`HeatmapUNet` -- U-Net heatmap regression (primary method).
- H3 :class:`SpatialConfigurationNet` -- SpatialConfiguration-Net
  (Payer et al., 2016).

A :func:`get_model` factory returns the requested model by key.
"""

from __future__ import annotations

from typing import Dict, Union

import torch
import torch.nn as nn

from common.blocks import ConvBNReLU, DoubleConv, Down, Up

NUM_LANDMARKS: int = 4
INPUT_SIZE: int = 256


class CoordRegressionNet(nn.Module):
    """H1: strided-conv backbone -> global average pool -> FC -> 8 coords.

    Outputs 8 values (4 points x 2 coords) squashed to ``[0, 1]`` by a sigmoid,
    interpreted as normalized ``(x, y)`` on the 256x256 grid. This is the simple
    direct-regression baseline.
    """

    def __init__(self, in_channels: int = 1, width: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = width, width * 2, width * 4, width * 8, width * 16
        # 5 strided blocks: 256 -> 128 -> 64 -> 32 -> 16 -> 8.
        self.backbone = nn.Sequential(
            ConvBNReLU(in_channels, c1, stride=2),
            ConvBNReLU(c1, c2, stride=2),
            ConvBNReLU(c2, c3, stride=2),
            ConvBNReLU(c3, c4, stride=2),
            ConvBNReLU(c4, c5, stride=2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c5, c4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(c4, NUM_LANDMARKS * 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 8)`` normalized coordinates in ``[0, 1]``."""
        feats = self.backbone(x)
        pooled = self.gap(feats)
        return self.head(pooled)


class HeatmapUNet(nn.Module):
    """H2: U-Net producing 4 raw heatmap channels at 256x256.

    Encoder/decoder with skip connections; the final 1x1 conv emits one channel
    per landmark (CSV order) with NO activation (raw heatmap regression). This is
    the primary method, robust to the label-noise tail.
    """

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
        self.outc = nn.Conv2d(b, NUM_LANDMARKS, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 4, 256, 256)`` raw heatmaps."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class SpatialConfigurationNet(nn.Module):
    """H3: SpatialConfiguration-Net (Payer et al., 2016).

    Two interacting components:

    - **Appearance block**: a few conv layers producing 4 local-appearance
      heatmaps ``H_app`` (256x256).
    - **Spatial-configuration block**: ``H_app`` is downsampled (factor
      ``downsample``), passed through large-kernel convolutions so that each
      landmark's response can be informed by the others' positions, then
      upsampled back to 256x256 to yield ``H_spatial``.

    The outputs are combined by element-wise multiplication (Eq. 3 of the
    paper): ``H_final = H_app * H_spatial``.

    ``forward`` returns ``H_final``; pass ``return_components=True`` to also
    obtain ``H_app`` and ``H_spatial`` for visualization.
    """

    def __init__(
        self,
        in_channels: int = 1,
        appearance_width: int = 64,
        spatial_width: int = 64,
        downsample: int = 8,
        spatial_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.downsample = downsample

        # --- Appearance block: a few full-resolution conv layers -> 4 maps ---
        aw = appearance_width
        self.appearance = nn.Sequential(
            ConvBNReLU(in_channels, aw),
            ConvBNReLU(aw, aw),
            ConvBNReLU(aw, aw),
            nn.Conv2d(aw, NUM_LANDMARKS, kernel_size=1),
        )

        # --- Spatial-configuration block: large-kernel convs at low res -------
        sw = spatial_width
        pad = spatial_kernel // 2
        self.spatial = nn.Sequential(
            ConvBNReLU(NUM_LANDMARKS, sw, kernel_size=spatial_kernel, padding=pad),
            ConvBNReLU(sw, sw, kernel_size=spatial_kernel, padding=pad),
            nn.Conv2d(sw, NUM_LANDMARKS, kernel_size=spatial_kernel, padding=pad),
        )
        self.pool = nn.AvgPool2d(kernel_size=downsample, stride=downsample)
        self.upsample = nn.Upsample(
            scale_factor=downsample, mode="bilinear", align_corners=False
        )

    def forward(
        self, x: torch.Tensor, return_components: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return ``H_final`` ``(B, 4, 256, 256)``; optionally the components."""
        h_app = self.appearance(x)

        h_spatial = self.spatial(self.pool(h_app))
        h_spatial = self.upsample(h_spatial)
        # Guard against rounding so spatial matches appearance exactly.
        if h_spatial.shape[-2:] != h_app.shape[-2:]:
            h_spatial = nn.functional.interpolate(
                h_spatial, size=h_app.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        h_final = h_app * h_spatial
        if return_components:
            return {"final": h_final, "appearance": h_app, "spatial": h_spatial}
        return h_final


def get_model(name: str, **kwargs: object) -> nn.Module:
    """Factory returning a Part A model by key.

    Parameters
    ----------
    name:
        One of ``"coord"`` (H1), ``"unet"`` (H2), ``"scn"`` (H3).
    **kwargs:
        Forwarded to the selected model's constructor.

    Returns
    -------
    torch.nn.Module
    """
    key = name.lower()
    builders = {
        "coord": CoordRegressionNet,
        "unet": HeatmapUNet,
        "scn": SpatialConfigurationNet,
    }
    if key not in builders:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(builders)}")
    return builders[key](**kwargs)  # type: ignore[arg-type]
