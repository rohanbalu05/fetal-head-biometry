"""Gaussian heatmap encoding and decoding for landmark detection.

Encodes landmark coordinates into 2D Gaussian heatmaps for heatmap-regression
models and decodes predicted heatmaps back into sub-pixel landmark coordinates.

Channel order matches the ground-truth CSV exactly: ``ofd_1, ofd_2, bpd_1,
bpd_2``. No reordering or relabeling of points occurs here.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def coords_to_heatmaps(
    points_xy: np.ndarray,
    out_hw: Tuple[int, int] = (256, 256),
    sigma: float = 4.0,
) -> np.ndarray:
    """Render landmark coordinates as a stack of Gaussian heatmaps.

    Parameters
    ----------
    points_xy:
        Array-like of shape ``(4, 2)`` with ``(x, y)`` pixel coordinates in the
        output resolution, in CSV order ``ofd_1, ofd_2, bpd_1, bpd_2``.
    out_hw:
        Output ``(H, W)`` of each heatmap (network resolution, default 256x256).
    sigma:
        Standard deviation (px) of the Gaussian blob.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(4, H, W)``; each channel's peak is
        normalized to ``1.0``.
    """
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    height, width = int(out_hw[0]), int(out_hw[1])

    ys = np.arange(height, dtype=np.float32).reshape(height, 1)
    xs = np.arange(width, dtype=np.float32).reshape(1, width)

    heatmaps = np.zeros((points.shape[0], height, width), dtype=np.float32)
    two_sigma_sq = 2.0 * float(sigma) ** 2
    for i, (px, py) in enumerate(points):
        gauss = np.exp(-(((xs - px) ** 2) + ((ys - py) ** 2)) / two_sigma_sq)
        peak = float(gauss.max())
        if peak > 0.0:
            gauss /= peak  # normalize peak to 1.0
        heatmaps[i] = gauss
    return heatmaps


def heatmaps_to_coords(heatmaps: np.ndarray, window: int = 7) -> np.ndarray:
    """Decode heatmaps to sub-pixel ``(x, y)`` coordinates.

    For each channel the integer argmax is located, then refined to sub-pixel
    accuracy via an intensity-weighted centroid computed in a small window
    centered on the argmax.

    Parameters
    ----------
    heatmaps:
        Array of shape ``(C, H, W)``.
    window:
        Side length (px) of the square refinement window (odd; default 7).

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(C, 2)`` of ``(x, y)`` coordinates.
    """
    hm = np.asarray(heatmaps, dtype=np.float32)
    if hm.ndim != 3:
        raise ValueError(f"expected (C, H, W), got shape {hm.shape}")
    channels, height, width = hm.shape
    radius = max(int(window) // 2, 0)

    coords = np.zeros((channels, 2), dtype=np.float32)
    for c in range(channels):
        plane = hm[c]
        y0, x0 = np.unravel_index(int(np.argmax(plane)), plane.shape)

        x_lo, x_hi = max(0, x0 - radius), min(width, x0 + radius + 1)
        y_lo, y_hi = max(0, y0 - radius), min(height, y0 + radius + 1)
        patch = plane[y_lo:y_hi, x_lo:x_hi]
        grid_y, grid_x = np.mgrid[y_lo:y_hi, x_lo:x_hi]

        total = float(patch.sum())
        if total > 0.0:
            cx = float((patch * grid_x).sum() / total)
            cy = float((patch * grid_y).sum() / total)
        else:
            cx, cy = float(x0), float(y0)
        coords[c] = (cx, cy)
    return coords
