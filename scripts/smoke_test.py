"""Smoke test for the landmark data pipeline (dataset + heatmap encode/decode).

Loads one training batch, verifies that points recovered from the target
heatmaps match the ground-truth points to within a couple of pixels, and saves a
visual overlay (``scripts/smoke_overlay.png``) for three samples showing the
resized image with GT + recovered points and the four heatmap channels.

Run directly::

    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common.dataset import INPUT_HW, LandmarkDataset  # noqa: E402
from common.heatmaps import heatmaps_to_coords  # noqa: E402
from common.metrics import per_point_error  # noqa: E402

OVERLAY_PNG: Path = Path(__file__).resolve().parent / "smoke_overlay.png"
CHANNEL_LABELS = ("ofd_1", "ofd_2", "bpd_1", "bpd_2")


def _display_image(image_chw: torch.Tensor) -> np.ndarray:
    """Min-max normalize a standardized ``(1, H, W)`` tensor for display."""
    arr = image_chw.squeeze(0).numpy()
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def save_overlay(
    images: torch.Tensor,
    heatmaps: torch.Tensor,
    points: torch.Tensor,
    recovered: List[np.ndarray],
    names: List[str],
    n: int = 3,
) -> None:
    """Save a grid: per sample, [image+points] then the 4 heatmap channels."""
    n = min(n, images.shape[0])
    n_cols = 1 + len(CHANNEL_LABELS)
    fig, axes = plt.subplots(n, n_cols, figsize=(n_cols * 2.8, n * 2.8))
    axes = np.atleast_2d(axes)

    for r in range(n):
        gt = points[r].numpy()
        rec = recovered[r]

        ax = axes[r, 0]
        ax.imshow(_display_image(images[r]), cmap="gray")
        ax.scatter(gt[:, 0], gt[:, 1], facecolors="none", edgecolors="lime",
                   s=80, linewidths=1.5, label="GT")
        ax.scatter(rec[:, 0], rec[:, 1], c="red", marker="x", s=60,
                   linewidths=1.5, label="recovered")
        ax.set_xlim(0, INPUT_HW[1])
        ax.set_ylim(INPUT_HW[0], 0)
        err = float(per_point_error(rec, gt).max())
        ax.set_title(f"{names[r]}\nmax err={err:.2f}px", fontsize=7)
        ax.axis("off")
        if r == 0:
            ax.legend(loc="lower right", fontsize=6, framealpha=0.6)

        for c, label in enumerate(CHANNEL_LABELS):
            axc = axes[r, c + 1]
            axc.imshow(heatmaps[r, c].numpy(), cmap="hot")
            axc.scatter([gt[c, 0]], [gt[c, 1]], facecolors="none",
                        edgecolors="lime", s=60, linewidths=1.2)
            axc.set_title(label, fontsize=7)
            axc.axis("off")

    fig.suptitle("Landmark pipeline smoke test (GT=green, recovered=red x)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OVERLAY_PNG, dpi=130)
    plt.close(fig)
    print(f"overlay saved -> {OVERLAY_PNG}")


def main() -> None:
    """Run the smoke test and report the max recovery error across the batch."""
    dataset = LandmarkDataset(split="train", augment=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    images, heatmaps, points, _orig_wh, names = next(iter(loader))
    batch_size = images.shape[0]

    recovered: List[np.ndarray] = []
    max_err = 0.0
    for i in range(batch_size):
        rec = heatmaps_to_coords(heatmaps[i].numpy())
        recovered.append(rec)
        errs = per_point_error(rec, points[i].numpy())
        max_err = max(max_err, float(errs.max()))

    save_overlay(images, heatmaps, points, recovered, list(names), n=3)

    status = "OK (<=2px)" if max_err <= 2.0 else "WARNING (>2px)"
    print(f"Max recovery error across batch ({batch_size} samples): "
          f"{max_err:.4f} px  [{status}]")


if __name__ == "__main__":
    main()
