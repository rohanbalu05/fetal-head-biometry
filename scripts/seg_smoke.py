"""Smoke test for Part B: segmentation models + classical baseline + decode.

1. Instantiates :class:`SegUNet` and :class:`SegUNetAttn`, runs a dummy
   ``(2, 1, 256, 256)`` batch, asserts ``(2, 1, 256, 256)`` outputs, and prints
   parameter counts.
2. Loads one real validation image, runs :func:`segment_classical`, decodes the
   resulting mask with :func:`mask_to_biometry`, prints the two axis lengths, and
   saves ``scripts/seg_classical_demo.png`` (image | classical mask |
   fitted ellipse + axes).

Run directly::

    python scripts/seg_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import data_utils, splits  # noqa: E402
from common.dataset import INPUT_HW, SegmentationDataset  # noqa: E402
from part_b_segmentation.classical import segment_classical  # noqa: E402
from part_b_segmentation.geometry_decode import mask_to_biometry  # noqa: E402
from part_b_segmentation.models import get_seg_model  # noqa: E402

DEMO_PNG: Path = Path(__file__).resolve().parent / "seg_classical_demo.png"


def count_params(model: torch.nn.Module) -> int:
    """Return the total number of parameters in ``model``."""
    return sum(p.numel() for p in model.parameters())


def check_models() -> None:
    """Instantiate both seg models, run a dummy batch, assert shapes, print sizes."""
    dummy = torch.randn(2, 1, 256, 256)
    print("Part B model check (input: (2, 1, 256, 256))")
    print("-" * 60)
    for tag, key in (("H2", "unet"), ("H3", "unet_attn")):
        model = get_seg_model(key).eval()
        with torch.no_grad():
            out = model(dummy)
        assert tuple(out.shape) == (2, 1, 256, 256), (
            f"{key} output {tuple(out.shape)} != (2, 1, 256, 256)"
        )
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0, "not in [0,1]"
        print(f"{tag}  {key:<10}{model.__class__.__name__:<14}"
              f"out={tuple(out.shape)}  params={count_params(model):,}")
    print("-" * 60)


def check_dataset() -> None:
    """Sanity-check one SegmentationDataset item (shapes + binary mask)."""
    ds = SegmentationDataset(split="val", augment=False)
    image, mask, orig_wh, name = ds[0]
    uniques = torch.unique(mask).tolist()
    assert tuple(image.shape) == (1, 256, 256)
    assert tuple(mask.shape) == (1, 256, 256)
    assert set(uniques).issubset({0.0, 1.0}), f"mask not binary: {uniques}"
    print(f"SegmentationDataset[0]: {name} image={tuple(image.shape)} "
          f"mask={tuple(mask.shape)} mask_vals={uniques} orig_wh={orig_wh.tolist()}")


def run_classical_demo() -> None:
    """Run the classical baseline on one val image and save the demo figure."""
    name = splits.image_names_for("val")[0]
    img = cv2.imread(str(data_utils.IMAGES_DIR / name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {name}")

    out_h, out_w = INPUT_HW
    resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    mask = segment_classical(resized)
    biometry: Optional[dict] = mask_to_biometry(mask)

    overlay = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    if biometry is not None:
        cv2.ellipse(overlay, biometry["ellipse"], (0, 255, 0), 2)
        maj = biometry["major_axis"]
        minr = biometry["minor_axis"]
        cv2.line(overlay,
                 tuple(np.round(maj[0]).astype(int)),
                 tuple(np.round(maj[1]).astype(int)), (0, 255, 255), 2)  # major: cyan
        cv2.line(overlay,
                 tuple(np.round(minr[0]).astype(int)),
                 tuple(np.round(minr[1]).astype(int)), (255, 255, 0), 2)  # minor: yellow
        major_len = biometry["major_len"]
        minor_len = biometry["minor_len"]
    else:
        major_len = minor_len = float("nan")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    axes[0].imshow(resized, cmap="gray")
    axes[0].set_title(f"image\n{name}", fontsize=9)
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("classical mask\n(Otsu+morph+largest CC)", fontsize=9)
    axes[2].imshow(overlay)
    axes[2].set_title(f"fitted ellipse + axes\nmajor={major_len:.1f}px  minor={minor_len:.1f}px",
                      fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Part B classical baseline (cyan=major axis, yellow=minor axis)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(DEMO_PNG, dpi=130)
    plt.close(fig)

    print(f"classical baseline on {name}: "
          f"major_len={major_len:.2f}px  minor_len={minor_len:.2f}px")
    print(f"demo saved -> {DEMO_PNG}")


def main() -> None:
    """Run the Part B smoke test."""
    check_models()
    check_dataset()
    run_classical_demo()


if __name__ == "__main__":
    main()
