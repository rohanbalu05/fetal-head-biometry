"""Classical (non-learned) cranium segmentation baseline -- Part B hypothesis H1.

A pure computer-vision pipeline (no deep learning) that shows what is achievable
before any training: Otsu thresholding, morphological cleanup, and
largest-connected-component selection.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def segment_classical(
    image_gray_uint8: np.ndarray,
    blur_ksize: int = 5,
    morph_ksize: int = 7,
) -> np.ndarray:
    """Segment the cranium from a grayscale ultrasound with classical CV.

    Pipeline (each step documented inline):

    1. **Gaussian blur** -- suppress ultrasound speckle so Otsu picks a stable
       global threshold.
    2. **Otsu threshold** -- automatically split bright (bone/tissue) foreground
       from the darker background.
    3. **Morphological closing** -- bridge gaps along the bright skull ring so it
       forms a connected blob.
    4. **Morphological opening** -- remove small isolated speckles left over.
    5. **Keep largest connected component** -- discard spurious blobs and retain
       the single largest region (the cranium).

    Parameters
    ----------
    image_gray_uint8:
        Grayscale image, ``uint8`` in ``[0, 255]`` (any size).
    blur_ksize:
        Odd Gaussian kernel size for denoising.
    morph_ksize:
        Diameter (px) of the elliptical structuring element for morphology.

    Returns
    -------
    numpy.ndarray
        Binary ``uint8`` mask (same H x W as the input) with the cranium region
        set to 255 and background to 0.
    """
    if image_gray_uint8.ndim != 2:
        raise ValueError("expected a single-channel grayscale image")
    img = image_gray_uint8.astype(np.uint8)

    # 1. Denoise to stabilize the Otsu threshold against speckle.
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(img, (k, k), 0)

    # 2. Otsu threshold -> bright foreground.
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3-4. Morphological closing then opening with an elliptical kernel.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5. Keep only the largest connected component.
    return _keep_largest_component(opened)


def _keep_largest_component(mask_uint8: np.ndarray) -> np.ndarray:
    """Return a mask containing only the largest non-background blob."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_uint8 > 0).astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return np.zeros_like(mask_uint8)

    # Label 0 is background; pick the foreground label with the largest area.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask_uint8)
    out[labels == largest] = 255
    return out
