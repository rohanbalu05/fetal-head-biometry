"""Bridge from a cranium segmentation to the two biometry diameters.

Given a binary cranium mask, fit an ellipse to its largest contour and expose
the ellipse parameters and the major/minor axis endpoints and lengths. The
ellipse major/minor axes are the geometric stand-ins for the two diameters
derived from a segmentation (Part B), independent of the Part A CSV landmarks.

Axis endpoint geometry is delegated to :func:`common.geometry.ellipse_to_points`.
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from common.geometry import ellipse_to_points


def mask_to_biometry(binary_mask: np.ndarray) -> Optional[Dict[str, object]]:
    """Fit an ellipse to a cranium mask and return its biometry decomposition.

    Parameters
    ----------
    binary_mask:
        Binary mask (any nonzero value = foreground), shape ``(H, W)``.

    Returns
    -------
    dict or None
        ``None`` if no ellipse can be fit (no contour, or fewer than 5 contour
        points). Otherwise a dict with keys:

        - ``ellipse``    : ``((cx, cy), (d1, d2), angle_deg)`` from
          ``cv2.fitEllipse``.
        - ``major_axis`` : ``(p1, p2)`` endpoints of the major axis.
        - ``minor_axis`` : ``(p1, p2)`` endpoints of the minor axis.
        - ``major_len``  : major-axis length (px).
        - ``minor_len``  : minor-axis length (px).
    """
    mask = (np.asarray(binary_mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None

    ellipse = cv2.fitEllipse(contour)
    (_, _), (d1, d2), _ = ellipse
    axes = ellipse_to_points(ellipse)

    return {
        "ellipse": ellipse,
        "major_axis": axes["major"],
        "minor_axis": axes["minor"],
        "major_len": float(max(d1, d2)),
        "minor_len": float(min(d1, d2)),
    }
