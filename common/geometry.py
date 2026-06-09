"""Geometry helpers converting between ellipse parameters and clinical measurements.

Converts a fitted head ellipse to/from biometry endpoints and measurements:
biparietal diameter (BPD) and occipitofrontal diameter (OFD) endpoints, head
circumference (HC), and the cephalic index.

IMPORTANT (per task rules): any "which axis is major/minor" reasoning in this
module is for INTERNAL metric/measurement purposes only. It must never modify,
reorder, or feed back into the training labels, which are used exactly as given
in the ground-truth CSV (order: ofd_1, ofd_2, bpd_1, bpd_2).

Unless noted otherwise, all lengths are in PIXELS.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

# HC approximation coefficient: HC ~= 1.62 * (BPD + OFD).
HC_COEFF: float = 1.62


def _euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(math.hypot(float(p1[0] - p2[0]), float(p1[1] - p2[1])))


def points_to_axes(points_xy: np.ndarray) -> Tuple[float, float]:
    """Return ``(ofd_len, bpd_len)`` in pixels from the 4 landmark points.

    Parameters
    ----------
    points_xy:
        Array-like of shape ``(4, 2)`` in CSV order
        ``ofd_1, ofd_2, bpd_1, bpd_2``.

    Returns
    -------
    tuple of float
        ``(ofd_len, bpd_len)`` Euclidean distances per pair, in pixels.
    """
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError(f"expected 4 points, got {pts.shape[0]}")
    ofd_len = _euclidean(pts[0], pts[1])
    bpd_len = _euclidean(pts[2], pts[3])
    return ofd_len, bpd_len


def cephalic_index(bpd_len: float, ofd_len: float) -> float:
    """Cephalic index = ``100 * BPD / OFD`` (percent; dimensionless ratio).

    Computed directly from the provided lengths without any axis swapping, so
    the value reflects the CSV labels as-is.
    """
    if ofd_len == 0:
        return float("nan")
    return 100.0 * float(bpd_len) / float(ofd_len)


def hc_estimate(bpd_len: float, ofd_len: float) -> float:
    """Head-circumference estimate in PIXELS: ``1.62 * (BPD + OFD)``."""
    return HC_COEFF * (float(bpd_len) + float(ofd_len))


def ellipse_to_points(
    ellipse: Tuple[Tuple[float, float], Tuple[float, float], float],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Convert a ``cv2.fitEllipse`` result to major/minor axis endpoints.

    Parameters
    ----------
    ellipse:
        The ``((cx, cy), (d1, d2), angle_deg)`` tuple returned by
        ``cv2.fitEllipse`` (``d1, d2`` are full axis lengths; ``angle_deg`` is
        the rotation of the first axis from the x-axis).

    Returns
    -------
    dict
        ``{"major": (p1, p2), "minor": (p1, p2)}`` where each ``p`` is an
        ``(x, y)`` ``float64`` numpy array. Major/minor is decided purely by
        axis length (internal metric use only).
    """
    (cx, cy), (d1, d2), angle_deg = ellipse
    theta = math.radians(float(angle_deg))

    # First axis (d1) is along (cos, sin); second axis (d2) is perpendicular.
    ux, uy = math.cos(theta), math.sin(theta)
    vx, vy = -math.sin(theta), math.cos(theta)
    center = np.array([float(cx), float(cy)], dtype=np.float64)

    half1, half2 = float(d1) / 2.0, float(d2) / 2.0
    axis1 = (center - half1 * np.array([ux, uy]), center + half1 * np.array([ux, uy]))
    axis2 = (center - half2 * np.array([vx, vy]), center + half2 * np.array([vx, vy]))

    if d1 >= d2:
        return {"major": axis1, "minor": axis2}
    return {"major": axis2, "minor": axis1}
