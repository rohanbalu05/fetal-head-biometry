"""Visualization utilities for overlay rendering.

Small, reusable helpers that draw predictions over a grayscale ultrasound image
and return a BGR image (OpenCV convention) ready to write with ``cv2.imwrite``.
The drawing style matches the per-image overlays the Part A / Part B testers
already produce:

- :func:`draw_landmarks` -- four colour-coded landmark points (CSV order), with
  optional OFD/BPD chords (the Part A landmark overlay).
- :func:`draw_mask_biometry` -- predicted mask outline + fitted ellipse + axis
  endpoints (the Part B segmentation overlay).

Point order is the canonical CSV order ``ofd_1, ofd_2, bpd_1, bpd_2`` and is
never reordered. All coordinates are in the image's own pixel space.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

# Per-landmark BGR colours / labels (CSV order), as used by the Part A overlay.
LANDMARK_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 255),      # ofd_1 -- red
    (0, 165, 255),    # ofd_2 -- orange
    (0, 255, 0),      # bpd_1 -- green
    (255, 255, 0),    # bpd_2 -- cyan
)
LANDMARK_LABELS: Tuple[str, ...] = ("ofd_1", "ofd_2", "bpd_1", "bpd_2")

# Pair colours used by the Part B endpoint overlay (ofd pair / bpd pair).
ENDPOINT_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (0, 165, 255), (0, 165, 255),  # ofd_1, ofd_2 -- orange
    (0, 0, 255), (0, 0, 255),      # bpd_1, bpd_2 -- red
)

_OFD_CHORD_COLOR = (0, 200, 255)   # OFD chord -- amber
_BPD_CHORD_COLOR = (0, 255, 0)     # BPD chord -- green
_MASK_COLOR = (255, 0, 0)          # predicted mask outline -- blue
_ELLIPSE_COLOR = (0, 255, 255)     # fitted ellipse -- yellow
_RING_COLOR = (255, 255, 255)      # white ring around each point

# A ``cv2.fitEllipse`` result: ((cx, cy), (d1, d2), angle_deg).
Ellipse = Tuple[Tuple[float, float], Tuple[float, float], float]


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    """Return a BGR copy of ``gray`` (pass through if already 3-channel)."""
    arr = np.asarray(gray)
    if arr.ndim == 3:
        return arr.copy()
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)


def _radius_thickness(height: int, width: int) -> Tuple[int, int]:
    """Point radius and line thickness scaled to the image size."""
    radius = max(3, int(round(0.006 * max(height, width))))
    return radius, max(1, radius // 2)


def _draw_points(
    canvas: np.ndarray,
    points_xy: np.ndarray,
    colors: Sequence[Tuple[int, int, int]],
    labels: Optional[Sequence[str]],
    radius: int,
    thickness: int,
) -> None:
    """Draw filled, white-ringed, optionally-labelled points onto ``canvas``."""
    for idx, (x, y) in enumerate(points_xy):
        px, py = int(round(float(x))), int(round(float(y)))
        color = colors[idx % len(colors)]
        cv2.circle(canvas, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), radius, _RING_COLOR, 1, cv2.LINE_AA)
        if labels is not None and idx < len(labels) and labels[idx]:
            cv2.putText(
                canvas, labels[idx], (px + radius + 2, py - radius - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )


def draw_landmarks(
    gray: np.ndarray,
    points_xy: np.ndarray,
    draw_chords: bool = True,
    labels: Optional[Sequence[str]] = LANDMARK_LABELS,
) -> np.ndarray:
    """Draw the four predicted landmarks on a grayscale image; return BGR.

    Parameters
    ----------
    gray:
        Grayscale image ``(H, W)`` (or an existing BGR image).
    points_xy:
        ``(4, 2)`` array of ``(x, y)`` in the image's own pixel coordinates,
        in CSV order ``ofd_1, ofd_2, bpd_1, bpd_2``.
    draw_chords:
        If true, join the OFD pair (points 0-1) and BPD pair (points 2-3) with
        thin chords.
    labels:
        Optional per-point text labels (defaults to the CSV names); pass ``None``
        to omit text.
    """
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    canvas = _to_bgr(gray)
    height, width = canvas.shape[:2]
    radius, thickness = _radius_thickness(height, width)

    if draw_chords and len(pts) >= 4:
        p = [(int(round(x)), int(round(y))) for x, y in pts[:4]]
        cv2.line(canvas, p[0], p[1], _OFD_CHORD_COLOR, thickness, cv2.LINE_AA)
        cv2.line(canvas, p[2], p[3], _BPD_CHORD_COLOR, thickness, cv2.LINE_AA)

    _draw_points(canvas, pts, LANDMARK_COLORS, labels, radius, thickness)
    return canvas


def draw_mask_biometry(
    gray: np.ndarray,
    mask: np.ndarray,
    ellipse: Optional[Ellipse] = None,
    points_xy: Optional[np.ndarray] = None,
    labels: Optional[Sequence[str]] = LANDMARK_LABELS,
) -> np.ndarray:
    """Draw a predicted mask outline + fitted ellipse + endpoints; return BGR.

    Parameters
    ----------
    gray:
        Grayscale image ``(H, W)`` (or an existing BGR image).
    mask:
        Binary mask (any nonzero = foreground). If its shape differs from the
        image it is nearest-neighbour resized to match.
    ellipse:
        A ``cv2.fitEllipse`` tuple ``((cx, cy), (d1, d2), angle_deg)`` in the
        image's pixel coordinates, or ``None`` to skip the ellipse.
    points_xy:
        Optional ``(4, 2)`` axis endpoints (CSV order) to mark.
    labels:
        Optional per-point labels (defaults to the CSV names); ``None`` to omit.
    """
    canvas = _to_bgr(gray)
    height, width = canvas.shape[:2]
    radius, thickness = _radius_thickness(height, width)

    region = np.asarray(mask)
    if region.shape[:2] != (height, width):
        region = cv2.resize(
            region.astype(np.uint8), (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    binary = (region > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(canvas, contours, -1, _MASK_COLOR, thickness, cv2.LINE_AA)

    if ellipse is not None:
        (cx, cy), (d1, d2), angle = ellipse
        cv2.ellipse(
            canvas, (int(round(cx)), int(round(cy))),
            (int(round(d1 / 2.0)), int(round(d2 / 2.0))),
            float(angle), 0.0, 360.0, _ELLIPSE_COLOR, thickness, cv2.LINE_AA,
        )

    if points_xy is not None:
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        _draw_points(canvas, pts, ENDPOINT_COLORS, labels, radius, thickness)
    return canvas
