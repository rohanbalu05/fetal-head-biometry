"""Evaluation metrics for landmark detection and segmentation.

Provides pixel-distance error for landmarks (mean radial error, per-point error,
and success-rate-at-threshold) and the Dice coefficient for segmentation masks
(used by Part B).
"""

from __future__ import annotations

import numpy as np


def per_point_error(pred_xy: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
    """Per-point Euclidean distance (px) between predicted and GT points.

    Parameters
    ----------
    pred_xy, gt_xy:
        Array-like of shape ``(N, 2)`` with matching point order.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of shape ``(N,)`` of per-point distances in pixels.
    """
    pred = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    gt = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")
    return np.linalg.norm(pred - gt, axis=1)


def mean_radial_error(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    """Mean per-point Euclidean distance (px)."""
    return float(np.mean(per_point_error(pred_xy, gt_xy)))


def success_rate_at_threshold(errors: np.ndarray, thresh_px: float) -> float:
    """Percentage of points whose error is within ``thresh_px``.

    Parameters
    ----------
    errors:
        Array-like of per-point distances (px), e.g. from :func:`per_point_error`.
    thresh_px:
        Distance threshold in pixels.

    Returns
    -------
    float
        Percentage (0-100) of points with ``error <= thresh_px``.
    """
    err = np.asarray(errors, dtype=np.float64).ravel()
    if err.size == 0:
        return 0.0
    return float(100.0 * np.mean(err <= float(thresh_px)))


def dice(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    """Dice coefficient between two binary masks (Part B).

    Inputs are treated as binary via ``> 0``. Returns a value in ``[0, 1]``.
    """
    pred = np.asarray(pred_mask) > 0
    gt = np.asarray(gt_mask) > 0
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")
    intersection = float(np.logical_and(pred, gt).sum())
    denom = float(pred.sum() + gt.sum())
    return (2.0 * intersection + eps) / (denom + eps)
