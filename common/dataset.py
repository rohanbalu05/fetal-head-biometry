"""PyTorch Datasets for fetal-head ultrasound (Part A landmarks, Part B masks).

``LandmarkDataset`` pairs each ultrasound image with the four ground-truth
landmark points from the CSV (order ``ofd_1, ofd_2, bpd_1, bpd_2``) and the
corresponding Gaussian target heatmaps.

``SegmentationDataset`` pairs each image with its FILLED cranium mask (from
``data/derived/masks_filled/``) for Part B segmentation.

Label policy (per task rules): the four CSV points are used EXACTLY as given.
The only transforms applied to coordinates are (a) the deterministic resize
scaling from each image's real size to the 256x256 network resolution and
(b) optional geometric augmentation that is applied CONSISTENTLY to both the
image and the points. No swapping, reordering, or "normalization" of labels is
performed.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common import data_utils, splits
from common.heatmaps import coords_to_heatmaps

# Network input resolution.
INPUT_HW: Tuple[int, int] = (256, 256)

# CSV columns for the four points, in canonical order.
_POINT_COLS = (
    ("ofd_1_x", "ofd_1_y"),
    ("ofd_2_x", "ofd_2_y"),
    ("bpd_1_x", "bpd_1_y"),
    ("bpd_2_x", "bpd_2_y"),
)


class LandmarkDataset(Dataset):
    """Dataset of (image, heatmaps, points) for a patient-grouped split.

    Each item is a tuple::

        (image, heatmaps, points, orig_wh, image_name)

    where

    - ``image``     : ``float32`` tensor ``(1, 256, 256)``, normalized to
      ``[0, 1]`` then per-image standardized.
    - ``heatmaps``  : ``float32`` tensor ``(4, 256, 256)`` Gaussian targets.
    - ``points``    : ``float32`` tensor ``(4, 2)`` of ``(x, y)`` coordinates in
      the 256x256 frame (after resize scaling and any augmentation).
    - ``orig_wh``   : ``int64`` tensor ``(2,)`` of the image's real ``(W, H)``.
    - ``image_name``: ``str`` filename.

    Parameters
    ----------
    split:
        ``"train"``, ``"val"`` or ``"test"``.
    sigma:
        Gaussian heatmap sigma (px) in the 256x256 frame.
    augment:
        Enable geometric + intensity augmentation. Ignored unless
        ``split == "train"``.
    seed:
        Seed for the augmentation RNG and (if needed) the split build.
    """

    def __init__(
        self,
        split: str = "train",
        sigma: float = 4.0,
        augment: bool = False,
        seed: int = 42,
        rot_deg: float = 15.0,
        scale_jitter: float = 0.10,
        trans_frac: float = 0.05,
        intensity_jitter: float = 0.20,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.split = split
        self.sigma = float(sigma)
        self.augment = bool(augment) and split == "train"
        self.rot_deg = float(rot_deg)
        self.scale_jitter = float(scale_jitter)
        self.trans_frac = float(trans_frac)
        self.intensity_jitter = float(intensity_jitter)
        self._rng = np.random.default_rng(seed)

        wanted = set(splits.image_names_for(split, seed=seed))
        gt = pd.read_csv(data_utils.GT_CSV)
        gt = gt[gt["image_name"].isin(wanted)].reset_index(drop=True)
        self.frame = gt
        self.image_names = list(gt["image_name"])

    def __len__(self) -> int:
        return len(self.frame)

    def _read_points(self, row: "pd.Series") -> np.ndarray:
        """Return the 4 CSV points as a ``(4, 2)`` float array (real-pixel coords)."""
        pts = np.array(
            [[float(row[xc]), float(row[yc])] for xc, yc in _POINT_COLS],
            dtype=np.float32,
        )
        return pts

    def _augment(
        self, image01: np.ndarray, points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply a consistent affine + intensity jitter to image and points.

        Operates in the 256x256 frame; points are clipped to remain in bounds.
        """
        height, width = image01.shape[:2]
        center = (width / 2.0, height / 2.0)

        angle = float(self._rng.uniform(-self.rot_deg, self.rot_deg))
        scale = float(1.0 + self._rng.uniform(-self.scale_jitter, self.scale_jitter))
        tx = float(self._rng.uniform(-self.trans_frac, self.trans_frac) * width)
        ty = float(self._rng.uniform(-self.trans_frac, self.trans_frac) * height)

        mat = cv2.getRotationMatrix2D(center, angle, scale)
        mat[0, 2] += tx
        mat[1, 2] += ty

        warped = cv2.warpAffine(
            image01, mat, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )

        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        homog = np.concatenate([points, ones], axis=1)  # (4, 3)
        pts_aug = (homog @ mat.T).astype(np.float32)     # (4, 2)
        pts_aug[:, 0] = np.clip(pts_aug[:, 0], 0.0, width - 1.0)
        pts_aug[:, 1] = np.clip(pts_aug[:, 1], 0.0, height - 1.0)

        # Intensity jitter: contrast + brightness, kept in [0, 1].
        contrast = float(1.0 + self._rng.uniform(-self.intensity_jitter, self.intensity_jitter))
        brightness = float(self._rng.uniform(-self.intensity_jitter, self.intensity_jitter))
        warped = np.clip(warped * contrast + brightness, 0.0, 1.0)

        return warped.astype(np.float32), pts_aug

    @staticmethod
    def _standardize(image01: np.ndarray) -> np.ndarray:
        """Per-image standardization: zero mean, unit std (guarded)."""
        mean = float(image01.mean())
        std = float(image01.std())
        std = std if std > 1e-6 else 1.0
        return ((image01 - mean) / std).astype(np.float32)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        row = self.frame.iloc[index]
        image_name = str(row["image_name"])

        # Read at REAL size (grayscale).
        img = cv2.imread(str(data_utils.IMAGES_DIR / image_name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_name}")
        orig_h, orig_w = img.shape[:2]

        # Deterministic resize to 256x256 and scale points by (256/W, 256/H).
        out_h, out_w = INPUT_HW
        resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        points = self._read_points(row)
        points[:, 0] *= out_w / float(orig_w)
        points[:, 1] *= out_h / float(orig_h)

        image01 = resized.astype(np.float32) / 255.0
        if self.augment:
            image01, points = self._augment(image01, points)

        heatmaps = coords_to_heatmaps(points, out_hw=INPUT_HW, sigma=self.sigma)
        image_std = self._standardize(image01)

        image_t = torch.from_numpy(image_std).unsqueeze(0)        # (1, H, W)
        heatmaps_t = torch.from_numpy(heatmaps)                   # (4, H, W)
        points_t = torch.from_numpy(points.astype(np.float32))    # (4, 2)
        orig_wh_t = torch.tensor([orig_w, orig_h], dtype=torch.long)

        return image_t, heatmaps_t, points_t, orig_wh_t, image_name


class SegmentationDataset(Dataset):
    """Dataset of (image, cranium mask) pairs for Part B segmentation.

    Each item is a tuple::

        (image, mask, orig_wh, image_name)

    where

    - ``image``     : ``float32`` tensor ``(1, 256, 256)``, normalized to
      ``[0, 1]`` then per-image standardized (same scheme as
      :class:`LandmarkDataset`).
    - ``mask``      : ``float32`` tensor ``(1, 256, 256)`` with values in
      ``{0, 1}`` -- the FILLED cranium mask from ``data/derived/masks_filled/``
      (not the thin outline), resized with NEAREST interpolation.
    - ``orig_wh``   : ``int64`` tensor ``(2,)`` of the image's real ``(W, H)``.
    - ``image_name``: ``str`` filename.

    Parameters
    ----------
    split:
        ``"train"``, ``"val"`` or ``"test"``.
    augment:
        Enable geometric + intensity augmentation. The SAME affine is applied to
        the image and the mask. Ignored unless ``split == "train"``.
    seed:
        Seed for the augmentation RNG and (if needed) the split build.
    """

    def __init__(
        self,
        split: str = "train",
        augment: bool = False,
        seed: int = 42,
        rot_deg: float = 15.0,
        scale_jitter: float = 0.10,
        trans_frac: float = 0.05,
        intensity_jitter: float = 0.20,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.split = split
        self.augment = bool(augment) and split == "train"
        self.rot_deg = float(rot_deg)
        self.scale_jitter = float(scale_jitter)
        self.trans_frac = float(trans_frac)
        self.intensity_jitter = float(intensity_jitter)
        self._rng = np.random.default_rng(seed)
        self.image_names = list(splits.image_names_for(split, seed=seed))

    def __len__(self) -> int:
        return len(self.image_names)

    @staticmethod
    def _filled_mask_path(image_name: str):
        """Return the filled-mask path for an image filename."""
        return data_utils.MASKS_FILLED_DIR / data_utils.mask_path_for(image_name).name

    def _augment(
        self, image01: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply a consistent affine to image+mask, plus image-only intensity jitter."""
        height, width = image01.shape[:2]
        center = (width / 2.0, height / 2.0)

        angle = float(self._rng.uniform(-self.rot_deg, self.rot_deg))
        scale = float(1.0 + self._rng.uniform(-self.scale_jitter, self.scale_jitter))
        tx = float(self._rng.uniform(-self.trans_frac, self.trans_frac) * width)
        ty = float(self._rng.uniform(-self.trans_frac, self.trans_frac) * height)

        mat = cv2.getRotationMatrix2D(center, angle, scale)
        mat[0, 2] += tx
        mat[1, 2] += ty

        img_aug = cv2.warpAffine(
            image01, mat, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        mask_aug = cv2.warpAffine(
            mask, mat, (width, height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )

        contrast = float(1.0 + self._rng.uniform(-self.intensity_jitter, self.intensity_jitter))
        brightness = float(self._rng.uniform(-self.intensity_jitter, self.intensity_jitter))
        img_aug = np.clip(img_aug * contrast + brightness, 0.0, 1.0)

        return img_aug.astype(np.float32), mask_aug.astype(np.float32)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        image_name = self.image_names[index]

        img = cv2.imread(str(data_utils.IMAGES_DIR / image_name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_name}")
        orig_h, orig_w = img.shape[:2]

        mask_path = self._filled_mask_path(image_name)
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise FileNotFoundError(f"Could not read filled mask: {mask_path}")

        out_h, out_w = INPUT_HW
        resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask_raw, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        image01 = resized.astype(np.float32) / 255.0
        if self.augment:
            image01, mask = self._augment(image01, mask)
        mask = (mask > 0.5).astype(np.float32)

        image_std = LandmarkDataset._standardize(image01)

        image_t = torch.from_numpy(image_std).unsqueeze(0)   # (1, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0)         # (1, H, W)
        orig_wh_t = torch.tensor([orig_w, orig_h], dtype=torch.long)

        return image_t, mask_t, orig_wh_t, image_name
