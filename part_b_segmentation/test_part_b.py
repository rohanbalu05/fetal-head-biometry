"""Label-free folder inference for the Part B segmentation->geometry pipeline.

Runs a trained ``unet`` / ``unet_attn`` cranium segmenter over an arbitrary
folder of images, fits an ellipse to each predicted mask, and writes the four
derived biometry endpoints back in the image's OWN original pixel space. It
needs no labels or masks and never reads the ground-truth CSV unless ``--labels``
is explicitly supplied, so it can be graded on a hidden image set.

Pipeline (per image, matching ``common.dataset.SegmentationDataset``):

1. Read the image at its ORIGINAL size (grayscale); record ``(orig_w, orig_h)``.
2. Resize a COPY to 256x256 (``INTER_LINEAR``), scale to ``[0, 1]``, then apply
   the same per-image standardization (zero mean, unit std) used in training.
3. Run inference; threshold the sigmoid output at 0.5 -> binary cranium mask.
4. Derive biometry via ``geometry_decode.mask_to_biometry`` (largest contour ->
   ``cv2.fitEllipse`` -> major/minor axis endpoints) in the 256 frame.
5. Denormalize every endpoint from the 256 grid back to that image's own
   original pixel size -- a PER-IMAGE scale, since image sizes vary.

Axis -> column mapping (geometry only, for output formatting):
the four endpoints are denormalized to original pixels FIRST, then the LONGER
original-pixel pair goes into the ``bpd_*`` columns and the SHORTER into the
``ofd_*`` columns. This is NOT clinical naming: a documented audit (see report)
found the dataset's ``bpd`` pair is the longer measurement in ~84% of rows -- a
naming inversion that is recorded but never corrected, and that was measured in
ORIGINAL pixels. Since the 256 -> original denormalize is anisotropic, the
longer/shorter decision must be made post-denormalize. No CSV label is read or
altered to do this.

Outputs (under ``--out-dir``):

- ``predictions.csv`` with the EXACT ground-truth column layout
  (``image_name, ofd_1_x, ofd_1_y, ofd_2_x, ofd_2_y, bpd_1_x, bpd_1_y,
  bpd_2_x, bpd_2_y``), integer pixel coordinates in each image's original space.
- ``<stem>_overlay.png`` per image: predicted mask outline + fitted ellipse +
  the four derived endpoints, drawn on the original image.

Images whose mask yields no fittable ellipse are not crashed on: a sentinel row
(all coords ``-1``) is written, a warning is printed, and that image is excluded
from the optional metrics.

Example::

    python part_b_segmentation/test_part_b.py \\
        --model unet --weights part_b_segmentation/weights/part_b_unet_best.pth \\
        --images data/images --out-dir runs/unet_infer

    # with optional metrics against the ground-truth CSV:
    python part_b_segmentation/test_part_b.py --model unet \\
        --weights part_b_segmentation/weights/part_b_unet_best.pth \\
        --images data/images --out-dir runs/unet_infer \\
        --labels data/ground_truth.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# The finished checkpoints were trained on Linux (Colab) and pickle PosixPath
# objects inside their saved 'args'; those cannot be instantiated on Windows.
# Map PosixPath -> WindowsPath so torch.load can unpickle them. We only ever use
# the tensor weights, never these pickled paths.
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc]

# Make project packages importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from common.metrics import (  # noqa: E402
    mean_radial_error,
    per_point_error,
    success_rate_at_threshold,
)
from part_b_segmentation.geometry_decode import mask_to_biometry  # noqa: E402
from part_b_segmentation.models import get_seg_model  # noqa: E402

# Network input resolution (must match training/dataset).
INPUT_HW: Tuple[int, int] = (256, 256)

# CSV columns for the four points, in canonical order (never reordered).
POINT_NAMES: Tuple[str, ...] = ("ofd_1", "ofd_2", "bpd_1", "bpd_2")
CSV_FIELDNAMES: List[str] = ["image_name"] + [
    f"{name}_{axis}" for name in POINT_NAMES for axis in ("x", "y")
]
_POINT_COLS: Tuple[Tuple[str, str], ...] = tuple(
    (f"{name}_x", f"{name}_y") for name in POINT_NAMES
)

# Sentinel written when no ellipse can be fit for an image.
SENTINEL: int = -1

# Success-rate thresholds (px), reported only when --labels is given.
SR_THRESHOLDS: Tuple[int, ...] = (5, 10, 20)

# Recognised image extensions for folder discovery (case-insensitive).
IMAGE_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Label-free folder inference for Part B segmentation->geometry. "
            "Writes a predictions CSV (in ground-truth column order) and a "
            "per-image overlay PNG; computes metrics only if --labels is given."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=["unet", "unet_attn"], required=True,
        help="Which Part B architecture the weights belong to.",
    )
    parser.add_argument(
        "--weights", type=Path, required=True,
        help="Path to the .pth checkpoint (training dict or raw state_dict).",
    )
    parser.add_argument(
        "--images", type=Path, required=True,
        help="Folder of images to run inference on (read at original size).",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Folder for predictions.csv and the per-image overlay PNGs.",
    )
    parser.add_argument(
        "--labels", type=Path, default=None,
        help="Optional ground-truth CSV. If given, metrics (mean radial error "
             "and SR@5/10/20) are printed for images matched by name.",
    )
    return parser.parse_args()


def select_device() -> torch.device:
    """Return CUDA if available, else CPU, and announce the choice."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device


def load_model(
    model_key: str, weights_path: Path, device: torch.device
) -> nn.Module:
    """Build the model and load weights from either checkpoint format.

    Handles both a training checkpoint (a dict with a ``model_state`` key) and a
    raw ``state_dict`` (our slim weights-only copies).
    """
    if not weights_path.is_file():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    # Try a full unpickle first; if it fails for any reason, fall back to a
    # weights-only load (all we need for inference: the tensor weights).
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        print("Load path: full checkpoint (weights_only=False)")
    except Exception as exc:  # noqa: BLE001 - report and retry weights-only
        print(f"  full unpickle failed ({type(exc).__name__}: {exc}); "
              f"retrying with weights_only=True")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        print("Load path: weights-only fallback (weights_only=True)")

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        fmt = "training checkpoint (model_state)"
    else:
        state_dict = checkpoint
        fmt = "raw state_dict"
    print(f"Checkpoint format: {fmt}")

    model = get_seg_model(model_key).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def list_images(folder: Path) -> List[Path]:
    """Return sorted image paths in ``folder`` (recognised extensions only)."""
    if not folder.is_dir():
        raise NotADirectoryError(f"images folder not found: {folder}")
    paths = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(paths, key=lambda p: p.name)


def preprocess(gray: np.ndarray) -> torch.Tensor:
    """Resize a COPY to 256x256 and standardize, matching the training pipeline.

    Returns a ``(1, 1, 256, 256)`` float32 tensor. The input ``gray`` is left
    untouched (we operate on a resized copy).
    """
    out_h, out_w = INPUT_HW
    resized = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    image01 = resized.astype(np.float32) / 255.0
    mean = float(image01.mean())
    std = float(image01.std())
    std = std if std > 1e-6 else 1.0
    standardized = (image01 - mean) / std
    return torch.from_numpy(standardized).unsqueeze(0).unsqueeze(0)


def assign_points_for_csv(
    bio: Dict[str, object], orig_w: int, orig_h: int
) -> np.ndarray:
    """Denormalize the four ellipse endpoints, then assign pairs to CSV columns
    by ORIGINAL-pixel pair length.

    Returns ``(4, 2)`` as ``[ofd_1, ofd_2, bpd_1, bpd_2]`` already in original
    pixels.

    Mapping rationale (output formatting only, no label is touched): the dataset
    audit documented in the report found the ``bpd`` pair is the LONGER of the
    two measurements in ~84% of rows -- a naming inversion that is recorded but
    never corrected, and that was measured in ORIGINAL pixel space. Because the
    256 -> original denormalize is anisotropic (x and y scale by different
    factors), the longer fitted axis AT 256 can become the shorter pair in
    original pixels. The longer/shorter decision is therefore made AFTER
    denormalize, in original pixels: the LONGER original-pixel pair goes into the
    ``bpd_*`` columns and the SHORTER into the ``ofd_*`` columns. The existing
    within-pair point ordering is preserved.
    """
    # Denormalize each fitted axis's endpoints to original pixels (per-image).
    axis_1 = denormalize(
        np.asarray(bio["major_axis"], dtype=np.float64).reshape(2, 2),
        orig_w, orig_h,
    )
    axis_2 = denormalize(
        np.asarray(bio["minor_axis"], dtype=np.float64).reshape(2, 2),
        orig_w, orig_h,
    )
    len_1 = float(np.hypot(axis_1[0, 0] - axis_1[1, 0], axis_1[0, 1] - axis_1[1, 1]))
    len_2 = float(np.hypot(axis_2[0, 0] - axis_2[1, 0], axis_2[0, 1] - axis_2[1, 1]))

    longer, shorter = (axis_1, axis_2) if len_1 >= len_2 else (axis_2, axis_1)
    # ofd_* = shorter original-pixel pair; bpd_* = longer original-pixel pair.
    return np.array(
        [shorter[0], shorter[1], longer[0], longer[1]], dtype=np.float64
    )


def denormalize(points_grid: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    """Scale ``(4, 2)`` points from the 256 grid back to this image's pixels."""
    out_h, out_w = INPUT_HW
    out = points_grid.copy()
    out[:, 0] *= orig_w / float(out_w)
    out[:, 1] *= orig_h / float(out_h)
    return out


def draw_overlay(
    gray: np.ndarray,
    mask256: np.ndarray,
    ellipse256: Optional[Tuple[Tuple[float, float], Tuple[float, float], float]],
    points_px: np.ndarray,
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """Render mask outline + fitted ellipse + 4 endpoints on the original image.

    The mask/ellipse are computed at 256; the mask is upsampled (NEAREST) and the
    ellipse is drawn as a per-image-scaled polygon, so everything lines up with
    the denormalized endpoints.
    """
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out_h, out_w = INPUT_HW
    sx, sy = orig_w / float(out_w), orig_h / float(out_h)
    radius = max(3, int(round(0.006 * max(orig_h, orig_w))))
    thickness = max(1, radius // 2)

    # Predicted mask outline (upsample the 256 mask to original size).
    mask_full = cv2.resize(
        mask256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    )
    contours, _ = cv2.findContours(
        mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(canvas, contours, -1, (255, 0, 0), thickness, cv2.LINE_AA)

    # Fitted ellipse, scaled per-image from the 256 grid to original pixels.
    if ellipse256 is not None:
        (cx, cy), (d1, d2), angle = ellipse256
        poly = cv2.ellipse2Poly(
            (int(round(cx)), int(round(cy))),
            (int(round(d1 / 2.0)), int(round(d2 / 2.0))),
            int(round(angle)), 0, 360, 3,
        ).astype(np.float64)
        poly[:, 0] *= sx
        poly[:, 1] *= sy
        cv2.polylines(
            canvas, [poly.astype(np.int32)], True, (0, 255, 255),
            thickness, cv2.LINE_AA,
        )

    # Endpoints: ofd (shorter axis) and bpd (longer axis), distinct colors.
    pts = [(int(round(x)), int(round(y))) for x, y in points_px]
    styles = (
        ("ofd_1", (0, 165, 255)), ("ofd_2", (0, 165, 255)),
        ("bpd_1", (0, 0, 255)), ("bpd_2", (0, 0, 255)),
    )
    for (label, color), (px, py) in zip(styles, pts):
        cv2.circle(canvas, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), radius, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            canvas, label, (px + radius + 2, py - radius - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )
    return canvas


def read_labels(labels_csv: Path) -> Dict[str, np.ndarray]:
    """Read the ground-truth CSV into ``{image_name: (4, 2) points}``.

    Imported lazily so the CSV is touched ONLY when --labels is supplied.
    """
    import pandas as pd  # local import: keep the label-free path free of pandas

    frame = pd.read_csv(labels_csv)
    gt: Dict[str, np.ndarray] = {}
    for _, row in frame.iterrows():
        pts = np.array(
            [[float(row[xc]), float(row[yc])] for xc, yc in _POINT_COLS],
            dtype=np.float64,
        )
        gt[str(row["image_name"])] = pts
    return gt


def report_metrics(
    predictions: Dict[str, np.ndarray], labels_csv: Path
) -> None:
    """Print MRE and SR@5/10/20 for predictions matched to GT by image name.

    Only images with a valid (non-sentinel) prediction are scored.
    """
    gt = read_labels(labels_csv)
    matched_pred: List[np.ndarray] = []
    matched_gt: List[np.ndarray] = []
    for name, pred in predictions.items():
        if name in gt:
            matched_pred.append(pred.astype(np.float64))
            matched_gt.append(gt[name])

    n_matched = len(matched_pred)
    print(f"\nMetrics (labels: {labels_csv})")
    if n_matched == 0:
        print("  no scorable image_name overlap between predictions and labels.")
        return

    pred_arr = np.concatenate(matched_pred, axis=0)
    gt_arr = np.concatenate(matched_gt, axis=0)
    errors = per_point_error(pred_arr, gt_arr)
    mre = mean_radial_error(pred_arr, gt_arr)

    print(f"  matched images: {n_matched} ({len(errors)} points)")
    print(f"  mean radial error: {mre:.3f} px")
    for thresh in SR_THRESHOLDS:
        print(f"  SR@{thresh}: {success_rate_at_threshold(errors, thresh):.1f}%")


def run_inference(
    model: nn.Module,
    image_paths: List[Path],
    out_dir: Path,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Run inference over all images, writing the CSV and overlays.

    Returns ``{image_name: (4, 2) points}`` in original pixels for images that
    produced a valid ellipse (sentinel rows are excluded from this dict).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"
    predictions: Dict[str, np.ndarray] = {}
    n_ok = 0
    n_skipped = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for path in image_paths:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"  WARNING: could not read {path.name}; skipping.")
                continue
            orig_h, orig_w = gray.shape[:2]

            tensor = preprocess(gray).to(device)
            with torch.no_grad():
                probs = model(tensor)
            mask256 = (probs.detach().cpu().numpy()[0, 0] > 0.5).astype(np.uint8)

            bio = mask_to_biometry(mask256)
            if bio is None:
                # No fittable ellipse (e.g. empty mask): sentinel row, no crash.
                n_skipped += 1
                print(f"  WARNING: no ellipse for {path.name}; writing sentinel row.")
                row: Dict[str, object] = {"image_name": path.name}
                for xc, yc in _POINT_COLS:
                    row[xc] = SENTINEL
                    row[yc] = SENTINEL
                writer.writerow(row)
                overlay = draw_overlay(
                    gray, mask256, None, np.full((4, 2), SENTINEL, np.float64),
                    orig_w, orig_h,
                )
                cv2.imwrite(str(out_dir / f"{path.stem}_overlay.png"), overlay)
                continue

            points_px = assign_points_for_csv(bio, orig_w, orig_h)
            predictions[path.name] = points_px
            n_ok += 1

            row = {"image_name": path.name}
            for (xc, yc), (px, py) in zip(_POINT_COLS, points_px):
                row[xc] = int(round(float(px)))
                row[yc] = int(round(float(py)))
            writer.writerow(row)

            overlay = draw_overlay(
                gray, mask256, bio["ellipse"], points_px, orig_w, orig_h
            )
            cv2.imwrite(str(out_dir / f"{path.stem}_overlay.png"), overlay)

    print(f"Processed {n_ok + n_skipped} image(s): {n_ok} with geometry, "
          f"{n_skipped} sentinel.")
    print(f"Predictions CSV: {csv_path}")
    print(f"Overlays:        {out_dir}")
    return predictions


def main() -> None:
    """Entry point: load the model, run folder inference, optionally score."""
    args = parse_args()
    device = select_device()

    model = load_model(args.model, args.weights, device)
    image_paths = list_images(args.images)
    print(f"Model: {args.model} | images found: {len(image_paths)} "
          f"in {args.images}")
    if not image_paths:
        print("No images to process; exiting.")
        return

    predictions = run_inference(model, image_paths, args.out_dir, device)

    if args.labels is not None:
        report_metrics(predictions, args.labels)


if __name__ == "__main__":
    main()
