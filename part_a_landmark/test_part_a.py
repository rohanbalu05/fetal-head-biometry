"""Label-free folder inference for the Part A landmark models.

Runs a trained ``coord`` / ``unet`` / ``scn`` model over an arbitrary folder of
images and writes, for each image, the four predicted landmarks back in the
image's OWN original pixel space. It needs no labels and never reads the
ground-truth CSV unless ``--labels`` is explicitly supplied, so it can be graded
on a hidden image set.

Pipeline (per image, matching ``common.dataset.LandmarkDataset`` exactly):

1. Read the image at its ORIGINAL size (grayscale) and record ``(orig_w, orig_h)``.
2. Resize a COPY to 256x256 (``INTER_LINEAR``), scale to ``[0, 1]``, then apply
   the same per-image standardization (zero mean, unit std) used in training.
3. Run inference. For ``unet`` / ``scn`` decode the four heatmap channels via
   argmax + sub-pixel centroid (``common.heatmaps.heatmaps_to_coords``); for
   ``coord`` read the 8-vector directly and scale onto the 256 grid.
4. Denormalize every point from the 256 grid back to that image's own original
   pixel size -- a PER-IMAGE scale, since image sizes vary in this dataset.

Outputs (under ``--out-dir``):

- ``predictions.csv`` with the EXACT ground-truth column layout
  (``image_name, ofd_1_x, ofd_1_y, ofd_2_x, ofd_2_y, bpd_1_x, bpd_1_y,
  bpd_2_x, bpd_2_y``), integer pixel coordinates in each image's original space.
- ``<stem>_overlay.png`` per image: the four predicted points drawn on the
  original image.

Point order is the canonical CSV order ``ofd_1, ofd_2, bpd_1, bpd_2`` throughout;
no point is ever swapped, reordered, or relabeled.

Example::

    python part_a_landmark/test_part_a.py \\
        --model scn --weights part_a_landmark/weights/part_a_scn_best.pth \\
        --images data/images --out-dir runs/scn_infer

    # with optional metrics against the ground-truth CSV:
    python part_a_landmark/test_part_a.py --model scn \\
        --weights part_a_landmark/weights/part_a_scn_best.pth \\
        --images data/images --out-dir runs/scn_infer \\
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

from common.heatmaps import heatmaps_to_coords  # noqa: E402
from common.metrics import (  # noqa: E402
    mean_radial_error,
    per_point_error,
    success_rate_at_threshold,
)
from part_a_landmark.models import get_model  # noqa: E402

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

# Success-rate thresholds (px), reported only when --labels is given.
SR_THRESHOLDS: Tuple[int, ...] = (5, 10, 20)

# Recognised image extensions for folder discovery (case-insensitive).
IMAGE_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# BGR colors and short labels for the four overlay points.
_OVERLAY_STYLE: Tuple[Tuple[str, Tuple[int, int, int]], ...] = (
    ("ofd_1", (0, 0, 255)),      # red
    ("ofd_2", (0, 165, 255)),    # orange
    ("bpd_1", (0, 255, 0)),      # green
    ("bpd_2", (255, 255, 0)),    # cyan
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Label-free folder inference for Part A landmark models. Writes a "
            "predictions CSV (in ground-truth column order) and a per-image "
            "overlay PNG; computes metrics only if --labels is supplied."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=["coord", "unet", "scn"], required=True,
        help="Which Part A architecture the weights belong to.",
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
        "--sigma", type=float, default=4.0,
        help="Heatmap sigma (px); sets the sub-pixel centroid window for "
             "heatmap models. Ignored by the coord model.",
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

    model = get_model(model_key).to(device)
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


def centroid_window(sigma: float) -> int:
    """Map a heatmap sigma to an odd sub-pixel centroid window side (px).

    ``sigma=4`` -> ``7`` (the decoder default); scales sensibly otherwise.
    """
    return max(3, int(round(sigma)) * 2 - 1)


def decode_to_grid(
    model_key: str, outputs: torch.Tensor, window: int
) -> np.ndarray:
    """Decode raw model outputs to ``(4, 2)`` ``(x, y)`` on the 256 grid.

    ``coord`` outputs are normalized ``[0, 1]`` and scaled by the grid size;
    heatmap outputs are decoded with argmax + sub-pixel centroid refinement.
    """
    if model_key == "coord":
        vec = outputs.detach().cpu().numpy().reshape(4, 2)
        scale = np.array([INPUT_HW[1], INPUT_HW[0]], dtype=np.float32)
        return (vec * scale).astype(np.float32)
    heat = outputs.detach().cpu().numpy()[0]  # (4, H, W)
    return heatmaps_to_coords(heat, window=window).astype(np.float32)


def denormalize(points_grid: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    """Scale ``(4, 2)`` points from the 256 grid back to this image's pixels."""
    out_h, out_w = INPUT_HW
    out = points_grid.copy()
    out[:, 0] *= orig_w / float(out_w)
    out[:, 1] *= orig_h / float(out_h)
    return out


def draw_overlay(gray: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    """Render the four predicted points (and the OFD/BPD chords) on the image."""
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    height, width = gray.shape[:2]
    radius = max(3, int(round(0.006 * max(height, width))))
    thickness = max(1, radius // 2)

    pts = [(int(round(x)), int(round(y))) for x, y in points_px]
    # Faint chords linking the OFD pair and the BPD pair, for the eyeball check.
    cv2.line(canvas, pts[0], pts[1], (0, 200, 255), thickness, cv2.LINE_AA)
    cv2.line(canvas, pts[2], pts[3], (0, 255, 0), thickness, cv2.LINE_AA)

    for (label, color), (px, py) in zip(_OVERLAY_STYLE, pts):
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
    import pandas as pd  # local import: keep label-free path free of pandas

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
    """Print MRE and SR@5/10/20 for predictions matched to GT by image name."""
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
        print("  no image_name overlap between predictions and labels; "
              "nothing to score.")
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
    model_key: str,
    image_paths: List[Path],
    out_dir: Path,
    device: torch.device,
    window: int,
) -> Dict[str, np.ndarray]:
    """Run inference over all images, writing the CSV and overlays.

    Returns ``{image_name: (4, 2) points}`` in each image's original pixels.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"
    predictions: Dict[str, np.ndarray] = {}

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
                outputs = model(tensor)
            points_grid = decode_to_grid(model_key, outputs, window)
            points_px = denormalize(points_grid, orig_w, orig_h)
            predictions[path.name] = points_px

            row: Dict[str, object] = {"image_name": path.name}
            for (xc, yc), (px, py) in zip(_POINT_COLS, points_px):
                row[xc] = int(round(float(px)))
                row[yc] = int(round(float(py)))
            writer.writerow(row)

            overlay = draw_overlay(gray, points_px)
            cv2.imwrite(str(out_dir / f"{path.stem}_overlay.png"), overlay)

    print(f"Processed {len(predictions)} image(s).")
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

    window = centroid_window(args.sigma)
    predictions = run_inference(
        model, args.model, image_paths, args.out_dir, device, window
    )

    if args.labels is not None:
        report_metrics(predictions, args.labels)


if __name__ == "__main__":
    main()
