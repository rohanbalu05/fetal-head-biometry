"""Ground-truth consistency check for the fetal-head biometry dataset.

For every image this script:

1. Fills the thin outline annotation into a solid head region.
2. Fits an ellipse (``cv2.fitEllipse``) to the largest contour of that region.
3. Measures how far each of the four CSV landmark points (OFD/BPD endpoints)
   lies from the filled head region (0 px if the point is inside).

Per-image results are written to ``data/derived/gt_consistency.csv`` and a
summary is printed. A contact sheet of the 24 worst (most inconsistent) images
is saved to ``data/derived/outliers_contact_sheet.png``.

Run directly::

    python analysis/ground_truth_check.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Make ``common`` importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common.data_utils import (  # noqa: E402
    DERIVED_DIR,
    GT_CSV,
    IMAGES_DIR,
    fill_mask,
    mask_path_for,
)

GT_CONSISTENCY_CSV: Path = DERIVED_DIR / "gt_consistency.csv"
CONTACT_SHEET_PNG: Path = DERIVED_DIR / "outliers_contact_sheet.png"

CLEAN_THRESH_PX: float = 30.0
POINT_LABELS: Tuple[str, str, str, str] = ("ofd1", "ofd2", "bpd1", "bpd2")


def largest_contour(filled: np.ndarray) -> Optional[np.ndarray]:
    """Return the largest external contour of a filled mask, or ``None``."""
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def dist_to_region(contour: np.ndarray, point: Tuple[float, float]) -> float:
    """Distance (px) from ``point`` to the region bounded by ``contour``.

    Returns 0.0 when the point lies inside (or on) the contour, otherwise the
    Euclidean distance to the nearest contour edge.
    """
    signed = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True)
    return 0.0 if signed >= 0 else -signed


def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def analyze() -> pd.DataFrame:
    """Build the per-image consistency table and save it to ``gt_consistency.csv``."""
    gt = pd.read_csv(GT_CSV)
    rows: List[dict] = []

    for _, r in gt.iterrows():
        image_name = str(r["image_name"])
        ofd1 = (float(r["ofd_1_x"]), float(r["ofd_1_y"]))
        ofd2 = (float(r["ofd_2_x"]), float(r["ofd_2_y"]))
        bpd1 = (float(r["bpd_1_x"]), float(r["bpd_1_y"]))
        bpd2 = (float(r["bpd_2_x"]), float(r["bpd_2_y"]))
        points = [ofd1, ofd2, bpd1, bpd2]

        mask_path = mask_path_for(image_name)
        filled = fill_mask(mask_path)
        contour = largest_contour(filled)

        if contour is None:
            dists = [float("nan")] * 4
            major = minor = float("nan")
        else:
            dists = [dist_to_region(contour, p) for p in points]
            if len(contour) >= 5:
                (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
                major = float(max(axis_a, axis_b))
                minor = float(min(axis_a, axis_b))
            else:
                major = minor = float("nan")

        ofd_len = _euclidean(ofd1, ofd2)
        bpd_len = _euclidean(bpd1, bpd2)
        max_dist = float(np.nanmax(dists)) if np.any(~np.isnan(dists)) else float("nan")
        mean_dist = float(np.nanmean(dists)) if np.any(~np.isnan(dists)) else float("nan")

        rows.append(
            {
                "image_name": image_name,
                "max_point_dist_px": max_dist,
                "mean_point_dist_px": mean_dist,
                "ofd_len_px": ofd_len,
                "bpd_len_px": bpd_len,
                "ellipse_major_px": major,
                "ellipse_minor_px": minor,
                "clean": bool(max_dist <= CLEAN_THRESH_PX) if not math.isnan(max_dist) else False,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "image_name",
            "max_point_dist_px",
            "mean_point_dist_px",
            "ofd_len_px",
            "bpd_len_px",
            "ellipse_major_px",
            "ellipse_minor_px",
            "clean",
        ],
    )
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(GT_CONSISTENCY_CSV, index=False)
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print aggregate consistency statistics for the dataset."""
    total = len(df)
    pct_clean = 100.0 * df["clean"].mean() if total else 0.0

    gt = pd.read_csv(GT_CSV)
    pt_cols = [
        ("ofd_1_x", "ofd_1_y"),
        ("ofd_2_x", "ofd_2_y"),
        ("bpd_1_x", "bpd_1_y"),
        ("bpd_2_x", "bpd_2_y"),
    ]
    # Recompute all individual point distances for point-level percentages.
    all_dists: List[float] = []
    gt_indexed = gt.set_index("image_name")
    for _, row in df.iterrows():
        name = row["image_name"]
        filled = fill_mask(mask_path_for(name))
        contour = largest_contour(filled)
        if contour is None:
            continue
        g = gt_indexed.loc[name]
        for xc, yc in pt_cols:
            all_dists.append(dist_to_region(contour, (float(g[xc]), float(g[yc]))))
    arr = np.asarray(all_dists, dtype=float)
    n_pts = arr.size
    pct_gt50 = 100.0 * np.mean(arr > 50.0) if n_pts else 0.0
    pct_gt150 = 100.0 * np.mean(arr > 150.0) if n_pts else 0.0

    median_bpd = float(df["bpd_len_px"].median())
    median_ofd = float(df["ofd_len_px"].median())
    n_bpd_gt_ofd = int((df["bpd_len_px"] > df["ofd_len_px"]).sum())
    pct_bpd_gt_ofd = 100.0 * n_bpd_gt_ofd / total if total else 0.0

    print("\n===== Ground-truth consistency summary =====")
    print(f"total images:            {total}")
    print(f"clean (max dist<=30px):  {df['clean'].sum()} ({pct_clean:.1f}%)")
    print(f"points off head >50px:   {pct_gt50:.1f}%  ({int(np.sum(arr > 50.0))}/{n_pts})")
    print(f"points off head >150px:  {pct_gt150:.1f}%  ({int(np.sum(arr > 150.0))}/{n_pts})")
    print(f"median BPD length:       {median_bpd:.1f} px")
    print(f"median OFD length:       {median_ofd:.1f} px")
    print(f"BPD length > OFD length: {n_bpd_gt_ofd} ({pct_bpd_gt_ofd:.1f}%)")
    print(f"per-image CSV:           {GT_CONSISTENCY_CSV}")
    print("============================================\n")


def save_contact_sheet(df: pd.DataFrame, n_worst: int = 24) -> None:
    """Save a grid of the ``n_worst`` images with the largest point distances."""
    worst = df.sort_values("max_point_dist_px", ascending=False).head(n_worst)
    gt = pd.read_csv(GT_CSV).set_index("image_name")

    n_cols = 6
    n_rows = int(math.ceil(n_worst / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 3.2))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(axes, worst.iterrows()):
        name = row["image_name"]
        img = cv2.imread(str(IMAGES_DIR / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            ax.set_title(f"{name}\n(missing image)", fontsize=7)
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # GT mask outline in red.
        outline = cv2.imread(str(mask_path_for(name)), cv2.IMREAD_GRAYSCALE)
        if outline is not None:
            rgb[outline > 127] = (255, 0, 0)

        ax.imshow(rgb)

        g = gt.loc[name]
        ofd = [(float(g["ofd_1_x"]), float(g["ofd_1_y"])),
               (float(g["ofd_2_x"]), float(g["ofd_2_y"]))]
        bpd = [(float(g["bpd_1_x"]), float(g["bpd_1_y"])),
               (float(g["bpd_2_x"]), float(g["bpd_2_y"]))]

        ax.plot([ofd[0][0], ofd[1][0]], [ofd[0][1], ofd[1][1]],
                color="yellow", linewidth=1.2, zorder=2)
        ax.plot([bpd[0][0], bpd[1][0]], [bpd[0][1], bpd[1][1]],
                color="cyan", linewidth=1.2, zorder=2)

        pts = ofd + bpd
        colors = ["yellow", "yellow", "cyan", "cyan"]
        for (px, py), label, color in zip(pts, POINT_LABELS, colors):
            ax.scatter([px], [py], c=color, s=18, edgecolors="black",
                       linewidths=0.5, zorder=3)
            ax.text(px + 5, py, label, color=color, fontsize=6,
                    zorder=4,
                    bbox=dict(facecolor="black", alpha=0.4, pad=0.3, edgecolor="none"))

        ax.set_xlim(0, rgb.shape[1])
        ax.set_ylim(rgb.shape[0], 0)
        ax.set_title(f"{name}\nmax dist={row['max_point_dist_px']:.0f}px", fontsize=7)
        ax.axis("off")

    fig.suptitle(f"{n_worst} worst ground-truth outliers (highest max point distance)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONTACT_SHEET_PNG, dpi=120)
    plt.close(fig)
    print(f"contact sheet:           {CONTACT_SHEET_PNG}")


def main() -> None:
    """Run the full ground-truth consistency check."""
    df = analyze()
    save_contact_sheet(df)
    print_summary(df)


if __name__ == "__main__":
    main()
