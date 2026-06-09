"""Part A vs Part B cross-check and clean-subset metrics (headline analysis).

Runs BOTH finished pipelines on the held-out TEST split (read-only) and reports:

1. Two independent sets of the four landmarks per image, each in that image's
   ORIGINAL pixel space:
   - Part A (landmark SCN): heatmap decode + per-image denormalize, reusing the
     EXACT functions in ``part_a_landmark/test_part_a.py``.
   - Part B (segmentation U-Net -> ellipse): ``mask_to_biometry`` + longer-axis
     -> bpd assignment + per-image denormalize, reusing the EXACT functions in
     ``part_b_segmentation/test_part_b.py``.
   Nothing is reimplemented here; both testers and this script share identical
   denormalize + axis-assignment so the comparison is fair.

2. Part A accuracy vs GT three ways (FULL / CLEAN / NOISY test subsets), where
   CLEAN is ``gt_consistency.clean == True`` (max landmark->head dist <= 30px).
3. Part B accuracy vs GT the same three ways (identical footing).
4. A-vs-B agreement, independent of GT: per-point distance (pair-matched), plus
   the mean absolute A-vs-B difference of the cephalic index and the HC estimate.
5. Figures: a Bland-Altman cross-check (BPD + OFD panels) and a grouped MRE bar
   chart (A vs B on full/clean/noisy).

This script trains nothing and only READS the dataset / CSV / split. The internal
"which axis is longer" logic is used purely for metrics and is never written back
to any label.

Example::

    python analysis/fusion.py            # default SCN + unet weights, test split
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# The finished checkpoints were trained on Linux (Colab) and pickle PosixPath
# objects inside their saved 'args'; those cannot be instantiated on Windows.
# Map PosixPath -> WindowsPath so torch.load can unpickle them. We only ever use
# the tensor weights, never these pickled paths.
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc]

# Make project packages importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: write files, never open a window

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from common import data_utils  # noqa: E402
from common.geometry import cephalic_index, hc_estimate, points_to_axes  # noqa: E402
from common.metrics import (  # noqa: E402
    mean_radial_error,
    per_point_error,
    success_rate_at_threshold,
)
# Reuse the EXACT tested inference logic (do not reimplement differently).
from part_a_landmark import test_part_a as a_test  # noqa: E402
from part_b_segmentation import test_part_b as b_test  # noqa: E402

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_FIG_DIR: Path = REPO_ROOT / "analysis" / "figures"
GT_CONSISTENCY_CSV: Path = REPO_ROOT / "data" / "derived" / "gt_consistency.csv"
CLEAN_THRESH_PX: float = 30.0
DPI: int = 150


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Cross-check Part A (landmark) vs Part B (segmentation->geometry) on "
            "the held-out test split, with full/clean/noisy metric breakdowns. "
            "Read-only; trains nothing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--part_a_model", choices=["coord", "unet", "scn"], default="scn",
        help="Part A architecture to load.",
    )
    parser.add_argument(
        "--part_a_weights", type=Path,
        default=REPO_ROOT / "part_a_landmark" / "weights" / "part_a_scn_best.pth",
        help="Part A checkpoint (.pth).",
    )
    parser.add_argument(
        "--part_b_model", choices=["unet", "unet_attn"], default="unet",
        help="Part B architecture to load.",
    )
    parser.add_argument(
        "--part_b_weights", type=Path,
        default=REPO_ROOT / "part_b_segmentation" / "weights" / "part_b_unet_best.pth",
        help="Part B checkpoint (.pth).",
    )
    parser.add_argument(
        "--images-dir", type=Path, default=data_utils.IMAGES_DIR,
        help="Folder of source images (read at original size).",
    )
    parser.add_argument(
        "--labels", type=Path, default=data_utils.GT_CSV,
        help="Ground-truth CSV (read-only).",
    )
    parser.add_argument(
        "--gt-consistency", type=Path, default=GT_CONSISTENCY_CSV,
        help="Per-image label-quality CSV providing the 'clean' flag.",
    )
    parser.add_argument(
        "--split", type=Path, default=data_utils.SPLIT_JSON,
        help="split.json; the 'test' split is used.",
    )
    parser.add_argument(
        "--sigma", type=float, default=4.0,
        help="Part A heatmap sigma (sets the sub-pixel centroid window).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_FIG_DIR,
        help="Folder for the cross-check figures.",
    )
    return parser.parse_args()


def load_test_names(split_path: Path) -> List[str]:
    """Return the test-split image filenames from split.json."""
    with open(split_path, "r", encoding="utf-8") as fh:
        split = json.load(fh)
    return list(split["test"])


def load_clean_map(consistency_path: Path) -> Dict[str, bool]:
    """Return ``{image_name: clean_bool}`` from the audit CSV.

    Uses the explicit ``clean`` column if present, else derives it from the
    max landmark->head distance column at the 30px threshold.
    """
    frame = pd.read_csv(consistency_path)
    cols = list(frame.columns)
    name_col = next((c for c in cols if "image" in c.lower()), cols[0])

    clean_col = next((c for c in cols if "clean" in c.lower()), None)
    dist_col = next(
        (c for c in cols if "max" in c.lower() and "dist" in c.lower()), None
    )
    if clean_col is not None:
        flags = frame[clean_col].astype(str).str.strip().str.lower().isin(
            ("true", "1", "yes")
        )
        print(f"clean flag source: column '{clean_col}'")
    elif dist_col is not None:
        flags = pd.to_numeric(frame[dist_col], errors="coerce") <= CLEAN_THRESH_PX
        print(f"clean flag source: derived from '{dist_col}' <= {CLEAN_THRESH_PX}px")
    else:
        raise ValueError(f"no 'clean' or max-distance column in {consistency_path}")

    return {str(n): bool(f) for n, f in zip(frame[name_col], flags)}


def infer_part_a(
    model: torch.nn.Module,
    model_key: str,
    gray: np.ndarray,
    window: int,
    device: torch.device,
) -> np.ndarray:
    """Part A points (4, 2) in original pixels, via the tester's own functions."""
    orig_h, orig_w = gray.shape[:2]
    tensor = a_test.preprocess(gray).to(device)
    with torch.no_grad():
        outputs = model(tensor)
    grid = a_test.decode_to_grid(model_key, outputs, window)
    return a_test.denormalize(grid, orig_w, orig_h)


def infer_part_b(
    model: torch.nn.Module, gray: np.ndarray, device: torch.device
) -> Optional[np.ndarray]:
    """Part B points (4, 2) in original pixels, or None if no ellipse fits."""
    orig_h, orig_w = gray.shape[:2]
    tensor = b_test.preprocess(gray).to(device)
    with torch.no_grad():
        probs = model(tensor)
    mask256 = (probs.detach().cpu().numpy()[0, 0] > 0.5).astype(np.uint8)
    bio = b_test.mask_to_biometry(mask256)
    if bio is None:
        return None
    return b_test.assign_points_for_csv(bio, orig_w, orig_h)


def subset_metrics(
    preds: Dict[str, np.ndarray], gt: Dict[str, np.ndarray], names: Sequence[str]
) -> Optional[Dict[str, float]]:
    """Pool the four points over ``names`` (where a prediction exists) and score."""
    pred_pts: List[np.ndarray] = []
    gt_pts: List[np.ndarray] = []
    for name in names:
        if name in preds and name in gt:
            pred_pts.append(np.asarray(preds[name], dtype=np.float64))
            gt_pts.append(np.asarray(gt[name], dtype=np.float64))
    if not pred_pts:
        return None
    pred_arr = np.concatenate(pred_pts, axis=0)
    gt_arr = np.concatenate(gt_pts, axis=0)
    errors = per_point_error(pred_arr, gt_arr)
    return {
        "n_images": float(len(pred_pts)),
        "n_points": float(errors.size),
        "mre": mean_radial_error(pred_arr, gt_arr),
        "sr5": success_rate_at_threshold(errors, 5),
        "sr10": success_rate_at_threshold(errors, 10),
        "sr20": success_rate_at_threshold(errors, 20),
    }


def print_block(title: str, metrics: Optional[Dict[str, float]]) -> None:
    """Print one metric line (or a no-images note)."""
    if metrics is None:
        print(f"  {title:<7}: (no images)")
        return
    print(
        f"  {title:<7}: n={int(metrics['n_images']):>3} imgs / "
        f"{int(metrics['n_points']):>4} pts | MRE={metrics['mre']:8.3f}px | "
        f"SR@5={metrics['sr5']:6.2f}% SR@10={metrics['sr10']:6.2f}% "
        f"SR@20={metrics['sr20']:6.2f}%"
    )


def pair_matched_distances(a_pts: np.ndarray, b_pts: np.ndarray) -> List[float]:
    """Per-point A-vs-B distances with optimal endpoint matching within each pair.

    The OFD pair is rows 0,1 and the BPD pair is rows 2,3 (CSV order). Within a
    pair the two endpoints are unordered between pipelines, so we take the
    endpoint assignment that minimizes total distance before measuring.
    """
    dists: List[float] = []
    for i, j in ((0, 1), (2, 3)):
        a1, a2, b1, b2 = a_pts[i], a_pts[j], b_pts[i], b_pts[j]
        d11, d22 = float(np.hypot(*(a1 - b1))), float(np.hypot(*(a2 - b2)))
        d12, d21 = float(np.hypot(*(a1 - b2))), float(np.hypot(*(a2 - b1)))
        if d11 + d22 <= d12 + d21:
            dists.extend((d11, d22))
        else:
            dists.extend((d12, d21))
    return dists


def bland_altman_panel(
    ax: plt.Axes,
    mean_vals: np.ndarray,
    diff_vals: np.ndarray,
    label: str,
) -> Dict[str, float]:
    """Draw one Bland-Altman panel (A vs B) and return its summary stats."""
    mean_vals = np.asarray(mean_vals, dtype=np.float64)
    diff_vals = np.asarray(diff_vals, dtype=np.float64)
    md = float(np.mean(diff_vals))
    sd = float(np.std(diff_vals, ddof=1)) if diff_vals.size > 1 else 0.0
    loa_hi, loa_lo = md + 1.96 * sd, md - 1.96 * sd

    ax.scatter(mean_vals, diff_vals, s=16, alpha=0.5, color="#4c72b0",
               edgecolor="none")
    ax.axhline(md, color="blue", lw=1.6, label=f"mean diff = {md:.2f}px")
    ax.axhline(loa_hi, color="red", ls="--", lw=1.4,
               label=f"+1.96 SD = {loa_hi:.2f}px")
    ax.axhline(loa_lo, color="red", ls="--", lw=1.4,
               label=f"-1.96 SD = {loa_lo:.2f}px")
    ax.set_title(f"Bland-Altman: {label} length (Part A vs Part B)")
    ax.set_xlabel(f"mean of A and B {label} length (px)")
    ax.set_ylabel(f"A - B {label} length (px)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, ls=":", alpha=0.4)
    return {"mean_diff": md, "sd": sd, "loa_hi": loa_hi, "loa_lo": loa_lo}


def figure_bland_altman(
    ba_data: Dict[str, np.ndarray], out_dir: Path
) -> Tuple[Path, Dict[str, Dict[str, float]]]:
    """Two-panel Bland-Altman figure (BPD and OFD)."""
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))
    stats = {
        "bpd": bland_altman_panel(
            axes[0], ba_data["bpd_mean"], ba_data["bpd_diff"], "BPD"
        ),
        "ofd": bland_altman_panel(
            axes[1], ba_data["ofd_mean"], ba_data["ofd_diff"], "OFD"
        ),
    }
    fig.suptitle("Part A vs Part B cross-check (diameter lengths)",
                 fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out_dir / "fusion_bland_altman.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path, stats


def figure_clean_vs_full(
    a_metrics: Dict[str, Optional[Dict[str, float]]],
    b_metrics: Dict[str, Optional[Dict[str, float]]],
    out_dir: Path,
) -> Path:
    """Grouped bar chart of MRE for A and B over full/clean/noisy."""
    subsets = ("full", "clean", "noisy")

    def mre(metrics: Optional[Dict[str, float]]) -> float:
        return float(metrics["mre"]) if metrics else float("nan")

    a_vals = [mre(a_metrics[s]) for s in subsets]
    b_vals = [mre(b_metrics[s]) for s in subsets]

    x = np.arange(len(subsets))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    bars_a = ax.bar(x - width / 2, a_vals, width, label="Part A (landmark)",
                    color="#2ca02c", edgecolor="black")
    bars_b = ax.bar(x + width / 2, b_vals, width, label="Part B (segmentation)",
                    color="#ff7f0e", edgecolor="black")
    for bars in (bars_a, bars_b):
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1f}",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in subsets])
    ax.set_ylabel("mean radial error (px)")
    ax.set_title("Landmark MRE by test subset\n(error drops sharply on the "
                 "clean subset)")
    ax.legend()
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    path = out_dir / "fusion_clean_vs_full_mre.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    """Entry point: run both pipelines on the test split and report everything."""
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = a_test.select_device()
    print(f"Part A: {args.part_a_model} <- {args.part_a_weights}")
    print(f"Part B: {args.part_b_model} <- {args.part_b_weights}")

    a_model = a_test.load_model(args.part_a_model, args.part_a_weights, device)
    b_model = b_test.load_model(args.part_b_model, args.part_b_weights, device)
    window = a_test.centroid_window(args.sigma)

    test_names = load_test_names(args.split)
    gt = a_test.read_labels(args.labels)
    clean_map = load_clean_map(args.gt_consistency)

    clean_names = [n for n in test_names if clean_map.get(n, False)]
    noisy_names = [n for n in test_names if not clean_map.get(n, False)]
    print(f"\nTest images: {len(test_names)} "
          f"(clean={len(clean_names)}, noisy={len(noisy_names)})")

    # --- (1) Run both pipelines on every test image ------------------------
    a_preds: Dict[str, np.ndarray] = {}
    b_preds: Dict[str, np.ndarray] = {}
    b_failures: List[str] = []
    missing: List[str] = []
    for name in test_names:
        gray = cv2.imread(str(args.images_dir / name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            missing.append(name)
            continue
        a_preds[name] = infer_part_a(a_model, args.part_a_model, gray, window, device)
        b_pts = infer_part_b(b_model, gray, device)
        if b_pts is None:
            b_failures.append(name)
        else:
            b_preds[name] = b_pts
    if missing:
        print(f"  WARNING: {len(missing)} test image(s) unreadable: {missing}")
    if b_failures:
        print(f"  Part B produced no ellipse for {len(b_failures)} image(s): "
              f"{b_failures}")

    # --- (2) Part A accuracy vs GT (full / clean / noisy) ------------------
    a_metrics = {
        "full": subset_metrics(a_preds, gt, test_names),
        "clean": subset_metrics(a_preds, gt, clean_names),
        "noisy": subset_metrics(a_preds, gt, noisy_names),
    }
    print(f"\n=== Part A ({args.part_a_model}) vs GT ===")
    for key in ("full", "clean", "noisy"):
        print_block(key, a_metrics[key])

    # --- (3) Part B accuracy vs GT (full / clean / noisy) ------------------
    b_metrics = {
        "full": subset_metrics(b_preds, gt, test_names),
        "clean": subset_metrics(b_preds, gt, clean_names),
        "noisy": subset_metrics(b_preds, gt, noisy_names),
    }
    print(f"\n=== Part B ({args.part_b_model}) vs GT ===")
    for key in ("full", "clean", "noisy"):
        print_block(key, b_metrics[key])

    # --- (4) A-vs-B agreement (independent of GT) --------------------------
    agree_dists: List[float] = []
    ci_diffs: List[float] = []
    hc_diffs: List[float] = []
    ba: Dict[str, List[float]] = {
        k: [] for k in ("bpd_mean", "bpd_diff", "ofd_mean", "ofd_diff")
    }
    both_names = [n for n in test_names if n in a_preds and n in b_preds]
    for name in both_names:
        a_pts, b_pts = a_preds[name], b_preds[name]
        agree_dists.extend(pair_matched_distances(a_pts, b_pts))

        ofd_a, bpd_a = points_to_axes(a_pts)
        ofd_b, bpd_b = points_to_axes(b_pts)
        ci_diffs.append(abs(cephalic_index(bpd_a, ofd_a)
                            - cephalic_index(bpd_b, ofd_b)))
        hc_diffs.append(abs(hc_estimate(bpd_a, ofd_a) - hc_estimate(bpd_b, ofd_b)))
        ba["bpd_mean"].append((bpd_a + bpd_b) / 2.0)
        ba["bpd_diff"].append(bpd_a - bpd_b)
        ba["ofd_mean"].append((ofd_a + ofd_b) / 2.0)
        ba["ofd_diff"].append(ofd_a - ofd_b)

    agree = np.asarray(agree_dists, dtype=np.float64)
    ci_arr = np.asarray(ci_diffs, dtype=np.float64)
    hc_arr = np.asarray(hc_diffs, dtype=np.float64)
    print(f"\n=== A-vs-B agreement (independent of GT, n={len(both_names)} images) ===")
    print(f"  per-point distance: mean={np.mean(agree):.3f}px  "
          f"median={np.median(agree):.3f}px  ({agree.size} points)")
    print(f"  cephalic index |A-B|: mean={np.nanmean(ci_arr):.3f}  "
          f"median={np.nanmedian(ci_arr):.3f}")
    print(f"  HC estimate   |A-B|: mean={np.nanmean(hc_arr):.3f}px  "
          f"median={np.nanmedian(hc_arr):.3f}px")

    # --- (5) Figures -------------------------------------------------------
    ba_arrays = {k: np.asarray(v, dtype=np.float64) for k, v in ba.items()}
    ba_path, ba_stats = figure_bland_altman(ba_arrays, out_dir)
    bar_path = figure_clean_vs_full(a_metrics, b_metrics, out_dir)

    # --- Concise summary table --------------------------------------------
    def cell(metrics: Optional[Dict[str, float]], field: str) -> str:
        return f"{metrics[field]:.3f}" if metrics else "n/a"

    a_head = f"Part A ({args.part_a_model})"
    b_head = f"Part B ({args.part_b_model})"
    print("\n================== SUMMARY (px unless noted) ==================")
    print(f"{'metric':<22}{a_head:>20}{b_head:>20}")
    for key in ("full", "clean", "noisy"):
        print(f"{'MRE ' + key:<22}{cell(a_metrics[key], 'mre'):>20}"
              f"{cell(b_metrics[key], 'mre'):>20}")
    for key in ("full", "clean", "noisy"):
        print(f"{'SR@10 ' + key:<22}{cell(a_metrics[key], 'sr10'):>20}"
              f"{cell(b_metrics[key], 'sr10'):>20}")
    print("-" * 62)
    print(f"{'A-B agree mean':<22}{np.mean(agree):>20.3f}{'':>20}")
    print(f"{'A-B agree median':<22}{np.median(agree):>20.3f}{'':>20}")
    print(f"{'A-B |CI| mean':<22}{np.nanmean(ci_arr):>20.3f}{'':>20}")
    print(f"{'A-B |HC| mean (px)':<22}{np.nanmean(hc_arr):>20.3f}{'':>20}")
    print(f"{'BPD bias (A-B)':<22}{ba_stats['bpd']['mean_diff']:>20.3f}{'':>20}")
    print(f"{'OFD bias (A-B)':<22}{ba_stats['ofd']['mean_diff']:>20.3f}{'':>20}")
    print("=" * 62)

    print("\nFigures written:")
    for path in (ba_path, bar_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
