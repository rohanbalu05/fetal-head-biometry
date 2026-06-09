"""Figure-generation for the report (read-only EDA; trains nothing).

Produces the report PNGs into ``analysis/figures/`` from three already-existing,
read-only sources:

- the ground-truth label-quality audit ``data/derived/gt_consistency.csv``
  (per-image stats), and
- the three finished Part A training logs
  ``part_a_landmark/weights/part_a_{scn,unet,coord}_log.csv``,
- plus a small table of published Part A parameter counts / best val MRE.

It does NOT train and does NOT depend on any Part B weights (Part B is still
training). The dataset, masks, CSV and split are only ever READ.

Figures written:

1. ``partA_param_vs_mre.png``  -- parameter count vs best val MRE for the three
   hypotheses (the "smaller/structure-aware beats bigger" headline).
2. ``partA_training_curves.png`` -- val MRE vs epoch for every available model.
3. ``gt_label_quality.png``    -- multi-panel ground-truth audit summary.
4. ``gt_bpd_vs_ofd.png``       -- focused OFD-vs-BPD scatter (naming inversion).

Example::

    python analysis/eda.py --out-dir analysis/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

# --- Canonical read-only locations -----------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
GT_CONSISTENCY_CSV: Path = REPO_ROOT / "data" / "derived" / "gt_consistency.csv"
PART_A_WEIGHTS: Path = REPO_ROOT / "part_a_landmark" / "weights"

# Clean/noisy threshold used by the audit (max landmark->head distance, px).
CLEAN_THRESH_PX: float = 30.0

# Output raster settings.
DPI: int = 150

# Published Part A summary (parameter count, best val MRE in px). Provided in
# the report brief; figure 1 is a fixed comparison, not re-derived here.
PART_A_SUMMARY: Dict[str, Dict[str, float]] = {
    "coord": {"params": 1_702_376, "mre": 33.849},
    "unet": {"params": 13_390_404, "mre": 117.341},
    "scn": {"params": 301_000, "mre": 31.465},
}

# Part A training logs to read for the training-curve figure.
PART_A_LOG_MODELS: Sequence[str] = ("coord", "unet", "scn")

# Consistent per-model colors across figures.
MODEL_COLORS: Dict[str, str] = {
    "coord": "#1f77b4",
    "unet": "#ff7f0e",
    "scn": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the report EDA figures (read-only) from the ground-truth "
            "audit CSV and the Part A training logs. Trains nothing; needs no "
            "Part B weights."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "analysis" / "figures",
        help="Folder to write the figure PNGs into (created if missing).",
    )
    return parser.parse_args()


def guess_column(
    columns: Sequence[str], must_contain: Sequence[str]
) -> Optional[str]:
    """Return the first column whose lowercased name contains every substring.

    ``must_contain`` is a list of lowercase substrings that must all appear in
    the column name. Returns ``None`` if no column matches.
    """
    for col in columns:
        low = col.lower()
        if all(sub in low for sub in must_contain):
            return col
    return None


def map_consistency_columns(columns: Sequence[str]) -> Dict[str, Optional[str]]:
    """Best-guess map logical names -> actual gt_consistency columns and print it."""
    mapping = {
        "image_name": guess_column(columns, ["image"]),
        "max_dist": guess_column(columns, ["max", "dist"]),
        "ofd_len": guess_column(columns, ["ofd", "len"]),
        "bpd_len": guess_column(columns, ["bpd", "len"]),
        "clean": guess_column(columns, ["clean"]),
    }
    print("gt_consistency.csv column mapping:")
    print(f"  available columns: {list(columns)}")
    for logical, actual in mapping.items():
        status = actual if actual is not None else "<NOT FOUND>"
        print(f"  {logical:<11} -> {status}")
    return mapping


def _clean_mask(series: pd.Series) -> pd.Series:
    """Coerce a 'clean' column (bool / 'True' / 1) to a boolean Series."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def fig_param_vs_mre(out_dir: Path) -> Path:
    """Figure 1: parameter count vs best val MRE for the three hypotheses."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for model, stats in PART_A_SUMMARY.items():
        ax.scatter(
            stats["params"], stats["mre"], s=160,
            color=MODEL_COLORS[model], edgecolor="black", zorder=3,
            label=f"{model} ({stats['params']:,} params, {stats['mre']:.1f}px)",
        )
        ax.annotate(
            model, (stats["params"], stats["mre"]),
            textcoords="offset points", xytext=(8, 8), fontsize=11, weight="bold",
        )

    # Make the SCN point stand out: it has the fewest params AND the best MRE.
    scn = PART_A_SUMMARY["scn"]
    ax.scatter(
        scn["params"], scn["mre"], s=420, facecolors="none",
        edgecolors=MODEL_COLORS["scn"], linewidths=2.0, zorder=2,
    )
    ax.annotate(
        "fewest params, best MRE",
        (scn["params"], scn["mre"]),
        textcoords="offset points", xytext=(20, -34), fontsize=10,
        color=MODEL_COLORS["scn"], weight="bold",
        arrowprops=dict(arrowstyle="->", color=MODEL_COLORS["scn"], lw=1.5),
    )

    ax.set_xscale("log")
    ax.set_xlabel("parameter count (log scale)")
    ax.set_ylabel("best validation MRE (px)")
    ax.set_title("Part A: model size vs accuracy\n(smaller, structure-aware SCN wins)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(title="model", loc="upper left", fontsize=9)
    return _save(fig, out_dir / "partA_param_vs_mre.png")


def fig_training_curves(out_dir: Path) -> Path:
    """Figure 2: val MRE vs epoch for every Part A log that is present."""
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    plotted = 0
    for model in PART_A_LOG_MODELS:
        log_path = PART_A_WEIGHTS / f"part_a_{model}_log.csv"
        if not log_path.is_file():
            print(f"  WARNING: log missing for '{model}' ({log_path}); "
                  f"skipping its curve.")
            continue
        frame = pd.read_csv(log_path)
        mre_col = guess_column(frame.columns, ["val", "mre"]) \
            or guess_column(frame.columns, ["mre"])
        if mre_col is None:
            print(f"  WARNING: no val-MRE column in {log_path.name} "
                  f"(columns: {list(frame.columns)}); skipping.")
            continue
        epoch_col = guess_column(frame.columns, ["epoch"])
        epochs = frame[epoch_col] if epoch_col else range(1, len(frame) + 1)
        best = float(frame[mre_col].min())
        ax.plot(
            epochs, frame[mre_col], marker="o", ms=3, lw=1.8,
            color=MODEL_COLORS.get(model, None),
            label=f"{model} (best {best:.2f} px)",
        )
        plotted += 1

    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MRE (px)")
    ax.set_title("Part A: validation MRE per epoch")
    ax.grid(True, ls=":", alpha=0.5)
    if plotted:
        ax.legend(title="model", fontsize=9)
    else:
        ax.text(0.5, 0.5, "no Part A logs found", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="red")
    return _save(fig, out_dir / "partA_training_curves.png")


def fig_label_quality(
    df: pd.DataFrame, cols: Dict[str, Optional[str]], out_dir: Path
) -> Path:
    """Figure 3: multi-panel ground-truth audit (distance, split, OFD/BPD)."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # Panel (a): distribution of per-image max landmark->head distance.
    ax = axes[0]
    if cols["max_dist"]:
        dist = pd.to_numeric(df[cols["max_dist"]], errors="coerce").dropna()
        ax.hist(dist, bins=60, color="#4c72b0", alpha=0.85)
        ax.axvline(CLEAN_THRESH_PX, color="red", ls="--", lw=2,
                   label=f"clean threshold = {CLEAN_THRESH_PX:.0f}px")
        pct_clean = 100.0 * (dist <= CLEAN_THRESH_PX).mean()
        ax.set_title(f"(a) Max landmark->head distance\n"
                     f"{pct_clean:.1f}% within {CLEAN_THRESH_PX:.0f}px")
        ax.set_xlabel(f"max point distance (px)  [{cols['max_dist']}]")
        ax.set_ylabel("image count")
        ax.legend(fontsize=9)
        ax.grid(True, ls=":", alpha=0.4)
    else:
        _panel_unavailable(ax, "(a) max-distance column not found")

    # Panel (b): clean vs noisy proportion.
    ax = axes[1]
    clean_series = None
    if cols["clean"]:
        clean_series = _clean_mask(df[cols["clean"]])
    elif cols["max_dist"]:
        clean_series = pd.to_numeric(df[cols["max_dist"]], errors="coerce") \
            <= CLEAN_THRESH_PX
    if clean_series is not None:
        n_clean = int(clean_series.sum())
        n_noisy = int((~clean_series).sum())
        total = n_clean + n_noisy
        bars = ax.bar(
            ["clean", "noisy"], [n_clean, n_noisy],
            color=["#55a868", "#c44e52"], edgecolor="black",
        )
        for bar, count in zip(bars, (n_clean, n_noisy)):
            pct = 100.0 * count / total if total else 0.0
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
        ax.set_title("(b) Label-quality split")
        ax.set_ylabel("image count")
        ax.set_ylim(0, max(n_clean, n_noisy) * 1.18)
        ax.grid(True, axis="y", ls=":", alpha=0.4)
    else:
        _panel_unavailable(ax, "(b) clean/max-distance columns not found")

    # Panel (c): OFD vs BPD length scatter (bpd longer = above y=x).
    ax = axes[2]
    if cols["ofd_len"] and cols["bpd_len"]:
        ofd = pd.to_numeric(df[cols["ofd_len"]], errors="coerce")
        bpd = pd.to_numeric(df[cols["bpd_len"]], errors="coerce")
        valid = ofd.notna() & bpd.notna()
        ofd, bpd = ofd[valid], bpd[valid]
        pct_above = 100.0 * (bpd > ofd).mean() if len(ofd) else 0.0
        lim = float(max(ofd.max(), bpd.max())) * 1.05
        ax.scatter(ofd, bpd, s=12, alpha=0.5, color="#8172b3",
                   edgecolor="none")
        ax.plot([0, lim], [0, lim], color="black", ls="--", lw=1.5, label="y = x")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(f"(c) OFD vs BPD length\nbpd longer in {pct_above:.1f}% of images")
        ax.set_xlabel(f"OFD length (px)  [{cols['ofd_len']}]")
        ax.set_ylabel(f"BPD length (px)  [{cols['bpd_len']}]")
        ax.legend(fontsize=9)
        ax.grid(True, ls=":", alpha=0.4)
    else:
        _panel_unavailable(ax, "(c) ofd_len/bpd_len columns not found")

    fig.suptitle("Ground-truth label-quality audit (gt_consistency.csv)",
                 fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out_dir / "gt_label_quality.png")


def fig_bpd_vs_ofd(
    df: pd.DataFrame, cols: Dict[str, Optional[str]], out_dir: Path
) -> Optional[Path]:
    """Figure 4: focused OFD-vs-BPD scatter with the y=x line and % above it."""
    if not (cols["ofd_len"] and cols["bpd_len"]):
        print("  WARNING: ofd_len/bpd_len not found; skipping gt_bpd_vs_ofd.png.")
        return None

    ofd = pd.to_numeric(df[cols["ofd_len"]], errors="coerce")
    bpd = pd.to_numeric(df[cols["bpd_len"]], errors="coerce")
    valid = ofd.notna() & bpd.notna()
    ofd, bpd = ofd[valid], bpd[valid]
    above = bpd > ofd
    pct_above = 100.0 * above.mean() if len(ofd) else 0.0
    lim = float(max(ofd.max(), bpd.max())) * 1.05

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.scatter(ofd[above], bpd[above], s=16, alpha=0.55, color="#2ca02c",
               edgecolor="none", label=f"bpd longer ({pct_above:.1f}%)")
    ax.scatter(ofd[~above], bpd[~above], s=16, alpha=0.55, color="#d62728",
               edgecolor="none", label=f"ofd longer ({100 - pct_above:.1f}%)")
    ax.plot([0, lim], [0, lim], color="black", ls="--", lw=1.5, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"OFD length (px)  [{cols['ofd_len']}]")
    ax.set_ylabel(f"BPD length (px)  [{cols['bpd_len']}]")
    ax.set_title(f"OFD vs BPD per image\nbpd is the longer measurement in "
                 f"{pct_above:.1f}% of images (naming inversion)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    return _save(fig, out_dir / "gt_bpd_vs_ofd.png")


def _panel_unavailable(ax: plt.Axes, message: str) -> None:
    """Render a placeholder for a panel whose source column is missing."""
    print(f"  WARNING: {message}; rendering placeholder panel.")
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=11, color="red", wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def _save(fig: plt.Figure, path: Path) -> Path:
    """Save a figure at the project DPI with a tight bbox and close it."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def report_outputs(paths: Sequence[Path]) -> None:
    """Print each written figure with its real pixel dimensions."""
    print("\nFigures written:")
    for path in paths:
        with Image.open(path) as img:
            width, height = img.size
        print(f"  {path}  ({width} x {height} px)")


def main() -> None:
    """Entry point: build every figure and report the outputs."""
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    written: List[Path] = []

    # Figure 1: parameter count vs best val MRE (fixed summary table).
    written.append(fig_param_vs_mre(out_dir))

    # Figure 2: training curves from whichever logs are present.
    written.append(fig_training_curves(out_dir))

    # Figures 3 & 4: ground-truth audit (requires gt_consistency.csv).
    if GT_CONSISTENCY_CSV.is_file():
        df = pd.read_csv(GT_CONSISTENCY_CSV)
        cols = map_consistency_columns(df.columns)
        written.append(fig_label_quality(df, cols, out_dir))
        fig4 = fig_bpd_vs_ofd(df, cols, out_dir)
        if fig4 is not None:
            written.append(fig4)
    else:
        print(f"  WARNING: {GT_CONSISTENCY_CSV} not found; "
              f"skipping the label-quality figures.")

    report_outputs(written)


if __name__ == "__main__":
    main()
