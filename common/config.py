"""Central configuration: filesystem paths, image size, random seed, and hyperparameters.

A single, importable source of truth for the canonical constants used across
both pipelines. The values here MIRROR what the individual modules already use
(image size 256, heatmap sigma 4, seed 42, split ratios 0.7/0.15/0.15, the fixed
landmark order ``ofd_1, ofd_2, bpd_1, bpd_2``, and the data paths). This module
declares constants and paths only; it never reads the dataset or the CSV.

Note: existing modules currently hardcode these same values (e.g.
``common.data_utils`` for the paths, ``common.dataset`` for ``INPUT_HW``). They
are intentionally left as-is; this module simply matches them so future code can
import one canonical definition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

# --- Filesystem paths (repo root = parent of the common/ package) ----------
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
IMAGES_DIR: Path = DATA_DIR / "images"
MASKS_DIR: Path = DATA_DIR / "masks"
DERIVED_DIR: Path = DATA_DIR / "derived"
MASKS_FILLED_DIR: Path = DERIVED_DIR / "masks_filled"
GT_CSV: Path = DATA_DIR / "ground_truth.csv"
SPLIT_JSON: Path = DERIVED_DIR / "split.json"

# --- Image / network resolution --------------------------------------------
INPUT_SIZE: int = 256
INPUT_HW: Tuple[int, int] = (INPUT_SIZE, INPUT_SIZE)

# --- Training / decoding hyperparameters -----------------------------------
SEED: int = 42
HEATMAP_SIGMA: float = 4.0
SPLIT_RATIOS: Tuple[float, float, float] = (0.7, 0.15, 0.15)

# --- Canonical landmark order (never reordered; matches the GT CSV) ---------
NUM_LANDMARKS: int = 4
POINT_NAMES: Tuple[str, ...] = ("ofd_1", "ofd_2", "bpd_1", "bpd_2")
# Per-point (x, y) column-name pairs, in CSV order.
POINT_COLUMNS: Tuple[Tuple[str, str], ...] = tuple(
    (f"{name}_x", f"{name}_y") for name in POINT_NAMES
)
# Full ground-truth/prediction CSV header, in order.
CSV_COLUMNS: Tuple[str, ...] = ("image_name",) + tuple(
    f"{name}_{axis}" for name in POINT_NAMES for axis in ("x", "y")
)

# --- Evaluation / audit constants ------------------------------------------
SR_THRESHOLDS_PX: Tuple[int, ...] = (5, 10, 20)
CLEAN_THRESHOLD_PX: float = 30.0
HC_COEFF: float = 1.62
