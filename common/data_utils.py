"""Data utilities: filename parsing, patient-grouped splitting, and mask filling.

This module centralizes the dataset paths and the low-level helpers shared by the
landmark and segmentation pipelines:

- ``patient_id`` parses the patient identifier from an image filename.
- ``build_patient_grouped_split`` produces a leak-free train/val/test split keyed
  by unique patient id and persists it to ``data/derived/split.json``.
- ``fill_mask`` converts a thin outline annotation into a solid binary mask.
- ``save_filled_masks`` materializes the filled masks under ``data/derived/``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# --- Canonical paths -------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
IMAGES_DIR: Path = DATA_DIR / "images"
MASKS_DIR: Path = DATA_DIR / "masks"
DERIVED_DIR: Path = DATA_DIR / "derived"
MASKS_FILLED_DIR: Path = DERIVED_DIR / "masks_filled"
GT_CSV: Path = DATA_DIR / "ground_truth.csv"
SPLIT_JSON: Path = DERIVED_DIR / "split.json"

_PATIENT_RE = re.compile(r"^(\d+)")


def patient_id(filename: str) -> str:
    """Return the patient id (leading digits before the first underscore).

    Examples
    --------
    >>> patient_id("499_2HC.png")
    '499'
    >>> patient_id("000_HC.png")
    '000'
    """
    name = Path(filename).name
    match = _PATIENT_RE.match(name)
    if match is None:
        raise ValueError(f"Cannot parse patient id from {filename!r}")
    return match.group(1)


def list_image_names() -> List[str]:
    """Return the sorted list of image filenames, ignoring ``desktop.ini``."""
    names = [p.name for p in IMAGES_DIR.glob("*.png") if p.name.lower() != "desktop.ini"]
    return sorted(names)


def mask_path_for(image_name: str) -> Path:
    """Return the annotation (outline) mask path for a given image filename."""
    return MASKS_DIR / f"{Path(image_name).stem}_Annotation.png"


def build_patient_grouped_split(
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Dict[str, object]:
    """Split the dataset by unique patient id and save it to ``split.json``.

    Splitting is performed over unique patient ids (not images) so that no
    patient appears in more than one split. Image counts per split are printed
    and zero patient overlap is verified before returning.

    Parameters
    ----------
    seed:
        Seed for the patient-shuffling RNG (reproducible splits).
    ratios:
        ``(train, val, test)`` fractions of *patients*. They should sum to ~1;
        the test count absorbs any rounding remainder.

    Returns
    -------
    dict
        ``{"seed", "ratios", "train", "val", "test", "patients"}`` where the
        split keys map to sorted lists of image filenames and ``patients`` maps
        each split to its sorted list of patient ids.
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {ratios} (sum={sum(ratios)})")

    image_names = list_image_names()
    images_by_patient: Dict[str, List[str]] = {}
    for name in image_names:
        images_by_patient.setdefault(patient_id(name), []).append(name)

    patients = sorted(images_by_patient)
    rng = np.random.default_rng(seed)
    shuffled = list(patients)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(round(ratios[0] * n_total))
    n_val = int(round(ratios[1] * n_total))
    n_train = min(n_train, n_total)
    n_val = min(n_val, n_total - n_train)

    patient_split: Dict[str, List[str]] = {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }

    split: Dict[str, object] = {
        "seed": seed,
        "ratios": list(ratios),
        "patients": patient_split,
    }
    for name in ("train", "val", "test"):
        images: List[str] = []
        for pid in patient_split[name]:
            images.extend(images_by_patient[pid])
        split[name] = sorted(images)

    # --- Verify zero patient overlap --------------------------------------
    sets = {k: set(v) for k, v in patient_split.items()}
    overlaps = {
        f"{a}&{b}": sets[a] & sets[b]
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    max_overlap = max(len(v) for v in overlaps.values())

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_JSON, "w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=2)

    print("Patient-grouped split (seed=%d, ratios=%s)" % (seed, tuple(ratios)))
    print(f"  patients: total={n_total}, "
          f"train={len(patient_split['train'])}, "
          f"val={len(patient_split['val'])}, "
          f"test={len(patient_split['test'])}")
    print(f"  images:   total={len(image_names)}, "
          f"train={len(split['train'])}, "  # type: ignore[arg-type]
          f"val={len(split['val'])}, "      # type: ignore[arg-type]
          f"test={len(split['test'])}")     # type: ignore[arg-type]
    if max_overlap == 0:
        print("  patient overlap between splits: 0 (OK, no leakage)")
    else:
        for key, val in overlaps.items():
            if val:
                print(f"  WARNING: overlap {key}: {sorted(val)}")
    print(f"  saved -> {SPLIT_JSON}")

    return split


def fill_mask(mask_path: str | Path) -> np.ndarray:
    """Read a thin outline annotation and return a solid (filled) ``uint8`` mask.

    The outline is binarized, its external contours are detected, and those
    contours are drawn filled, producing the solid head region (values 0/255).

    Parameters
    ----------
    mask_path:
        Path to the outline annotation PNG.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array (same H x W as the annotation) with the head region
        set to 255 and background to 0.
    """
    outline = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if outline is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    _, binary = cv2.threshold(outline, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled = np.zeros_like(binary)
    if contours:
        cv2.drawContours(filled, contours, -1, color=255, thickness=cv2.FILLED)
    return filled.astype(np.uint8)


def save_filled_masks(out_dir: str | Path = MASKS_FILLED_DIR) -> Path:
    """Fill every annotation outline and write the results to ``out_dir``.

    Each filled mask keeps its original annotation filename. The output
    directory (default ``data/derived/masks_filled/``) is created if needed.

    Returns
    -------
    pathlib.Path
        The directory the filled masks were written to.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(
        p for p in MASKS_DIR.glob("*.png") if p.name.lower() != "desktop.ini"
    )
    for i, mask_file in enumerate(mask_files, start=1):
        filled = fill_mask(mask_file)
        cv2.imwrite(str(out_dir / mask_file.name), filled)
        if i % 100 == 0 or i == len(mask_files):
            print(f"  filled {i}/{len(mask_files)} masks", end="\r")
    print(f"\n  saved {len(mask_files)} filled masks -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    build_patient_grouped_split()
    save_filled_masks()
