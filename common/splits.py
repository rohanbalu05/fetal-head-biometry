"""Patient-grouped train/validation/test splitting.

Thin loader around :func:`common.data_utils.build_patient_grouped_split`. The
split keeps all images from the same patient within a single split, preventing
patient-level leakage. The split is cached on disk at ``data/derived/split.json``
and rebuilt on demand.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

from common import data_utils


def load_split(
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    rebuild: bool = False,
) -> Dict[str, object]:
    """Return the patient-grouped split, building and caching it if needed.

    Parameters
    ----------
    seed, ratios:
        Forwarded to :func:`common.data_utils.build_patient_grouped_split` when
        the split must be built.
    rebuild:
        If ``True``, ignore any cached ``split.json`` and rebuild.

    Returns
    -------
    dict
        The split dict (see ``build_patient_grouped_split``).
    """
    if data_utils.SPLIT_JSON.exists() and not rebuild:
        with open(data_utils.SPLIT_JSON, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return data_utils.build_patient_grouped_split(seed=seed, ratios=ratios)


def image_names_for(split_name: str, **kwargs: object) -> List[str]:
    """Return the sorted image filenames belonging to ``split_name``.

    Parameters
    ----------
    split_name:
        One of ``"train"``, ``"val"``, ``"test"``.
    **kwargs:
        Forwarded to :func:`load_split` (``seed``, ``ratios``, ``rebuild``).
    """
    if split_name not in ("train", "val", "test"):
        raise ValueError(f"split_name must be train/val/test, got {split_name!r}")
    split = load_split(**kwargs)  # type: ignore[arg-type]
    return list(split[split_name])  # type: ignore[index]
