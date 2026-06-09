"""Smoke test for the Part A model definitions.

Instantiates all three hypotheses (H1/H2/H3), runs a dummy
``(2, 1, 256, 256)`` batch through each, and prints output shapes and parameter
counts as a small table. Asserts the expected output shapes.

Run directly::

    python scripts/model_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from part_a_landmark.models import get_model  # noqa: E402


def count_params(model: torch.nn.Module) -> int:
    """Return the total number of parameters in ``model``."""
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    """Run all three models on a dummy batch and print a summary table."""
    torch.manual_seed(0)
    dummy = torch.randn(2, 1, 256, 256)

    specs = [
        ("H1", "coord", (2, 8)),
        ("H2", "unet", (2, 4, 256, 256)),
        ("H3", "scn", (2, 4, 256, 256)),
    ]

    rows = []
    for tag, key, expected in specs:
        model = get_model(key)
        model.eval()
        with torch.no_grad():
            out = model(dummy)
        out_shape: Tuple[int, ...] = tuple(out.shape)
        assert out_shape == expected, (
            f"{tag} ({key}) output shape {out_shape} != expected {expected}"
        )
        rows.append((tag, key, model.__class__.__name__, out_shape, count_params(model)))

    # Extra: confirm H3 exposes appearance/spatial components.
    scn = get_model("scn")
    scn.eval()
    with torch.no_grad():
        comps = scn(dummy, return_components=True)
    assert set(comps) == {"final", "appearance", "spatial"}
    assert tuple(comps["appearance"].shape) == (2, 4, 256, 256)
    assert tuple(comps["spatial"].shape) == (2, 4, 256, 256)

    print("\nPart A model smoke test (input: (2, 1, 256, 256))")
    print("-" * 74)
    print(f"{'H':<4}{'key':<7}{'class':<26}{'output shape':<22}{'params':>12}")
    print("-" * 74)
    for tag, key, cls, shape, n in rows:
        print(f"{tag:<4}{key:<7}{cls:<26}{str(shape):<22}{n:>12,}")
    print("-" * 74)
    print("H3 components: final/appearance/spatial all (2, 4, 256, 256) OK")
    print("All output-shape assertions passed.")


if __name__ == "__main__":
    main()
