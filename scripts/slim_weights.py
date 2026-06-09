"""Create slim, weights-only copies of the five trained checkpoints and verify them.

Each finished training checkpoint bundles the optimizer/scheduler/args alongside
the model weights. For release we only need the model weights, so this script
extracts the ``model_state`` tensors and re-saves them as a plain ``state_dict``.

The point is CORRECTNESS verification of the exact artifacts we ship: every
slim file is reloaded with ``weights_only=True`` and pushed back into its model
class with ``strict=True`` so a missing/renamed key is caught immediately.
Smaller files are a welcome side effect, not the goal.

This script only ever touches weight files; it never reads the dataset or CSV.

Run::

    python scripts/slim_weights.py
"""

from __future__ import annotations

import os
import pathlib
import sys
from pathlib import Path
from typing import Callable, List, NamedTuple

# The finished checkpoints were trained on Linux (Colab) and pickle PosixPath
# objects inside their saved 'args'; those cannot be instantiated on Windows.
# Map PosixPath -> WindowsPath so torch.load can unpickle them (same shim the
# testers use). No-op on Linux/Colab. We only ever keep the tensor weights.
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc]

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from part_a_landmark.models import get_model  # noqa: E402
from part_b_segmentation.models import get_seg_model  # noqa: E402


class Pair(NamedTuple):
    """One (source checkpoint -> slim weights) job plus its architecture."""

    label: str
    source: Path
    slim: Path
    build: Callable[[], torch.nn.Module]


def make_pairs() -> List[Pair]:
    """Return the five hardcoded (source -> slim) pairs with their builders."""
    a_dir = REPO_ROOT / "part_a_landmark" / "weights"
    b_dir = REPO_ROOT / "part_b_segmentation" / "weights"
    return [
        Pair("part_a_coord", a_dir / "part_a_coord_best.pth",
             a_dir / "slim" / "part_a_coord.pth", lambda: get_model("coord")),
        Pair("part_a_unet", a_dir / "part_a_unet_best.pth",
             a_dir / "slim" / "part_a_unet.pth", lambda: get_model("unet")),
        Pair("part_a_scn", a_dir / "part_a_scn_best.pth",
             a_dir / "slim" / "part_a_scn.pth", lambda: get_model("scn")),
        Pair("part_b_unet", b_dir / "part_b_unet_best.pth",
             b_dir / "slim" / "part_b_unet.pth", lambda: get_seg_model("unet")),
        Pair("part_b_unet_attn", b_dir / "part_b_unet_attn_best.pth",
             b_dir / "slim" / "part_b_unet_attn.pth",
             lambda: get_seg_model("unet_attn")),
    ]


def size_mb(path: Path) -> float:
    """Return a file's size in mebibytes."""
    return path.stat().st_size / (1024.0 * 1024.0)


def slim_one(pair: Pair) -> None:
    """Extract the model weights from a full checkpoint and save them slim."""
    checkpoint = torch.load(pair.source, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        # Already a raw state_dict (e.g. re-running on an existing slim file).
        state_dict = checkpoint

    pair.slim.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, pair.slim)

    src_mb, slim_mb = size_mb(pair.source), size_mb(pair.slim)
    reduction = 100.0 * (1.0 - slim_mb / src_mb) if src_mb > 0 else 0.0
    print(f"  {pair.label:<16} {src_mb:8.2f} MB -> {slim_mb:7.2f} MB  "
          f"({reduction:5.1f}% smaller)")


def verify_one(pair: Pair) -> bool:
    """Reload the slim file weights-only and strict-load it into its model."""
    try:
        state_dict = torch.load(pair.slim, map_location="cpu", weights_only=True)
        model = pair.build()
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:  # noqa: BLE001 - surface the full load error
        print(f"  FAIL {pair.label}: {type(exc).__name__}: {exc}")
        return False
    print(f"  OK {pair.label} (strict load, {len(state_dict)} tensors)")
    return True


def main() -> None:
    """Slim every available checkpoint, then verify each slim file round-trips."""
    pairs = make_pairs()

    present = [p for p in pairs if p.source.is_file()]
    for pair in pairs:
        if pair.source not in (p.source for p in present):
            print(f"  WARNING: source missing, skipping: {pair.source}")

    print("Slimming checkpoints (model_state -> raw state_dict):")
    for pair in present:
        slim_one(pair)

    print("\nVerifying slim files (weights_only=True + strict load):")
    results = [verify_one(pair) for pair in present]

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} slim checkpoints verified.")
    if n_ok != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
