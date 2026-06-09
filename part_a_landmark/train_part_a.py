"""CLI trainer for the three Part A landmark models (coord / unet / scn).

A single training loop drives all three hypotheses, differing only in the loss
and the way predictions are decoded to pixel coordinates:

- ``coord`` : MSE on normalized ``(x, y)`` (8 values), targets = GT points / 256.
- ``unet``/``scn`` : MSE on heatmaps vs. ``coords_to_heatmaps`` targets.

After every epoch the *real* validation metric is computed in pixels by decoding
predictions back to four ``(x, y)`` points (argmax + sub-pixel for heatmaps,
denormalization for coord) and measuring mean radial error (MRE) and
success-rate at 5/10/20 px.

Label policy (standing rule): the four CSV points are used exactly as given, in
CSV order ``ofd_1, ofd_2, bpd_1, bpd_2``. Nothing here swaps or normalizes the
labels beyond the deterministic resize scaling done by the dataset.

Checkpointing supports automatic resume (e.g. after a Colab disconnect): every
epoch writes ``part_a_{model}_last.pth`` and the best-MRE epoch writes
``part_a_{model}_best.pth``.

Example::

    python part_a_landmark/train_part_a.py --model scn --smoke
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Make project packages importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from common.dataset import INPUT_HW, LandmarkDataset  # noqa: E402
from common.heatmaps import heatmaps_to_coords  # noqa: E402
from common.metrics import (  # noqa: E402
    mean_radial_error,
    per_point_error,
    success_rate_at_threshold,
)
from part_a_landmark.models import get_model  # noqa: E402

SR_THRESHOLDS: Tuple[int, ...] = (5, 10, 20)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "weights"
# Normalization scale for the coord model: (W, H) of the network grid.
_SCALE = np.array([INPUT_HW[1], INPUT_HW[0]], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Part A landmark model.")
    parser.add_argument("--model", choices=["coord", "unet", "scn"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma", type=float, default=4.0, help="heatmap sigma (px)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny end-to-end run (2 epochs, bs 2, small subset)")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_predictions(model_key: str, outputs: torch.Tensor) -> np.ndarray:
    """Decode raw model outputs to pixel coordinates ``(B, 4, 2)`` on the 256 grid.

    For ``coord`` the normalized outputs are denormalized; for heatmap models
    each channel is decoded via argmax + sub-pixel centroid refinement.
    """
    if model_key == "coord":
        arr = outputs.detach().cpu().numpy().reshape(-1, 4, 2)
        return arr * _SCALE
    heat = outputs.detach().cpu().numpy()  # (B, 4, H, W)
    points = np.zeros((heat.shape[0], 4, 2), dtype=np.float32)
    for i in range(heat.shape[0]):
        points[i] = heatmaps_to_coords(heat[i])
    return points


def compute_loss(
    model_key: str,
    outputs: torch.Tensor,
    heatmaps: torch.Tensor,
    points: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Return the training loss for one batch given the model type."""
    if model_key == "coord":
        scale = torch.tensor(_SCALE, device=device)
        target = (points.to(device) / scale).reshape(points.shape[0], -1)
        return criterion(outputs, target)
    return criterion(outputs, heatmaps.to(device))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_key: str,
) -> Tuple[float, Dict[int, float]]:
    """Compute validation MRE (px) and success rates at the SR thresholds."""
    model.eval()
    pred_list: List[np.ndarray] = []
    gt_list: List[np.ndarray] = []
    for images, heatmaps, points, _orig_wh, _names in loader:
        outputs = model(images.to(device))
        pred_list.append(decode_predictions(model_key, outputs))
        gt_list.append(points.numpy())

    preds = np.concatenate(pred_list, axis=0).reshape(-1, 2)
    gts = np.concatenate(gt_list, axis=0).reshape(-1, 2)
    mre = mean_radial_error(preds, gts)
    errors = per_point_error(preds, gts)
    sr = {t: success_rate_at_threshold(errors, t) for t in SR_THRESHOLDS}
    return mre, sr


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    model_key: str,
) -> float:
    """Run one training epoch; return the average per-sample loss."""
    model.train()
    total_loss = 0.0
    total_n = 0
    for images, heatmaps, points, _orig_wh, _names in loader:
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = compute_loss(model_key, outputs, heatmaps, points, criterion, device)
        loss.backward()
        optimizer.step()
        bs = images.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs
    return total_loss / max(total_n, 1)


def build_loaders(
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val data loaders (subset in smoke mode)."""
    train_ds: torch.utils.data.Dataset = LandmarkDataset(
        split="train", sigma=args.sigma, augment=True, seed=args.seed
    )
    val_ds: torch.utils.data.Dataset = LandmarkDataset(
        split="val", sigma=args.sigma, augment=False, seed=args.seed
    )
    if args.smoke:
        train_ds = Subset(train_ds, range(min(16, len(train_ds))))
        val_ds = Subset(val_ds, range(min(8, len(val_ds))))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=pin, drop_last=False,
    )
    return train_loader, val_loader


def append_log(log_path: Path, row: Dict[str, object]) -> None:
    """Append one epoch's metrics to the CSV log, writing a header if new."""
    fieldnames = ["epoch", "train_loss", "val_mre", "sr@5", "sr@10", "sr@20", "lr"]
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """Entry point: parse args, build everything, and run the training loop."""
    args = parse_args()
    if args.smoke:
        args.epochs = 2
        args.batch_size = 2

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model} | epochs: {args.epochs} | batch-size: {args.batch_size} "
          f"| lr: {args.lr} | sigma: {args.sigma} | smoke: {args.smoke}")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"part_a_{args.model}_best.pth"
    last_path = out_dir / f"part_a_{args.model}_last.pth"
    log_path = out_dir / f"part_a_{args.model}_log.csv"

    train_loader, val_loader = build_loaders(args)
    print(f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    model = get_model(args.model).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    start_epoch = 0
    best_mre = float("inf")
    # Auto-resume (disabled in smoke mode so re-runs always exercise the loop).
    if not args.smoke and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_mre = float(ckpt.get("best_mre", float("inf")))
        print(f"Resumed from {last_path} at epoch {start_epoch} "
              f"(best val MRE so far: {best_mre:.3f} px)")

    if start_epoch >= args.epochs:
        print("Nothing to do: start epoch >= requested epochs (training already complete).")
        return

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.model
        )
        val_mre, sr = evaluate(model, val_loader, device, args.model)
        scheduler.step(val_mre)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"epoch {epoch + 1}/{args.epochs} | train_loss {train_loss:.6f} | "
            f"val_mre {val_mre:.3f}px | "
            f"sr@5 {sr[5]:.1f}% sr@10 {sr[10]:.1f}% sr@20 {sr[20]:.1f}% | lr {lr:.2e}"
        )
        append_log(log_path, {
            "epoch": epoch + 1,
            "train_loss": f"{train_loss:.6f}",
            "val_mre": f"{val_mre:.4f}",
            "sr@5": f"{sr[5]:.2f}",
            "sr@10": f"{sr[10]:.2f}",
            "sr@20": f"{sr[20]:.2f}",
            "lr": f"{lr:.3e}",
        })

        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "val_mre": val_mre,
            "best_mre": min(best_mre, val_mre),
            "args": vars(args),
        }
        torch.save(ckpt, last_path)
        if val_mre < best_mre:
            best_mre = val_mre
            ckpt["best_mre"] = best_mre
            torch.save(ckpt, best_path)
            print(f"  new best val MRE {best_mre:.3f}px -> saved {best_path.name}")

    print(f"Done. best val MRE: {best_mre:.3f}px")
    print(f"checkpoints: {last_path.name}, {best_path.name} | log: {log_path.name}")


if __name__ == "__main__":
    main()
