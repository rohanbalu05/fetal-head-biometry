"""CLI trainer for the two Part B segmentation models (unet / unet_attn).

Mirrors the Part A trainer's structure. A single loop trains either learned
segmenter with a combined Dice + BCE loss and, every epoch, reports the real
validation metrics: mean Dice (predictions thresholded at 0.5) and a simple
geometry sanity check (mean ``|major_len - minor_len|`` of the ellipse fit to the
predicted mask, in the 256 grid). Dice is the primary metric for checkpointing.

Label policy (standing rule): Part B trains only on images and the derived
FILLED cranium masks. Nothing here reads, swaps, or normalizes the CSV biometry
labels.

Checkpointing supports automatic resume (Colab-disconnect safe): every epoch
writes ``part_b_{model}_last.pth`` and the best-Dice epoch writes
``part_b_{model}_best.pth``.

Example::

    python part_b_segmentation/train_part_b.py --model unet --smoke
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

from common.dataset import SegmentationDataset  # noqa: E402
from common.metrics import dice as dice_score  # noqa: E402
from part_b_segmentation.geometry_decode import mask_to_biometry  # noqa: E402
from part_b_segmentation.models import get_seg_model  # noqa: E402

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "weights"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Part B segmentation model.")
    parser.add_argument("--model", choices=["unet", "unet_attn"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
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


def dice_loss(probs: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice loss (``1 - Dice``) on per-pixel probabilities in ``[0, 1]``."""
    dims = (1, 2, 3)
    numerator = 2.0 * (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (numerator + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def combined_loss(
    probs: torch.Tensor, target: torch.Tensor, bce: nn.Module
) -> torch.Tensor:
    """Combined loss = BCE + soft Dice (summed)."""
    return bce(probs, target) + dice_loss(probs, target)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch; return the average per-sample loss."""
    model.train()
    total_loss = 0.0
    total_n = 0
    for images, masks, _orig_wh, _names in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        probs = model(images)
        loss = combined_loss(probs, masks, bce)
        loss.backward()
        optimizer.step()
        bs = images.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    bce: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Return ``(val_loss, mean_dice, mean_abs_axis_diff)`` over the val set."""
    model.eval()
    total_loss = 0.0
    total_n = 0
    dices: List[float] = []
    axis_diffs: List[float] = []
    for images, masks, _orig_wh, _names in loader:
        images = images.to(device)
        probs = model(images)
        total_loss += float(combined_loss(probs, masks.to(device), bce).item()) * images.shape[0]
        total_n += images.shape[0]

        probs_np = probs.detach().cpu().numpy()
        masks_np = masks.numpy()
        for i in range(probs_np.shape[0]):
            pred = (probs_np[i, 0] > 0.5).astype(np.uint8)
            gt = (masks_np[i, 0] > 0.5).astype(np.uint8)
            dices.append(dice_score(pred, gt))
            bio = mask_to_biometry(pred)
            if bio is not None:
                axis_diffs.append(abs(float(bio["major_len"]) - float(bio["minor_len"])))

    val_loss = total_loss / max(total_n, 1)
    mean_dice = float(np.mean(dices)) if dices else 0.0
    mean_axis_diff = float(np.mean(axis_diffs)) if axis_diffs else float("nan")
    return val_loss, mean_dice, mean_axis_diff


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """Build train/val data loaders (subset in smoke mode)."""
    train_ds: torch.utils.data.Dataset = SegmentationDataset(
        split="train", augment=True, seed=args.seed
    )
    val_ds: torch.utils.data.Dataset = SegmentationDataset(
        split="val", augment=False, seed=args.seed
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
    fieldnames = ["epoch", "train_loss", "val_loss", "val_dice", "mean_axis_diff", "lr"]
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
          f"| lr: {args.lr} | smoke: {args.smoke}")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"part_b_{args.model}_best.pth"
    last_path = out_dir / f"part_b_{args.model}_last.pth"
    log_path = out_dir / f"part_b_{args.model}_log.csv"

    train_loader, val_loader = build_loaders(args)
    print(f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    model = get_seg_model(args.model).to(device)
    bce = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    start_epoch = 0
    best_dice = float("-inf")
    # Auto-resume (disabled in smoke mode so re-runs always exercise the loop).
    if not args.smoke and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_dice = float(ckpt.get("best_dice", float("-inf")))
        print(f"Resumed from {last_path} at epoch {start_epoch} "
              f"(best val Dice so far: {best_dice:.4f})")

    if start_epoch >= args.epochs:
        print("Nothing to do: start epoch >= requested epochs (training already complete).")
        return

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, bce, device)
        val_loss, val_dice, axis_diff = evaluate(model, val_loader, bce, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"epoch {epoch + 1}/{args.epochs} | train_loss {train_loss:.6f} | "
            f"val_loss {val_loss:.6f} | val_dice {val_dice:.4f} | "
            f"axis_diff {axis_diff:.2f}px | lr {lr:.2e}"
        )
        append_log(log_path, {
            "epoch": epoch + 1,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_dice": f"{val_dice:.4f}",
            "mean_axis_diff": f"{axis_diff:.4f}",
            "lr": f"{lr:.3e}",
        })

        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "val_dice": val_dice,
            "best_dice": max(best_dice, val_dice),
            "args": vars(args),
        }
        torch.save(ckpt, last_path)
        if val_dice > best_dice:
            best_dice = val_dice
            ckpt["best_dice"] = best_dice
            torch.save(ckpt, best_path)
            print(f"  new best val Dice {best_dice:.4f} -> saved {best_path.name}")

    print(f"Done. best val Dice: {best_dice:.4f}")
    print(f"checkpoints: {last_path.name}, {best_path.name} | log: {log_path.name}")


if __name__ == "__main__":
    main()
