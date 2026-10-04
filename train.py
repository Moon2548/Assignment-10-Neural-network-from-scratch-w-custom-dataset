"""Train the custom UNet from scratch for single-class lane segmentation.

Features:
  - 48x48 input and output resolution for minimal memory footprint
  - BCE + Dice Loss for handling foreground/background imbalance
  - TensorBoard logging for loss and IoU metrics
  - Automatic loss curve visualization plot export (loss_curve.png)
  - Saves best and last checkpoint weights (.pt)
"""
import argparse
import os
from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import LaneSegDataset
from model import UNet


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combines BCE for pixel-level classification and Dice Loss for overlap optimization."""

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        return (self.bce_weight * self.bce(logits, targets)
                + (1.0 - self.bce_weight) * dice_loss(logits, targets))


@torch.no_grad()
def compute_iou(logits, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(logits) > threshold).float()
    preds = preds.reshape(preds.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def build_datasets(args):
    train_images = Path(args.data_root) / "images" / "train"
    train_labels = Path(args.data_root) / "labels" / "train"
    val_images = Path(args.data_root) / "images" / "val"
    val_labels = Path(args.data_root) / "labels" / "val"

    print("Loading training dataset into memory...")
    train_ds = LaneSegDataset(train_images, train_labels, args.img_size, augment=True, cache_in_memory=True)
    print("Loading validation dataset into memory...")
    val_ds = LaneSegDataset(val_images, val_labels, args.img_size, augment=False, cache_in_memory=True)
    return train_ds, val_ds


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_iou, n_batches = 0.0, 0.0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_iou += compute_iou(logits, masks)
            n_batches += 1

    n_batches = max(n_batches, 1)
    return total_loss / n_batches, total_iou / n_batches


def save_training_plot(history, output_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    plt.plot(epochs, history["val_loss"], "r-s", label="Val Loss", markersize=3)
    plt.title("Training and Validation Loss (BCE + Dice)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # IoU plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_iou"], "b-o", label="Train IoU", markersize=3)
    plt.plot(epochs, history["val_iou"], "g-s", label="Val IoU", markersize=3)
    plt.title("Training and Validation IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=200)
    plt.close()
    print(f"Saved training plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train custom UNet for lane segmentation.")
    parser.add_argument("--data-root", default="./data", help="Root dir containing images/ and labels/.")
    parser.add_argument("--img-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4, help="Recommended 2-4 for local machines.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--run-name", default="unet_lane")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds = build_datasets(args)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=False)

    model = UNet(in_channels=3, num_classes=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir) / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(args.log_dir) / args.run_name))

    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}
    best_val_loss = float("inf")
    best_val_iou = 0.0

    print(f"\n--- Starting Training ({args.epochs} epochs, batch size {args.batch_size}) ---")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_iou = run_epoch(model, val_loader, criterion, device, optimizer=None)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("IoU/train", train_iou, epoch)
        writer.add_scalar("IoU/val", val_iou, epoch)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} | "
              f"val_loss={val_loss:.4f} val_iou={val_iou:.4f}")

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou
        }, checkpoint_dir / "last.pt")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_iou = val_iou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_iou
            }, checkpoint_dir / "best.pt")

    writer.close()

    # Save visualization plot
    save_training_plot(history, checkpoint_dir / "loss_curve.png")
    save_training_plot(history, Path(args.log_dir) / args.run_name / "loss_curve.png")

    print("\n==========================================")
    print("Training Complete!")
    print(f"Best Val Loss: {best_val_loss:.4f} (IoU: {best_val_iou:.4f})")
    print(f"Best Checkpoint: {checkpoint_dir / 'best.pt'}")
    print(f"Last Checkpoint: {checkpoint_dir / 'last.pt'}")
    print("==========================================")


if __name__ == "__main__":
    main()
