"""Run lane segmentation inference, record memory footprint, and save binary masks & visual snapshots.

Outputs:
  - Binary masks saved to disk at native resolution (1280x720) under run/<run-name>/masks/
  - Snapshot visual comparisons (original, GT, predicted mask, overlay) under snapshots/
  - Memory footprint benchmark report saved to run/<run-name>/memory_footprint.json
"""
import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import psutil
import torch

from dataset import list_images, load_yolo_seg_polygons, polygons_to_mask, label_path_for_image
from model import UNet


def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def load_model(checkpoint_path, device):
    model = UNet(in_channels=3, num_classes=1)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess(image_rgb, img_size):
    resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return tensor


@torch.no_grad()
def predict_mask(model, image_path, img_size, device, threshold=0.5):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    tensor = preprocess(image_rgb, img_size).to(device)
    logits = model(tensor)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask_small = (prob > threshold).astype(np.uint8) * 255
    mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return image_bgr, mask_full, prob


def create_snapshot(image_bgr, gt_mask, pred_mask, output_path):
    """Generate a 4-panel visual comparison: Original, GT Overlay, Pred Mask, Pred Overlay."""
    h, w = image_bgr.shape[:2]

    # Ground truth overlay (Green)
    gt_overlay = image_bgr.copy()
    gt_color = np.zeros_like(image_bgr)
    gt_color[:, :, 1] = 255  # Green
    gt_bool = (gt_mask > 0)
    gt_overlay[gt_bool] = cv2.addWeighted(gt_overlay, 0.6, gt_color, 0.4, 0)[gt_bool]
    cv2.putText(gt_overlay, "Ground Truth", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Predicted mask overlay (Cyan / Sky Blue)
    pred_overlay = image_bgr.copy()
    pred_color = np.zeros_like(image_bgr)
    pred_color[:, :, 0] = 255  # Blue
    pred_color[:, :, 1] = 255  # Green
    pred_bool = (pred_mask > 0)
    pred_overlay[pred_bool] = cv2.addWeighted(pred_overlay, 0.6, pred_color, 0.4, 0)[pred_bool]
    cv2.putText(pred_overlay, "Prediction Overlay", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

    # Original text
    orig = image_bgr.copy()
    cv2.putText(orig, "Original Image", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Binary mask BGR
    mask_bgr = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(mask_bgr, "Predicted Binary Mask", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Combine into 2x2 grid
    row1 = np.hstack([orig, gt_overlay])
    row2 = np.hstack([mask_bgr, pred_overlay])
    grid = np.vstack([row1, row2])

    # Downscale grid slightly for easy viewing
    grid_view = cv2.resize(grid, (1280, 720), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_path), grid_view)


def main():
    parser = argparse.ArgumentParser(description="Lane segmentation inference, memory benchmarking & snapshots.")
    parser.add_argument("--images-dir", default="./data/images/val", help="Path to input images directory.")
    parser.add_argument("--labels-dir", default="./data/labels/val", help="Path to GT labels (for snapshot comparisons).")
    parser.add_argument("--checkpoint", default="checkpoints/unet_lane/best.pt")
    parser.add_argument("--img-size", type=int, default=48)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--run-name", default="val_exp")
    parser.add_argument("--output-root", default="run")
    parser.add_argument("--snapshots-dir", default="snapshots")
    parser.add_argument("--num-snapshots", type=int, default=5, help="Number of snapshot comparison images to generate.")
    args = parser.parse_args()

    mem_before = get_process_memory_mb()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"RAM before loading model: {mem_before:.2f} MB")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    model = load_model(checkpoint_path, device)
    mem_after_model = get_process_memory_mb()
    num_params = sum(p.numel() for p in model.parameters())

    print(f"Model parameters: {num_params:,}")
    print(f"Checkpoint file size: {model_file_size_mb:.2f} MB")
    print(f"RAM after loading model: {mem_after_model:.2f} MB (Delta: {mem_after_model - mem_before:.2f} MB)")

    out_masks_dir = Path(args.output_root) / args.run_name / "masks"
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    snapshots_dir = Path(args.snapshots_dir)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(args.images_dir)
    if not images:
        print(f"No images found under {args.images_dir}")
        return

    print(f"\nRunning inference on {len(images)} images...")
    inference_times = []
    peak_mem = mem_after_model

    # Warmup
    dummy = torch.randn(1, 3, args.img_size, args.img_size).to(device)
    for _ in range(5):
        _ = model(dummy)

    for idx, image_path in enumerate(images):
        t0 = time.perf_counter()
        image_bgr, pred_mask, prob = predict_mask(model, image_path, args.img_size, device, args.threshold)
        t1 = time.perf_counter()
        inference_times.append(t1 - t0)

        current_mem = get_process_memory_mb()
        if current_mem > peak_mem:
            peak_mem = current_mem

        # Save binary mask
        out_mask_path = out_masks_dir / f"{image_path.stem}.png"
        cv2.imwrite(str(out_mask_path), pred_mask)

        # Generate comparison snapshots for the first N images
        if idx < args.num_snapshots and Path(args.labels_dir).exists():
            h, w = image_bgr.shape[:2]
            lbl_path = label_path_for_image(image_path, args.labels_dir)
            polygons = load_yolo_seg_polygons(lbl_path, w, h, target_class_id=0)
            gt_mask = polygons_to_mask(polygons, w, h)

            snap_path = snapshots_dir / f"snapshot_{idx + 1:02d}_{image_path.stem}.jpg"
            create_snapshot(image_bgr, gt_mask, pred_mask, snap_path)

    avg_time_ms = (sum(inference_times) / len(inference_times)) * 1000.0
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0

    memory_report = {
        "device": str(device),
        "total_parameters": num_params,
        "checkpoint_file_size_mb": round(model_file_size_mb, 2),
        "ram_before_load_mb": round(mem_before, 2),
        "ram_model_loaded_mb": round(mem_after_model, 2),
        "ram_model_weights_delta_mb": round(mem_after_model - mem_before, 2),
        "peak_ram_during_inference_mb": round(peak_mem, 2),
        "average_inference_time_ms": round(avg_time_ms, 2),
        "throughput_fps": round(fps, 1),
        "input_resolution": f"{args.img_size}x{args.img_size}",
        "output_mask_resolution": "1280x720 (native)",
        "num_images_processed": len(images)
    }

    report_path = Path(args.output_root) / args.run_name / "memory_footprint.json"
    with open(report_path, "w") as f:
        json.dump(memory_report, f, indent=2)

    print("\n=== Inference & Memory Footprint Report ===")
    print(f"Device                     : {device}")
    print(f"Total Parameters           : {num_params:,}")
    print(f"Model Checkpoint Size      : {model_file_size_mb:.2f} MB")
    print(f"RAM Footprint (Model Load) : {mem_after_model - mem_before:.2f} MB")
    print(f"Peak RAM During Inference  : {peak_mem:.2f} MB")
    print(f"Average Latency per Frame  : {avg_time_ms:.2f} ms")
    print(f"Throughput                 : {fps:.1f} FPS")
    print(f"Binary masks exported to   : {out_masks_dir}")
    print(f"Snapshots saved to         : {snapshots_dir}")
    print(f"Detailed report saved to   : {report_path}")


if __name__ == "__main__":
    main()
