"""Evaluate predicted lane masks against YOLO-seg ground truth via pixel-wise IoU.

Calculates:
  - Detection Flag (Yes/No): Frame is positive if IoU >= 0.6 (or user-defined threshold)
  - Detection Rate: Percentage of frames meeting the detection criteria
  - Average IoU of detected results: Mean IoU over positive detection frames
  - Average IoU of all images: Overall mean IoU across the entire test set

Outputs:
  - Formatted terminal report
  - JSON metrics file (summary and per-image results)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import list_images, load_yolo_seg_polygons, polygons_to_mask, label_path_for_image


def pixel_iou(pred_mask, gt_mask, eps=1e-6):
    pred = (pred_mask > 0)
    gt = (gt_mask > 0)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union + eps)


def main():
    parser = argparse.ArgumentParser(description="Evaluate lane masks via pixel-wise IoU.")
    parser.add_argument("--images-dir", default="./data/images/val", help="Original images dir.")
    parser.add_argument("--labels-dir", default="./data/labels/val", help="YOLO-seg polygon label dir.")
    parser.add_argument("--pred-masks-dir", default="run/val_exp/masks", help="Predicted mask PNGs directory.")
    parser.add_argument("--iou-threshold", type=float, default=0.6, help="Threshold for positive detection (default: 0.6).")
    parser.add_argument("--output-json", default="run/val_exp/metrics.json", help="Path to save evaluation JSON.")
    args = parser.parse_args()

    images = list_images(args.images_dir)
    if not images:
        print(f"No images found under {args.images_dir}")
        return

    results = []
    missing_preds = 0

    print(f"Evaluating {len(images)} validation images...")
    for image_path in images:
        pred_path = Path(args.pred_masks_dir) / f"{image_path.stem}.png"
        if not pred_path.exists():
            missing_preds += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]

        label_path = label_path_for_image(image_path, args.labels_dir)
        polygons = load_yolo_seg_polygons(label_path, w, h, target_class_id=0)
        gt_mask = polygons_to_mask(polygons, w, h)

        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred_mask is None:
            missing_preds += 1
            continue
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        iou = pixel_iou(pred_mask, gt_mask)
        detected = bool(iou >= args.iou_threshold)
        results.append({
            "image": image_path.name,
            "iou": round(iou, 4),
            "detected": detected
        })

    n = len(results)
    if n == 0:
        print("No predictions were evaluated.")
        return

    detected_results = [r for r in results if r["detected"]]
    detection_rate = len(detected_results) / n if n else 0.0
    avg_iou_detected = (sum(r["iou"] for r in detected_results) / len(detected_results)
                         if detected_results else 0.0)
    avg_iou_all = sum(r["iou"] for r in results) / n if n else 0.0

    summary = {
        "num_images_evaluated": n,
        "num_missing_predictions": missing_preds,
        "iou_threshold": args.iou_threshold,
        "num_detected": len(detected_results),
        "detection_rate_pct": round(detection_rate * 100.0, 2),
        "avg_iou_detected": round(avg_iou_detected, 4),
        "avg_iou_all": round(avg_iou_all, 4)
    }

    print("\n==============================================")
    print("      LANE SEGMENTATION EVALUATION REPORT      ")
    print("==============================================")
    print(f"Total Images Evaluated   : {n}")
    print(f"IoU Threshold (Positive) : >= {args.iou_threshold}")
    print(f"Positive Detections      : {len(detected_results)} / {n}")
    print(f"Detection Rate           : {detection_rate * 100.0:.2f}%")
    print(f"Average IoU (Detected)   : {avg_iou_detected:.4f} ({avg_iou_detected * 100.0:.2f}%)")
    print(f"Average IoU (All Images) : {avg_iou_all:.4f} ({avg_iou_all * 100.0:.2f}%)")
    print("==============================================")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "per_image": results}, f, indent=2)
        print(f"Saved evaluation metrics to {out_path}")


if __name__ == "__main__":
    main()
