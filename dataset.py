"""Dataset utilities for lane segmentation from Ultralytics YOLO-seg polygon labels.

Supports loading polygon annotations specifically for lane segmentation (class 0),
rasterizing them to binary masks, applying photometric augmentations, and caching
resized samples in memory for ultra-fast training on CPU or local GPU.
"""
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(images_dir):
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return []
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def label_path_for_image(image_path, labels_dir):
    return Path(labels_dir) / f"{Path(image_path).stem}.txt"


def load_yolo_seg_polygons(label_path, img_w, img_h, target_class_id=0):
    """Parse a YOLO-seg label file into pixel-coordinate polygons.
    Filters specifically for target_class_id (default 0 for lane polygon).
    """
    label_path = Path(label_path)
    polygons = []
    if not label_path.exists():
        return polygons
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class_id + at least 3 (x, y) coordinate pairs
                continue
            class_id = int(float(parts[0]))
            if target_class_id is not None and class_id != target_class_id:
                continue
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= img_w
            coords[:, 1] *= img_h
            polygons.append(coords.astype(np.int32))
    return polygons


def polygons_to_mask(polygons, img_w, img_h):
    """Rasterize a list of polygon vertices onto a binary mask (0 or 1)."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if polygons:
        cv2.fillPoly(mask, polygons, color=1)
    return mask


def augment_image(image):
    """Slight photometric augmentation to mimic outdoor sunlight and sensor noise.

    Applies:
      1. White balance gain shift (per-channel scaling)
      2. Brightness shift (additive delta)
      3. Light Gaussian blur
    """
    img = image.astype(np.float32)

    # 1. White balance shift
    if random.random() < 0.5:
        gains = np.random.uniform(0.90, 1.10, size=3).astype(np.float32)
        img = img * gains[np.newaxis, np.newaxis, :]

    # 2. Brightness shift
    if random.random() < 0.5:
        delta = np.random.uniform(-25.0, 25.0)
        img = img + delta

    img = np.clip(img, 0, 255).astype(np.uint8)

    # 3. Gaussian blur
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


class LaneSegDataset(Dataset):
    """Lane segmentation dataset with polygon rasterization and in-memory sample caching."""

    def __init__(self, images_dir, labels_dir, img_size=48, augment=False, cache_in_memory=True):
        self.images = list_images(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.augment = augment
        self.cache_in_memory = cache_in_memory
        self.cached_samples = []

        if self.cache_in_memory and self.images:
            for image_path in self.images:
                img_rgb, mask = self._load_and_preprocess(image_path)
                self.cached_samples.append((img_rgb, mask))

    def _load_and_preprocess(self, image_path):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        label_path = label_path_for_image(image_path, self.labels_dir)
        polygons = load_yolo_seg_polygons(label_path, w, h, target_class_id=0)
        mask = polygons_to_mask(polygons, w, h)

        # Resize to network input/output resolution
        image_resized = cv2.resize(image_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        return image_resized, mask_resized

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.cache_in_memory and self.cached_samples:
            image, mask = self.cached_samples[idx]
            image = image.copy()
        else:
            image_path = self.images[idx]
            image, mask = self._load_and_preprocess(image_path)

        if self.augment:
            image = augment_image(image)

        image_t = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).contiguous()
        return image_t, mask_t
