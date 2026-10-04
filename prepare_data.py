"""Split PSU-reservoir dataset (Assignment 8) into 65% train and 35% validation/test splits.
"""
import os
import random
import shutil
from pathlib import Path


def prepare_dataset(
    src_root="d:/PSU/1-69/eco/rattachai/assignment8/assignment8/data/yolo_seg_dataset",
    dst_root="data",
    train_ratio=0.65,
    seed=42,
):
    src_images_dir = Path(src_root) / "images" / "train"
    src_labels_dir = Path(src_root) / "labels" / "train"

    dst_root = Path(dst_root)
    train_img_dir = dst_root / "images" / "train"
    val_img_dir = dst_root / "images" / "val"
    train_lbl_dir = dst_root / "labels" / "train"
    val_lbl_dir = dst_root / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    img_files = sorted(src_images_dir.glob("*.jpg"))
    if not img_files:
        raise RuntimeError(f"No images found in {src_images_dir}")

    total_images = len(img_files)
    print(f"Total source images found: {total_images}")

    # Reproducible shuffle
    rng = random.Random(seed)
    shuffled_indices = list(range(total_images))
    rng.shuffle(shuffled_indices)

    train_count = int(total_images * train_ratio)
    val_count = total_images - train_count

    train_indices = set(shuffled_indices[:train_count])
    print(f"Splitting into {train_count} train (65%) and {val_count} val (35%) frames...")

    for idx, img_path in enumerate(img_files):
        stem = img_path.stem
        lbl_path = src_labels_dir / f"{stem}.txt"

        if idx in train_indices:
            target_img_dir = train_img_dir
            target_lbl_dir = train_lbl_dir
        else:
            target_img_dir = val_img_dir
            target_lbl_dir = val_lbl_dir

        shutil.copy2(img_path, target_img_dir / img_path.name)
        if lbl_path.exists():
            shutil.copy2(lbl_path, target_lbl_dir / lbl_path.name)

    print("Dataset preparation complete:")
    print(f"  Train images: {len(list(train_img_dir.glob('*.*')))}")
    print(f"  Train labels: {len(list(train_lbl_dir.glob('*.txt')))}")
    print(f"  Val images  : {len(list(val_img_dir.glob('*.*')))}")
    print(f"  Val labels  : {len(list(val_lbl_dir.glob('*.txt')))}")


if __name__ == "__main__":
    prepare_dataset()
