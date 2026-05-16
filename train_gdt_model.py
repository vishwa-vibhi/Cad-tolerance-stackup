"""
GD&T Symbol Detection — Model Training Script

Trains a YOLOv8n model on the downloaded Roboflow dataset.
Handles the case where only a 'train' folder exists (auto-splits 80/20).
Fixes relative paths in data.yaml to absolute paths.

Usage:
    python train_gdt_model.py
    python train_gdt_model.py --dataset "gd-t symbol detection.yolov8"
    python train_gdt_model.py --epochs 150 --model yolov8s.pt
"""

import os
import sys
import shutil
import random
import argparse
import yaml


# ============================================================
# Config
# ============================================================

DEFAULT_DATASET_DIR  = "gd-t symbol detection.yolov8"
OUTPUT_MODEL_PATH    = "models/gdt_yolov8.pt"
TRAIN_OUTPUT_DIR     = "runs/gdt_train"
PREPARED_DATASET_DIR = "gdt_dataset_prepared"

DEFAULT_BASE_MODEL = "yolov8n.pt"
DEFAULT_EPOCHS     = 100
DEFAULT_IMG_SIZE   = 640
DEFAULT_BATCH      = 8      # small batch — 101 images total
VAL_SPLIT          = 0.15   # 15% of train images → validation


# ============================================================
# Step 1: Prepare dataset (fix paths, create val split)
# ============================================================

def prepare_dataset(dataset_dir, output_dir=PREPARED_DATASET_DIR):
    """
    Prepare the dataset for training:
      - Creates train/valid splits if no valid folder exists
      - Writes a corrected data.yaml with absolute paths

    Returns path to the corrected data.yaml.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    output_dir  = os.path.abspath(output_dir)

    train_images = os.path.join(dataset_dir, "train", "images")
    train_labels = os.path.join(dataset_dir, "train", "labels")
    valid_images = os.path.join(dataset_dir, "valid", "images")

    print(f"\nPreparing dataset from: {dataset_dir}")

    # ── Check if valid split already exists ──────────────────────────────
    has_valid = os.path.isdir(valid_images) and len(os.listdir(valid_images)) > 0

    if has_valid:
        print(f"  Found existing valid split — using as-is")
        out_train_images = train_images
        out_valid_images = valid_images
    else:
        print(f"  No valid split found — creating 80/20 split from train set")
        out_train_images, out_valid_images = _create_split(
            train_images, train_labels, output_dir
        )

    # ── Read original data.yaml for class names ───────────────────────────
    orig_yaml = os.path.join(dataset_dir, "data.yaml")
    with open(orig_yaml, 'r') as f:
        orig_data = yaml.safe_load(f)

    nc    = orig_data.get("nc", 11)
    names = orig_data.get("names", [])

    # ── Write corrected data.yaml with absolute paths ─────────────────────
    os.makedirs(output_dir, exist_ok=True)
    new_yaml_path = os.path.join(output_dir, "data.yaml")

    new_data = {
        "train": out_train_images,
        "val":   out_valid_images,
        "nc":    nc,
        "names": names,
    }

    with open(new_yaml_path, 'w') as f:
        yaml.dump(new_data, f, default_flow_style=False, allow_unicode=True)

    print(f"  data.yaml written: {new_yaml_path}")
    print(f"  Classes ({nc}): {names}")
    print(f"  Train images: {out_train_images}")
    print(f"  Valid images: {out_valid_images}")

    return new_yaml_path


def _create_split(train_images_dir, train_labels_dir, output_dir):
    """
    Split train images 80/20 into new train/valid folders under output_dir.
    Returns (new_train_images_path, new_valid_images_path).
    """
    all_images = sorted([
        f for f in os.listdir(train_images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    random.seed(42)
    random.shuffle(all_images)

    n_val   = max(1, int(len(all_images) * VAL_SPLIT))
    val_set = set(all_images[:n_val])
    trn_set = set(all_images[n_val:])

    print(f"  Total images: {len(all_images)} → train: {len(trn_set)}, val: {len(val_set)}")

    # Create output dirs
    new_train_img = os.path.join(output_dir, "train", "images")
    new_train_lbl = os.path.join(output_dir, "train", "labels")
    new_valid_img = os.path.join(output_dir, "valid", "images")
    new_valid_lbl = os.path.join(output_dir, "valid", "labels")

    for d in [new_train_img, new_train_lbl, new_valid_img, new_valid_lbl]:
        os.makedirs(d, exist_ok=True)

    def copy_pair(img_name, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
        stem = os.path.splitext(img_name)[0]
        # Copy image
        shutil.copy2(
            os.path.join(src_img_dir, img_name),
            os.path.join(dst_img_dir, img_name)
        )
        # Copy label if it exists
        lbl_name = stem + ".txt"
        lbl_src  = os.path.join(src_lbl_dir, lbl_name)
        if os.path.exists(lbl_src):
            shutil.copy2(lbl_src, os.path.join(dst_lbl_dir, lbl_name))

    for img in trn_set:
        copy_pair(img, train_images_dir, train_labels_dir, new_train_img, new_train_lbl)
    for img in val_set:
        copy_pair(img, train_images_dir, train_labels_dir, new_valid_img, new_valid_lbl)

    return new_train_img, new_valid_img


# ============================================================
# Step 2: Train
# ============================================================

def train(data_yaml, base_model=DEFAULT_BASE_MODEL, epochs=DEFAULT_EPOCHS,
          img_size=DEFAULT_IMG_SIZE, batch=DEFAULT_BATCH):
    """Train YOLOv8 on the prepared dataset."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics not installed. Run:\n"
            "  pip install ultralytics"
        )

    print(f"\n{'='*60}")
    print(f"Training YOLOv8 GD&T Detector")
    print(f"  Base model:  {base_model}")
    print(f"  data.yaml:   {data_yaml}")
    print(f"  Epochs:      {epochs}")
    print(f"  Image size:  {img_size}")
    print(f"  Batch size:  {batch}")
    print(f"{'='*60}\n")

    model = YOLO(base_model)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        project=TRAIN_OUTPUT_DIR,
        name="gdt",
        exist_ok=True,
        # Augmentation tuned for small engineering drawing dataset
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.2,
        degrees=3.0,       # small rotation only
        translate=0.05,
        scale=0.3,
        flipud=0.0,        # GD&T symbols are orientation-sensitive
        fliplr=0.2,
        mosaic=0.5,
        mixup=0.0,
        # Training settings
        patience=25,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    best_weights = os.path.join(TRAIN_OUTPUT_DIR, "gdt", "weights", "best.pt")
    if not os.path.exists(best_weights):
        # Search for it
        for root, dirs, files in os.walk(TRAIN_OUTPUT_DIR):
            for f in files:
                if f == "best.pt":
                    best_weights = os.path.join(root, f)
                    break

    return best_weights


# ============================================================
# Step 3: Save model to standard location
# ============================================================

def save_model(best_weights, output_path=OUTPUT_MODEL_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copy2(best_weights, output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nModel saved → {output_path}  ({size_mb:.1f} MB)")
    return output_path


# ============================================================
# Step 4: Validate
# ============================================================

def validate_model(model_path, data_yaml):
    try:
        from ultralytics import YOLO
    except ImportError:
        return

    print(f"\nValidating: {model_path}")
    model   = YOLO(model_path)
    metrics = model.val(data=data_yaml, verbose=False)

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    print(f"{'='*60}")
    print(f"\nTarget: mAP50 ≥ 0.70 for good GD&T detection performance")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train GD&T YOLO model")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_DIR,
                        help=f"Path to Roboflow dataset folder (default: '{DEFAULT_DATASET_DIR}')")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL,
                        help=f"YOLOv8 base model (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help=f"Batch size (default: {DEFAULT_BATCH})")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE,
                        help=f"Image size (default: {DEFAULT_IMG_SIZE})")
    parser.add_argument("--validate-only", action="store_true",
                        help="Skip training, only validate existing model")
    args = parser.parse_args()

    # ── Validate-only mode ────────────────────────────────────────────────
    if args.validate_only:
        if not os.path.exists(OUTPUT_MODEL_PATH):
            print(f"No model found at {OUTPUT_MODEL_PATH}. Train first.")
            sys.exit(1)
        data_yaml = os.path.join(PREPARED_DATASET_DIR, "data.yaml")
        if not os.path.exists(data_yaml):
            data_yaml = prepare_dataset(args.dataset)
        validate_model(OUTPUT_MODEL_PATH, data_yaml)
        return

    # ── Check dataset exists ──────────────────────────────────────────────
    if not os.path.isdir(args.dataset):
        print(f"ERROR: Dataset folder not found: '{args.dataset}'")
        print(f"Expected folder: '{DEFAULT_DATASET_DIR}'")
        sys.exit(1)

    # ── Step 1: Prepare ───────────────────────────────────────────────────
    data_yaml = prepare_dataset(args.dataset)

    # ── Step 2: Train ─────────────────────────────────────────────────────
    best_weights = train(
        data_yaml  = data_yaml,
        base_model = args.model,
        epochs     = args.epochs,
        img_size   = args.imgsz,
        batch      = args.batch,
    )

    if not os.path.exists(best_weights):
        print(f"ERROR: Training finished but best.pt not found at: {best_weights}")
        sys.exit(1)

    # ── Step 3: Save ──────────────────────────────────────────────────────
    save_model(best_weights, OUTPUT_MODEL_PATH)

    # ── Step 4: Validate ──────────────────────────────────────────────────
    validate_model(OUTPUT_MODEL_PATH, data_yaml)

    print(f"\n{'='*60}")
    print("DONE — next steps:")
    print(f"  1. Model is at: {OUTPUT_MODEL_PATH}")
    print(f"  2. Run the pipeline normally — GD&T detection is automatic")
    print(f"     python src/pipeline.py data/category_1/cad1_001.png results/test")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
