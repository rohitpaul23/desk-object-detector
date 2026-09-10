"""
scripts/train.py

Fine-tune a YOLOv8 detector on the cable/device dataset and write
results/metrics.json + models/best.pt.

Usage:
    python scripts/train.py                        # all defaults
    python scripts/train.py --model yolov8s.pt --epochs 150 --name run2
    python scripts/train.py --help

Hardware used: NVIDIA GeForce RTX 3050 6 GB (local GPU)
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths (all relative to the repo root, i.e. one level above this file)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.resolve()
DATA_YAML = REPO_ROOT / "data" / "data.yaml"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 on cable/device dataset")
    p.add_argument("--model", default="yolov8n.pt",
                   help="Base weights (default: yolov8n.pt). Use yolov8s.pt for larger model.")
    p.add_argument("--data", default=str(DATA_YAML),
                   help="Path to data.yaml (default: data/data.yaml)")
    p.add_argument("--epochs", type=int, default=100,
                   help="Training epochs (default: 100)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Input image size (default: 640)")
    p.add_argument("--batch", type=int, default=8,
                   help="Batch size (default: 8, suitable for RTX 3050 6GB)")
    p.add_argument("--lr0", type=float, default=0.01,
                   help="Initial learning rate (default: 0.01)")
    p.add_argument("--project", default=str(RESULTS_DIR / "train_runs"),
                   help="Directory for training run artifacts")
    p.add_argument("--name", default="run1",
                   help="Run name — increment on retries (run2, run3...) to preserve history")
    return p.parse_args()


def print_summary(metrics: dict):
    print("\n" + "=" * 55)
    print("  TRAINING SUMMARY")
    print("=" * 55)
    print(f"  Model            : {metrics['model']}")
    print(f"  Epochs           : {metrics['epochs']}")
    print(f"  Image size       : {metrics['imgsz']}")
    print(f"  Batch size       : {metrics['batch']}")
    print(f"  LR (initial)     : {metrics['lr0']}")
    print(f"  Hardware         : {metrics['hardware']}")
    print(f"  Training time    : {metrics['train_time_seconds']:.0f}s "
          f"({metrics['train_time_seconds']/60:.1f} min)")
    print("-" * 55)
    print(f"  Precision        : {metrics['precision']:.4f}")
    print(f"  Recall           : {metrics['recall']:.4f}")
    print(f"  mAP@0.5          : {metrics['map50']:.4f}")
    print(f"  mAP@0.5:0.95     : {metrics['map50_95']:.4f}")
    print("=" * 55)

    # Sanity check: warn if suspiciously high on a small dataset
    if metrics["map50"] > 0.95:
        print("\n  ⚠️  WARNING: mAP@0.5 > 0.95 on a ~60-image dataset.")
        print("     This is a leakage red flag. Verify that split_dataset.py")
        print("     was actually run and no block appears in both train and val.\n")


def main():
    args = parse_args()

    # Resolve hardware info
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            hardware = f"GPU: {gpu_name}"
        else:
            hardware = "CPU only"
    except Exception:
        hardware = "unknown"

    print(f"\n[train.py] Hardware detected: {hardware}")
    print(f"[train.py] Loading base weights: {args.model}")
    print(f"[train.py] Data config        : {args.data}")
    print(f"[train.py] Run output dir     : {args.project}/{args.name}\n")

    model = YOLO(args.model)

    # --- Train ---
    t_start = time.time()
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        project=args.project,
        name=args.name,
        exist_ok=False,        # fail loudly if run name already exists — don't silently overwrite
        augment=True,          # use Ultralytics default augmentation pipeline
        mosaic=0.5,            # reduce mosaic on small dataset (default 1.0 is too aggressive)
        mixup=0.0,             # no mixup — too few images
        degrees=10.0,          # light rotation augmentation
        fliplr=0.5,            # horizontal flip
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        verbose=True,
    )
    train_time = time.time() - t_start

    # --- Validate on val split ---
    print("\n[train.py] Running validation on val split...")
    val_results = model.val(data=args.data, imgsz=args.imgsz, split="val")

    # Extract metrics (Ultralytics returns a Results object)
    precision = float(val_results.box.mp)     # mean precision
    recall = float(val_results.box.mr)        # mean recall
    map50 = float(val_results.box.map50)      # mAP@0.5
    map50_95 = float(val_results.box.map)     # mAP@0.5:0.95

    metrics = {
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "lr0": args.lr0,
        "augmentations": "mosaic=0.5, mixup=0.0, fliplr=0.5, degrees=10, HSV jitter",
        "train_time_seconds": round(train_time, 1),
        "hardware": hardware,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "map50": round(map50, 6),
        "map50_95": round(map50_95, 6),
    }

    # --- Save metrics.json ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[train.py] Metrics saved to: {metrics_path}")

    # --- Copy best weights ---
    run_dir = Path(args.project) / args.name
    best_src = run_dir / "weights" / "best.pt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_dst = MODELS_DIR / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"[train.py] Best weights saved to: {best_dst}")
    else:
        print(f"[train.py] ⚠️  Could not find best.pt at {best_src}")

    print_summary(metrics)


if __name__ == "__main__":
    main()
