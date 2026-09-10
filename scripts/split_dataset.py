"""
scripts/split_dataset.py

Splits annotated images + labels into train/val by CAPTURE BLOCK (not by
random per-image shuffle) to prevent near-duplicate leakage across the split.

Images within a block (same b01_, b02_... prefix) were shot in the same
session with minor angle/distance variation — shuffling them would put
near-identical frames into both train and val.

Usage:
    python scripts/split_dataset.py

Output:
    dataset/images/train/    dataset/images/val/
    dataset/labels/train/    dataset/labels/val/
    dataset/dataset.yaml

The per-class instance counts printed at the end should be pasted into
README.md -> Measured results -> Dataset.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Reproducible shuffle within blocks
random.seed(42)

REPO_ROOT = Path(__file__).parent.parent.resolve()
RAW_IMAGES_DIR = REPO_ROOT / "data" / "raw_images"
LABELS_DIR = REPO_ROOT / "data" / "labels"
CLASSES_TXT = REPO_ROOT / "data" / "classes.txt"

DATASET_DIR = REPO_ROOT / "dataset"
IMAGES_TRAIN = DATASET_DIR / "images" / "train"
IMAGES_VAL = DATASET_DIR / "images" / "val"
LABELS_TRAIN = DATASET_DIR / "labels" / "train"
LABELS_VAL = DATASET_DIR / "labels" / "val"

VAL_RATIO = 0.20  # 80/20 split within each block


def load_classes() -> list[str]:
    with open(CLASSES_TXT) as f:
        return [l.strip() for l in f if l.strip()]


def setup_dirs():
    for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        d.mkdir(parents=True, exist_ok=True)


def count_instances(label_files: list[Path], class_names: list[str]) -> dict:
    counts = defaultdict(int)
    for lf in label_files:
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
    return {class_names[i]: counts[i] for i in range(len(class_names))}


def main():
    class_names = load_classes()
    setup_dirs()

    # Collect valid image-label pairs
    image_files = sorted(
        list(RAW_IMAGES_DIR.glob("*.jpg")) + list(RAW_IMAGES_DIR.glob("*.png"))
    )

    valid_pairs = []
    skipped = []
    for img in image_files:
        lbl = LABELS_DIR / f"{img.stem}.txt"
        if lbl.exists():
            valid_pairs.append((img, lbl))
        else:
            skipped.append(img.name)

    if skipped:
        print(f"\n⚠️  {len(skipped)} images skipped (no matching label file):")
        for s in skipped:
            print(f"   {s}")

    # Group by block prefix (characters before first '_')
    blocks: dict[str, list] = defaultdict(list)
    for img, lbl in valid_pairs:
        prefix = img.stem.split("_")[0] if "_" in img.stem else "default"
        blocks[prefix].append((img, lbl))

    train_pairs = []
    val_pairs = []

    print(f"\n{'Block':<10} {'Total':>6} {'Train':>6} {'Val':>6}")
    print("-" * 32)

    for block_id in sorted(blocks):
        pairs = blocks[block_id]
        random.shuffle(pairs)
        n_val = max(1, round(len(pairs) * VAL_RATIO))
        v = pairs[:n_val]
        t = pairs[n_val:]
        val_pairs.extend(v)
        train_pairs.extend(t)
        print(f"{block_id:<10} {len(pairs):>6} {len(t):>6} {len(v):>6}")

    # Copy files
    for img, lbl in train_pairs:
        shutil.copy2(img, IMAGES_TRAIN / img.name)
        shutil.copy2(lbl, LABELS_TRAIN / lbl.name)

    for img, lbl in val_pairs:
        shutil.copy2(img, IMAGES_VAL / img.name)
        shutil.copy2(lbl, LABELS_VAL / lbl.name)

    # Per-class instance counts per split
    train_labels = [LABELS_TRAIN / p[1].name for p in train_pairs]
    val_labels = [LABELS_VAL / p[1].name for p in val_pairs]
    train_counts = count_instances(train_labels, class_names)
    val_counts = count_instances(val_labels, class_names)

    print(f"\n{'':=<50}")
    print("  SPLIT SUMMARY")
    print(f"{'':=<50}")
    print(f"  Total images : {len(valid_pairs)}  (train={len(train_pairs)}, val={len(val_pairs)})")
    print(f"\n  Per-class instance counts:")
    print(f"  {'Class':<12} {'Train':>8} {'Val':>8} {'Total':>8}")
    print(f"  {'':-<40}")
    for cls in class_names:
        tr = train_counts.get(cls, 0)
        va = val_counts.get(cls, 0)
        print(f"  {cls:<12} {tr:>8} {va:>8} {tr+va:>8}")
    print(f"{'':=<50}")
    print("  [OK] Paste the above table into README.md -> Measured results -> Dataset\n")

    # Verify no overlap
    train_names = {p[0].name for p in train_pairs}
    val_names = {p[0].name for p in val_pairs}
    overlap = train_names & val_names
    if overlap:
        print(f"  [FAIL] LEAKAGE DETECTED - {len(overlap)} images in both splits!")
        for x in sorted(overlap):
            print(f"     {x}")
    else:
        print("  [OK]  No image appears in both train and val splits.\n")

    # Write dataset.yaml (paths relative to repo root for portability)
    yaml_lines = [
        f"path: {DATASET_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "",
        "names:",
    ]
    for i, cls in enumerate(class_names):
        yaml_lines.append(f"  {i}: {cls}")
    yaml_lines.append("")

    yaml_path = DATASET_DIR / "dataset.yaml"
    yaml_path.write_text("\n".join(yaml_lines))
    print(f"  Dataset config: {yaml_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
