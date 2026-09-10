import os
import shutil
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Source paths
BASE_DIR = Path(__file__).parent.resolve()
RAW_IMAGES_DIR = BASE_DIR / 'data' / 'raw_images'
LABELS_DIR = BASE_DIR / 'data' / 'labels'

# Output dataset directory
DATASET_DIR = BASE_DIR / 'dataset'
IMAGES_TRAIN = DATASET_DIR / 'images' / 'train'
IMAGES_VAL = DATASET_DIR / 'images' / 'val'
LABELS_TRAIN = DATASET_DIR / 'labels' / 'train'
LABELS_VAL = DATASET_DIR / 'labels' / 'val'

# Split ratio
VAL_RATIO = 0.20  # 80% train, 20% val

def setup_directories():
    for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        d.mkdir(parents=True, exist_ok=True)

def main():
    setup_directories()
    
    # Collect all image files that have matching label files
    image_files = sorted(list(RAW_IMAGES_DIR.glob('*.jpg')) + list(RAW_IMAGES_DIR.glob('*.png')))
    
    valid_pairs = []
    skipped_no_label = []
    
    for img_path in image_files:
        lbl_path = LABELS_DIR / f"{img_path.stem}.txt"
        if lbl_path.exists():
            valid_pairs.append((img_path, lbl_path))
        else:
            skipped_no_label.append(img_path.name)
            
    print(f"Total images found: {len(image_files)}")
    print(f"Valid image-label pairs: {len(valid_pairs)}")
    if skipped_no_label:
        print(f"Skipped {len(skipped_no_label)} images with no label file: {skipped_no_label}")

    # Group by block prefix (e.g. b01)
    block_pairs = {}
    for img_path, lbl_path in valid_pairs:
        # Prefix before first underscore (e.g. 'b01')
        prefix = img_path.name.split('_')[0] if '_' in img_path.name else 'default'
        block_pairs.setdefault(prefix, []).append((img_path, lbl_path))
        
    train_count = 0
    val_count = 0
    
    for block_id, pairs in block_pairs.items():
        random.shuffle(pairs)
        n_val = max(1, int(len(pairs) * VAL_RATIO)) if len(pairs) >= 5 else int(len(pairs) * VAL_RATIO)
        
        val_set = pairs[:n_val]
        train_set = pairs[n_val:]
        
        for img_p, lbl_p in train_set:
            shutil.copy2(img_p, IMAGES_TRAIN / img_p.name)
            shutil.copy2(lbl_p, LABELS_TRAIN / lbl_p.name)
            train_count += 1
            
        for img_p, lbl_p in val_set:
            shutil.copy2(img_p, IMAGES_VAL / img_p.name)
            shutil.copy2(lbl_p, LABELS_VAL / lbl_p.name)
            val_count += 1

    print("\n--- Dataset Split Completed ---")
    print(f"Train split: {train_count} images & labels -> {IMAGES_TRAIN.relative_to(BASE_DIR)}")
    print(f"Val split:   {val_count} images & labels -> {IMAGES_VAL.relative_to(BASE_DIR)}")

    # Create dataset.yaml
    yaml_content = f"""path: {DATASET_DIR.as_posix()}
train: images/train
val: images/val

names:
  0: cable
  1: device
"""
    yaml_path = DATASET_DIR / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    print(f"\nGenerated YOLO dataset configuration file: {yaml_path.relative_to(BASE_DIR)}")

if __name__ == '__main__':
    main()
