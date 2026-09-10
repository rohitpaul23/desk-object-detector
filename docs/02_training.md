# Phase 2 — Train the detector

## Goal

Fine-tune a YOLO-family detector on `data/train/` and evaluate on
`data/val/`, producing honest metrics (precision, recall, mAP@0.5, and
mAP@0.5:0.95 if the framework provides it). This is A2 in the assignment
brief — do not skip reporting exact numbers, including mediocre ones.

## Prerequisites

- `data/train/images`, `data/train/labels`, `data/val/images`,
  `data/val/labels` populated (Phase 1 acceptance criteria met).
- `data/classes.txt` present.
- `ultralytics` installed (`pip install ultralytics`).

## What to build

### 1. `data/data.yaml`

Create a YOLO dataset config:

```yaml
path: ../data          # relative to this file's location, adjust as needed
train: train/images
val: val/images
names:
  0: cable
  1: device
```

Generate the `names` mapping by reading `data/classes.txt` — do not
hardcode it separately, to avoid drift from the source-of-truth file.

### 2. `scripts/train.py`

A CLI script using the `ultralytics` Python API (`from ultralytics import
YOLO`). Required arguments (use `argparse`):

| arg | default | purpose |
|---|---|---|
| `--model` | `yolov8n.pt` | base weights (n = nano, appropriate for a small dataset + fast local iteration; allow override to `yolov8s.pt` etc.) |
| `--data` | `data/data.yaml` | dataset config |
| `--epochs` | `100` | adjust down if overfitting on ~85 images is visible early |
| `--imgsz` | `640` | must match what `preprocess`/export later assume |
| `--batch` | `8` | small dataset, small batch is fine |
| `--lr0` | `0.01` | initial LR, ultralytics default is reasonable to start |
| `--project` | `results/train_runs` | where ultralytics writes run artifacts |
| `--name` | `run1` | run name, increment (`run2`, `run3`...) on retries — **do not overwrite prior runs**, the commit history needs to show the run that failed and the one that worked |

Behavior:
1. Load base weights, call `.train(...)` with the above args plus
   `augment=True` and light augmentation appropriate for a small dataset
   (`ultralytics` defaults are a reasonable starting point — avoid
   aggressive augmentations like heavy mosaic on only ~85 images unless
   validated to help).
2. After training, run `.val()` on the val split explicitly and capture:
   precision, recall, mAP50, mAP50-95.
3. Write these metrics to `results/metrics.json`:
   ```json
   {
     "model": "yolov8n.pt",
     "epochs": 100,
     "imgsz": 640,
     "batch": 8,
     "train_time_seconds": 0,
     "hardware": "<fill from actual run>",
     "precision": 0.0,
     "recall": 0.0,
     "map50": 0.0,
     "map50_95": 0.0
   }
   ```
4. Save best weights to `models/best.pt` (copy from ultralytics' own
   `runs/.../weights/best.pt` output).
5. Print a human-readable summary table to stdout at the end.

### 3. Sanity checks to run before trusting the numbers

- If `map50` comes back above ~0.95 on an 80-image self-captured
  dataset, **do not treat this as success** — it is the leakage red flag
  the assignment brief explicitly warns about. Re-check that
  `split_dataset.py` was actually run (not a stale/empty val split) and
  that no block appears in both `BLOCK_TO_SPLIT` train and val.
- If `cable` recall is dramatically worse than `device` recall, that's
  expected (thin-object class is harder) and should be reported honestly
  in the README, not tuned away by inflating epochs until it memorizes
  the small val set.

## Acceptance criteria

- [ ] `results/metrics.json` exists with real (non-placeholder) numbers
- [ ] `models/best.pt` exists
- [ ] Training command + exact args used are pasted into `README.md`
      → "How to reproduce" and "Measured results" → Training
- [ ] At least two commits exist for this phase: one for the first
      training attempt (even if the numbers are bad or the run
      mid-configured), one for the working version — per the brief's
      requirement that commit history show real iteration, not a
      squashed final state
