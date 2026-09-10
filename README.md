# Artikate CV Engineer Assessment — cable / device Detector

## Objective

Train a two-class object detector (`cable`, `device`) on a self-captured
image dataset from a real desk/workspace environment, covering variation
in lighting, camera angle, distance, background, and occlusion. Export
the trained model to ONNX, produce a reduced-precision version, benchmark
FP32 against it, and report honest, documented failure cases.

This repository is built incrementally and committed as work progresses —
see commit history for the real sequence (dataset → first training run →
failures → fixes → export → benchmark).

## Status

> Update this section as phases complete. Do not leave it stale before
> submission — the reviewer reads this first.

- [x] Repo structure created
- [x] Class definitions and annotation convention locked
- [x] `split_dataset.py` written (moved to `scripts/split_dataset.py`)
- [x] Dataset captured — 60 images, single block (b01), one session
- [x] Dataset annotated in LabelImg (YOLO format, `data/labels/`)
- [x] Train/val split executed — 48 train / 12 val (see Measured results below)
- [ ] Model trained (Phase 2 — see `docs/02_training.md`)
- [ ] ONNX export + verification (Phase 3 — see `docs/03_export_quantize_benchmark.md`)
- [ ] Quantization + benchmark (Phase 3)
- [ ] Failure analysis (Phase 4 — see `docs/04_failure_analysis.md`)
- [ ] Part D deployment design written
- [ ] Screen recording linked below
- [ ] ANSWERS.md complete (Parts B, C, D)

## Repository structure

```
artikate-cv-assignment/
├── data/
│   ├── raw_images/        # all captured .jpg, named <block>_<seq>.jpg
│   ├── labels/             # YOLO .txt labels, same basename as image
│   ├── classes.txt         # class list, index order = YOLO class id
│   ├── train/images, train/labels
│   └── val/images, val/labels
├── scripts/
│   ├── split_dataset.py    # done — splits by capture block, not shuffle
│   ├── train.py             # Phase 2
│   ├── export_onnx.py       # Phase 3
│   ├── quantize.py          # Phase 3
│   ├── benchmark.py         # Phase 3
│   └── failure_analysis.py  # Phase 4
├── docs/
│   ├── 01_dataset.md
│   ├── 02_training.md
│   ├── 03_export_quantize_benchmark.md
│   └── 04_failure_analysis.md
├── models/                  # trained weights, ONNX, quantized model
├── results/                  # metrics.json, benchmark tables, failure case images
├── README.md
└── ANSWERS.md                # Parts B, C, D written answers
```

## Classes

| id | name | definition |
|---|---|---|
| 0 | `cable` | Any wire/cord, including charging cables, earphone wires, USB cables. Boxed separately from any device/brick it connects to, even when overlapping. |
| 1 | `device` | Any discrete electronic item with a body/casing: phone, mouse, USB hub, USB drive, speaker, power bank, charging case, extension/power strip, charger adapter brick, router. |

**Excluded (never annotated):** non-electronic items — stationery, eyewear, watches, tools, tissue box, loose batteries, scissors, wall-mounted switchboards/fixtures.

## Annotation convention

See `docs/01_dataset.md` for the full ruleset and rationale. Summary:
one box per physically distinct cable, tight around its full visible
extent even when coiled/crossing; charger brick and its cable are two
separate instances of two different classes.

## Environment / setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install ultralytics onnx onnxruntime onnxruntime-tools numpy opencv-python matplotlib
```

Hardware used: Local laptop — NVIDIA GeForce RTX 3050 6 GB, Windows 11

## How to reproduce (fill in exact commands as each phase completes)

```bash
# 1. Split dataset by block into train/val
python scripts/split_dataset.py

# 2. Train (RTX 3050 6GB, ~15-20 min for 100 epochs at imgsz=640)
python scripts/train.py --model yolov8n.pt --epochs 100 --batch 8 --name run1

# 3. Export + verify ONNX
python scripts/export_onnx.py

# 4. Quantize (INT8 preferred, FP16 fallback)
python scripts/quantize.py

# 5. Benchmark FP32 vs reduced precision
python scripts/benchmark.py

# 6. Failure analysis
python scripts/failure_analysis.py
```

## Measured results

> Fill in after each phase. Do not round away decimals that matter —
> report exact numbers, including negative/embarrassing ones.

**Dataset**
- Total images: `60` (train: `48`, val: `12`)
- Split method: by capture block (all images are block `b01`); 20% of block assigned to val, 80% to train — guaranteed by `scripts/split_dataset.py` which verifies zero overlap at runtime
- Per-class instance counts:

| Class | Train | Val | Total |
|---|---|---|---|
| `cable` | 253 | 73 | 326 |
| `device` | 316 | 94 | 410 |

**Training (A2)**
- Base weights / model: `<>`
- Image size / epochs / batch / LR schedule: `<>`
- Training time and hardware: `<>`
- Precision / Recall / mAP@0.5 / mAP@0.5:0.95 (val): `<>`

**Export & quantization (A3)**
- ONNX vs PyTorch output match method and result: `<>`
- Reduced precision used (INT8 or FP16) and why: `<>`
- Benchmark table (latency mean/p95, file size, val accuracy — FP32 vs reduced): `<>`

## Assumptions

1. Train/val split is by **capture block**, not a random shuffle of
   individual images, because images within a block are near-duplicates
   (same setup, minor angle/distance changes) — shuffling would leak
   near-identical content across the split. See `scripts/split_dataset.py`.
2. Charger adapters/bricks are annotated as `device`; their cords are
   annotated as a separate `cable` instance, even when the two boxes
   overlap.
3. Non-electronic items visible in frame (stationery, eyewear, tools,
   etc.) are deliberately left unannotated as background clutter, not
   a labeling gap.
4. `<add more as decisions are made during training/export/quantization>`

## Known gaps

`<fill in honestly before submission — what you didn't get to and why>`

## Screen recording

`<link here>`
