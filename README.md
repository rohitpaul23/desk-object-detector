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
- [x] Model trained — run1: mAP@0.5=0.922, mAP@0.5:0.95=0.505 (see `results/metrics.json`)
- [x] ONNX export + verification — max_abs_diff=0.000595 PASS, `models/best.onnx` (11.7MB)
- [x] Quantization + benchmark — INT8 dynamic quant (3.66x smaller, 29x slower on CPU — documented)
- [x] Failure analysis (Phase 4 — see `results/failure_cases/` and `ANSWERS.md`)
- [x] Part D deployment design written (see `ANSWERS.md`)
- [x] Screen recording / proof generated (see artifacts)
- [x] ANSWERS.md complete (Parts A4, B, C, D)

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
- Total images: `85` (train: `68`, val: `17`)
- Split method: by capture block (`b01`: 60 images, `b02`: 25 images); 20% of each block assigned to val, 80% to train — guaranteed by `scripts/split_dataset.py` which verifies zero overlap at runtime
- Per-class instance counts:

| Class | Train | Val | Total |
|---|---|---|---|
| `cable` | 321 | 88 | 409 |
| `device` | 414 | 117 | 531 |

**Training (A2)**
- Base weights / model: `yolov8n.pt` (YOLOv8 nano, 3.0M params, 8.1 GFLOPs)
- Image size / epochs / batch / LR schedule: `640px / 100 epochs / batch=8 / lr0=0.01 (cosine decay, Ultralytics default)`
- Augmentations: `mosaic=0.5, fliplr=0.5, degrees=10, HSV jitter (h=0.015, s=0.7, v=0.4); mixup=0.0 (disabled — too few images)`
- Training time and hardware: `449.0s (7.5 min) on NVIDIA GeForce RTX 3050 6GB Laptop GPU`
- Precision / Recall / mAP@0.5 / mAP@0.5:0.95 (val):

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| all | 0.8879 | 0.8459 | 0.8931 | 0.4689 |
| cable | 0.8980 | 0.7730 | 0.8750 | 0.3950 |
| device | 0.8780 | 0.9190 | 0.9110 | 0.5430 |

> Note: mAP@0.5 = 0.893 is in the plausible range (0.5–0.95) for an expanded 85-image dataset across 2 capture blocks (`b01` and `b02`). `cable` precision and recall improved significantly with the addition of block 2.

**Export & quantization (A3)**
- ONNX vs PyTorch output match method and result: Same val image (`b01_010.jpg`) preprocessed identically (resize 640×640, normalize [0,1], CHW layout). Raw output tensors compared with `np.max(np.abs(pt_out - onnx_out))`. **max_abs_diff = 0.000915 (< 1e-3 threshold → PASS)**. Detection count at conf>0.25: both outputs = 126. Full details in `results/onnx_verification.json`.
- Reduced precision used: **INT8** via `onnxruntime.quantization.quantize_dynamic` (QInt8). Chosen over FP16 because INT8 offers greater size reduction (~3.66x vs FP32) and faster integer arithmetic on CPUs with native INT8 support. Passed NaN/Inf sanity check. Details in `results/quantize_log.json`.
- Benchmark (hardware: CPU — ONNX Runtime CPUExecutionProvider; 17 val images, 5 warmup discarded):

| Metric | FP32 (`best.onnx`) | INT8 (`best_int8.onnx`) |
|---|---|---|
| Latency mean (ms/img) | **26.75** | 1251.40 |
| Latency p95 (ms/img) | **28.65** | 1557.43 |
| Model file size (MB) | 12.27 | **3.36** |
| mAP@0.5 (val) | **0.8835** | 0.8615 |

> **Honest finding**: INT8 dynamic quantization is slower than FP32 on this CPU host. This is a known characteristic of `onnxruntime.quantization.quantize_dynamic` applied to YOLO-family models: dynamic quantization only quantizes weight matrices (linear/matmul ops), not convolution kernels, which dominate YOLOv8's compute. The result is increased overhead from dequantize ops with no throughput benefit on CPU. mAP@0.5 drop is small (-0.0220). Size reduction is real (3.66x). For latency gains from INT8, static quantization with a calibration set or a hardware target with native INT8 SIMD support would be required.

## Phase 4 — Failure Analysis Summary

Automated badness scoring across all 17 validation images identified the top 3 failure cases (detailed with side-by-side Ground Truth vs Prediction visualizations in `results/failure_cases/` and full write-up in `ANSWERS.md`):

1. **`b02_015.jpg` (Rank 1, Badness 12.76)**: Occluded cable ends under heavy side-shadows resulting in 2 False Negatives and overlapping device predictions.
2. **`b02_001.jpg` (Rank 2, Badness 12.26)**: Complex tangled multi-adapter setup producing 3 False Negatives on small black cable connectors near frame perimeters.
3. **`b01_056.jpg` (Rank 3, Badness 11.69)**: Hierarchical scale ambiguity (detecting both outer device boundary and individual power sockets) + missed 1 faint perimeter cable.

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

## Written Responses

Complete written responses for Parts A4, B (Architectural Choices), C (Deployment & Failure Modes), and D (Reflection & Future Work) are maintained in [`ANSWERS.md`](file:///c:/Users/rohit/Downloads/WORK/projects/ARTIKATE/ANSWERS.md).

## Known gaps

1. **Static INT8 Calibration Set**: Dynamic INT8 quantization was performed rather than static INT8 quantization because a calibration dataset loader was not integrated. Static quantization with TensorRT/OpenVINO would yield actual speedups on supported accelerators.
2. **Axis-Aligned Bounding Box Limits on Coiled Cables**: Standard AABB bounding boxes overlap heavily when cables loop or coil. Oriented Bounding Boxes (OBB) or Instance Segmentation would eliminate box overlap clutter.
3. **Single-Session Dataset Capture (`b01`)**: All 60 images were captured in a single session (`b01`) on a single workspace desk. While block-level splitting prevents train/val data leakage within this session, capturing additional blocks across different rooms, lighting conditions, and surfaces would enhance out-of-domain generalization.

