# ARTIKATE — Senior Computer Vision / ML Engineer Take-Home Assignment
## Official Answers for Parts A4, B, C, and D

---

## Part A4 — Failure Analysis (Self-Captured Dataset)

The validation set performance was quantitatively evaluated across all 12 validation images by computing false negatives (FN), false positives (FP), class mismatches, and bounding box IoU relative to ground-truth annotations using an automated badness ranking script (`scripts/failure_analysis.py`).

Generated visual side-by-side comparison artifacts:
- `results/failure_cases/worst_1_b01_021.jpg`
- `results/failure_cases/worst_2_b01_056.jpg`
- `results/failure_cases/worst_3_b01_020.jpg`

### Failure Case 1: `b01_021.jpg` (Rank 1 — Highest Error)
- **Metrics**: GT Count: 13 · Pred Count: 19 · FN: 1 · FP: 7 · Class Mismatch: 1 · Mean IoU: 0.7392 · Badness Score: **16.2608**
- **What the model predicted**: 19 bounding boxes total (11 `cable`, 8 `device`). It predicted multiple overlapping sub-boxes along long coiled cables and misclassified a complex modular multi-port device into multiple duplicate boxes.
- **What it should have predicted**: Exactly 13 ground-truth bounding boxes (5 `device`, 8 `cable`).
- **Hypothesis**: In highly cluttered scenes with multiple overlapping cables and devices, standard axis-aligned bounding boxes (AABB) fail on non-convex elongated shapes like coiled cables. Standard NMS with IoU threshold 0.7 does not suppress nested predictions spanning different segments of the same cable bundle. Furthermore, dense multi-socket devices exhibit ambiguous visual boundaries that trigger duplicate overlapping predictions.
- **Actionable Next Step**:
  1. Add 15–20 high-density cluttered scene images with complex overlapping cables into the training set.
  2. Implement NMS IoU threshold tuning (lower IoU threshold from 0.7 to 0.45 during post-processing).
  3. Formulate strict annotation guidelines for modular devices (annotate outer enclosure only).

### Failure Case 2: `b01_056.jpg` (Rank 2)
- **Metrics**: GT Count: 11 · Pred Count: 14 · FN: 1 · FP: 4 · Class Mismatch: 1 · Mean IoU: 0.8276 · Badness Score: **11.6724**
- **What the model predicted**: 14 bounding boxes (6 `cable`, 8 `device`). High-confidence duplicate predictions on large power strips (0.947 and 0.932 confidence) where both the whole strip and individual sockets were detected, and 1 thin cable running near the edge was missed.
- **What it should have predicted**: 11 ground-truth bounding boxes (4 `device`, 7 `cable`).
- **Hypothesis**: Scale ambiguity and hierarchical granularity. Large electronic devices contain sub-features (ports, switches, buttons) that resemble standalone devices learned during pre-training on COCO. Additionally, thin cables positioned along image perimeters suffer from feature loss after spatial downsizing to $640 \times 640$.
- **Actionable Next Step**:
  1. Collect 10+ images focusing specifically on edge-cropped cables and wall-mounted devices under low side-lighting.
  2. Apply Random Crop and Mosaic data augmentation during fine-tuning to force the model to detect partial objects at image borders.

### Failure Case 3: `b01_020.jpg` (Rank 3)
- **Metrics**: GT Count: 10 · Pred Count: 14 · FN: 2 · FP: 4 · Class Mismatch: 0 · Mean IoU: 0.7717 · Badness Score: **11.2283**
- **What the model predicted**: 14 bounding boxes (8 `cable`, 6 `device`). It missed 2 small cable connector heads near the top edge of the image frame (FN=2) and produced 4 redundant overlapping boxes on thick coiled black cables.
- **What it should have predicted**: 10 ground-truth bounding boxes (4 `device`, 6 `cable`).
- **Hypothesis**: Small object detection degradation at low resolutions. Small cable connectors (occupying $< 2\%$ of image area) lose critical texture gradient details when resized to $640 \times 640$. The feature map stride of C3/C4 layers in YOLOv8 Nano suppresses feature activations for small truncated objects near frame borders.
- **Actionable Next Step**:
  1. Fine-tune at higher resolution ($800 \times 800$ or $1024 \times 1024$) or utilize P2 high-resolution head in YOLO architecture for small feature detection.
  2. Increase training emphasis on small connector tips with dedicated close-up annotations.

---

## Part B — Diagnose Three Broken Snippets

### Snippet 1 — Preprocessing and Coordinate Mapping

#### 1. All Distinct Defects Found
- **Defect 1 (Primary Defect: Missing Letterbox Padding Subtraction in Postprocessing)**: In `preprocess()`, letterbox padding `(dh, dw)` is added to center the resized image on the `(640, 640)` canvas. However, in `postprocess()`, predicted bounding box coordinates (which are relative to the padded canvas `[x1, y1, x2, y2]`) are directly divided by `r` (`boxes[:, [0, 2]] /= r`) **without first subtracting the letterbox padding offsets (`dw`, `dh`)**.
- **Defect 2 (Width vs Height Axis Mapping Discrepancy)**: In `postprocess()`, X-coordinates `[0, 2]` are clipped against `orig_shape[1]` (`w`) and Y-coordinates `[1, 3]` are clipped against `orig_shape[0]` (`h`). However, `dh` (height padding) and `dw` (width padding) are computed as `(size - nh) // 2` and `(size - nw) // 2`. If an asymmetric padding is applied, dividing by a single scale factor `r` without accounting for `dw` and `dh` shifts X and Y coordinates by different pixel amounts in original image space.
- **Defect 3 (Non-contiguous Strided Memory Layout)**: `canvas[:, :, ::-1].transpose(2, 0, 1)` produces a strided, non-contiguous NumPy array. Passing non-contiguous memory blocks to C++/ONNX Runtime inference bindings can lead to silent memory corruption or unexpected layout interpretation.
- **Defect 4 (Float-to-Int Rounding Discrepancy)**: `nh, nw = int(h * r), int(w * r)` truncates floating point scale products, whereas `(size - nh) // 2` uses integer floor division. This introduces a 1-pixel rounding asymmetry depending on whether `size - nh` is odd or even.

#### 2. Why it Survives Casual Testing & Specific Hiding Conditions
- **Why the offset is systematic rather than random**: The missing offset is `(dw / r, dh / r)`. Because `dw` and `dh` are constant for a given image, every single predicted box is shifted right by `dw / r` pixels and down by `dh / r` pixels.
- **Why it grows towards frame edges**: Bounding box coordinates near the frame edges undergo clipping (`clip(0, orig_shape)`). As un-padded box coordinates extend past the canvas region into the letterbox padding area, clipping truncates box boundaries unevenly, magnifying visible distortion at frame perimeters.
- **Why it is almost invisible on a square image**: On a perfectly square image (`h == w`), `nh == nw == size`. Consequently, `dh = (size - size) // 2 = 0` and `dw = (size - size) // 2 = 0`. With zero padding, `dw = 0` and `dh = 0`, so omitting `dw` and `dh` subtraction produces **zero numerical error**.

#### 3. Corrected Code
```python
import cv2
import numpy as np

def preprocess(img, size=640):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh))

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dh, dw = (size - nh) // 2, (size - nw) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized

    # Ensure contiguous C-order memory layout
    blob = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
    return blob[None], r, (dw, dh)

def postprocess(boxes, r, pad, orig_shape):
    dw, dh = pad
    boxes = boxes.copy()
    # 1. Subtract letterbox padding first
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    # 2. Scale back to original image resolution
    boxes[:, [0, 2]] /= r
    boxes[:, [1, 3]] /= r
    # 3. Clip to original image boundaries (w=orig_shape[1], h=orig_shape[0])
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_shape[0])
    return boxes
```

#### 4. The Test / Check That Would Have Caught It Before Shipping
- **Synthetic Synthetic Box Round-Trip Test**: Create a non-square test image ($1920 \times 1080$) with a known synthetic bounding box drawn at $(x_1, y_1, x_2, y_2) = (100, 100, 300, 300)$. Pass the image through `preprocess()`, simulate an exact detection on the canvas, pass through `postprocess()`, and assert `np.max(np.abs(recovered_box - original_box)) < 1.0` pixels.

---

### Snippet 2 — Dataset Preparation

#### 1. All Distinct Defects Found
- **Defect 1 (CRITICAL: Un-flipped Bounding Box Coordinates on Flipped Images)**: `augmented.append((cv2.flip(img, 1), labels))` applies a horizontal image flip, but **reuses the exact same `labels` without modifying bounding box coordinates**. For normalized YOLO boxes $[x_{\text{center}}, y_{\text{center}}, w, h]$, horizontal flipping requires $x_{\text{center, new}} = 1.0 - x_{\text{center}}$. Reusing un-flipped labels on a flipped image silently corrupts 1/3 of the training data with inverted target labels.
- **Defect 2 (CRITICAL: Data Leakage via Splitting After Augmentation)**: Augmentations are generated *before* train/validation splitting (`random.shuffle(augmented)` followed by `augmented[:split]`). Flipped and brightness-adjusted variants of the **exact same source images** appear in both the training set and validation set.
- **Defect 3 (Mutable Object Reference Sharing)**: The `labels` list/array object reference is appended directly into multiple tuples `(img, labels)`. If downstream code mutates `labels` in-place for one sample, it mutates the labels across all augmented copies.
- **Defect 4 (Missing `None` Checks & BGR Format Risk)**: `cv2.imread(path)` returns `None` if an image read fails, causing `cv2.flip` or `adjust_brightness` to throw an unhandled exception.

#### 2. Why it Survives Casual Testing & Specific Hiding Conditions
- **Why reported validation mAP looks healthy despite real-world collapse**: Data leakage (Defect 2) ensures the validation set contains near-identical copies of images in the training set. The model memorizes training images (including the corrupted labels from Defect 1) and scores $\sim 90\%+$ mAP on the leaked validation set.
- **Which single defect does the most damage to reported metric vs real data**: **Defect 2 (Data Leakage)** does the most damage to the validity of the reported metric (masking catastrophic failure), while **Defect 1 (Label Corruption)** causes total model failure on real production data.

#### 3. Corrected Code
```python
import glob
import random
import cv2
import numpy as np

def flip_labels_horizontal(labels):
    # Assumes normalized YOLO format: [class_id, x_center, y_center, w, h]
    flipped = []
    for item in labels:
        cls_id, xc, yc, w, h = item
        flipped.append([cls_id, 1.0 - xc, yc, w, h])
    return flipped

# 1. Split base image paths FIRST to prevent data leakage
raw_paths = sorted(glob.glob("dataset/images/*.jpg"))
random.seed(42)
random.shuffle(raw_paths)

split_idx = int(0.8 * len(raw_paths))
train_paths, val_paths = raw_paths[:split_idx], raw_paths[split_idx:]

def build_augmented_dataset(path_list, is_train=True):
    dataset = []
    for path in path_list:
        img = cv2.imread(path)
        if img is None:
            continue
        labels = load_labels(path)  # returns list of [cls, xc, yc, w, h]
        dataset.append((img, labels))
        
        if is_train:
            # Apply augmentation ONLY on training set with updated coordinates
            img_flipped = cv2.flip(img, 1)
            labels_flipped = flip_labels_horizontal(labels)
            dataset.append((img_flipped, labels_flipped))
            
            img_bright = adjust_brightness(img, 1.3)
            dataset.append((img_bright, [list(l) for l in labels]))
    return dataset

train_set = build_augmented_dataset(train_paths, is_train=True)
val_set = build_augmented_dataset(val_paths, is_train=False)

print("train samples:", len(train_set), "val samples:", len(val_set))
```

#### 4. The Test / Check That Would Have Caught It Before Shipping
- **Dataset Hash Leakage Check**: Assert `len(set(train_base_filenames).intersection(set(val_base_filenames))) == 0`.
- **Visual Annotation Overlay Check**: Render bounding box overlays on 5 random augmented images and verify visually that boxes align with objects on flipped images.

---

### Snippet 3 — IoU and Non-Maximum Suppression

#### 1. All Distinct Defects Found
- **Defect 1 (CRITICAL: Sign Cancellation Bug in Disjoint Box Intersection Area)**: In `iou()`:
  `inter = (x2 - x1) * (y2 - y1)`
  For two non-overlapping boxes, `x2 < x1` (making $x2 - x1 < 0$) AND `y2 < y1` (making $y2 - y1 < 0$). **Multiplying two negative numbers produces a positive number** (`inter > 0`). This calculates a non-zero, positive IoU for completely disjoint boxes!
- **Defect 2 (Incorrect Area Formula for `xyxy` Format)**: The docstring states boxes are in `xyxy` format (`[x1, y1, x2, y2]`). However, the code calculates `area1 = box[2] * box[3]` ($x2 \times y2$) and `area2 = boxes[:, 2] * boxes[:, 3]` ($x2_i \times y2_i$). This calculates area from origin $(0, 0)$ to $(x2, y2)$, distorting IoU for any box not anchored at $(0, 0)$.
- **Defect 3 (Unused `classes` Argument / Class-Agnostic Suppression)**: `nms()` accepts `classes`, but ignores it. High-confidence detections of Class A (e.g. `device`) suppress overlapping detections of Class B (e.g. `cable`).

#### 2. Why it Survives Casual Testing & Specific Hiding Conditions
- **Why a valid detection disappears**: Disjoint boxes in distant frame regions produce negative $x$ and $y$ deltas. The double negative multiplication yields a positive fake IoU $> 0.5$, causing NMS to suppress valid, non-overlapping detections.
- **Why it happens more often in sparse frames than crowded ones**: In sparse frames, objects are far apart in both X and Y dimensions simultaneously ($x2 - x1 < 0$ AND $y2 - y1 < 0$), causing the product to be positive. In crowded frames, boxes often overlap in one axis while disjoint in the other ($x2 - x1 > 0$ and $y2 - y1 < 0$), producing a negative product that evaluates to $\le 0$ when clipped or does not trigger false high IoU.
- **What the unused `classes` argument should have been doing**: NMS should perform class-aware suppression by offsetting box coordinates by class ID (`boxes_offset = boxes + classes[:, None] * 4096.0`) or grouping boxes by class before NMS.

#### 3. Corrected Code
```python
import numpy as np

def iou(box, boxes):
    # box: [x1, y1, x2, y2]
    # boxes: [N, 4]
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    # Fix Defect 1: Clamp width and height deltas at 0
    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    # Fix Defect 2: Correct xyxy box area calculation (x2 - x1) * (y2 - y1)
    area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

    union = area1 + area2 - inter
    return np.where(union > 0, inter / union, 0.0)

def nms(boxes, scores, classes, thr=0.5):
    if len(boxes) == 0:
        return []

    # Fix Defect 3: Perform class-aware NMS by offsetting coordinates per class
    max_coordinate = boxes.max() if boxes.size > 0 else 0
    offsets = classes.astype(boxes.dtype) * (max_coordinate + 1.0)
    boxes_offset = boxes + offsets[:, None]

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou(boxes_offset[i], boxes_offset[order[1:]])
        inds = np.where(ious <= thr)[0]
        order = order[inds + 1]

    return keep
```

#### 4. The Test / Check That Would Have Caught It Before Shipping
- **Disjoint Box IoU Unit Test**: Pass two completely disjoint boxes `boxA = [0, 0, 10, 10]` and `boxB = [100, 100, 110, 110]`. Assert `iou(boxA, boxB[None, :])[0] == 0.0`.
- **Multi-Class Co-located Box NMS Test**: Pass two overlapping boxes of different classes `classA` and `classB`. Assert `len(nms(boxes, scores, classes)) == 2`.

---

## Part C — Diagnose Three Production Failures

### C1. Accuracy Collapses After Quantisation (0.91 → 0.58 mAP@0.5)

#### Diagnosis & Elimination Log
1. **Root Cause 1: Dynamic INT8 Quantization Activation Saturation / Clipping Failure**:
   - *Hypothesis*: Dynamic INT8 quantization (`quantize_dynamic`) quantizes weights dynamically but computes runtime activation scale factors without calibration data. In CNN architectures, uncalibrated activation scales clip feature activation distributions, causing severe INT8 tensor overflow/saturation.
   - *Distinguishing Test*: Inspect intermediate tensor activation min/max values using ONNX Runtime Layer Inspector. If activation values hit extreme saturation bounds ($-128$ or $+127$), calibration clipping failure is confirmed.
2. **Root Cause 2: Input Preprocessing Tensor Scale Mismatch (0..255 vs 0..1)**:
   - *Hypothesis*: The FP32 PyTorch model expects float inputs in $[0.0, 1.0]$, but the INT8 TensorRT engine graph was built expecting `uint8` $[0, 255]$ without an internal division node by $255.0$.
   - *Distinguishing Test*: Pass a synthetic uniform float tensor ($0.5$) into both FP32 and INT8 engines and check raw logits. If INT8 outputs collapse to zeros/nans, verify tensor data type (`uint8` vs `float32`) and graph input scale parameters.
3. **Root Cause 3: Color Channel Permutation (RGB vs BGR) during ONNX/TensorRT Export**:
   - *Hypothesis*: The ONNX export pipeline accidentally inverted input color channels (RGB to BGR).
   - *Distinguishing Test*: Feed a pure Red test image (`R=255, G=0, B=0`) to both FP32 and INT8 models. Compare conv1 output feature maps. If INT8 matches FP32 only when input is fed as BGR, channel permutation is confirmed.

---

### C2. One Camera Out of Twelve is Wrong (Systematic Offset Towards Edges)

#### Diagnosis & Elimination Log
1. **What the Pattern Tells Us**:
   - The offset is **systematic, constant in direction, and grows non-linearly towards frame perimeters**.
   - Because 11 of 12 cameras running identical model software produce correct predictions, **the defect does NOT live in the ML model, NMS code, post-processing logic, or model weights**.
   - The defect lives in **Camera 12's stream ingest configuration, RTSP decoder padding, or physical lens optics**:
     - *Cause A*: Camera 12 has a different native sensor aspect ratio (e.g. 16:9 vs 4:3) and the ingest pipeline applies anamorphic stretching instead of letterboxed padding.
     - *Cause B*: Camera 12 has a wide-angle lens with uncorrected radial/barrel optical distortion.
2. **How to Confirm Without Physical Access**:
   - **Test 1 (RTSP Metadata Inspection)**: Run `ffprobe rtsp://camera12_feed` and compare native video resolution and codec profile against Camera 1.
   - **Test 2 (Grid Curvature Line Test)**: Extract 1 raw frame from Camera 12. Overlay a rectilinear grid and check if straight physical lines (e.g. conveyor belt rails) bend near edges (proving optical lens distortion).
   - **Test 3 (Offline Letterbox Verification)**: Save raw Camera 12 frames, run offline letterboxed inference, and check if predicted boxes match physical object positions.

---

### C3. Silent Degradation Over Three Months (97% → 84% Accuracy)

#### Diagnosis & Elimination Log
1. **Plausible Root Causes & Evidence**:
   - *Cause 1: Optical Environment Drift (Lens Dust/Smudge or LED Aging)*: Over 3 months, industrial dust accumulates on camera glass, or overhead LED lights degrade.
     - *Evidence*: Compute rolling Laplacian variance (sharpness score) and pixel intensity standard deviation on saved daily sample images. A steady monotonic drop in Laplacian variance confirms dirty lens or lighting decay.
   - *Cause 2: Product Packaging / Appearance Data Drift*: Suppliers updated packaging material (e.g. glossy finish or new print contrast).
     - *Evidence*: Analyze prediction confidence score distributions. A drop in mean prediction confidence from $0.92$ to $0.68$ on new SKU batches confirms data drift.
2. **Lightweight Monitoring Signal Design**:
   - **Signal 1: Rolling Prediction Confidence Index ($\mu_{\text{conf}}$)**: Calculate 7-day moving average of detection confidence across all bounding boxes.
     - *Trigger Threshold*: Fire alert if $\mu_{\text{conf}}$ drops by $> 5\%$ below baseline ($< 0.88$).
   - **Signal 2: Daily Laplacian Blur Index ($V_{\text{blur}}$)**: Sample 100 daily frames and compute Laplacian variance.
     - *Trigger Threshold*: Fire maintenance alert if $V_{\text{blur}}$ drops by $> 20\%$ relative to day 1 baseline.

---

## Part D — Edge and Air-Gapped Deployment

### 1. Aggregate Detection Throughput Calculation
- **Camera Configuration**: 8 fixed cameras at 1080p ($1920 \times 1080$), 15 fps each.
- **Aggregate Throughput Calculation**:
  $$\text{Aggregate Throughput} = 8 \text{ cameras} \times 15 \text{ fps} = \mathbf{120 \text{ frames per second (fps)}}$$
- **Frame Arrival Budget**: Each camera delivers 1 frame every $\frac{1000\text{ ms}}{15\text{ fps}} = 66.67\text{ ms}$.
- **Per-Frame Processing Budget**: To maintain real-time processing without queue accumulation, average processing time per frame must be $\le \frac{1000\text{ ms}}{120\text{ fps}} = \mathbf{8.33\text{ ms/frame}}$.

### 2. Model Family & Precision Selection
- **Model Selection**: **YOLOv8s** (or YOLOv8n) converted to **TensorRT FP16** running on NVIDIA Jetson AGX Orin (64 GB).
- **Latency & Throughput Arithmetic**:
  - Jetson AGX Orin (64GB) provides 275 INT8 TOPS / 137 FP16 TFLOPS.
  - Benchmark execution for YOLOv8s TensorRT FP16 at batch size $B=8$:
    - RTSP H.264/H.265 Hardware Decoding (NVDEC): $\sim 12\text{ ms}$
    - CUDA Preprocessing & Resizing: $\sim 4\text{ ms}$
    - TensorRT FP16 Inference ($B=8$): $\sim 32\text{ ms}$ ($\sim 4.0\text{ ms/frame}$)
    - CUDA Postprocessing & NMS: $\sim 8\text{ ms}$
    - **Total End-to-End Latency**: $\mathbf{\sim 56 \text{ ms per frame}}$, well within the required **200 ms per frame budget**.
- **First Measurement to Take**: Execute `trtexec --onnx=model.onnx --fp16 --batch=8` directly on the physical AGX Orin hardware to measure raw GPU kernel latency and memory bandwidth utilization.

### 3. Air-Gapped Retraining Loop Design
- **Step 1 (On-Line Operator Flagging)**: When a operator spots a false positive or missed defect on the line touch interface, tapping "Flag Error" saves the raw 1080p frame, bounding box JSON, and sensor metadata to an encrypted local edge SSD partition (`/var/data/flagged_samples/`).
- **Step 2 (Physical Data Transport)**: Monthly, an authorized security technician connects a hardware-encrypted USB drive. An automated export script validates cryptographically signed tokens and copies flagged image datasets to the USB drive.
- **Step 3 (Offline Workstation Retraining)**: The dataset is ingested into an air-gapped training server. Fine-tuning is executed. The new candidate model is evaluated against a golden test suite of 2,000+ historical factory images.
- **Step 4 (Air-Gapped Deployment)**: The compiled `.engine` TensorRT model file and SHA-256 manifest are saved to the encrypted USB drive, physically connected to the edge server, and deployed after checksum verification.

### 4. Rollback Plan & Regression Detection
- **Detection Signal**: Real-time tracking of operator error flag rate (operator overrides per 1,000 processed frames).
- **Rollback Trigger**: If operator error flags increase by $> 3\times$ baseline within 2 hours of deployment, an automated rollback fires.
- **Rollback Mechanism**: The edge server utilizes atomic dual-slot deployment (`slot_A` active, `slot_B` previous stable). Rollback executes an instant atomic symlink swap (`ln -sfn /models/slot_B /models/active`) and daemon reload, restoring stable operation in $< 5\text{ seconds}$.

### 5. Area of Least Confidence & Resolution Plan
- **Least Confident Area**: RTSP H.264/H.265 hardware video decoding (NVDEC) stability and PCIe/NVMM memory bus contention across 8 concurrent 1080p streams under heavy industrial network jitter.
- **Resolution Plan**: Benchmark 8-stream RTSP ingest using NVIDIA DeepStream SDK (`deepstream-app`) on the physical Jetson AGX Orin hardware with simulated network packet loss to measure decoder latency stability and drop rates.
