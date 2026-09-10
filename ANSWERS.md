# ARTIKATE — Computer Vision Assignment Responses

## Part A4 — Failure Analysis

The validation set performance was quantitatively evaluated across all 12 validation images by computing false negatives (FN), false positives (FP), class mismatches, and bounding box IoU relative to ground-truth annotations. The three worst-performing images were identified using an automated badness ranking script (`scripts/failure_analysis.py`).

Generated visual side-by-side comparison artifacts:
- `results/failure_cases/worst_1_b01_021.jpg`
- `results/failure_cases/worst_2_b01_056.jpg`
- `results/failure_cases/worst_3_b01_020.jpg`

---

### Failure Case 1: `b01_021.jpg` (Rank 1 — Highest Error)

| Metric | Value |
|---|---|
| Ground Truth Count | 13 instances |
| Prediction Count | 19 instances |
| False Negatives (FN) | 1 |
| False Positives (FP) | 7 |
| Class Mismatches | 1 |
| Mean Matched IoU | 0.7392 |
| Badness Score | **16.2608** |

- **What the model predicted**: 19 bounding boxes total (11 `cable` and 8 `device`). It predicted multiple overlapping sub-boxes along the length of long coiled cables and misclassified a complex modular multi-port device as multiple distinct `device` predictions (e.g., confidence 0.909, 0.875, 0.792).
- **What it should have predicted**: Exactly 13 ground-truth bounding boxes (5 `device`, 8 `cable`).
- **Hypothesis**: In highly cluttered scenes with multiple overlapping cables and devices, standard axis-aligned bounding boxes (AABB) fail on non-convex elongated shapes like coiled cables. Standard NMS (Non-Maximum Suppression) with IoU threshold 0.7 does not suppress nested predictions spanning different segments of the same cable bundle. Furthermore, dense multi-socket devices exhibit ambiguous visual boundaries that trigger duplicate overlapping predictions.
- **Concrete Next Step**:
  1. Add 15–20 high-density cluttered scene images with complex overlapping cables into the training set.
  2. Implement NMS IoU threshold tuning (lower IoU threshold from 0.7 to 0.45 during post-processing).
  3. Formulate strict annotation guidelines for modular devices (annotate outer enclosure only).

---

### Failure Case 2: `b01_056.jpg` (Rank 2)

| Metric | Value |
|---|---|
| Ground Truth Count | 11 instances |
| Prediction Count | 14 instances |
| False Negatives (FN) | 1 |
| False Positives (FP) | 4 |
| Class Mismatches | 1 |
| Mean Matched IoU | 0.8276 |
| Badness Score | **11.6724** |

- **What the model predicted**: 14 bounding boxes (6 `cable`, 8 `device`). High-confidence duplicate predictions on large power strips (0.947 and 0.932 confidence) where both the whole strip and individual sockets were detected, and 1 thin cable running near the edge was missed.
- **What it should have predicted**: 11 ground-truth bounding boxes (4 `device`, 7 `cable`).
- **Hypothesis**: Scale ambiguity and hierarchical granularity. Large electronic devices contain sub-features (ports, switches, buttons) that resemble standalone devices learned during pre-training on COCO. Additionally, thin cables positioned along image perimeters suffer from feature loss after spatial downsizing to $640 \times 640$.
- **Concrete Next Step**:
  1. Collect 10+ images focusing specifically on edge-cropped cables and wall-mounted devices under low side-lighting.
  2. Apply Random Crop and Mosaic data augmentation during fine-tuning to force the model to detect partial objects at image borders.

---

### Failure Case 3: `b01_020.jpg` (Rank 3)

| Metric | Value |
|---|---|
| Ground Truth Count | 10 instances |
| Prediction Count | 14 instances |
| False Negatives (FN) | 2 |
| False Positives (FP) | 4 |
| Class Mismatches | 0 |
| Mean Matched IoU | 0.7717 |
| Badness Score | **11.2283** |

- **What the model predicted**: 14 bounding boxes (8 `cable`, 6 `device`). It missed 2 small cable connector heads near the top edge of the image frame (FN=2) and produced 4 redundant overlapping boxes on thick coiled black cables.
- **What it should have predicted**: 10 ground-truth bounding boxes (4 `device`, 6 `cable`).
- **Hypothesis**: Small object detection degradation at low resolutions. Small cable connectors (occupying $< 2\%$ of image area) lose critical texture gradient details when resized to $640 \times 640$. The feature map stride of C3/C4 layers in YOLOv8 Nano suppresses feature activations for small truncated objects near frame borders.
- **Concrete Next Step**:
  1. Fine-tune at higher resolution ($800 \times 800$ or $1024 \times 1024$) or utilize P2 high-resolution head in YOLO architecture for small feature detection.
  2. Increase training emphasis on small connector tips with dedicated close-up annotations.

---

## Part B — Architectural Choices & Trade-offs

### B1. Model Architecture Selection
- **Choice**: YOLOv8 Nano (`yolov8n.pt`, 3.15M parameters).
- **Rationale**: For real-time edge deployment on CPU/mobile devices, YOLOv8n strikes an optimal Pareto balance between latency ($< 55\text{ ms}$ CPU inference per frame) and detection mAP. Heavier backbones (e.g., YOLOv8m/l) improve mAP by 3–5 percentage points but increase CPU latency by $5\times$ to $12\times$, exceeding real-time processing constraints ($> 200\text{ ms}$).

### B2. Loss Function Formulation
- **Bounding Box Regression**: Combined CIoU (Complete IoU) loss + DFL (Distribution Focal Loss). CIoU penalizes box center distance, aspect ratio discrepancy, and overlap area simultaneously. DFL models box boundary locations as continuous probability distributions, significantly improving localization precision on flexible, non-rigid objects like cables.
- **Classification Loss**: Binary Cross-Entropy (BCE) with Sigmoidal focal activation.

### B3. Post-Processing & NMS
- **Thresholds**: Confidence threshold $\text{conf} = 0.25$, NMS IoU threshold $\text{iou} = 0.70$.
- **Trade-off**: High IoU threshold ($0.70$) prevents suppressing valid adjacent objects in dense arrangements but causes duplicate predictions on elongated/coiled cables (as observed in Failure Case 1). Lowering IoU to $0.45$ eliminates redundant predictions but risks suppressing tightly packed parallel cables.

---

## Part C — Real-World Deployment & Failure Modes

### C1. Domain Shift & Environmental Distribution Variations
In real-world production deployment (e.g., industrial inspections or consumer mobile apps), models encounter significant distribution shifts:
- **Lighting variations**: Outdoor sunlight, harsh shadows, or low-light environments degrade color histograms and edge gradients.
- **Background clutter**: Complex office desks, industrial floors, or patterned carpets introduce high-frequency background noise.
- **Camera hardware variations**: Lens distortion, motion blur, and varying sensor noise parameters from different smartphone cameras.
- **Mitigation Strategy**: Implement domain randomization in training pipelines (heavy HSV jitter, contrast augmentation, motion blur synthesis) and maintain an automated trigger to route low-confidence production predictions to human review for continuous fine-tuning.

### C2. Edge System Constraints & Hardware Latency
- **CPU Resource Limits**: On low-cost edge CPUs without AVX-512 / VNNI hardware extensions, INT8 matrix multiplication emulation overhead can render quantized models slower than FP32 implementations (as demonstrated in our benchmark: 1528 ms for INT8 vs 52.75 ms for FP32 on CPU without native INT8 vector execution units).
- **Memory Footprint**: FP32 model consumes 12.27 MB disk space and ~85 MB RAM runtime memory, making it highly suitable for constrained microcontrollers and mobile apps.

### C3. Monitoring, Telemetry, and Data Drift Detection
- **Prediction Drift**: Track moving average confidence distributions and class prediction ratios over time. A sudden drop in average prediction confidence indicates domain shift.
- **Input Data Sanity Checks**: Enforce automated validation pipelines that reject out-of-range brightness, corrupted frames, or extreme aspect ratio inputs before model inference.

---

## Part D — Reflection & Future Work

### D1. Scaling Dataset from 60 to 1,000 Images
- **Data Engine Strategy**:
  1. **Stratified Sampling**: Collect 500 images across diverse environmental buckets (indoor office, outdoor daylight, low light, high clutter, clean desk).
  2. **Active Learning Loop**: Deploy the current baseline model to auto-annotate unlabelled video streams. Retrieve only high-uncertainty frames ($0.30 \le \text{conf} \le 0.60$) for manual annotation verification, reducing human labeling effort by $70\%$.
  3. **Strict Annotation Protocols**: Define explicit rules for composite items (e.g., power strips vs individual plugs) to eliminate class hierarchy noise.

### D2. Synthetic Data & Generative AI Augmentation
- **Diffusion-Based Data Generation**: Utilize control-net / diffusion models to synthesize photo-realistic backgrounds behind segmented cable and device foreground assets.
- **3D Render Pipelines**: Render synthetic 3D CAD models of cables and electronic devices in Blender with randomized physics-based cable tangling and ray-traced lighting variations.

### D3. Architectural Extensions
- **Oriented Bounding Boxes (OBB)**: Replace standard axis-aligned bounding boxes (AABB) with OBB ($x, y, w, h, \theta$) or Instance Segmentation (YOLOv8-Seg) to accurately capture diagonal and curved cable trajectories without bounding box overlap contamination.
