# Phase 3 — Export, quantize, benchmark

Covers A3 in the assignment brief: ONNX export + verification, a
reduced-precision version, and a three-way FP32 vs reduced-precision
benchmark (latency, size, accuracy).

## Prerequisites

- `models/best.pt` from Phase 2.
- `pip install onnx onnxruntime` (add `onnxruntime-gpu` instead if
  benchmarking on GPU; state explicitly in README which one was used).

## What to build

### 1. `scripts/export_onnx.py`

- Load `models/best.pt` via ultralytics (`YOLO(...).export(format="onnx",
  imgsz=640, opset=12)` or equivalent) → produces `models/best.onnx`.
- **Verification step (required, not optional):** run the same image
  through both the PyTorch model and the ONNX Runtime session, compare
  outputs numerically (e.g. max absolute difference on raw prediction
  tensors, or IoU + class agreement on final decoded boxes after
  postprocessing). Print the comparison result. This directly answers
  the brief's "say how you confirmed it — not just that you did" —
  a script that exports and never checks does not satisfy this.
- Save verification result to `results/onnx_verification.json`:
  ```json
  {
    "image_used": "data/val/images/<name>.jpg",
    "max_abs_diff": 0.0,
    "boxes_match": true,
    "method": "description of exactly how comparison was done"
  }
  ```

### 2. `scripts/quantize.py`

- Prefer INT8 if the toolchain supports static/dynamic quantization
  cleanly for this model (`onnxruntime.quantization.quantize_dynamic` or
  `quantize_static` with a small calibration set drawn from
  `data/train/images`). If INT8 proves unreliable or unsupported for
  this architecture in the time available, fall back to FP16
  (`onnxconverter_common.float16.convert_float_to_float16`) and **say so
  explicitly** in output — do not silently downgrade without logging why.
- Output: `models/best_quantized.onnx` (or `models/best_fp16.onnx`,
  name accordingly) plus a printed/saved note on which precision was
  used and why.

### 3. `scripts/benchmark.py`

- Load both `models/best.onnx` (FP32) and the quantized/FP16 model via
  ONNX Runtime.
- On the same hardware, same val images, measure per-image latency over
  enough repeated runs to get a stable estimate (e.g. 50+ inferences
  after a few warmup runs, discard warmup). Report **mean and p95**
  latency, not just mean — the brief explicitly asks for both.
- Compare file sizes (`os.path.getsize`).
- Re-run accuracy (mAP@0.5 or at minimum precision/recall) on the val
  set for both versions to quantify any accuracy drop — a number, not a
  description.
- Write `results/benchmark.json`:
  ```json
  {
    "hardware": "<fill in>",
    "fp32": {"latency_mean_ms": 0.0, "latency_p95_ms": 0.0, "size_mb": 0.0, "map50": 0.0},
    "reduced_precision": {"type": "int8|fp16", "latency_mean_ms": 0.0, "latency_p95_ms": 0.0, "size_mb": 0.0, "map50": 0.0}
  }
  ```
- Also print a markdown table version to stdout, formatted so it can be
  pasted directly into `README.md` under "Measured results".

## Acceptance criteria

- [ ] `models/best.onnx` and the reduced-precision model both exist
- [ ] `results/onnx_verification.json` shows a real numeric comparison,
      not a placeholder
- [ ] `results/benchmark.json` has real mean **and** p95 latency for
      both versions, real file sizes, real accuracy numbers
- [ ] The three-way trade-off table is pasted into `README.md`
- [ ] If accuracy dropped after quantization, the exact magnitude is
      stated in README — not glossed over as "similar performance"
