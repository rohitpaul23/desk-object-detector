"""
scripts/benchmark.py

Benchmark FP32 (best.onnx) vs reduced-precision (best_int8.onnx / best_fp16.onnx)
on the same hardware and same val images.

Reports:
  - Latency: mean and p95 per image (50+ inferences, 5 warmup discarded)
  - Model file size
  - mAP@0.5 on val split for both models (via Ultralytics .val())
  - Markdown table ready to paste into README.md

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --fp32 models/best.onnx --quantized models/best_int8.onnx

Outputs:
    results/benchmark.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.resolve()
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
VAL_IMAGES_DIR = REPO_ROOT / "dataset" / "images" / "val"
DATA_YAML = REPO_ROOT / "data" / "data.yaml"

WARMUP_RUNS = 5
BENCH_RUNS = 50  # per image; 12 val images × 50 = 600 total inferences per model


def parse_args():
    # Auto-detect quantized model
    int8 = MODELS_DIR / "best_int8.onnx"
    fp16 = MODELS_DIR / "best_fp16.onnx"
    quant_default = str(int8) if int8.exists() else str(fp16)

    p = argparse.ArgumentParser(description="Benchmark FP32 vs quantized ONNX model")
    p.add_argument("--fp32", default=str(MODELS_DIR / "best.onnx"),
                   help="FP32 ONNX model path")
    p.add_argument("--quantized", default=quant_default,
                   help="Quantized ONNX model path (INT8 or FP16)")
    p.add_argument("--warmup", type=int, default=WARMUP_RUNS,
                   help=f"Warmup inferences to discard (default: {WARMUP_RUNS})")
    p.add_argument("--bench-runs", type=int, default=BENCH_RUNS,
                   help=f"Timed inferences per image (default: {BENCH_RUNS})")
    return p.parse_args()


def preprocess_image(img_path: Path, imgsz: int = 640) -> np.ndarray:
    import cv2
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(np.expand_dims(img, 0))


def benchmark_model(onnx_path: Path, val_images: list[Path],
                    warmup: int, bench_runs: int) -> dict:
    """
    Measure per-image inference latency for an ONNX model.
    Returns dict with mean_ms, p95_ms, and raw latencies.
    """
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = sess.get_inputs()[0].name

    # Preload images
    imgs = [preprocess_image(p) for p in val_images]

    # Warmup (discard)
    print(f"  Warming up ({warmup} runs)...")
    for i in range(warmup):
        sess.run(None, {input_name: imgs[i % len(imgs)]})

    # Timed benchmark
    latencies_ms = []
    print(f"  Benchmarking ({bench_runs} runs per image × {len(imgs)} images)...")
    for img in imgs:
        for _ in range(bench_runs):
            t0 = time.perf_counter()
            sess.run(None, {input_name: img})
            latencies_ms.append((time.perf_counter() - t0) * 1000)

    latencies_ms = np.array(latencies_ms)
    return {
        "mean_ms": round(float(np.mean(latencies_ms)), 3),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "min_ms": round(float(np.min(latencies_ms)), 3),
        "max_ms": round(float(np.max(latencies_ms)), 3),
        "total_inferences": len(latencies_ms),
    }


def get_map50_onnx(onnx_path: Path, precision_type: str) -> float:
    """
    Compute mAP@0.5 on the val split using Ultralytics YOLO with the ONNX model.
    Returns mAP50 float.
    """
    from ultralytics import YOLO
    print(f"  Running val accuracy check on {onnx_path.name}...")
    model = YOLO(str(onnx_path), task="detect")
    results = model.val(data=str(DATA_YAML), imgsz=640, split="val", verbose=False)
    return round(float(results.box.map50), 6)


def print_markdown_table(fp32_result: dict, quant_result: dict, quant_type: str):
    print("\n" + "=" * 72)
    print("  BENCHMARK TABLE (copy into README.md)")
    print("=" * 72)
    print()
    print(f"| Metric | FP32 (`best.onnx`) | {quant_type} (`best_{quant_type.lower()}.onnx`) |")
    print("|---|---|---|")
    print(f"| Latency mean (ms/img) | {fp32_result['latency']['mean_ms']} | {quant_result['latency']['mean_ms']} |")
    print(f"| Latency p95 (ms/img)  | {fp32_result['latency']['p95_ms']} | {quant_result['latency']['p95_ms']} |")
    print(f"| Model file size (MB)  | {fp32_result['size_mb']} | {quant_result['size_mb']} |")
    print(f"| mAP@0.5 (val)         | {fp32_result['map50']} | {quant_result['map50']} |")

    map_drop = round(fp32_result["map50"] - quant_result["map50"], 4)
    speedup = round(fp32_result["latency"]["mean_ms"] / quant_result["latency"]["mean_ms"], 2)
    size_ratio = round(fp32_result["size_mb"] / quant_result["size_mb"], 2)

    print()
    print(f"  Speedup (mean latency): {speedup}x")
    print(f"  Size reduction:         {size_ratio}x")
    print(f"  mAP@0.5 drop:           {map_drop:+.4f}")
    print("=" * 72)


def main():
    args = parse_args()
    fp32_path = Path(args.fp32)
    quant_path = Path(args.quantized)

    for p in [fp32_path, quant_path]:
        if not p.exists():
            print(f"[benchmark.py] ERROR: Model not found: {p}")
            return

    val_images = sorted(VAL_IMAGES_DIR.glob("*.jpg"))
    if not val_images:
        print("[benchmark.py] ERROR: No val images found.")
        return

    # Determine quantization type from filename
    quant_type = "INT8" if "int8" in quant_path.name else "FP16"

    # Detect hardware
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        hardware = "GPU (CUDA)" if "CUDAExecutionProvider" in providers else "CPU"
    except Exception:
        hardware = "unknown"

    print(f"\n[benchmark.py] Hardware   : {hardware}")
    print(f"[benchmark.py] FP32 model : {fp32_path.name}")
    print(f"[benchmark.py] Quant model: {quant_path.name} ({quant_type})")
    print(f"[benchmark.py] Val images : {len(val_images)}")
    print(f"[benchmark.py] Warmup     : {args.warmup} runs | Bench: {args.bench_runs} runs/image\n")

    # --- FP32 benchmark ---
    print(f"[benchmark.py] Benchmarking FP32 model...")
    fp32_latency = benchmark_model(fp32_path, val_images, args.warmup, args.bench_runs)
    fp32_size_mb = round(os.path.getsize(fp32_path) / 1e6, 3)
    print(f"  FP32 — mean: {fp32_latency['mean_ms']}ms  p95: {fp32_latency['p95_ms']}ms  size: {fp32_size_mb}MB")

    # --- Quantized benchmark ---
    print(f"\n[benchmark.py] Benchmarking {quant_type} model...")
    quant_latency = benchmark_model(quant_path, val_images, args.warmup, args.bench_runs)
    quant_size_mb = round(os.path.getsize(quant_path) / 1e6, 3)
    print(f"  {quant_type} — mean: {quant_latency['mean_ms']}ms  p95: {quant_latency['p95_ms']}ms  size: {quant_size_mb}MB")

    # --- mAP@0.5 accuracy for both ---
    print(f"\n[benchmark.py] Computing val accuracy for FP32 model...")
    fp32_map50 = get_map50_onnx(fp32_path, "FP32")
    print(f"  FP32 mAP@0.5: {fp32_map50}")

    print(f"\n[benchmark.py] Computing val accuracy for {quant_type} model...")
    quant_map50 = get_map50_onnx(quant_path, quant_type)
    print(f"  {quant_type} mAP@0.5: {quant_map50}")

    # --- Assemble results ---
    fp32_result = {
        "model": fp32_path.name,
        "latency": fp32_latency,
        "size_mb": fp32_size_mb,
        "map50": fp32_map50,
    }
    quant_result = {
        "model": quant_path.name,
        "type": quant_type,
        "latency": quant_latency,
        "size_mb": quant_size_mb,
        "map50": quant_map50,
    }

    benchmark_out = {
        "hardware": hardware,
        "warmup_runs_discarded": args.warmup,
        "bench_runs_per_image": args.bench_runs,
        "total_inferences_per_model": fp32_latency["total_inferences"],
        "fp32": fp32_result,
        "reduced_precision": quant_result,
        "map50_drop": round(fp32_map50 - quant_map50, 6),
        "mean_latency_speedup_x": round(fp32_latency["mean_ms"] / quant_latency["mean_ms"], 3),
        "size_reduction_x": round(fp32_size_mb / quant_size_mb, 3),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "benchmark.json"
    with open(out_path, "w") as f:
        json.dump(benchmark_out, f, indent=2)
    print(f"\n[benchmark.py] Results saved to: {out_path}")

    print_markdown_table(fp32_result, quant_result, quant_type)


if __name__ == "__main__":
    main()
