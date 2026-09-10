"""
scripts/quantize.py

Produce a reduced-precision version of models/best.onnx.

Strategy:
  1. Try INT8 dynamic quantization via onnxruntime.quantization.quantize_dynamic
  2. If INT8 fails or produces NaN outputs on a sanity check → fall back to FP16
     via onnxconverter_common.float16.convert_float_to_float16
  3. Log explicitly which path was taken and why.

Usage:
    python scripts/quantize.py
    python scripts/quantize.py --onnx models/best.onnx

Outputs:
    models/best_int8.onnx   (preferred)
    OR
    models/best_fp16.onnx   (fallback)
    results/quantize_log.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.resolve()
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
VAL_IMAGES_DIR = REPO_ROOT / "dataset" / "images" / "val"


def parse_args():
    p = argparse.ArgumentParser(description="Quantize best.onnx to INT8 or FP16")
    p.add_argument("--onnx", default=str(MODELS_DIR / "best.onnx"),
                   help="Source FP32 ONNX model (default: models/best.onnx)")
    return p.parse_args()


def preprocess_image(img_path: Path, imgsz: int = 640) -> np.ndarray:
    import cv2
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(np.expand_dims(img, 0))


def sanity_check_onnx(onnx_path: Path, test_image: Path) -> tuple[bool, str]:
    """
    Run one inference through the quantized model.
    Returns (ok, reason) — ok=False if NaN/Inf present in output.
    """
    import onnxruntime as ort
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        img = preprocess_image(test_image)
        input_name = sess.get_inputs()[0].name
        # INT8 sessions expect float32 input still
        outputs = sess.run(None, {input_name: img})
        out = outputs[0]
        if np.any(np.isnan(out)) or np.any(np.isinf(out)):
            return False, "Output contains NaN or Inf values"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def try_int8(onnx_src: Path, out_path: Path, test_image: Path) -> tuple[bool, str]:
    """Attempt INT8 dynamic quantization. Returns (success, reason)."""
    print("[quantize.py] Attempting INT8 dynamic quantization...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            model_input=str(onnx_src),
            model_output=str(out_path),
            weight_type=QuantType.QInt8,
        )
        ok, reason = sanity_check_onnx(out_path, test_image)
        if not ok:
            out_path.unlink(missing_ok=True)
            return False, f"INT8 sanity check failed: {reason}"
        print(f"[quantize.py] INT8 quantization succeeded. Output: {out_path}")
        return True, "INT8 dynamic quantization via onnxruntime.quantization.quantize_dynamic"
    except Exception as e:
        if out_path.exists():
            out_path.unlink()
        return False, f"INT8 failed with exception: {e}"


def try_fp16(onnx_src: Path, out_path: Path, test_image: Path) -> tuple[bool, str]:
    """Attempt FP16 conversion. Returns (success, reason)."""
    print("[quantize.py] Attempting FP16 conversion via onnxconverter_common...")
    try:
        import onnx
        from onnxconverter_common.float16 import convert_float_to_float16
        model = onnx.load(str(onnx_src))
        fp16_model = convert_float_to_float16(model, keep_io_types=True)
        onnx.save(fp16_model, str(out_path))
        # FP16 model still accepts float32 input when keep_io_types=True
        ok, reason = sanity_check_onnx(out_path, test_image)
        if not ok:
            out_path.unlink(missing_ok=True)
            return False, f"FP16 sanity check failed: {reason}"
        print(f"[quantize.py] FP16 conversion succeeded. Output: {out_path}")
        return True, "FP16 conversion via onnxconverter_common.float16.convert_float_to_float16 (keep_io_types=True)"
    except ImportError:
        return False, "onnxconverter_common not installed — run: pip install onnxconverter-common"
    except Exception as e:
        if out_path.exists():
            out_path.unlink()
        return False, f"FP16 failed with exception: {e}"


def main():
    args = parse_args()
    onnx_src = Path(args.onnx)

    if not onnx_src.exists():
        print(f"[quantize.py] ERROR: Source ONNX not found: {onnx_src}")
        print("              Run scripts/export_onnx.py first.")
        return

    val_images = sorted(VAL_IMAGES_DIR.glob("*.jpg"))
    if not val_images:
        print("[quantize.py] ERROR: No val images found.")
        return
    test_image = val_images[0]

    print(f"\n[quantize.py] Source model : {onnx_src}")
    print(f"[quantize.py] Sanity image : {test_image.name}")

    log = {
        "source_onnx": str(onnx_src),
        "int8_attempted": False,
        "int8_success": False,
        "int8_reason": "",
        "fp16_attempted": False,
        "fp16_success": False,
        "fp16_reason": "",
        "chosen_precision": None,
        "output_model": None,
        "decision_rationale": "",
    }

    # --- Try INT8 first ---
    int8_out = MODELS_DIR / "best_int8.onnx"
    log["int8_attempted"] = True
    int8_ok, int8_reason = try_int8(onnx_src, int8_out, test_image)
    log["int8_success"] = int8_ok
    log["int8_reason"] = int8_reason

    if int8_ok:
        log["chosen_precision"] = "INT8"
        log["output_model"] = str(int8_out)
        log["decision_rationale"] = (
            "INT8 dynamic quantization succeeded and passed NaN/Inf sanity check. "
            "Chosen over FP16 because INT8 offers greater size reduction (~4x vs FP32) "
            "and faster CPU inference via integer arithmetic."
        )
        print(f"\n[quantize.py] Chosen: INT8 — {int8_out.name}")
    else:
        print(f"[quantize.py] INT8 failed: {int8_reason}")
        print("[quantize.py] Falling back to FP16...")

        fp16_out = MODELS_DIR / "best_fp16.onnx"
        log["fp16_attempted"] = True
        fp16_ok, fp16_reason = try_fp16(onnx_src, fp16_out, test_image)
        log["fp16_success"] = fp16_ok
        log["fp16_reason"] = fp16_reason

        if fp16_ok:
            log["chosen_precision"] = "FP16"
            log["output_model"] = str(fp16_out)
            log["decision_rationale"] = (
                f"INT8 failed ({int8_reason}). "
                "FP16 conversion used as fallback — reduces model size ~2x with minimal "
                "accuracy impact; keep_io_types=True preserves float32 I/O compatibility."
            )
            print(f"\n[quantize.py] Chosen: FP16 — {fp16_out.name}")
        else:
            print(f"[quantize.py] FP16 also failed: {fp16_reason}")
            print("[quantize.py] ERROR: Both quantization paths failed. See log for details.")

    # --- Save log ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / "quantize_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n[quantize.py] Log saved to: {log_path}")
    print(f"[quantize.py] Chosen precision: {log['chosen_precision']}")
    print(f"[quantize.py] Output model:     {log['output_model']}")
    print(f"[quantize.py] Rationale:        {log['decision_rationale']}")


if __name__ == "__main__":
    main()
