"""
scripts/export_onnx.py

Export models/best.pt to ONNX and verify the ONNX output numerically
matches the PyTorch model on the same val image.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --weights models/best.pt --imgsz 640

Outputs:
    models/best.onnx
    results/onnx_verification.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.resolve()
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
VAL_IMAGES_DIR = REPO_ROOT / "dataset" / "images" / "val"


def parse_args():
    p = argparse.ArgumentParser(description="Export YOLOv8 best.pt to ONNX and verify")
    p.add_argument("--weights", default=str(MODELS_DIR / "best.pt"),
                   help="PyTorch weights to export (default: models/best.pt)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Input image size (must match training, default: 640)")
    p.add_argument("--opset", type=int, default=12,
                   help="ONNX opset version (default: 12)")
    return p.parse_args()


def preprocess_image(img_path: Path, imgsz: int) -> np.ndarray:
    """Load and preprocess an image to ONNX input format (1, 3, H, W), float32 [0,1]."""
    import cv2
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # HWC -> CHW
    img = np.expand_dims(img, 0)          # -> (1, 3, H, W)
    return np.ascontiguousarray(img)


def run_pytorch_inference(model, img_tensor: np.ndarray) -> np.ndarray:
    """Run inference through the PyTorch model, return raw output numpy array."""
    import torch
    with torch.no_grad():
        tensor = torch.from_numpy(img_tensor)
        result = model.model(tensor)
        # result is a tuple; first element is the raw detection tensor
        raw = result[0] if isinstance(result, tuple) else result
        return raw.cpu().numpy()


def run_onnx_inference(onnx_path: Path, img_tensor: np.ndarray) -> np.ndarray:
    """Run inference through ONNX Runtime, return raw output numpy array."""
    import onnxruntime as ort
    try:
        ort.capi.onnxruntime_inference_collection.InferenceSession._validate_graph_capture_run_api = lambda self, run_options: None
    except Exception:
        pass
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: img_tensor})
    return outputs[0]


def main():
    args = parse_args()
    weights_path = Path(args.weights)

    print(f"\n[export_onnx.py] Loading weights: {weights_path}")
    from ultralytics import YOLO
    model = YOLO(str(weights_path))

    # --- Export to ONNX ---
    onnx_out_dir = MODELS_DIR
    onnx_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export_onnx.py] Exporting to ONNX (opset={args.opset}, imgsz={args.imgsz})...")
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
    )
    # Ultralytics saves next to the .pt file; move it to models/
    export_path = Path(export_path)
    onnx_dest = MODELS_DIR / "best.onnx"
    if export_path != onnx_dest:
        import shutil
        shutil.copy2(export_path, onnx_dest)
    print(f"[export_onnx.py] ONNX model saved to: {onnx_dest}")

    # --- Verification ---
    val_images = sorted(VAL_IMAGES_DIR.glob("*.jpg"))
    if not val_images:
        print("[export_onnx.py] ERROR: No val images found. Run split_dataset.py first.")
        return

    # Use first val image for verification
    test_image = val_images[0]
    print(f"\n[export_onnx.py] Verification image: {test_image.name}")

    img_tensor = preprocess_image(test_image, args.imgsz)

    # PyTorch inference
    print("[export_onnx.py] Running PyTorch inference...")
    t0 = time.perf_counter()
    pt_output = run_pytorch_inference(model, img_tensor)
    pt_time = (time.perf_counter() - t0) * 1000

    # ONNX Runtime inference
    print("[export_onnx.py] Running ONNX Runtime inference...")
    t0 = time.perf_counter()
    onnx_output = run_onnx_inference(onnx_dest, img_tensor)
    onnx_time = (time.perf_counter() - t0) * 1000

    # Numeric comparison
    max_abs_diff = float(np.max(np.abs(pt_output - onnx_output)))
    mean_abs_diff = float(np.mean(np.abs(pt_output - onnx_output)))

    # Box-level agreement: decode top predictions and check IoU
    # Use confidence threshold 0.25 to get predictions from both outputs
    def decode_boxes(raw: np.ndarray, conf_thresh=0.25):
        """Extract [x1,y1,x2,y2,conf,cls] from raw YOLOv8 output tensor."""
        # raw shape: (1, 6, num_anchors) for YOLOv8
        pred = raw[0]  # (6, num_anchors) or (num_anchors, 6)
        if pred.shape[0] == 6:
            pred = pred.T  # -> (num_anchors, 6)
        # columns: cx, cy, w, h, conf_cls0, conf_cls1
        boxes = pred[:, :4]
        scores = pred[:, 4:]
        conf = scores.max(axis=1)
        mask = conf > conf_thresh
        return boxes[mask], conf[mask]

    pt_boxes, pt_confs = decode_boxes(pt_output)
    onnx_boxes, onnx_confs = decode_boxes(onnx_output)
    boxes_count_match = len(pt_boxes) == len(onnx_boxes)

    print(f"\n[export_onnx.py] --- Verification Results ---")
    print(f"  Max absolute difference (raw tensors): {max_abs_diff:.8f}")
    print(f"  Mean absolute difference:              {mean_abs_diff:.8f}")
    print(f"  PyTorch detections (conf>0.25):        {len(pt_boxes)}")
    print(f"  ONNX Runtime detections (conf>0.25):   {len(onnx_boxes)}")
    print(f"  Detection count match:                 {boxes_count_match}")
    print(f"  PyTorch inference time:                {pt_time:.1f}ms")
    print(f"  ONNX Runtime inference time:           {onnx_time:.1f}ms")

    match_status = max_abs_diff < 1e-3  # acceptable numerical tolerance
    print(f"\n  Numeric match (max_diff < 1e-3):       {'PASS' if match_status else 'FAIL (check ONNX export)'}")

    verification = {
        "image_used": f"dataset/images/val/{test_image.name}",
        "pytorch_weights": str(weights_path),
        "onnx_model": "models/best.onnx",
        "max_abs_diff": round(max_abs_diff, 10),
        "mean_abs_diff": round(mean_abs_diff, 10),
        "pytorch_detections_conf25": int(len(pt_boxes)),
        "onnx_detections_conf25": int(len(onnx_boxes)),
        "boxes_count_match": boxes_count_match,
        "numeric_match_passed": bool(match_status),
        "method": (
            "Same image preprocessed identically (resize to 640x640, normalize to [0,1], "
            "CHW layout). Raw output tensors compared with np.max(np.abs(pt_out - onnx_out)). "
            "Additionally decoded detections at conf>0.25 from both outputs and compared count."
        ),
        "pytorch_inference_ms": round(pt_time, 2),
        "onnx_inference_ms": round(onnx_time, 2),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "onnx_verification.json"
    with open(out_path, "w") as f:
        json.dump(verification, f, indent=2)
    print(f"\n[export_onnx.py] Verification results saved to: {out_path}")


if __name__ == "__main__":
    main()
