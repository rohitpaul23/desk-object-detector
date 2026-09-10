import os
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

def box_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def load_gt_boxes(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                x1 = (cx - w / 2.0) * img_w
                y1 = (cy - h / 2.0) * img_h
                x2 = (cx + w / 2.0) * img_w
                y2 = (cy + h / 2.0) * img_h
                boxes.append({'cls': cls_id, 'bbox': [x1, y1, x2, y2]})
    return boxes

def main():
    val_img_dir = Path("dataset/images/val")
    val_lbl_dir = Path("dataset/labels/val")
    model_path = Path("models/best.pt")
    out_dir = Path("results/failure_cases")
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = ["cable", "device"]
    if os.path.exists("data/classes.txt"):
        with open("data/classes.txt") as f:
            class_names = [line.strip() for line in f if line.strip()]

    print(f"Loading model from {model_path}...")
    model = YOLO(str(model_path))

    results_list = []

    img_paths = sorted(list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png")))
    print(f"Analyzing {len(img_paths)} validation images...")

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
        gt_boxes = load_gt_boxes(lbl_path, w, h)

        # Run inference
        preds = model.predict(str(img_path), conf=0.25, verbose=False)[0]

        pred_boxes = []
        for box in preds.boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            pred_boxes.append({'cls': cls_id, 'conf': conf, 'bbox': xyxy})

        # Match GT and Pred
        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)
        matched_ious = []
        class_mismatches = 0

        for p_idx, p in enumerate(pred_boxes):
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, g in enumerate(gt_boxes):
                iou = box_iou(p['bbox'], g['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx

            if best_iou >= 0.5 and best_g_idx != -1:
                if p['cls'] == gt_boxes[best_g_idx]['cls']:
                    if not gt_matched[best_g_idx]:
                        gt_matched[best_g_idx] = True
                        pred_matched[p_idx] = True
                        matched_ious.append(best_iou)
                else:
                    class_mismatches += 1

        fn_count = sum(1 for m in gt_matched if not m)
        fp_count = sum(1 for m in pred_matched if not m)
        mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

        # Badness score computation
        badness_score = (fn_count * 2.5) + (fp_count * 1.5) + (class_mismatches * 3.0) + ((1.0 - mean_iou) if gt_boxes else 0.0)

        results_list.append({
            'filename': img_path.name,
            'img_path': str(img_path),
            'gt_count': len(gt_boxes),
            'pred_count': len(pred_boxes),
            'fn_count': fn_count,
            'fp_count': fp_count,
            'class_mismatches': class_mismatches,
            'mean_matched_iou': round(mean_iou, 4),
            'badness_score': round(badness_score, 4),
            'gt_boxes': [{'class': class_names[g['cls']], 'bbox': [round(x, 1) for x in g['bbox']]} for g in gt_boxes],
            'pred_boxes': [{'class': class_names[p['cls']], 'conf': round(p['conf'], 3), 'bbox': [round(x, 1) for x in p['bbox']]} for p in pred_boxes]
        })

    # Sort by badness score descending
    results_list.sort(key=lambda x: x['badness_score'], reverse=True)

    worst_3 = results_list[:3]

    print("\n--- TOP 3 FAILURE CASES ---")
    for idx, item in enumerate(worst_3, 1):
        print(f"Rank {idx}: {item['filename']} | Badness Score: {item['badness_score']} | FN: {item['fn_count']} | FP: {item['fp_count']} | Class Mismatches: {item['class_mismatches']} | Mean IoU: {item['mean_matched_iou']}")

    # Generate visual artifacts for top 3 failure cases
    for idx, item in enumerate(worst_3, 1):
        img_orig = cv2.imread(item['img_path'])
        h, w = img_orig.shape[:2]

        # Draw Ground Truth on Left Panel
        img_gt = img_orig.copy()
        cv2.putText(img_gt, "GROUND TRUTH", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        for g in item['gt_boxes']:
            x1, y1, x2, y2 = map(int, g['bbox'])
            cls_name = g['class']
            cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label_text = f"GT: {cls_name}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img_gt, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1)
            cv2.putText(img_gt, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Draw Predictions on Right Panel
        img_pred = img_orig.copy()
        cv2.putText(img_pred, "PREDICTION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        for p in item['pred_boxes']:
            x1, y1, x2, y2 = map(int, p['bbox'])
            cls_name = p['class']
            conf = p['conf']
            color = (0, 0, 255) if cls_name == "cable" else (255, 165, 0)
            cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 3)
            label_text = f"PRED: {cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img_pred, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(img_pred, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Combine GT and Pred side-by-side
        combined = np.hstack((img_gt, img_pred))
        save_name = f"worst_{idx}_{item['filename']}"
        cv2.imwrite(str(out_dir / save_name), combined)
        print(f"Saved side-by-side visualization: {out_dir / save_name}")

    summary_file = out_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'worst_3_cases': worst_3,
            'all_val_results': results_list
        }, f, indent=2)

    print(f"\nSaved summary to {summary_file}")

if __name__ == "__main__":
    main()
