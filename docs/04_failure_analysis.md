# Phase 4 — Failure analysis (A4)

## Goal

Identify the three validation images the trained model handles worst,
and write an honest, specific diagnosis for each — this is explicitly
weighted heavily in the brief; a submission claiming near-perfect
results reads as a leaked split, not as strength.

## Prerequisites

- `models/best.pt` (or `models/best.onnx`) from Phase 2/3.
- `data/val/images` + `data/val/labels`.

## What to build

### `scripts/failure_analysis.py`

1. Run inference with the trained model on every image in
   `data/val/images`.
2. For each image, compute a simple "badness" score against ground
   truth labels — e.g. count of missed ground-truth boxes (false
   negatives) + count of spurious predicted boxes with no matching
   ground truth (false positives, IoU below threshold with all GT
   boxes) + any large IoU error on matched boxes. Rank images by this
   score.
3. Select the **three worst** images.
4. For each of the three, save a side-by-side visualization to
   `results/failure_cases/`: ground-truth boxes drawn in one color,
   predicted boxes in another, on the same image — so the mismatch is
   visible at a glance, not just implied by numbers.
5. Print (and save to `results/failure_cases/summary.json`) for each
   image: filename, predicted boxes (class + confidence), ground-truth
   boxes, and the computed badness score components (FN count, FP
   count, IoU errors).

### What the script does NOT do

It does not write the hypothesis text — that's a human judgment call
(annotation inconsistency vs. class imbalance vs. scale vs. occlusion
vs. insufficient examples of that condition) that belongs in
`ANSWERS.md`/README, informed by looking at the actual visualizations
the script produces. Do not have the agentic IDE auto-generate the
hypothesis prose from the numbers alone — it needs to reflect an actual
look at the image content.

## Acceptance criteria

- [ ] `results/failure_cases/` contains 3 annotated comparison images
- [ ] `results/failure_cases/summary.json` has real per-image numbers
- [ ] For each of the 3 images, README/ANSWERS.md states: what the
      model predicted, what it should have predicted, and a specific
      hypothesis for why (not a generic "the model struggled")
- [ ] For each, a stated next step (what to capture/change to fix it) —
      e.g. "more low-contrast cable examples like this one" rather than
      "collect more data" generically
