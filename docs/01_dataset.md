# Phase 1 — Dataset (reference doc, mostly already executed manually)

This phase is largely done by hand (shooting + LabelImg annotation) before
handing off to an agentic IDE. This doc exists so the IDE has full context
on the data contract the later phases depend on — do not re-derive these
decisions, just enforce them if writing any dataset-adjacent code.

## Classes

```
0: cable
1: device
```
Defined in `data/classes.txt`, one per line, index = line number
(0-indexed). This file is the source of truth for class order — any
script that maps class index to name must read it, never hardcode
`["cable", "device"]` in more than one place.

## Annotation convention (already applied by hand in LabelImg)

- One bounding box per physically distinct `cable` instance, drawn tight
  around its full visible extent, even if coiled, bent, or crossing
  another cable or object. Two different cables that visually overlap
  still get two separate boxes.
- `device` = any discrete electronic item with a body/casing (phone,
  mouse, USB hub/drive, speaker, power bank, charging case, extension
  strip, charger brick, router).
- A charger's brick and its cord are two separate annotated instances:
  brick → `device`, cord → `cable` — even when their boxes overlap.
- Non-electronic items (stationery, eyewear, watches, tools, tissue box,
  loose batteries, scissors, wall-mounted switchboards) are never
  annotated — leave them unboxed as background.

## File naming contract

Raw images live in `data/raw_images/` named `<block>_<seq>.jpg`, e.g.
`b01_001.jpg`, `b01_002.jpg`, `b02_001.jpg`. The `<block>` prefix
identifies the physical capture setup/session (tidy desk, cluttered desk,
lamp-only lighting, etc.) and is the unit the train/val split operates on
— **never split by individual shuffled image**, since images within a
block are near-duplicates and would leak across the split.

Label files live in `data/labels/`, one `.txt` per image, same basename,
YOLO format (`class_id x_center y_center width height`, normalized 0–1).

## Split script contract (`scripts/split_dataset.py` — already written)

- Reads `BLOCK_TO_SPLIT: dict[str, "train"|"val"]` mapping block prefix
  to split.
- Copies each image + label into `data/train/{images,labels}/` or
  `data/val/{images,labels}/` accordingly.
- Prints per-block image counts, warns on any image missing a label
  file, and prints per-class instance counts per split for the README.
- **If asked to modify this script:** preserve the block-level split
  logic. Do not silently change it to a random per-image shuffle — that
  would reintroduce the leakage problem this project explicitly guards
  against (see README → Assumptions #1).

## Acceptance criteria before moving to Phase 2

- [ ] `data/train/images` and `data/val/images` both non-empty
- [ ] Every image in `train/` and `val/` has a matching `.txt` label file
      (script warns if not — resolve before training)
- [ ] Per-class instance counts printed by the split script are pasted
      into `README.md` → "Measured results" → Dataset
- [ ] At least ~10 images across the val set are ones the person
      capturing expects to be hard (backlit, low-contrast, crowded,
      thin cable against clutter) — needed later for Phase 4 failure
      analysis. If this isn't true yet, flag it back to the user rather
      than proceeding silently.
