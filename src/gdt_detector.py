"""
Stage 3-A: GD&T Symbol Detection using YOLOv8.

Detects GD&T symbols visually from the image using a trained YOLO model,
producing detections in the same format as Stage 3 classified entries so
they can be merged with OCR-based classification.

Classes detected (11 from Roboflow dataset):
    angularity, circularity, concentricity, cylindricity, flatness,
    gdt (generic feature control frame), parallelism, perpendicularity,
    position, profile of line, profile of surface, straightness

The detections are merged with OCR output in merge_detections():
  - YOLO wins on symbol type (visual classification is more reliable)
  - OCR provides the numeric value inside the feature control frame
  - Non-overlapping OCR entries are kept as-is

Usage (CLI):
    python src/gdt_detector.py <image_path> [model_path]

Usage (API):
    from src.gdt_detector import detect_gdt, merge_detections
"""

import os
import sys
import json
import math

# ============================================================
# GD&T class → pipeline annotation type mapping
# ============================================================

# Maps YOLO class names to the pipeline's VALID_TYPES vocabulary.
# All GD&T symbols map to "tolerance" since they represent geometric
# tolerances in the stack-up analysis.
GDT_CLASS_TO_TYPE = {
    "angularity":         "tolerance",
    "circularity":        "tolerance",
    "concentricity":      "tolerance",
    "cylindricity":       "tolerance",
    "flatness":           "tolerance",
    "gdt":                "tolerance",
    "parallelism":        "tolerance",
    "perpendicularity":   "tolerance",
    "position":           "tolerance",
    "profile of line":    "tolerance",
    "profile of surface": "tolerance",
    "straightness":       "tolerance",
}

# Sub-type label stored in parsed.gdt_symbol for downstream use
GDT_SYMBOL_LABELS = {
    "angularity":         "angularity",
    "circularity":        "circularity",
    "concentricity":      "concentricity",
    "cylindricity":       "cylindricity",
    "flatness":           "flatness",
    "gdt":                "feature_control_frame",
    "parallelism":        "parallelism",
    "perpendicularity":   "perpendicularity",
    "position":           "position",
    "profile of line":    "profile_of_line",
    "profile of surface": "profile_of_surface",
    "straightness":       "straightness",
}

# Default model path (relative to workspace root)
DEFAULT_MODEL_PATH = "models/gdt_yolov8.pt"

# Minimum confidence threshold for YOLO detections
MIN_CONFIDENCE = 0.35

# IoU threshold for overlap detection during merge
IOU_THRESHOLD = 0.3


# ============================================================
# Lazy model loader
# ============================================================

_model = None
_model_path = None


def get_model(model_path=DEFAULT_MODEL_PATH):
    """Lazy-load the YOLO model (loads once per process)."""
    global _model, _model_path
    if _model is None or _model_path != model_path:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for GD&T detection. "
                "Install with: pip install ultralytics"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"GD&T model not found at '{model_path}'. "
                f"Run train_gdt_model.py first, or set model_path correctly."
            )
        print(f"  Loading GD&T YOLO model from: {model_path}")
        _model = YOLO(model_path)
        _model_path = model_path
    return _model


# ============================================================
# Geometry helpers
# ============================================================

def _box_iou(box_a, box_b):
    """
    Compute IoU between two [x, y, w, h] boxes.

    Args:
        box_a, box_b: [x, y, w, h] format boxes.

    Returns:
        IoU float in [0, 1].
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    # Convert to x1,y1,x2,y2
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    # Intersection
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _boxes_overlap(box_a, box_b, threshold=IOU_THRESHOLD):
    """Return True if two boxes overlap above the IoU threshold."""
    return _box_iou(box_a, box_b) >= threshold


# ============================================================
# Core detection function
# ============================================================

def detect_gdt(image_path, model_path=DEFAULT_MODEL_PATH, min_confidence=MIN_CONFIDENCE):
    """
    Run YOLO GD&T detection on an image.

    Args:
        image_path:     Path to the original PNG image.
        model_path:     Path to the trained YOLOv8 .pt model file.
        min_confidence: Minimum detection confidence (default 0.35).

    Returns:
        List of detection dicts, each with keys:
            id, box [x,y,w,h], text, raw_text, confidence, type,
            parsed, source ("yolo")
        Returns [] if model not found (graceful degradation).
    """
    print(f"\n=== STAGE 3-A: GD&T Symbol Detection (YOLO) ===")

    # Graceful degradation if model doesn't exist yet
    if not os.path.exists(model_path):
        print(f"  WARNING: GD&T model not found at '{model_path}' — skipping YOLO stage")
        print(f"  Run python train_gdt_model.py to train the model first.")
        return []

    try:
        model = get_model(model_path)
    except (ImportError, FileNotFoundError) as e:
        print(f"  WARNING: {e} — skipping YOLO stage")
        return []

    # Run inference
    results = model(image_path, conf=min_confidence, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        names = result.names  # {class_id: class_name}

        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, "gdt").lower()

            # YOLO returns xyxy format
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)

            ann_type = GDT_CLASS_TO_TYPE.get(cls_name, "tolerance")
            symbol   = GDT_SYMBOL_LABELS.get(cls_name, cls_name)

            detection = {
                "id":         None,   # assigned during merge
                "box":        [x, y, w, h],
                "text":       cls_name,   # placeholder; OCR value merged in later
                "raw_text":   cls_name,
                "confidence": round(conf, 4),
                "type":       ann_type,
                "parsed":     {
                    "gdt_symbol":       symbol,
                    "tolerance_string": cls_name,
                },
                "source":     "yolo",
            }
            detections.append(detection)

    print(f"  Detected {len(detections)} GD&T symbols")
    for d in detections:
        x, y, w, h = d["box"]
        print(f"    ({x},{y}) {w}x{h}  conf={d['confidence']:.2f}  {d['text']}")

    return detections


# ============================================================
# Merge YOLO detections with OCR structured entries
# ============================================================

def merge_detections(ocr_entries, yolo_detections, iou_threshold=IOU_THRESHOLD):
    """
    Merge YOLO GD&T detections with OCR-classified entries.

    Merge strategy:
      1. For each YOLO detection, find any OCR entry whose box overlaps
         (IoU >= iou_threshold).
      2. If overlap found: YOLO wins on type/symbol, OCR provides the
         numeric text value (e.g. "0.05" from inside the feature control frame).
      3. If no overlap: YOLO detection is added as a new entry.
      4. OCR entries that don't overlap any YOLO detection are kept unchanged.

    This means:
      - GD&T symbols get accurate visual classification from YOLO
      - Their numeric tolerance values come from OCR
      - All other annotations (dimensions, BOM, etc.) are unaffected

    Args:
        ocr_entries:     List of classified entry dicts from Stage 3 validation.
        yolo_detections: List of detection dicts from detect_gdt().

    Returns:
        Merged list of entry dicts, re-numbered with sequential IDs.
    """
    if not yolo_detections:
        return ocr_entries

    merged = []
    used_ocr_ids = set()   # OCR entry indices consumed by a YOLO match

    for yolo_det in yolo_detections:
        yolo_box = yolo_det["box"]
        best_ocr_idx = None
        best_iou = 0.0

        # Find the best-overlapping OCR entry
        for idx, ocr_entry in enumerate(ocr_entries):
            ocr_box = ocr_entry.get("box", [])
            if len(ocr_box) < 4:
                continue
            iou = _box_iou(yolo_box, ocr_box)
            if iou > best_iou:
                best_iou = iou
                best_ocr_idx = idx

        if best_ocr_idx is not None and best_iou >= iou_threshold:
            # Merge: YOLO type + OCR text value
            ocr_entry = ocr_entries[best_ocr_idx]
            merged_entry = dict(ocr_entry)   # copy OCR entry
            merged_entry["type"]       = yolo_det["type"]
            merged_entry["confidence"] = max(ocr_entry.get("confidence", 0),
                                             yolo_det["confidence"])
            # Preserve OCR text but enrich parsed with GD&T symbol info
            merged_entry["parsed"] = {
                **ocr_entry.get("parsed", {}),
                "gdt_symbol":       yolo_det["parsed"]["gdt_symbol"],
                "tolerance_string": ocr_entry.get("text", yolo_det["text"]),
            }
            merged_entry["source"] = "yolo+ocr"
            merged.append(merged_entry)
            used_ocr_ids.add(best_ocr_idx)
        else:
            # No OCR overlap — add YOLO detection as standalone entry
            merged.append(dict(yolo_det))

    # Add all OCR entries that weren't consumed by a YOLO match
    for idx, ocr_entry in enumerate(ocr_entries):
        if idx not in used_ocr_ids:
            merged.append(dict(ocr_entry))

    # Re-assign sequential IDs
    for i, entry in enumerate(merged, 1):
        entry["id"] = i

    print(f"  Merge: {len(yolo_detections)} YOLO + {len(ocr_entries)} OCR "
          f"→ {len(merged)} total ({len(used_ocr_ids)} overlaps resolved)")

    return merged


# ============================================================
# CLI entry point
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/gdt_detector.py <image_path> [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_MODEL_PATH

    detections = detect_gdt(image_path, model_path=model_path)

    print(f"\nDetected {len(detections)} GD&T symbols:")
    for d in detections:
        print(f"  {d['parsed']['gdt_symbol']:<25} conf={d['confidence']:.3f}  "
              f"box={d['box']}")


if __name__ == "__main__":
    main()
