"""
Stage 2: OCR on engineering drawings.
Uses EasyOCR with engineering-drawing aware post-processing.

Improvements:
- CLAHE + unsharp masking via preprocessing module
- Region-aware upscaling: BOM table gets 3x, rest gets 2x
- BOM region detections merged with main detections (deduped)
- Expanded post-processing corrections
- Better junk filtering
"""

import cv2
import numpy as np
import os
import sys
import json

try:
    from preprocessing import preprocess, preprocess_for_ocr
except ImportError:
    from .preprocessing import preprocess, preprocess_for_ocr


# ============================================================
# Engineering codes that must NEVER be modified by post-processing
# ============================================================
PROTECTED_CODES = {
    'MS', 'CS', 'CI', 'FS', 'GM', 'CR', 'AL', 'BR',
    'MCS', 'HCS', 'LCS', 'SS', 'SPS', 'EN', 'IS',
    'DIA', 'PCD', 'CSK', 'TYP', 'EQUI', 'EQUI-SP',
    'OIL', 'HOLE', 'GROOVE', 'KEY', 'DEEP', 'THICK',
    'X-X', 'X', 'A-A', 'A', 'B-B', 'B', 'Y-Y',
    'BOLT', 'NUT', 'WASHER', 'PIN', 'STRAP',
    'PARTS', 'LIST', 'NAME', 'MATERIAL', 'QTY', 'NO',
    'PART', 'SL', 'NO.', 'MM', 'CM',
    'VALVE', 'SPRING', 'LEVER', 'ROCKER', 'ARM', 'BOX',
    'CONNECTING', 'ROD', 'COTTER', 'BRASS', 'JIB', 'SET', 'SCREW',
    'BODY', 'COVER', 'PLATE', 'SEAT', 'SLEEVE', 'COLLAR',
    'SPINDLE', 'HANDWHEEL', 'GLAND', 'BONNET', 'STUFFING',
    # Extended
    'PIVOT', 'FORK', 'BLOCK', 'HOLDER', 'SWIVEL', 'SHEAVE',
    'PISTON', 'BUSH', 'BUSHING', 'STUD', 'CAP', 'FLANGE',
    'SHAFT', 'GEAR', 'PULLEY', 'WHEEL', 'DISC', 'FRAME',
    'BASE', 'SUPPORT', 'BRACKET', 'CLAMP', 'HOUSING',
}

# ============================================================
# Known OCR misreads for engineering drawing text
# Applied as a final correction pass after post_process_text
# ============================================================
OCR_CORRECTIONS = {
    # Part names
    'BRAS':      'BRASS',
    'BRAS:':     'BRASS',
    'GLANC':     'GLAND',
    'GLANC:':    'GLAND',
    'NU:':       'NUT',
    'SPINDL':    'SPINDLE',
    'BONNIT':    'BONNET',
    'SLEVE':     'SLEEVE',
    'HANDWHEL':  'HANDWHEEL',
    'VLAVE':     'VALVE',
    'VLVE':      'VALVE',
    'SPRIG':     'SPRING',
    'CONROD':    'CONNECTING ROD',
    'SWIVEL PLALE': 'SWIVEL PLATE',
    'TOOL HULDER':  'TOOL HOLDER',
    'CENTAL BLOCK': 'CENTRAL BLOCK',
    'SHEAVE PLECE': 'SHEAVE PIECE',
    # Material names
    'BABBITT':   'BABBIT',
    'ALUMINIUM': 'ALUMINUM',
    'ALUMNUM':   'ALUMINUM',
    'BRONZ':     'BRONZE',
    # BOM headers
    'OTY':       'QTY',
    'OTY:':      'QTY',
    'OLY:':      'QTY',
    'OLY':       'QTY',
    'MATL':      'MATERIAL',
    'MAT.':      'MATERIAL',
    # Dimension artefacts
    'O15':       'Ø15',
    'O20':       'Ø20',
    'O25':       'Ø25',
    'O30':       'Ø30',
    'O40':       'Ø40',
    'O50':       'Ø50',
}


def is_protected(text):
    """Check if text is a known engineering code/word."""
    upper = text.upper().strip()
    if upper in PROTECTED_CODES:
        return True
    if upper.isalpha() and 1 <= len(upper) <= 3 and upper == text.upper():
        return True
    return False


def post_process_text(text):
    """Engineering-aware post processing. Preserves protected codes."""
    if not text:
        return text
    text = text.strip()
    if is_protected(text):
        return text

    has_letter = any(c.isalpha() for c in text)
    has_digit  = any(c.isdigit() for c in text)

    # Pure digits: O → 0
    if has_digit and not has_letter:
        text = text.replace('O', '0').replace('o', '0')
        text = text.replace('*', '×')
        text = text.replace('Ã—', '×')
        return text

    # Thread spec: "M3O" → "M30"
    if text.startswith('M') and len(text) >= 3 and 'O' in text[1:]:
        text = text[0] + text[1:].replace('O', '0')

    # Symbol fixes
    text = text.replace('*', '×')
    text = text.replace('Ã—', '×')
    text = text.replace('Ã\u00d7', '×')

    if text.startswith('@'):
        text = 'Ø' + text[1:]

    # Apply known OCR corrections (case-insensitive lookup)
    upper = text.upper().strip()
    if upper in OCR_CORRECTIONS:
        return OCR_CORRECTIONS[upper]

    return text


def is_likely_junk(text, confidence, box):
    """
    Filter out OCR noise. Returns True if text should be DISCARDED.
    """
    if not text:
        return True
    t = text.strip()
    if len(t) == 1 and not t.isalnum():
        return True
    if len(t) == 1 and t.isdigit() and confidence < 0.70:
        return True
    if box[3] < 4:
        return True
    if t == '0' and confidence < 0.8:
        return True
    # Pure punctuation strings
    if all(c in '.,;:!?-_/\\|()[]{}' for c in t):
        return True
    return False


def _boxes_overlap(box_a, box_b, threshold=0.4):
    """Check if two [x,y,w,h] boxes overlap above IoU threshold."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return (inter / union) >= threshold if union > 0 else False


# ============================================================
# EasyOCR setup
# ============================================================
_ocr_engine = None


def get_ocr():
    """Lazy-init EasyOCR engine."""
    global _ocr_engine
    if _ocr_engine is None:
        import easyocr
        print("  Initializing EasyOCR (first call only)...")
        _ocr_engine = easyocr.Reader(['en'], gpu=True)
    return _ocr_engine


# ============================================================
# OCR on a single image region
# ============================================================
def _run_ocr_on_image(ocr, img_bgr, scale_factor, offset_x=0, offset_y=0,
                      min_confidence=0.5, filter_junk=True, region_label=""):
    """
    Run EasyOCR on an image, scale coordinates back, return structured entries.

    Args:
        img_bgr:      BGR image to run OCR on (already upscaled if needed)
        scale_factor: How much the image was upscaled (to convert coords back)
        offset_x/y:   Pixel offset if this is a crop of the full image
        region_label: For logging

    Returns:
        List of structured entry dicts.
    """
    results = ocr.readtext(
        img_bgr,
        detail=1,
        paragraph=False,
        text_threshold=0.45,
        low_text=0.25,
        link_threshold=0.30,
        rotation_info=[90, 180, 270],
    )

    entries = []
    for bbox, raw_text, conf in results:
        if conf < min_confidence:
            continue

        text = post_process_text(raw_text)

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        # Scale back to original image coordinates
        x     = int(min(xs) / scale_factor) + offset_x
        y     = int(min(ys) / scale_factor) + offset_y
        w_box = int((max(xs) - min(xs)) / scale_factor)
        h_box = int((max(ys) - min(ys)) / scale_factor)
        box   = [x, y, w_box, h_box]

        if filter_junk and is_likely_junk(text, conf, box):
            continue

        entries.append({
            "box":        box,
            "text":       text,
            "raw_text":   raw_text,
            "confidence": float(conf),
            "region":     region_label,
        })

    return entries


# ============================================================
# Main OCR function
# ============================================================
def read_full_image(image_path, output_dir="results", min_confidence=0.4,
                    filter_junk=True):
    """
    Run EasyOCR on the full image with region-aware upscaling.

    Strategy:
    1. Run OCR on full image at 2x upscale (enhanced with CLAHE + sharpening)
    2. If BOM region detected, run OCR again at 3x upscale on that region
    3. Merge results, preferring higher-confidence detections for overlapping boxes
    """
    print(f"\n=== STAGE 2: Full-image OCR ===")
    print(f"Mode: EasyOCR (region-aware, CLAHE enhanced)")

    # Get enhanced images
    enhanced_bgr, bom_crop, bom_bbox = preprocess_for_ocr(image_path)
    original = cv2.imread(image_path)
    h, w = enhanced_bgr.shape[:2]

    ocr = get_ocr()

    # ── Pass 1: Full image at 2x ─────────────────────────────────────────
    scale_2x = 2
    upscaled_2x = cv2.resize(enhanced_bgr, (w * scale_2x, h * scale_2x),
                              interpolation=cv2.INTER_CUBIC)
    print(f"\nPass 1: Full image OCR ({w*scale_2x}x{h*scale_2x})...")
    main_entries = _run_ocr_on_image(
        ocr, upscaled_2x, scale_factor=scale_2x,
        min_confidence=min_confidence, filter_junk=filter_junk,
        region_label="main"
    )
    print(f"  → {len(main_entries)} detections")

    # ── Pass 2: BOM region at 3x (if detected) ───────────────────────────
    bom_entries = []
    if bom_crop is not None and bom_bbox is not None:
        bx, by, bw_r, bh_r = bom_bbox
        scale_3x = 3
        bom_h, bom_w = bom_crop.shape[:2]
        upscaled_bom = cv2.resize(bom_crop, (bom_w * scale_3x, bom_h * scale_3x),
                                   interpolation=cv2.INTER_CUBIC)
        print(f"\nPass 2: BOM region OCR at 3x ({bom_w*scale_3x}x{bom_h*scale_3x})...")
        bom_entries = _run_ocr_on_image(
            ocr, upscaled_bom, scale_factor=scale_3x,
            offset_x=bx, offset_y=by,
            min_confidence=min_confidence - 0.05,  # slightly lower threshold for BOM
            filter_junk=filter_junk,
            region_label="bom"
        )
        print(f"  → {len(bom_entries)} BOM detections")

    # ── Merge: BOM entries override main entries for overlapping boxes ────
    all_entries = list(main_entries)
    for bom_entry in bom_entries:
        overlaps = False
        for i, main_entry in enumerate(all_entries):
            if _boxes_overlap(bom_entry["box"], main_entry["box"]):
                # BOM pass has higher resolution — prefer it if confidence is similar
                if bom_entry["confidence"] >= main_entry["confidence"] - 0.1:
                    all_entries[i] = bom_entry
                overlaps = True
                break
        if not overlaps:
            all_entries.append(bom_entry)

    # ── Assign sequential IDs and log ─────────────────────────────────────
    print(f"\n  Filtering with min_confidence={min_confidence}, junk_filter={filter_junk}")
    print("-" * 60)

    structured = []
    for entry in all_entries:
        item = {
            "id":         len(structured) + 1,
            "box":        entry["box"],
            "text":       entry["text"],
            "raw_text":   entry["raw_text"],
            "confidence": entry["confidence"],
        }
        structured.append(item)
        x, y, w_box, h_box = item["box"]
        marker = "" if item["text"] == item["raw_text"] else f"  (raw: '{item['raw_text']}')"
        region = f"[{entry.get('region','?')}]"
        print(f"  [{item['id']:3d}] {region} ({x:3d},{y:3d}) {w_box:3d}x{h_box:3d} "
              f"conf={item['confidence']:.2f}  ->  '{item['text']}'{marker}")

    # ── Save JSON ─────────────────────────────────────────────────────────
    filename    = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{filename}_fullocr.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)

    # ── Visualization — colored semi-transparent filled boxes by confidence ──
    vis = original.copy()
    overlay = vis.copy()

    for item in structured:
        x, y, w_box, h_box = item["box"]
        conf = item["confidence"]

        # Color by confidence: green=high, yellow=medium, red=low (BGR)
        if conf > 0.9:
            color = (0, 200, 60)      # green
        elif conf >= 0.7:
            color = (0, 200, 220)     # yellow
        else:
            color = (50, 80, 220)     # red/orange

        # Filled semi-transparent rectangle
        cv2.rectangle(overlay, (x, y), (x + w_box, y + h_box), color, -1)
        # Solid border
        cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), color, 1)

    # Blend overlay (30% opacity fill)
    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

    # Draw text labels on top of blended image
    for item in structured:
        x, y, w_box, h_box = item["box"]
        conf = item["confidence"]
        if conf > 0.9:
            color = (0, 200, 60)
        elif conf >= 0.7:
            color = (0, 200, 220)
        else:
            color = (50, 80, 220)

        label = item["text"][:18]
        # Small dark background behind text for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        ty = max(y - 2, th + 2)
        cv2.rectangle(vis, (x, ty - th - 2), (x + tw + 2, ty + 1), (0, 0, 0), -1)
        cv2.putText(vis, label, (x + 1, ty - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Highlight BOM region with orange border
    if bom_bbox is not None:
        bx, by, bw_r, bh_r = bom_bbox
        cv2.rectangle(vis, (bx, by), (bx + bw_r, by + bh_r), (0, 165, 255), 2)
        cv2.putText(vis, "BOM TABLE", (bx, max(by - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

    # Legend (bottom-left)
    legend = [("High conf >0.9", (0, 200, 60)),
              ("Med conf 0.7-0.9", (0, 200, 220)),
              ("Low conf <0.7", (50, 80, 220))]
    lx, ly = 6, vis.shape[0] - 10
    for lbl, col in legend:
        cv2.rectangle(vis, (lx, ly - 10), (lx + 12, ly), col, -1)
        cv2.putText(vis, lbl, (lx + 15, ly - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1)
        lx += 120

    vis_path = os.path.join(output_dir, f"{filename}_fullocr.png")
    cv2.imwrite(vis_path, vis)

    print("-" * 60)
    print(f"Saved JSON: {output_path}")
    print(f"Saved viz:  {vis_path}")
    bom_note = f" | BOM region: {len(bom_entries)} extra detections" if bom_entries else ""
    print(f"Total: {len(structured)} regions kept{bom_note}")

    return structured


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/vlm_reader.py <image_path>")
    else:
        read_full_image(sys.argv[1])
