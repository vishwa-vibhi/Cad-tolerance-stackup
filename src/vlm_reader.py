"""
Stage 2: OCR on engineering drawings.
Uses EasyOCR with engineering-drawing aware post-processing.
Includes junk filtering to remove single-digit OCR noise.
"""

import cv2
import os
import sys
import json
from preprocessing import preprocess


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
    has_digit = any(c.isdigit() for c in text)

    # pure digits: O -> 0, * -> ×
    if has_digit and not has_letter:
        text = text.replace('O', '0').replace('o', '0')
        return text.replace('*', '×')

    # thread spec like "M3O" -> "M30"
    if text.startswith('M') and len(text) >= 3 and 'O' in text[1:]:
        text = text[0] + text[1:].replace('O', '0')

    text = text.replace('*', '×')
    # Fix broken UTF-8 multiplication symbol (EasyOCR encoding artefact)
    text = text.replace('Ã—', '×')
    text = text.replace('Ã\u00d7', '×')

    if text.startswith('@'):
        text = 'Ø' + text[1:]

    return text


def is_likely_junk(text, confidence, box):
    """
    Filter out OCR noise. Returns True if text should be DISCARDED.

    Rules:
    - Single character non-alphanumeric -> junk
    - Single digit with confidence < 0.85 -> junk
    - Tiny bounding box (< 6px tall) -> junk
    - "0" alone -> usually OCR noise from a circle
    """
    if not text:
        return True
    t = text.strip()
    if len(t) == 1 and not t.isalnum():
        return True
    if len(t) == 1 and t.isdigit() and confidence < 0.85:
        return True
    if box[3] < 6:
        return True
    if t == '0':
        return True
    return False


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
# Main OCR function
# ============================================================
def read_full_image(image_path, output_dir="results", min_confidence=0.5,
                    filter_junk=True):
    """Run EasyOCR on whole image with engineering-drawing optimizations."""
    print(f"\n=== STAGE 2: Full-image OCR ===")
    print(f"Mode: EasyOCR (engineering tuned)")

    binary = preprocess(image_path, save_result=False)
    original = cv2.imread(image_path)

    # contrast boost
    enhanced = cv2.convertScaleAbs(original, alpha=1.3, beta=10)

    # upscale 2x for small text
    h, w = enhanced.shape[:2]
    upscaled = cv2.resize(enhanced, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    ocr = get_ocr()
    print(f"\nRunning EasyOCR on upscaled image ({w*2}x{h*2})...")
    results = ocr.readtext(
        upscaled,
        detail=1,
        paragraph=False,
        text_threshold=0.6,
        low_text=0.35,
        link_threshold=0.4,
        rotation_info=[90, 180, 270]
    )

    print(f"  Detected {len(results)} raw regions")
    print(f"  Filtering with min_confidence={min_confidence}, junk_filter={filter_junk}")
    print("-" * 60)

    structured = []
    junk_count = 0
    for i, (bbox, raw_text, conf) in enumerate(results, 1):
        if conf < min_confidence:
            continue

        text = post_process_text(raw_text)

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x = int(min(xs) / 2)
        y = int(min(ys) / 2)
        w_box = int((max(xs) - min(xs)) / 2)
        h_box = int((max(ys) - min(ys)) / 2)
        box = [x, y, w_box, h_box]

        if filter_junk and is_likely_junk(text, conf, box):
            junk_count += 1
            continue

        item = {
            "id": len(structured) + 1,
            "box": box,
            "text": text,
            "raw_text": raw_text,
            "confidence": float(conf)
        }
        structured.append(item)
        marker = "" if text == raw_text else f"  (raw: '{raw_text}')"
        print(f"  [{item['id']:3d}] ({x:3d},{y:3d}) {w_box:3d}x{h_box:3d} "
              f"conf={conf:.2f}  ->  '{text}'{marker}")

    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{filename}_fullocr.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(structured, f, indent=2)

    # visualization
    vis = original.copy()
    for item in structured:
        x, y, w_box, h_box = item["box"]
        cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        cv2.putText(vis, item["text"], (x, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    vis_path = os.path.join(output_dir, f"{filename}_fullocr.png")
    cv2.imwrite(vis_path, vis)

    print("-" * 60)
    print(f"Saved JSON: {output_path}")
    print(f"Saved viz:  {vis_path}")
    print(f"Total: {len(structured)} regions kept ({junk_count} filtered as junk)")

    return structured


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/vlm_reader.py <image_path>")
    else:
        read_full_image(sys.argv[1])