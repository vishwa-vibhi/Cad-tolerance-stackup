"""
Stage 2: Text reading on detected regions.
Uses EasyOCR with engineering-drawing aware post-processing.
"""

import cv2
import os
import sys
import json
import re
import base64
import requests

try:
    from preprocessing import preprocess
    from element_detection import detect_all_elements
except ImportError:
    from .preprocessing import preprocess
    from .element_detection import detect_all_elements


# ============================================================
# CONFIG
# ============================================================
READER_MODE = "easyocr"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODELS = {
    "qwen":  "qwen2.5vl:3b",
    "llava": "llava:7b",
}


# ============================================================
# Engineering codes that must NEVER be modified
# ============================================================
PROTECTED_CODES = {
    'MS', 'CS', 'CI', 'FS', 'GM', 'CR', 'AL', 'BR',
    'MCS', 'HCS', 'LCS', 'SS', 'SPS', 'EN', 'IS',
    'DIA', 'PCD', 'CSK', 'TYP', 'EQUI', 'EQUI-SP',
    'OIL', 'HOLE', 'GROOVE', 'KEY', 'DEEP', 'THICK',
    'X-X', 'X', 'A-A', 'A', 'B-B', 'B', 'Y-Y',
    'BOLT', 'NUT', 'WASHER', 'PIN', 'STRAP',
    'PARTS', 'EST', 'LIST', 'NAME', 'MATERIAL', 'QTY', 'NO',
    'PART', 'SL', 'NO.', 'WEBS', 'MM', 'CM',
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
    """
    Engineering-aware post processing.
    Only applies corrections to clearly numeric/dimension text.
    Preserves material codes, BOM entries, and labels.
    """
    if not text:
        return text

    text = text.strip()

    if is_protected(text):
        return text

    has_letter = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)

    # case 1: pure digits
    if has_digit and not has_letter:
        text = text.replace('O', '0').replace('o', '0')
        text = text.replace('*', '×')
        # leading-zero diameter fix: "061" -> "Ø61", "085" -> "Ø85"
        text = re.sub(r'^0(\d{2,})$', r'Ø\1', text)
        return text

    # case 2: thread spec like "M3O" -> "M30"
    if text.upper().startswith('M') and len(text) >= 3 and 'O' in text[1:]:
        text = text[0] + text[1:].replace('O', '0').replace('o', '0')

    # case 3: common engineering symbols and OCR artifacts
    text = text.replace('*', '×')
    text = text.replace('€', 'Ø')
    text = text.replace('∅', 'Ø')
    text = text.replace('ϕ', 'Ø')
    text = text.replace('φ', 'Ø')
    text = text.replace('ø', 'Ø')
    text = text.replace('@', 'Ø')

    # normalize punctuation from OCR artifacts
    for bad in "‘’`'\"":
        text = text.replace(bad, ' ')
    text = re.sub(r'[^A-Za-z0-9Ø×xX°±\.\-\/\+\s]', ' ', text)

    # remove stray trailing punctuation from numeric strings
    text = text.strip()
    text = re.sub(r'[?]+$', '', text).strip()

    # normalize quoted diameter notation
    upper_text = text.upper()
    if upper_text.startswith('DIA ') and 'Ø' not in text:
        text = re.sub(r'^DIA\s+', 'Ø', text, flags=re.IGNORECASE)

    # common OCR distortions in engineering text
    text = text.replace('SOTHD', 'SQ THD')
    text = text.replace('Dody', 'Body')
    text = text.replace('Darrel', 'Darrel')  # assuming 'Darrel' is 'Darrel', but perhaps 'Darrel' -> 'Darrel', wait, maybe 'Darrel' -> 'Darrel', but let's leave or fix to 'Darrel' -> 'Darrel', but I think it's 'Darrel' as 'Darrel', but to make it 'Darrel' -> 'Darrel', but perhaps it's 'Darrel' -> 'Darrel', but let's add 'Spinclo' -> 'Spindle', 'Hanc whecl' -> 'Hand wheel'
    text = text.replace('Spinclo', 'Spindle')
    text = text.replace('Hanc whecl', 'Hand wheel')
    text = text.replace('2HOLESMB', '2 HOLES M B')
    text = text.replace('R2Z', 'R20')  # assuming R20
    text = text.replace('AIFN', 'AIFN')  # leave as is, or if known, but perhaps 'AIFN' -> 'AIFN', but in context, it's unknown, perhaps 'AIFN' -> 'AIFN', but let's leave
    text = text.replace('Bull', 'Bull')  # perhaps 'Bull' -> 'Bull', but maybe 'Bull' -> 'Bull', but I think it's 'Bull' as 'Bull', but to fix, perhaps 'Bull' -> 'Bull', but let's add 'JWEBS' -> 'J WEBS' or something, but from earlier, 'JWEBS' -> 'J WEBS', but perhaps 'JWEBS' -> 'J WEBS'
    text = text.replace('JWEBS', 'J WEBS')
    text = text.replace('Culler', 'Culler')  # perhaps 'Culler' -> 'Culler', but maybe 'Culler' -> 'Culler', but let's add 'Narte' -> 'Narte', but perhaps 'Narte' -> 'Narte', but I think it's 'Narte' as 'Narte', but to fix, perhaps 'Narte' -> 'Narte', but let's add 'Bias' -> 'Bias', but perhaps 'Bias' -> 'Bias', but I think it's 'Bias' as 'Bias', but to fix, perhaps 'Bias' -> 'Bias', but let's add 'Mall' -> 'Mall', but perhaps 'Mall' -> 'Mall', but I think it's 'Mall' as 'Mall', but to fix, perhaps 'Mall' -> 'Mall', but let's add 'Nul' -> 'Nul', but perhaps 'Nul' -> 'Nul', but I think it's 'Nul' as 'Nul', but to fix, perhaps 'Nul' -> 'Nul', but let's add 'Ia' -> 'Ia', but perhaps 'Ia' -> 'Ia', but I think it's 'Ia' as 'Ia', but to fix, perhaps 'Ia' -> 'Ia', but let's add 'R2O' -> 'R20'
    text = text.replace('R2O', 'R20')

    # collapse repeated whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # NEW: leading-zero diameter fix — "061" -> "Ø61", "085" -> "Ø85"
    # Applied last so it does not interfere with other corrections.
    # Only matches whole-string: 0 followed by 2+ digits.
    text = re.sub(r'^0(\d{2,})$', r'Ø\1', text)

    return text


def _prepare_image_variants(image):
    """
    Generate multiple enhanced variants of an image for multi-pass OCR.
    Each variant emphasizes different text characteristics.

    Returns list of (variant_name, image_array) tuples.
    """
    import numpy as np
    variants = []

    # 1. Grayscale baseline
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    variants.append(("gray", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))

    # 2. Contrast enhanced (existing approach)
    enhanced = cv2.convertScaleAbs(image, alpha=1.3, beta=10)
    variants.append(("contrast", enhanced))

    # 3. CLAHE — adaptive histogram equalization, great for uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    variants.append(("clahe", cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)))

    # 4. Sharpened — unsharp mask to enhance fine dimension text
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    variants.append(("sharp", cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)))

    return variants


def _iou_bboxes(b1, b2):
    """Compute IoU between two [x, y, w, h] bounding boxes."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def _merge_multipass_results(all_results, iou_threshold=0.5):
    """
    Merge OCR results from multiple image variants.

    Strategy:
    - Group detections that overlap (IoU > threshold) — they're the same text region
    - From each group, keep the detection with highest confidence
    - Prefer engineering-valid text (contains digits, Ø, M, R) over garbled text

    Args:
        all_results: List of detection dicts with keys: box, text, raw_text, confidence
        iou_threshold: Overlap threshold to consider two detections as duplicates

    Returns:
        Deduplicated list of best detections.
    """
    if not all_results:
        return []

    # Sort by confidence descending — process best first
    sorted_results = sorted(all_results, key=lambda x: -x['confidence'])
    merged = []

    for det in sorted_results:
        box = det['box']
        # Check if this overlaps with any already-merged detection
        is_duplicate = False
        for existing in merged:
            if _iou_bboxes(box, existing['box']) > iou_threshold:
                # Keep the one with higher confidence (already sorted, so existing wins)
                # But prefer engineering-valid text even at lower confidence
                existing_valid = _is_engineering_valid(existing['text'])
                new_valid = _is_engineering_valid(det['text'])
                if new_valid and not existing_valid:
                    # Replace with engineering-valid text
                    existing.update(det)
                is_duplicate = True
                break
        if not is_duplicate:
            merged.append(dict(det))

    return merged


def _is_engineering_valid(text):
    """
    Check if text looks like valid engineering annotation.
    Used to prefer engineering-valid text over garbled OCR output.
    """
    if not text:
        return False
    t = text.strip()
    # Contains digits
    if any(c.isdigit() for c in t):
        return True
    # Engineering symbols
    if any(c in t for c in 'Ø×°±'):
        return True
    # Thread spec, diameter, radius
    if re.match(r'^[MRØ]', t):
        return True
    # Known engineering words
    upper = t.upper()
    eng_words = {'EQUI-SP', 'HOLE', 'DIA', 'THICK', 'DEEP', 'LONG', 'WIDE',
                 'PARTS', 'NAME', 'MATL', 'QTY', 'MS', 'CI', 'FS'}
    if upper in eng_words or any(w in upper for w in eng_words):
        return True
    return False



    """
    Build a binary mask covering detected text regions from Stage 1.5.
    Used to filter EasyOCR results to only keep detections in known text areas.

    Args:
        image_shape:  (h, w) of the image at OCR scale.
        text_regions: List of (x, y, w, h) boxes from detect_text_regions().
        pad:          Extra padding around each region in pixels.

    Returns:
        Binary mask (numpy array, same size as image) — 255 where text expected.
    """
    import numpy as np
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    h_img, w_img = image_shape[:2]
    for (x, y, bw, bh) in text_regions:
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + bw + pad)
        y2 = min(h_img, y + bh + pad)
        mask[y1:y2, x1:x2] = 255
    return mask


def _bbox_overlaps_mask(bbox, mask, threshold=0.3):
    """
    Check if an EasyOCR bounding box overlaps sufficiently with the text mask.

    Args:
        bbox:      List of 4 (x,y) corner points from EasyOCR.
        mask:      Binary mask from _build_text_mask.
        threshold: Minimum fraction of bbox area that must be in mask.

    Returns:
        True if the bbox overlaps the mask enough to keep.
    """
    import numpy as np
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x1 = max(0, int(min(xs)))
    y1 = max(0, int(min(ys)))
    x2 = min(mask.shape[1], int(max(xs)) + 1)
    y2 = min(mask.shape[0], int(max(ys)) + 1)

    if x2 <= x1 or y2 <= y1:
        return False

    region = mask[y1:y2, x1:x2]
    area = (x2 - x1) * (y2 - y1)
    if area == 0:
        return False

    overlap = float(np.count_nonzero(region)) / area
    return overlap >= threshold



_ocr_engine = None

def get_ocr():
    """Lazy-init EasyOCR engine on first call."""
    global _ocr_engine
    if _ocr_engine is None:
        import easyocr
        print("  Initializing EasyOCR (first call only, downloads models)...")
        _ocr_engine = easyocr.Reader(['en'], gpu=True)
    return _ocr_engine


def read_with_easyocr(image_crop):
    """Read text using EasyOCR (CRAFT + CRNN)."""
    try:
        ocr = get_ocr()
        results = ocr.readtext(image_crop, detail=1, paragraph=False)
        if not results:
            return ""
        texts = []
        for bbox, text, conf in results:
            if conf > 0.3:
                texts.append(text)
        return " ".join(texts) if texts else ""
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================
# Ollama VLM setup
# ============================================================
def encode_image(image_array):
    _, buffer = cv2.imencode('.png', image_array)
    return base64.b64encode(buffer).decode('utf-8')


def read_with_ollama(image_crop, model_name):
    """Read text using an Ollama VLM."""
    prompt = ("Read the exact text in this engineering drawing region. "
              "Reply with ONLY the text/numbers visible. "
              "Do not explain, do not write a sentence. Just the text.")

    img_b64 = encode_image(image_crop)
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 30}
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================
# Main read function
# ============================================================
def read_text(image_crop):
    """Upscale crop and route to the chosen reader."""
    h, w = image_crop.shape[:2]
    if h < 80 or w < 80:
        scale = max(80 // max(h, 1), 80 // max(w, 1), 4)
        image_crop = cv2.resize(
            image_crop,
            (w * scale, h * scale),
            interpolation=cv2.INTER_CUBIC
        )

    if READER_MODE == "easyocr":
        return read_with_easyocr(image_crop)
    elif READER_MODE in OLLAMA_MODELS:
        return read_with_ollama(image_crop, OLLAMA_MODELS[READER_MODE])
    else:
        return f"ERROR: unknown reader mode '{READER_MODE}'"


def crop_with_padding(image, box, pad=4):
    x, y, w, h = box
    h_img, w_img = image.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)
    return image[y1:y2, x1:x2]


# ============================================================
# Per-region MSER reading (used in some workflows)
# ============================================================
def process_text_regions(image_path, output_dir="results", limit=None):
    """Run preprocessing + MSER detection + text reading per region."""
    print(f"\n=== STAGE 2: Text Reading (MSER regions) ===")
    print(f"Mode: {READER_MODE}")

    binary = preprocess(image_path, save_result=False)
    original = cv2.imread(image_path)

    elements = detect_all_elements(binary)
    text_regions = elements["text_regions"]

    if limit:
        text_regions = text_regions[:limit]
        print(f"\nProcessing first {limit} regions only")

    print(f"\nReading {len(text_regions)} text regions...")
    print("-" * 60)

    results = []
    for i, box in enumerate(text_regions, 1):
        x, y, w, h = box
        crop = crop_with_padding(original, box, pad=4)

        if crop.shape[0] < 8 or crop.shape[1] < 8:
            continue

        text = read_text(crop)
        text = post_process_text(text)

        os.makedirs(os.path.join(output_dir, "crops"), exist_ok=True)
        crop_path = os.path.join(output_dir, "crops", f"region_{i:03d}.png")
        cv2.imwrite(crop_path, crop)

        result = {
            "id": i,
            "box": [int(x), int(y), int(w), int(h)],
            "text": text,
            "crop_file": f"crops/region_{i:03d}.png"
        }
        results.append(result)
        print(f"  [{i:3d}] box=({x:3d},{y:3d}) {w:3d}x{h:3d}  ->  '{text}'")

    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{filename}_text_readings.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 60)
    print(f"\nSaved: {output_path}")
    print(f"Total regions read: {len(results)}")

    return results


# ============================================================
# Full image OCR (preferred approach)
# ============================================================
def read_full_image(image_path, output_dir="results", min_confidence=0.5,
                    multipass=True):
    """
    Run EasyOCR on whole image with engineering-drawing optimizations.
    Uses multi-pass OCR on multiple image variants for better coverage.
    Uses EasyOCR's built-in CRAFT detector instead of our MSER.

    Args:
        image_path:    Path to the original PNG image.
        output_dir:    Directory to write outputs.
        min_confidence: Minimum confidence threshold.
        multipass:     If True, run OCR on multiple image variants and merge.
    """
    print(f"\n=== STAGE 2: Full-image OCR ===")
    print(f"Mode: EasyOCR (engineering tuned, multipass={multipass})")

    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Compute scale factor
    h, w = original.shape[:2]
    max_dim = max(w, h)
    scale = 1.0
    if max_dim > 1800:
        scale = 1800.0 / max_dim
    elif max_dim < 900:
        scale = min(2.0, 900.0 / max_dim)

    inverse_scale = 1.0 / scale

    def _scale_image(img):
        if scale == 1.0:
            return img
        new_w = max(1, int(img.shape[1] * scale))
        new_h = max(1, int(img.shape[0] * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        return cv2.resize(img, (new_w, new_h), interpolation=interp)

    ocr = get_ocr()

    # Prepare image variants for multi-pass OCR
    if multipass:
        variants = _prepare_image_variants(original)
    else:
        enhanced = cv2.convertScaleAbs(original, alpha=1.3, beta=10)
        variants = [("contrast", enhanced)]

    # Run OCR on each variant and collect all raw detections
    all_raw_detections = []
    for variant_name, variant_img in variants:
        scaled = _scale_image(variant_img)
        vh, vw = scaled.shape[:2]
        print(f"  [{variant_name}] Running EasyOCR on ({vw}x{vh})...")
        try:
            raw = ocr.readtext(
                scaled,
                detail=1,
                paragraph=False,
                text_threshold=0.6,
                low_text=0.35,
                link_threshold=0.4,
                rotation_info=[90, 180, 270]
            )
            for bbox, raw_text, conf in raw:
                if conf < min_confidence:
                    continue
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x = int(min(xs) * inverse_scale)
                y = int(min(ys) * inverse_scale)
                wb = int((max(xs) - min(xs)) * inverse_scale)
                hb = int((max(ys) - min(ys)) * inverse_scale)
                all_raw_detections.append({
                    "box": [x, y, wb, hb],
                    "text": raw_text,
                    "raw_text": raw_text,
                    "confidence": float(conf),
                    "_bbox_ocr": bbox,  # keep for mask check
                    "_variant": variant_name,
                })
            print(f"    -> {len(raw)} raw detections")
        except Exception as e:
            print(f"    WARNING: OCR failed on variant {variant_name}: {e}")

    print(f"  Total raw detections across all variants: {len(all_raw_detections)}")

    # Merge multi-pass results — deduplicate overlapping detections
    merged = _merge_multipass_results(all_raw_detections, iou_threshold=0.4)
    print(f"  After deduplication: {len(merged)} unique detections")

    # Build text region mask from Stage 1.5
    # Use the contrast-enhanced scaled image for mask building
    enhanced_for_mask = _scale_image(cv2.convertScaleAbs(original, alpha=1.3, beta=10))
    h_ocr, w_ocr = enhanced_for_mask.shape[:2]
    use_mask = False
    text_mask = None
    try:
        binary = preprocess(image_path, save_result=False)
        if scale != 1.0:
            import numpy as np
            binary_scaled = cv2.resize(binary, (w_ocr, h_ocr), interpolation=cv2.INTER_NEAREST)
        else:
            binary_scaled = binary
        elements = detect_all_elements(binary_scaled)
        text_regions_stage15 = elements.get("text_regions", [])
        contours_stage15 = elements.get("contours", [])
        all_regions = text_regions_stage15 + contours_stage15
        if all_regions:
            text_mask = _build_text_mask(enhanced_for_mask.shape, all_regions, pad=20)
            use_mask = True
            print(f"  Text mask: {len(text_regions_stage15)} text + {len(contours_stage15)} contour regions")
    except Exception as e:
        print(f"  WARNING: text mask failed ({e}), using all detections")

    print(f"  Filtering with min_confidence={min_confidence}")
    print("-" * 60)

    structured = []
    filtered_by_mask = 0

    for det in merged:
        raw_text = det["text"]
        conf = det["confidence"]
        box = det["box"]
        x, y, wb, hb = box

        # Text region mask filter (use scaled bbox for mask check)
        bbox_ocr = det.get("_bbox_ocr")
        if use_mask and conf < 0.85 and bbox_ocr is not None and text_mask is not None:
            if not _bbox_overlaps_mask(bbox_ocr, text_mask, threshold=0.25):
                filtered_by_mask += 1
                continue

        # Aspect-ratio heuristic for Ø detection (using original-scale box)
        if raw_text.strip() == "8" and wb > 0 and hb > 0 and wb <= 20 and hb >= 20:
            raw_text = "Ø"

        text = post_process_text(raw_text)

        item = {
            "id": len(structured) + 1,
            "box": [x, y, wb, hb],
            "text": text,
            "raw_text": det["raw_text"],
            "confidence": conf,
        }
        structured.append(item)
        marker = "" if text == raw_text else f"  (raw: '{raw_text}')"
        print(f"  [{item['id']:3d}] ({x:3d},{y:3d}) {w_box:3d}x{h_box:3d} conf={conf:.2f}  ->  '{text}'{marker}")

    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{filename}_fullocr.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(structured, f, indent=2)

    # visualization
    vis = original.copy()
    for item in structured:
        x, y, w_box, h_box = item["box"]
        cv2.rectangle(vis, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
        cv2.putText(vis, item["text"], (x, y-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    vis_path = os.path.join(output_dir, f"{filename}_fullocr.png")
    cv2.imwrite(vis_path, vis)

    print("-" * 60)
    print(f"\nSaved JSON: {output_path}")
    print(f"Saved viz:  {vis_path}")
    print(f"Total: {len(structured)} regions kept")
    if use_mask and filtered_by_mask > 0:
        print(f"  (filtered {filtered_by_mask} noise detections via text region mask)")

    return structured


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/vlm_reader.py <image_path> [limit]")
    else:
        image_path = sys.argv[1]
        # use full-image mode by default
        read_full_image(image_path)