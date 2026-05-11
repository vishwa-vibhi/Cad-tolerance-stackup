"""
Stage 3: Validation and Structuring
Converts raw OCR text strings into typed structured records.

Input:  results/<image>_fullocr.json (output of Stage 2)
Output: results/<image>_structured.json (typed records)
"""

import json
import os
import sys
import re


# ============================================================
# Engineering material codes (from Bureau of Indian Standards)
# ============================================================
MATERIAL_CODES = {
    'MS':  'Mild Steel',
    'CS':  'Cast Steel',
    'CI':  'Cast Iron',
    'FS':  'Forged Steel',
    'GM':  'Gunmetal',
    'CR':  'Chromium',
    'AL':  'Aluminium',
    'BR':  'Brass',
    'MCS': 'Medium Carbon Steel',
    'HCS': 'High Carbon Steel',
    'LCS': 'Low Carbon Steel',
    'SS':  'Stainless Steel',
    'SPS': 'Spring Steel',
    'EN':  'EN-Series Steel',
    'IS':  'Indian Standard',
}

# Common engineering labels/keywords
LABELS = {
    'DIA', 'PCD', 'CSK', 'TYP', 'EQUI', 'EQUI-SP', 'EQUI SP',
    'OIL', 'HOLE', 'GROOVE', 'KEY', 'KEYWAY', 'DEEP', 'THICK',
    'BOLT', 'NUT', 'WASHER', 'PIN', 'STRAP',
    'PARTS', 'LIST', 'NAME', 'MATERIAL', 'QTY', 'NO',
}

# Common part names that appear in BOM tables
PART_NAMES = {
    'BODY', 'COVER', 'PLATE', 'SEAT', 'SLEEVE', 'COLLAR',
    'SPINDLE', 'HANDWHEEL', 'GLAND', 'BONNET', 'STUFFING',
    'VALVE', 'SPRING', 'LEVER', 'ROCKER', 'ARM', 'BOX',
    'CONNECTING ROD', 'ROD', 'COTTER', 'BRASS', 'JIB', 'SCREW',
    'SHAFT', 'FORK', 'CENTRAL BLOCK', 'PIN', 'BUSH', 'BEARING',
    'FLANGE', 'CRANK', 'PULLEY', 'GEAR', 'PISTON',
}

# Section/view markers
SECTION_MARKERS = {
    'X-X', 'X', 'A-A', 'A', 'B-B', 'B', 'Y-Y', 'Y',
    'C-C', 'C', 'D-D', 'D',
}


# ============================================================
# Classification functions
# ============================================================

def parse_thread_spec(text):
    """
    Parse 'M30', 'M30 × 2.5', 'M16', 'M12 × 1.5'
    Returns: {"type": "thread_spec", ...} or None
    """
    # M followed by digits, optionally × pitch
    m = re.match(r'^M\s*(\d+)\s*[×x*]?\s*(\d+(?:\.\d+)?)?$', text.strip(), re.IGNORECASE)
    if m:
        size = int(m.group(1))
        pitch = float(m.group(2)) if m.group(2) else None
        return {
            "type": "thread_spec",
            "thread_size": f"M{size}",
            "diameter_mm": size,
            "pitch_mm": pitch,
            "raw": text
        }
    return None


def parse_dimension_with_tolerance(text):
    """
    Parse various tolerance formats:
    - 75±0.5 (bilateral)
    - 60+0.15-0.00 (limit)
    - 25H7 (fit class)
    - Ø50H7/h6
    Returns: {"type": "dimension_with_tolerance", ...} or None
    """
    text = text.strip().replace(' ', '')

    # bilateral: 75±0.5
    m = re.match(r'^(Ø)?(\d+(?:\.\d+)?)±(\d+(?:\.\d+)?)$', text)
    if m:
        return {
            "type": "dimension_with_tolerance",
            "tolerance_format": "bilateral",
            "is_diameter": bool(m.group(1)),
            "nominal": float(m.group(2)),
            "tolerance_plus": float(m.group(3)),
            "tolerance_minus": float(m.group(3)),
            "raw": text
        }

    # limit: 60+0.15-0.00 or 60+0.15/-0.00
    m = re.match(r'^(Ø)?(\d+(?:\.\d+)?)\+(\d+(?:\.\d+)?)[-/]\s*-?(\d+(?:\.\d+)?)$', text)
    if m:
        return {
            "type": "dimension_with_tolerance",
            "tolerance_format": "limit",
            "is_diameter": bool(m.group(1)),
            "nominal": float(m.group(2)),
            "tolerance_plus": float(m.group(3)),
            "tolerance_minus": float(m.group(4)),
            "raw": text
        }

    # fit class: Ø25H7 or 25H7/h6
    m = re.match(r'^(Ø)?(\d+(?:\.\d+)?)([HhJjKkMmNnPpRrSs]\d+)(/[HhJjKkMmNnPpRrSs]\d+)?$', text)
    if m:
        return {
            "type": "dimension_with_tolerance",
            "tolerance_format": "fit_class",
            "is_diameter": bool(m.group(1)),
            "nominal": float(m.group(2)),
            "fit_hole": m.group(3),
            "fit_shaft": m.group(4)[1:] if m.group(4) else None,
            "raw": text
        }

    return None


def parse_diameter(text):
    """
    Parse 'Ø50', 'Ø6', 'Ø25'
    Returns: {"type": "diameter", ...} or None
    """
    m = re.match(r'^Ø(\d+(?:\.\d+)?)$', text.strip())
    if m:
        return {
            "type": "diameter",
            "value_mm": float(m.group(1)),
            "raw": text
        }
    return None


def parse_radius(text):
    """
    Parse 'R25', 'R3', 'R 16'
    Returns: {"type": "radius", ...} or None
    """
    m = re.match(r'^R\s*(\d+(?:\.\d+)?)$', text.strip(), re.IGNORECASE)
    if m:
        return {
            "type": "radius",
            "value_mm": float(m.group(1)),
            "raw": text
        }
    return None


def parse_pcd(text):
    """
    Parse 'PCD 50', '120 PCD', etc
    Returns: {"type": "pcd", ...} or None
    """
    m = re.match(r'^(\d+(?:\.\d+)?)\s*PCD$|^PCD\s*(\d+(?:\.\d+)?)$', text.strip(), re.IGNORECASE)
    if m:
        val = m.group(1) or m.group(2)
        return {
            "type": "pitch_circle_diameter",
            "value_mm": float(val),
            "raw": text
        }
    return None


def parse_hole_callout(text):
    """
    Parse '3 HOLES, DIA 6', '4 HOLES M10', '2 HOLES, M15'
    Returns: {"type": "hole_callout", ...} or None
    """
    text_upper = text.upper().replace(',', ' ').replace('.', ' ')
    # N HOLES DIA D
    m = re.search(r'(\d+)\s*HOLES?\s*[,]?\s*DIA\s*(\d+(?:\.\d+)?)', text_upper)
    if m:
        return {
            "type": "hole_callout",
            "count": int(m.group(1)),
            "diameter_mm": float(m.group(2)),
            "thread": False,
            "raw": text
        }
    # N HOLES M-size
    m = re.search(r'(\d+)\s*HOLES?\s*[,]?\s*M(\d+)', text_upper)
    if m:
        return {
            "type": "hole_callout",
            "count": int(m.group(1)),
            "diameter_mm": int(m.group(2)),
            "thread": True,
            "thread_spec": f"M{m.group(2)}",
            "raw": text
        }
    return None


def parse_keyway(text):
    """
    Parse 'KEY WAY 12 × 8', 'KEYWAY 8x4'
    """
    text_upper = text.upper().replace('×', 'x').replace('*', 'x')
    m = re.search(r'KEY\s*WAY\s*(\d+)\s*x\s*(\d+)', text_upper)
    if m:
        return {
            "type": "keyway",
            "width_mm": int(m.group(1)),
            "depth_mm": int(m.group(2)),
            "raw": text
        }
    return None


def parse_simple_dimension(text):
    """
    Parse plain dimensions like '75', '60', '40', '12.5'
    Returns: {"type": "dimension", ...} or None
    """
    m = re.match(r'^\d+(\.\d+)?$', text.strip())
    if m:
        return {
            "type": "dimension",
            "value_mm": float(text.strip()),
            "raw": text
        }
    return None


def parse_section_marker(text):
    """X-X, A-A, etc."""
    if text.strip().upper() in SECTION_MARKERS:
        return {
            "type": "section_marker",
            "label": text.strip().upper(),
            "raw": text
        }
    return None


def parse_material_code(text):
    """MS, FS, CI, GM, etc."""
    upper = text.strip().upper()
    if upper in MATERIAL_CODES:
        return {
            "type": "material_code",
            "code": upper,
            "name": MATERIAL_CODES[upper],
            "raw": text
        }
    return None


def parse_part_name(text):
    """Match common part names from BOM."""
    upper = text.strip().upper()
    # exact match
    if upper in PART_NAMES:
        return {
            "type": "part_name",
            "name": text.strip().title(),
            "raw": text
        }
    # partial match (e.g. 'Spring seat' contains 'SPRING' or 'SEAT')
    for pn in PART_NAMES:
        if pn in upper:
            return {
                "type": "part_name",
                "name": text.strip().title(),
                "raw": text,
                "matched_keyword": pn
            }
    return None


def parse_label(text):
    """DIA, EQUI-SP, OIL HOLE, etc."""
    upper = text.strip().upper()
    for label in LABELS:
        if label in upper:
            return {
                "type": "label",
                "label": upper,
                "raw": text
            }
    return None


def parse_small_number(text):
    """
    Classify small standalone numeric tokens when an image has many of them.

    In assembly drawings, BOM QTY, item numbers, and leader callouts often
    appear as many single- or double-digit tokens ('1', '2', '8', ...).
    Treating all of those as millimeter dimensions inflates the 'dimension'
    bucket and hurts downstream association.
    """
    t = (text or "").strip()
    if not t.isdigit() or not (1 <= len(t) <= 2):
        return None
    return {
        "type": "small_number",
        "value": int(t),
        "raw": text,
    }


# ============================================================
# Main classifier
# ============================================================

def classify_text(text, *, context=None):
    """
    Try parsers in order. Return first match.
    Order matters - more specific patterns checked first.
    """
    if not text or len(text.strip()) == 0:
        return {"type": "empty", "raw": text}

    ctx = context or {}
    many_small_numbers = bool(ctx.get("many_small_numbers"))

    # Try specialized parsers first
    parsers = [
        parse_dimension_with_tolerance,
        parse_thread_spec,
        parse_diameter,
        parse_radius,
        parse_pcd,
        parse_hole_callout,
        parse_keyway,
        parse_section_marker,
        parse_material_code,
        parse_label,
        parse_part_name,
        # If an image has lots of single-digit numeric tokens (common in BOM qty,
        # item numbers, leader callouts), avoid over-classifying them as
        # millimeter dimensions.
        parse_small_number if many_small_numbers else None,
        parse_simple_dimension,  # most generic last
    ]

    for parser in parsers:
        if parser is None:
            continue
        result = parser(text)
        if result:
            return result

    return {"type": "unknown", "raw": text}


# ============================================================
# Process JSON file
# ============================================================

def structure_ocr_output(json_path, output_dir="results"):
    """Read raw OCR output, classify each region, save structured."""
    with open(json_path, 'r') as f:
        regions = json.load(f)

    print(f"\n=== STAGE 3: Structuring ===")
    print(f"Input:  {json_path}")
    print(f"Items:  {len(regions)}")
    print("-" * 70)

    # Heuristic context for classification.
    # If an image contains many single-digit numbers, they are often BOM qty,
    # item numbers, or callouts—not true dimensions.
    small_numeric_tokens = 0
    for r in regions:
        t = (r.get("text") or "").strip()
        if t.isdigit() and 1 <= len(t) <= 2:
            small_numeric_tokens += 1
    context = {
        "many_small_numbers": small_numeric_tokens >= 12,
        "small_numeric_tokens": small_numeric_tokens,
    }

    structured = []
    type_counts = {}

    for region in regions:
        text = region.get('text', '')
        classification = classify_text(text, context=context)

        item = {
            "id": region.get('id'),
            "box": region.get('box'),
            "confidence": region.get('confidence'),
            **classification
        }
        structured.append(item)

        t = classification['type']
        type_counts[t] = type_counts.get(t, 0) + 1

        # only print non-unknown to keep output clean
        if t not in ('unknown', 'empty'):
            print(f"  [{region.get('id'):3d}] '{text}'  ->  {t}: {classification}")

    # save
    base = os.path.basename(json_path).replace('_fullocr.json', '')
    output_path = os.path.join(output_dir, f"{base}_structured.json")
    with open(output_path, 'w') as f:
        json.dump(structured, f, indent=2)

    # summary
    print("-" * 70)
    print(f"\nClassification summary:")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = count * 100 // len(regions)
        print(f"  {t:30s}  {count:4d}  ({pct}%)")

    print(f"\nSaved: {output_path}")
    return structured


def batch_structure(input_dir="results", output_dir="results"):
    """Run structuring on every _fullocr.json in input_dir."""
    json_files = [f for f in os.listdir(input_dir)
                  if f.endswith('_fullocr.json')]

    print(f"\n{'='*70}")
    print(f"BATCH STRUCTURING")
    print(f"Found {len(json_files)} OCR output files")
    print(f"{'='*70}")

    for f in sorted(json_files):
        try:
            structure_ocr_output(os.path.join(input_dir, f), output_dir)
        except Exception as e:
            print(f"ERROR on {f}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python src/validation.py <ocr_json_path>     # single file")
        print("  python src/validation.py --batch              # all files in results/")
    elif sys.argv[1] == "--batch":
        batch_structure()
    else:
        structure_ocr_output(sys.argv[1])