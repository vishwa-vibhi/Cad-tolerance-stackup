"""
Stage 3: Validation & Structuring for the CAD Tolerance Stack-Up Analysis pipeline.

This module sits between Stage 2 (EasyOCR output, _fullocr.json) and Stage 4
(geometric association). It consumes _fullocr.json files and produces
_structured.json files by:

  1. Normalising known OCR artefacts in the `text` field of each entry.
  2. Classifying each normalised text string into one of 14 engineering
     annotation types using a priority-ordered rule chain.
  3. Extracting typed sub-fields (`parsed`) from each classified entry.
  4. Writing a well-structured _structured.json per image.

The module is pure Python (stdlib only) and is designed to process all 36
images in under 60 seconds on the target hardware.

Usage (CLI):
    python src/validation.py <input_dir> <output_dir>

Usage (API):
    from src.validation import validate_file, validate_batch
"""

import re
import json
import os
import sys
import pathlib


# ============================================================
# Engineering codes that must NEVER be modified by normalisation
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


# ============================================================
# Compiled regex constants (module-level for performance)
# ============================================================

# Priority 1: hole_callout — must contain HOLE + numeric/DIA
RE_HOLE = re.compile(r'HOLE', re.IGNORECASE)
RE_HOLE_NUMERIC = re.compile(r'(\d+|DIA)', re.IGNORECASE)

# Priority 2: thread_spec — M followed by digits, optional pitch
RE_THREAD = re.compile(r'^M\d+(\s*[×x]\s*\d+(\.\d+)?)?$', re.IGNORECASE)

# Priority 3: diameter_callout — Ø prefix or DIA prefix (no compound note)
RE_DIAMETER = re.compile(r'^(Ø\d+(\.\d+)?|DIA\s+\d+(\.\d+)?)$', re.IGNORECASE)

# Priority 3.5: radius_callout — R followed by digits, optional decimal (uppercase R only)
RE_RADIUS = re.compile(r'^R\d+(\.\d+)?$')

# Priority 4: dimension_with_note — number + THICK/DEEP/LONG/WIDE
RE_DIM_NOTE = re.compile(r'(\d+(\.\d+)?)\s*(THICK|DEEP|LONG|WIDE)|^DIA\s+\d+.*\s+(THICK|DEEP|LONG|WIDE)', re.IGNORECASE)

# Priority 5: tolerance — ±, +x/-y, H7/h6 patterns
RE_TOLERANCE = re.compile(r'^[±]\d|^\+\d.*\/\s*-\d|^[A-Z]\d+\/[a-z]\d+|^[A-Z]\d+$', re.IGNORECASE)

# Priority 6: spacing_annotation
RE_SPACING = re.compile(r'^EQUI[-\s]SP$', re.IGNORECASE)

# Priority 10: section_marker — only in Cat 1 and Cat 3
RE_SECTION = re.compile(r'^[A-Z]-[A-Z]$|^[A-Z]$')

# Priority 11: balloon_number — single digit 1-9, only in Cat 2 and Cat 3
RE_BALLOON = re.compile(r'^[1-9]$')

# Priority 12: quantity — 1-2 digit number, Cat 2 only
RE_QUANTITY = re.compile(r'^[1-9]\d?$')

# Priority 13: dimension_value — bare number
RE_DIMENSION = re.compile(r'^\d+(\.\d+)?$')


# ============================================================
# Set constants
# ============================================================

BOM_HEADERS = {
    'PARTS LIST', 'NAME', 'MATERIAL', 'QTY', 'NO', 'SL NO', 'PART NO',
    # Expanded: common abbreviations and variants in Indian standard drawings
    'MATL', 'MAT', 'SL.NO', 'SL. NO', 'PART NAME', 'PART NO.', 'NO.',
}

MATERIAL_CODES = {'MS', 'CI', 'FS', 'GM', 'CS', 'CR', 'AL', 'BR'}

# Full material names (as opposed to 2-letter codes) — used in BOM tables
MATERIAL_NAMES = {
    'BABBIT', 'BRASS', 'NI-CR STEEL', 'CD-AG', 'CAST IRON',
    'MILD STEEL', 'HIGH CARBON STEEL', 'LOW CARBON STEEL',
    'STAINLESS STEEL', 'ALUMINUM', 'BRONZE', 'COPPER',
}

PART_NAMES = {
    'VALVE', 'SPRING', 'PIN', 'BODY', 'SPINDLE', 'HANDWHEEL',
    'GLAND', 'BONNET', 'SLEEVE', 'COLLAR', 'COVER', 'PLATE',
    'SEAT', 'NUT', 'BOLT', 'WASHER',
    # Additional part names from dataset
    'FORK', 'BLOCK', 'PIECE', 'HOLDER', 'SWIVEL', 'SHEAVE',
    'ASSEMBLY', 'TOOL', 'CENTRAL', 'MODULE',
    # Compound part names from Category 2 assembly drawings
    'ARTICULATED ROD', 'COVER PLATE', 'ROD END', 'LOCK NUT',
    'LINK PIN', 'PISTON PIN', 'PISTON RING', 'ROD BUSH-UPPER',
    'ROD BUSH-LOWER', 'MASTER ROD BEARING', 'PISTON PIN PLUG',
    'PISTON', 'CONNECTING ROD', 'COTTER PIN',
}

VALID_TYPES = {
    'dimension_value', 'thread_spec', 'tolerance', 'diameter_callout',
    'hole_callout', 'section_marker', 'spacing_annotation', 'material_code',
    'part_name', 'bom_header', 'balloon_number', 'quantity',
    'dimension_with_note', 'unknown',
    # New types added by ocr-accuracy-improvements
    'radius_callout', 'material_name',
}


# ============================================================
# Internal helpers
# ============================================================

def _is_protected(token: str) -> bool:
    """
    Return True if the entire text string is a PROTECTED_CODE token.

    The check is a whole-string comparison: the stripped, uppercased token
    must be an exact member of PROTECTED_CODES. Used by normalise_text to
    skip correction on protected strings.

    Args:
        token: The text string to check.

    Returns:
        True if token (stripped and uppercased) is in PROTECTED_CODES,
        False otherwise.
    """
    return token.strip().upper() in PROTECTED_CODES


def _get_box_safe(entry: dict):
    """
    Safely extract (x, y, w, h) from an OCR entry's box field.

    Args:
        entry: A classified OCR entry dict.

    Returns:
        A 4-tuple of ints (x, y, w, h) if the box field is valid,
        or None if the box is missing, malformed, or non-numeric.
    """
    box = entry.get("box")
    if not isinstance(box, list) or len(box) < 4:
        return None
    try:
        return (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    except (TypeError, ValueError):
        return None


def _edit_distance(a: str, b: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings.

    Used for fuzzy part-name matching (threshold <= 1). Pure Python,
    O(len(a) * len(b)) time and space.

    Args:
        a: First string.
        b: Second string.

    Returns:
        The minimum number of single-character edits (insertions, deletions,
        substitutions) required to transform a into b.
    """
    len_a, len_b = len(a), len(b)
    # Handle degenerate cases
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    # Two-row rolling array: prev holds distances for row i-1, curr for row i
    prev = list(range(len_b + 1))
    curr = [0] * (len_b + 1)

    for i in range(1, len_a + 1):
        curr[0] = i
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, curr = curr, prev

    return prev[len_b]


# ============================================================
# Normalisation
# ============================================================

def normalise_text(text: str) -> str:
    """
    Apply OCR artefact corrections to a single text string.

    Corrections are applied in this order:
      1. Strip leading/trailing whitespace.
      2. If the entire stripped string is in PROTECTED_CODES, return unchanged.
      3a. Leading-zero diameter fix: "018" → "Ø18" (only for 0 + 2+ digits).
      3b. Degree symbol fix: '45"' → "45°".
      3c. THICK typo fix: "IHICK" → "THICK", "MICK" → "THICK" (case-insensitive).
      3d. Multiplication symbol fix: "*" → "×".

    The raw_text field is never modified; only the text field is normalised.

    Args:
        text: The raw text string from an OCR entry.

    Returns:
        The normalised text string.
    """
    # Step 1: strip whitespace
    t = text.strip()
    if not t:
        return ""

    # Step 2: protected code check — return unchanged
    if _is_protected(t):
        return t

    # Step 3a: leading-zero diameter fix — whole-string match only
    t = re.sub(r'^0(\d{2,})$', r'Ø\1', t)

    # Step 3b: degree symbol fix
    t = re.sub(r'(\d{1,2})"', r'\1°', t)

    # Step 3c: THICK typo fix (case-insensitive substring replacement)
    t = re.sub(r'IHICK', 'THICK', t, flags=re.IGNORECASE)
    t = re.sub(r'MICK', 'THICK', t, flags=re.IGNORECASE)

    # Step 3d: multiplication symbol fix
    t = t.replace('*', '×')

    return t


# ============================================================
# Category detection
# ============================================================

def detect_category(filename: str) -> int:
    """
    Infer drawing category from filename.

    The category is embedded in the filename by convention:
      - cad1_NNN → category 1 (part drawings)
      - cad2_NNN → category 2 (assembly drawings)
      - cad3_NNN → category 3 (mixed)

    Args:
        filename: The filename (basename or full path) to inspect.

    Returns:
        1, 2, or 3 if the category can be determined; 0 with a printed
        warning if the filename does not match any known pattern.
    """
    name = os.path.basename(filename).lower()
    if 'cad1_' in name:
        return 1
    if 'cad2_' in name:
        return 2
    if 'cad3_' in name:
        return 3
    print(f"WARNING: cannot determine category from filename '{filename}', defaulting to 0")
    return 0


# ============================================================
# Classification
# ============================================================

def classify(text: str, category: int) -> str:
    """
    Assign one of the 14 type strings to a normalised text string.

    Evaluates patterns in strict priority order to resolve ambiguities
    between overlapping patterns (e.g., hole_callout before diameter_callout,
    thread_spec before dimension_value). Two types are category-gated:
      - section_marker: categories 1 and 3 only
      - balloon_number: categories 2 and 3 only
      - quantity:       category 2 only

    Args:
        text:     The normalised text string to classify.
        category: The drawing category (1, 2, or 3; 0 for unknown).

    Returns:
        One of the 14 type strings defined in VALID_TYPES.
    """
    t = text.strip()
    if not t:
        return "unknown"

    # P1: hole_callout
    if RE_HOLE.search(t) and RE_HOLE_NUMERIC.search(t):
        return "hole_callout"

    # P2: thread_spec
    if RE_THREAD.match(t):
        return "thread_spec"

    # P3: diameter_callout
    if RE_DIAMETER.match(t):
        return "diameter_callout"

    # P3.5: radius_callout — R followed by digits (uppercase R only, no IGNORECASE)
    if RE_RADIUS.match(t):
        return "radius_callout"

    # P4: dimension_with_note
    if RE_DIM_NOTE.search(t):
        return "dimension_with_note"

    # P5: tolerance
    if RE_TOLERANCE.match(t):
        return "tolerance"

    # P6: spacing_annotation
    if RE_SPACING.match(t):
        return "spacing_annotation"

    # P7: bom_header
    if t.upper() in BOM_HEADERS:
        return "bom_header"

    # P8: material_code
    if t.upper() in MATERIAL_CODES:
        return "material_code"

    # P8.5: material_name (exact or fuzzy edit distance <= 2)
    upper_t = t.upper()
    for mat in MATERIAL_NAMES:
        if upper_t == mat or _edit_distance(upper_t, mat) <= 2:
            return "material_name"

    # P9: part_name (exact or fuzzy — compound names use threshold <= 2, single-word <= 1)
    for name in PART_NAMES:
        is_compound = ' ' in name or '-' in name
        threshold = 2 if is_compound else 1
        if upper_t == name or _edit_distance(upper_t, name) <= threshold:
            return "part_name"

    # P10: section_marker — Cat 1 and 3 only
    if category in (1, 3) and RE_SECTION.match(t):
        return "section_marker"

    # P11: balloon_number — Cat 2 and 3 only
    if category in (2, 3) and RE_BALLOON.match(t):
        return "balloon_number"

    # P12: quantity — Cat 2 only
    if category == 2 and RE_QUANTITY.match(t):
        return "quantity"

    # P13: dimension_value
    if RE_DIMENSION.match(t):
        return "dimension_value"

    # P13.5: single punctuation / noise characters → unknown immediately
    if len(t) <= 2 and not t.isalnum():
        return "unknown"

    # P14: fallback
    return "unknown"


# ============================================================
# Parsed field extraction
# ============================================================

def extract_parsed(type_: str, text: str) -> dict:
    """
    Extract typed sub-fields from a classified text string.

    Returns a dict whose keys match the schema defined for the given type:
      - dimension_value:    {"value": float}
      - thread_spec:        {"nominal": str, "pitch": float or None}
      - tolerance:          {"tolerance_string": str}
      - diameter_callout:   {"diameter": float or None}
      - hole_callout:       {"raw": str}
      - section_marker:     {"label": str}
      - spacing_annotation: {"annotation": "EQUI-SP"}
      - material_code:      {"code": str}  (uppercase)
      - part_name:          {"name": str}  (title case)
      - bom_header:         {"header": str} (uppercase)
      - balloon_number:     {"number": int}
      - quantity:           {"qty": int}
      - dimension_with_note:{"raw": str}
      - unknown:            {}

    Numeric conversions use try/except; on failure the field is set to None.

    Args:
        type_: The classification type string (one of VALID_TYPES).
        text:  The normalised text string.

    Returns:
        A dict of parsed sub-fields appropriate for the given type.
    """
    if type_ == "dimension_value":
        try:
            return {"value": float(text.strip())}
        except (ValueError, IndexError):
            return {"value": None}

    if type_ == "thread_spec":
        m = re.match(r'^(M\d+)(?:\s*[×x]\s*(\d+(?:\.\d+)?))?', text.strip(), re.IGNORECASE)
        if m:
            nominal = m.group(1).upper()
            pitch = float(m.group(2)) if m.group(2) else None
            return {"nominal": nominal, "pitch": pitch}
        return {"nominal": text.strip(), "pitch": None}

    if type_ == "tolerance":
        return {"tolerance_string": text.strip()}

    if type_ == "diameter_callout":
        t = text.strip()
        t = re.sub(r'^Ø', '', t)
        t = re.sub(r'^DIA\s+', '', t, flags=re.IGNORECASE)
        try:
            return {"diameter": float(t.split()[0])}
        except (ValueError, IndexError):
            return {"diameter": None}

    if type_ == "hole_callout":
        return {"raw": text.strip()}

    if type_ == "section_marker":
        return {"label": text.strip()}

    if type_ == "spacing_annotation":
        return {"annotation": "EQUI-SP"}

    if type_ == "material_code":
        return {"code": text.strip().upper()}

    if type_ == "part_name":
        return {"name": text.strip().title()}

    if type_ == "bom_header":
        return {"header": text.strip().upper()}

    if type_ == "balloon_number":
        try:
            return {"number": int(text.strip())}
        except (ValueError, IndexError):
            return {"number": None}

    if type_ == "quantity":
        try:
            return {"qty": int(text.strip())}
        except (ValueError, IndexError):
            return {"qty": None}

    if type_ == "dimension_with_note":
        return {"raw": text.strip()}

    if type_ == "radius_callout":
        try:
            return {"radius": float(text.strip()[1:])}  # strip leading 'R'
        except (ValueError, IndexError):
            return {"radius": None}

    if type_ == "material_name":
        return {"name": text.strip().title()}

    # unknown
    return {}


# ============================================================
# BOM row reconstruction
# ============================================================

def reconstruct_bom_rows(classified_entries: list, category: int) -> list:
    """
    Group spatially-adjacent BOM fragments into structured rows.

    Only runs for Category 2 (assembly drawings with BOM tables).
    Uses y-centroid proximity (Y_TOLERANCE=10px) to group entries into rows,
    then assigns roles by type within each row.

    Args:
        classified_entries: List of classified OCR entry dicts.
        category:           Drawing category (1, 2, or 3).

    Returns:
        List of BOM row dicts with keys: part_no, part_name, material, qty.
        Returns [] for non-Category-2 images or if no BOM entries found.
    """
    # Gate: only Category 2 images have BOM tables
    if category != 2:
        return []

    BOM_TYPES = {'balloon_number', 'part_name', 'material_code', 'material_name', 'quantity'}
    Y_TOLERANCE = 10  # pixels

    # Filter to BOM-relevant entries with valid boxes
    relevant = []
    for entry in classified_entries:
        if entry.get("type") not in BOM_TYPES:
            continue
        box = _get_box_safe(entry)
        if box is None:
            continue
        relevant.append((entry, box))

    if not relevant:
        return []

    # Compute y-centroid and sort top-to-bottom
    def y_center(item):
        _, box = item
        return box[1] + box[3] / 2.0

    relevant.sort(key=y_center)

    # Group into rows by y-centroid proximity
    rows = []
    current_row = [relevant[0]]
    current_y = y_center(relevant[0])

    for item in relevant[1:]:
        if abs(y_center(item) - current_y) <= Y_TOLERANCE:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]
            current_y = y_center(item)
    rows.append(current_row)

    # Assign roles within each row (sort by x, first match wins per role)
    bom_rows = []
    for row_items in rows:
        row_items.sort(key=lambda item: item[1][0])  # sort by x

        part_no = None
        part_name_val = None
        material = None
        qty = None

        for entry, _ in row_items:
            t = entry.get("type")
            parsed = entry.get("parsed", {})
            if t == "balloon_number" and part_no is None:
                part_no = parsed.get("number")
            elif t == "part_name" and part_name_val is None:
                part_name_val = parsed.get("name")
            elif t == "material_code" and material is None:
                material = parsed.get("code")
            elif t == "material_name" and material is None:
                material = parsed.get("name")
            elif t == "quantity" and qty is None:
                qty = parsed.get("qty")

        bom_rows.append({
            "part_no": part_no,
            "part_name": part_name_val,
            "material": material,
            "qty": qty,
        })

    return bom_rows


# ============================================================
# Output assembly
# ============================================================

def build_structured_output(source_file: str, category: int, classified_entries: list) -> dict:
    """
    Assemble the final output dict from processed entries.

    Computes total_detections (== len(classified_entries)) and a summary
    dict mapping each type string present to its count.

    Args:
        source_file:         Basename of the _fullocr.json file.
        category:            Drawing category (1, 2, 3, or 0).
        classified_entries:  List of structured record dicts, each containing
                             id, box, text, type, confidence, and parsed keys.

    Returns:
        A dict with keys: source_file, image_category, total_detections,
        classified, summary.
    """
    summary: dict = {}
    for entry in classified_entries:
        t = entry.get("type", "unknown")
        summary[t] = summary.get(t, 0) + 1

    bom_rows = reconstruct_bom_rows(classified_entries, category)

    return {
        "source_file": source_file,
        "image_category": category,
        "total_detections": len(classified_entries),
        "classified": classified_entries,
        "summary": summary,
        "bom_rows": bom_rows,
    }


# ============================================================
# File-level and batch processing
# ============================================================

def validate_file(fullocr_path: str, output_dir: str) -> dict:
    """
    Process a single _fullocr.json file.

    Reads the file, normalises and classifies each OCR entry, extracts
    parsed sub-fields, assembles the structured output dict, and writes
    it as a _structured.json file in output_dir.

    Does NOT raise on errors. Malformed JSON or missing fields are handled
    gracefully: errors are logged to stdout and an empty-result dict (or
    None for unrecoverable failures) is returned.

    Args:
        fullocr_path: Absolute or relative path to the _fullocr.json file.
        output_dir:   Directory where the _structured.json will be written.
                      Created with os.makedirs(..., exist_ok=True) if absent.

    Returns:
        The structured output dictionary (same content as the written file),
        or None if the file could not be processed.
    """
    fullocr_path = str(fullocr_path)
    basename = os.path.basename(fullocr_path)

    # Read and parse the JSON file
    try:
        with open(fullocr_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: malformed JSON in '{fullocr_path}': {e}")
        return None
    except OSError as e:
        print(f"ERROR: cannot read '{fullocr_path}': {e}")
        return None

    category = detect_category(basename)

    classified_entries = []
    for entry in entries:
        raw_text = entry.get("text", "")
        normalised = normalise_text(raw_text)
        type_ = classify(normalised, category)
        parsed = extract_parsed(type_, normalised)

        classified_entries.append({
            "id": entry.get("id", 0),
            "box": entry.get("box", []),
            "text": normalised,
            "type": type_,
            "confidence": entry.get("confidence", 0.0),
            "parsed": parsed,
        })

    result = build_structured_output(basename, category, classified_entries)

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    stem = basename
    if stem.endswith("_fullocr.json"):
        stem = stem[: -len("_fullocr.json")]
    output_path = os.path.join(output_dir, f"{stem}_structured.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def validate_batch(input_dir: str, output_dir: str) -> list:
    """
    Process all _fullocr.json files in input_dir.

    Iterates over every file matching *_fullocr.json in input_dir, calls
    validate_file for each, and collects results. Files that fail are
    skipped with an error message printed to stdout; the batch continues.

    Args:
        input_dir:  Directory containing _fullocr.json files.
        output_dir: Directory where _structured.json files will be written.

    Returns:
        List of structured output dicts (one per successfully processed file).
        Failed files contribute None entries (or are omitted, depending on
        caller needs).
    """
    results = []
    fullocr_files = sorted(pathlib.Path(input_dir).glob("*_fullocr.json"))
    total = len(fullocr_files)

    for idx, path in enumerate(fullocr_files, 1):
        try:
            result = validate_file(str(path), output_dir)
            if result is not None:
                # Print per-file summary
                count = result.get("total_detections", 0)
                summary = result.get("summary", {})
                summary_str = " ".join(f"{k}:{v}" for k, v in summary.items())
                print(f"[{idx}/{total}] {path.name} -> {count} entries | {summary_str}")
                results.append(result)
            else:
                print(f"[{idx}/{total}] {path.name} → FAILED (skipped)")
        except Exception as e:
            print(f"ERROR processing {path.name}: {e}")

    return results


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/validation.py <input_dir> <output_dir>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    results = validate_batch(input_dir, output_dir)
    print(f"Batch complete: {len(results)} files processed")
