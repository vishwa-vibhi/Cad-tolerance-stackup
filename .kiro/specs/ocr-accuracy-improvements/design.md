# Design Document: OCR Accuracy Improvements

## Overview

This document describes the design for targeted improvements to the CAD Tolerance Stack-Up Analysis pipeline OCR accuracy. The changes span two modules:

- **Stage 2** (`src/vlm_reader.py`): Add an aspect-ratio heuristic for the `8` to `Ø` (diameter symbol) correction and a leading-zero diameter fix in `post_process_text()`.
- **Stage 3** (`src/validation.py`): Add `radius_callout` and `material_name` types, expand `PART_NAMES` and `BOM_HEADERS`, and introduce `reconstruct_bom_rows()` for spatial BOM table reconstruction.

The goal is to raise classification accuracy for Category 2 assembly drawings from ~70% to >=80% for part names and >=70% for material names, while maintaining 100% backward compatibility with the existing 109 passing tests.

### Key Design Decisions

1. **Aspect-ratio check placement**: The bounding box is only available in `read_full_image()`, not in `post_process_text()`. The `8` to `Ø` heuristic is applied in `read_full_image()` before calling `post_process_text()`, keeping `post_process_text()` a pure text function.

2. **Leading-zero fix in Stage 2**: The `^0\d{2,}$` to `Ø\d+` correction already exists in `normalise_text()` (Stage 3). Adding it to `post_process_text()` (Stage 2) ensures the `_fullocr.json` output is cleaner and Stage 3 receives pre-corrected text.

3. **`material_name` before `part_name`**: Priority 8.5 ensures that strings like "Brass" or "Cast iron" are classified as `material_name` rather than falling through to `part_name` fuzzy matching.

4. **BOM row reconstruction uses y-coordinate grouping**: Entries within +/-10px of the same y-centroid are grouped into a row. Within each row, x-coordinate position determines the role (leftmost = part_no, then part_name, then material, rightmost = qty).

5. **No new third-party dependencies**: All new logic uses Python stdlib (`re`, `json`, `os`) and the existing `_edit_distance()` helper.

---

## Architecture

The pipeline stages and their data flow are unchanged. The improvements are additive within existing stages:

```
[Image] -> Stage 2 (vlm_reader.py) -> _fullocr.json -> Stage 3 (validation.py) -> _structured.json
```

### Stage 2 Changes (vlm_reader.py)

```
read_full_image()
  +-- EasyOCR raw results: (bbox, raw_text, conf)
  +-- NEW: aspect-ratio check on raw_text before post_process_text()
  |     if raw_text.strip() == "8" and w_box <= 20 and h_box >= 20:
  |         raw_text = "Ø"  # diameter symbol
  +-- post_process_text(raw_text)
        +-- existing: *, euro, empty-set, phi corrections
        +-- NEW: leading-zero diameter fix: ^0(\d{2,})$ -> Ø\1
```

### Stage 3 Changes (validation.py)

```
normalise_text()          -- unchanged (leading-zero fix already present)
classify()
  +-- P1:   hole_callout
  +-- P2:   thread_spec
  +-- P3:   diameter_callout
  +-- P3.5: NEW radius_callout
  +-- P4:   dimension_with_note
  +-- P5:   tolerance
  +-- P6:   spacing_annotation
  +-- P7:   bom_header          (expanded BOM_HEADERS set)
  +-- P8:   material_code
  +-- P8.5: NEW material_name   (new MATERIAL_NAMES set)
  +-- P9:   part_name           (expanded PART_NAMES, compound threshold <= 2)
  +-- P10:  section_marker
  +-- P11:  balloon_number
  +-- P12:  quantity
  +-- P13:  dimension_value
  +-- P14:  unknown
extract_parsed()          -- new cases for radius_callout, material_name
build_structured_output() -- adds bom_rows field
NEW: reconstruct_bom_rows()
```

---

## Components and Interfaces

### 2.1 `post_process_text(text: str) -> str` (modified)

**Location**: `src/vlm_reader.py`

**Change**: Add leading-zero diameter fix as the last correction step, after all existing symbol replacements.

```python
def post_process_text(text: str) -> str:
    # ... all existing logic unchanged ...

    # NEW: leading-zero diameter fix (mirrors normalise_text in Stage 3)
    # Applied last so it does not interfere with other corrections
    # Only matches whole-string: "061" -> "Ø61", "085" -> "Ø85"
    # Does NOT match "0" alone or "01" (single digit after zero)
    text = re.sub(r'^0(\d{2,})$', r'Ø\1', text)  # Ø = diameter symbol

    return text
```

**Rationale**: Applying this in Stage 2 means `_fullocr.json` already contains `Ø61` instead of `061`, making the Stage 3 input cleaner. The Stage 3 `normalise_text()` still applies the same fix as a safety net.

### 2.2 `read_full_image(image_path, output_dir, min_confidence)` (modified)

**Location**: `src/vlm_reader.py`

**Change**: Add aspect-ratio heuristic for `"8"` to diameter symbol before calling `post_process_text()`.

```python
# In the results loop, after unpacking (bbox, raw_text, conf):
xs = [pt[0] for pt in bbox]
ys = [pt[1] for pt in bbox]
w_box_ocr = max(xs) - min(xs)
h_box_ocr = max(ys) - min(ys)

# NEW: aspect-ratio heuristic -- tall narrow "8" is almost certainly a diameter symbol
# Guard against zero-dimension boxes (Requirement 11.1)
if (raw_text.strip() == "8"
        and w_box_ocr > 0 and h_box_ocr > 0
        and w_box_ocr <= 20 and h_box_ocr >= 20):
    raw_text = "Ø"  # diameter symbol

text = post_process_text(raw_text)
```

**Rationale**: The bounding box is only available at this point in the pipeline. The guard `w_box_ocr > 0 and h_box_ocr > 0` satisfies Requirement 11.1 (zero-dimension boxes skip the heuristic). The thresholds (width <= 20, height >= 20) are derived from the dataset: diameter symbols in these drawings are consistently taller than wide, while the digit "8" in dimension annotations is roughly square.

### 3.1 New Regex Constant `RE_RADIUS`

**Location**: `src/validation.py`, module-level constants section

```python
# Priority 3.5: radius_callout -- R followed by digits, optional decimal
RE_RADIUS = re.compile(r'^R\d+(\.\d+)?$')
```

**Note**: No `re.IGNORECASE` flag -- radius callouts in engineering drawings always use uppercase `R`. This avoids false positives from lowercase `r` in part names like "Rod end".

### 3.2 Expanded `BOM_HEADERS` Set

```python
BOM_HEADERS = {
    'PARTS LIST', 'NAME', 'MATERIAL', 'QTY', 'NO', 'SL NO', 'PART NO',
    # NEW additions
    'MATL', 'MAT', 'SL.NO', 'SL. NO', 'PART NAME', 'PART NO.', 'NO.',
}
```

**Rationale**: `MATL` is the abbreviation used in `cad2_007_fullocr.json`. `SL.NO` and `SL. NO` are common Indian standard drawing variants. `PART NAME` and `PART NO.` appear in K.L. Narayana textbook drawings.

### 3.3 New `MATERIAL_NAMES` Set

```python
MATERIAL_NAMES = {
    'BABBIT', 'BRASS', 'NI-CR STEEL', 'CD-AG', 'CAST IRON',
    'MILD STEEL', 'HIGH CARBON STEEL', 'LOW CARBON STEEL',
    'STAINLESS STEEL', 'ALUMINUM', 'BRONZE', 'COPPER',
}
```

**Note**: `BRASS` appears in both `MATERIAL_NAMES` and `PROTECTED_CODES`. Since `material_name` (P8.5) is evaluated before `part_name` (P9), and `material_code` (P8) only matches the 2-letter codes set, `BRASS` will correctly classify as `material_name`.

### 3.4 Expanded `PART_NAMES` Set

```python
PART_NAMES = {
    # existing single-word names
    'VALVE', 'SPRING', 'PIN', 'BODY', 'SPINDLE', 'HANDWHEEL',
    'GLAND', 'BONNET', 'SLEEVE', 'COLLAR', 'COVER', 'PLATE',
    'SEAT', 'NUT', 'BOLT', 'WASHER',
    'FORK', 'BLOCK', 'PIECE', 'HOLDER', 'SWIVEL', 'SHEAVE',
    'ASSEMBLY', 'TOOL', 'CENTRAL', 'MODULE',
    # NEW compound names (from Requirement 3.1)
    'ARTICULATED ROD', 'COVER PLATE', 'ROD END', 'LOCK NUT',
    'LINK PIN', 'PISTON PIN', 'PISTON RING', 'ROD BUSH-UPPER',
    'ROD BUSH-LOWER', 'MASTER ROD BEARING', 'PISTON PIN PLUG',
    # Additional compound names from dataset
    'PISTON', 'CONNECTING ROD', 'COTTER PIN',
}
```

### 3.5 Updated `VALID_TYPES` Set

```python
VALID_TYPES = {
    'dimension_value', 'thread_spec', 'tolerance', 'diameter_callout',
    'hole_callout', 'section_marker', 'spacing_annotation', 'material_code',
    'part_name', 'bom_header', 'balloon_number', 'quantity',
    'dimension_with_note', 'unknown',
    # NEW
    'radius_callout', 'material_name',
}
```

## Components and Interfaces

### 2.1 `post_process_text(text: str) -> str` (modified)

**Location**: `src/vlm_reader.py`

**Change**: Add leading-zero diameter fix as the last correction step, after all existing symbol replacements.

```python
def post_process_text(text: str) -> str:
    # ... all existing logic unchanged ...

    # NEW: leading-zero diameter fix (mirrors normalise_text in Stage 3)
    # Applied last so it does not interfere with other corrections
    # Only matches whole-string: "061" -> "Ø61", "085" -> "Ø85"
    # Does NOT match "0" alone or "01" (single digit after zero)
    text = re.sub(r'^0(\d{2,})$', r'Ø\1', text)  # Ø = diameter symbol

    return text
```

**Rationale**: Applying this in Stage 2 means `_fullocr.json` already contains `Ø61` instead of `061`, making the Stage 3 input cleaner. The Stage 3 `normalise_text()` still applies the same fix as a safety net.

### 2.2 `read_full_image(image_path, output_dir, min_confidence)` (modified)

**Location**: `src/vlm_reader.py`

**Change**: Add aspect-ratio heuristic for `"8"` to diameter symbol before calling `post_process_text()`.

```python
# In the results loop, after unpacking (bbox, raw_text, conf):
xs = [pt[0] for pt in bbox]
ys = [pt[1] for pt in bbox]
w_box_ocr = max(xs) - min(xs)
h_box_ocr = max(ys) - min(ys)

# NEW: aspect-ratio heuristic -- tall narrow "8" is almost certainly a diameter symbol
# Guard against zero-dimension boxes (Requirement 11.1)
if (raw_text.strip() == "8"
        and w_box_ocr > 0 and h_box_ocr > 0
        and w_box_ocr <= 20 and h_box_ocr >= 20):
    raw_text = "Ø"  # diameter symbol

text = post_process_text(raw_text)
```

**Rationale**: The bounding box is only available at this point in the pipeline. The guard `w_box_ocr > 0 and h_box_ocr > 0` satisfies Requirement 11.1 (zero-dimension boxes skip the heuristic). The thresholds (width <= 20, height >= 20) are derived from the dataset: diameter symbols in these drawings are consistently taller than wide, while the digit "8" in dimension annotations is roughly square.

### 3.1 New Regex Constant `RE_RADIUS`

**Location**: `src/validation.py`, module-level constants section

```python
# Priority 3.5: radius_callout -- R followed by digits, optional decimal
RE_RADIUS = re.compile(r'^R\d+(\.\d+)?$')
```

**Note**: No `re.IGNORECASE` flag -- radius callouts in engineering drawings always use uppercase `R`. This avoids false positives from lowercase `r` in part names like "Rod end".

### 3.2 Expanded `BOM_HEADERS` Set

```python
BOM_HEADERS = {
    'PARTS LIST', 'NAME', 'MATERIAL', 'QTY', 'NO', 'SL NO', 'PART NO',
    # NEW additions
    'MATL', 'MAT', 'SL.NO', 'SL. NO', 'PART NAME', 'PART NO.', 'NO.',
}
```

**Rationale**: `MATL` is the abbreviation used in `cad2_007_fullocr.json`. `SL.NO` and `SL. NO` are common Indian standard drawing variants. `PART NAME` and `PART NO.` appear in K.L. Narayana textbook drawings.

### 3.3 New `MATERIAL_NAMES` Set

```python
MATERIAL_NAMES = {
    'BABBIT', 'BRASS', 'NI-CR STEEL', 'CD-AG', 'CAST IRON',
    'MILD STEEL', 'HIGH CARBON STEEL', 'LOW CARBON STEEL',
    'STAINLESS STEEL', 'ALUMINUM', 'BRONZE', 'COPPER',
}
```

**Note**: `BRASS` appears in both `MATERIAL_NAMES` and `PROTECTED_CODES`. Since `material_name` (P8.5) is evaluated before `part_name` (P9), and `material_code` (P8) only matches the 2-letter codes set, `BRASS` will correctly classify as `material_name`.

### 3.4 Expanded `PART_NAMES` Set

```python
PART_NAMES = {
    # existing single-word names
    'VALVE', 'SPRING', 'PIN', 'BODY', 'SPINDLE', 'HANDWHEEL',
    'GLAND', 'BONNET', 'SLEEVE', 'COLLAR', 'COVER', 'PLATE',
    'SEAT', 'NUT', 'BOLT', 'WASHER',
    'FORK', 'BLOCK', 'PIECE', 'HOLDER', 'SWIVEL', 'SHEAVE',
    'ASSEMBLY', 'TOOL', 'CENTRAL', 'MODULE',
    # NEW compound names (from Requirement 3.1)
    'ARTICULATED ROD', 'COVER PLATE', 'ROD END', 'LOCK NUT',
    'LINK PIN', 'PISTON PIN', 'PISTON RING', 'ROD BUSH-UPPER',
    'ROD BUSH-LOWER', 'MASTER ROD BEARING', 'PISTON PIN PLUG',
    # Additional compound names from dataset
    'PISTON', 'CONNECTING ROD', 'COTTER PIN',
}
```

### 3.5 Updated `VALID_TYPES` Set

```python
VALID_TYPES = {
    'dimension_value', 'thread_spec', 'tolerance', 'diameter_callout',
    'hole_callout', 'section_marker', 'spacing_annotation', 'material_code',
    'part_name', 'bom_header', 'balloon_number', 'quantity',
    'dimension_with_note', 'unknown',
    # NEW
    'radius_callout', 'material_name',
}
```

### 3.6 Updated `classify()` Function

The priority chain gains two new entries. The function signature is unchanged.

```python
def classify(text: str, category: int) -> str:
    t = text.strip()
    if not t:
        return "unknown"

    # P1: hole_callout (unchanged)
    if RE_HOLE.search(t) and RE_HOLE_NUMERIC.search(t):
        return "hole_callout"

    # P2: thread_spec (unchanged)
    if RE_THREAD.match(t):
        return "thread_spec"

    # P3: diameter_callout (unchanged)
    if RE_DIAMETER.match(t):
        return "diameter_callout"

    # P3.5: NEW radius_callout
    if RE_RADIUS.match(t):
        return "radius_callout"

    # P4: dimension_with_note (unchanged)
    if RE_DIM_NOTE.search(t):
        return "dimension_with_note"

    # P5: tolerance (unchanged)
    if RE_TOLERANCE.match(t):
        return "tolerance"

    # P6: spacing_annotation (unchanged)
    if RE_SPACING.match(t):
        return "spacing_annotation"

    # P7: bom_header (expanded set)
    if t.upper() in BOM_HEADERS:
        return "bom_header"

    # P8: material_code (unchanged)
    if t.upper() in MATERIAL_CODES:
        return "material_code"

    # P8.5: NEW material_name (exact or fuzzy edit distance <= 2)
    upper_t = t.upper()
    for mat in MATERIAL_NAMES:
        if upper_t == mat or _edit_distance(upper_t, mat) <= 2:
            return "material_name"

    # P9: part_name (expanded set, compound threshold <= 2)
    for name in PART_NAMES:
        is_compound = ' ' in name or '-' in name
        threshold = 2 if is_compound else 1
        if upper_t == name or _edit_distance(upper_t, name) <= threshold:
            return "part_name"

    # P10: section_marker (unchanged)
    if category in (1, 3) and RE_SECTION.match(t):
        return "section_marker"

    # P11: balloon_number (unchanged)
    if category in (2, 3) and RE_BALLOON.match(t):
        return "balloon_number"

    # P12: quantity (unchanged)
    if category == 2 and RE_QUANTITY.match(t):
        return "quantity"

    # P13: dimension_value (unchanged)
    if RE_DIMENSION.match(t):
        return "dimension_value"

    # P13.5: single punctuation / noise (unchanged)
    if len(t) <= 2 and not t.isalnum():
        return "unknown"

    # P14: fallback
    return "unknown"
```

**Compound name detection**: A name is considered compound if it contains a space or hyphen. This covers `ARTICULATED ROD`, `ROD BUSH-UPPER`, etc. Single-word names retain the existing threshold of <= 1.

### 3.7 Updated `extract_parsed()` Function

Two new cases are added. The function signature is unchanged.

```python
if type_ == "radius_callout":
    t = text.strip()
    try:
        return {"radius": float(t[1:])}  # strip leading 'R'
    except (ValueError, IndexError):
        return {"radius": None}

if type_ == "material_name":
    return {"name": text.strip().title()}
```

### 3.8 New `reconstruct_bom_rows()` Function

**Location**: `src/validation.py`, after `extract_parsed()`

**Signature**:

```python
def reconstruct_bom_rows(classified_entries: list, category: int) -> list:
```

**Returns**: List of BOM_Row dicts, each with keys `part_no`, `part_name`, `material`, `qty`.

See Section 4 (BOM Row Reconstruction Algorithm) for full pseudocode.

### 3.9 Updated `build_structured_output()` Function

```python
def build_structured_output(source_file, category, classified_entries):
    summary = {}
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
        "bom_rows": bom_rows,   # NEW field
    }
```

---

## Data Models

### BOM_Row Dictionary

```json
{
  "part_no": 2,
  "part_name": "Rod End",
  "material": "MS",
  "qty": 1
}
```

| Field | Type | Source | Notes |
|-------|------|--------|-------|
| `part_no` | `int | null` | `balloon_number` entry | The part number from the BOM balloon |
| `part_name` | `str | null` | `part_name` entry | Title-cased part name |
| `material` | `str | null` | `material_code` or `material_name` entry | Code (e.g., "MS") or full name (e.g., "Babbit") |
| `qty` | `int | null` | `quantity` entry | Quantity as integer |

### Updated `_structured.json` Schema

```json
{
  "source_file": "cad2_005_fullocr.json",
  "image_category": 2,
  "total_detections": 63,
  "classified": [
    {
      "id": 7,
      "box": [627, 107, 35, 28],
      "text": "Ø61",
      "type": "diameter_callout",
      "confidence": 0.827,
      "parsed": {"diameter": 61.0}
    },
    {
      "id": 36,
      "box": [544, 408, 52, 16],
      "text": "Rod end",
      "type": "part_name",
      "confidence": 0.9999,
      "parsed": {"name": "Rod End"}
    },
    {
      "id": 44,
      "box": [647, 439, 38, 18],
      "text": "Brass",
      "type": "material_name",
      "confidence": 0.9999,
      "parsed": {"name": "Brass"}
    }
  ],
  "summary": {
    "diameter_callout": 3,
    "part_name": 6,
    "material_name": 2,
    "material_code": 3,
    "bom_header": 4,
    "balloon_number": 5,
    "quantity": 5,
    "unknown": 8
  },
  "bom_rows": [
    {"part_no": 1, "part_name": "Body", "material": "MS", "qty": 1},
    {"part_no": 2, "part_name": "Rod End", "material": "MS", "qty": 1},
    {"part_no": 3, "part_name": "Cover Plate", "material": "MS", "qty": 1},
    {"part_no": 4, "part_name": "Brasses", "material": "Brass", "qty": 2},
    {"part_no": 5, "part_name": "Bolt", "material": null, "qty": 4},
    {"part_no": 6, "part_name": "Nut", "material": null, "qty": 2},
    {"part_no": 7, "part_name": "Lock Nut", "material": null, "qty": 2}
  ]
}
```

**Backward compatibility**: The `bom_rows` field is additive. All existing fields (`source_file`, `image_category`, `total_detections`, `classified`, `summary`) are preserved unchanged. Existing tests that check for the presence of these fields will continue to pass.

---

## BOM Row Reconstruction Algorithm

### Overview

BOM tables in Category 2 drawings (K.L. Narayana textbook, Indian standard) are located in the lower-right quadrant of the image. Each row is a horizontal band of OCR entries. The algorithm groups entries by y-coordinate proximity, then assigns roles by type within each group.

### Pseudocode

```
function reconstruct_bom_rows(classified_entries, category):
    # Gate: only Category 2 images have BOM tables
    if category != 2:
        return []

    # Step 1: Filter to BOM-relevant entry types
    BOM_TYPES = {balloon_number, part_name, material_code, material_name, quantity}
    relevant = [e for e in classified_entries
                if e.type in BOM_TYPES
                and e.box is valid (length >= 4)]

    if relevant is empty:
        return []

    # Step 2: Compute y-centroid for each entry
    # box format: [x, y, width, height]
    for each entry in relevant:
        entry.y_center = entry.box[1] + entry.box[3] / 2

    # Step 3: Sort by y-centroid (top to bottom)
    relevant.sort(key=lambda e: e.y_center)

    # Step 4: Group entries into rows
    # Entries within Y_TOLERANCE pixels of the current row's y-centroid
    # belong to the same row
    Y_TOLERANCE = 10  # pixels
    rows = []
    current_row = [relevant[0]]
    current_y = relevant[0].y_center

    for each entry in relevant[1:]:
        if abs(entry.y_center - current_y) <= Y_TOLERANCE:
            current_row.append(entry)
        else:
            rows.append(current_row)
            current_row = [entry]
            current_y = entry.y_center

    rows.append(current_row)  # flush last row

    # Step 5: For each row, assign roles by type
    bom_rows = []
    for each row_entries in rows:
        # Sort entries within the row by x-coordinate (left to right)
        row_entries.sort(key=lambda e: e.box[0])

        part_no = null
        part_name = null
        material = null
        qty = null

        for each entry in row_entries:
            if entry.type == balloon_number and part_no is null:
                part_no = entry.parsed.number
            elif entry.type == part_name and part_name is null:
                part_name = entry.parsed.name
            elif entry.type in {material_code, material_name} and material is null:
                if entry.type == material_code:
                    material = entry.parsed.code
                else:
                    material = entry.parsed.name
            elif entry.type == quantity and qty is null:
                qty = entry.parsed.qty

        # Include row even if some fields are null (Requirement 6.4)
        bom_rows.append({
            "part_no": part_no,
            "part_name": part_name,
            "material": material,
            "qty": qty,
        })

    return bom_rows
```

### Spatial Adjacency Definition

Two entries are in the same BOM row if their y-centroids are within `Y_TOLERANCE = 10` pixels. This is tighter than the 20-pixel threshold in the requirements because the grouping is centroid-based (not edge-based), and the dataset shows BOM rows are typically 13-18 pixels tall with clear vertical separation.

The horizontal adjacency check from Requirement 6.2 (gap <= 50 pixels) is implicitly handled by the y-grouping: entries in the same horizontal band are assumed to belong to the same row regardless of horizontal gaps. This is appropriate because BOM columns can have large horizontal gaps between them.

### Role Assignment Rationale

Within a row, roles are assigned by type rather than by x-position rank. This is more robust than positional assignment because:
- Not all rows have all four fields (e.g., a bolt row may have no material code)
- The x-position of columns is consistent within a drawing but varies between drawings
- Type-based assignment handles missing fields gracefully

The `part_no is null` guard ensures that if two balloon numbers appear in the same y-band (unlikely but possible), only the first (leftmost) is used.

### Complexity

- Step 1 (filter): O(n)
- Step 2 (y-centroid): O(n)
- Step 3 (sort): O(n log n)
- Step 4 (grouping): O(n)
- Step 6 (role assignment): O(n) total across all rows

Total: O(n log n), satisfying Requirement 10.4 (O(n^2) or better).

### Example: cad2_005 BOM Table

From the real OCR data, the BOM entries (after classification improvements) would be grouped as follows. After y-grouping (Y_TOLERANCE=10) and role assignment:

```json
[
  {"part_no": 1, "part_name": "Body", "material": null, "qty": null},
  {"part_no": 2, "part_name": "Rod End", "material": "MS", "qty": null},
  {"part_no": 3, "part_name": "Cover Plate", "material": "MS", "qty": null},
  {"part_no": 4, "part_name": "Brasses", "material": "Brass", "qty": null}
]
```

Note: Some qty values are null because the OCR entries for those rows were classified as `balloon_number` (single digits) rather than `quantity`. This is a known limitation of the current classifier -- single digits in Category 2 are ambiguous between balloon numbers and quantities. The BOM row reconstruction captures what is available.

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system -- essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

This feature involves pure string transformation functions (`post_process_text`, `normalise_text`, `classify`, `extract_parsed`) and a spatial grouping algorithm (`reconstruct_bom_rows`). These are well-suited for property-based testing using [Hypothesis](https://hypothesis.readthedocs.io/).

### Property Reflection

Before writing properties, I reviewed the prework analysis for redundancy:

- Properties about `classify()` returning `radius_callout` for all matching strings (2.2) and `extract_parsed()` returning the correct radius (2.4) are complementary, not redundant -- one tests classification, the other tests extraction. They are combined into a single round-trip property.
- Properties about `classify()` returning `material_name` for exact matches (4.3) and fuzzy matches (4.4) are combined: exact match is edit distance 0, a special case of the fuzzy property.
- Properties about `classify()` returning `part_name` for exact matches (3.2) and fuzzy matches (3.3) are similarly combined.
- The `bom_rows == []` for non-Category-2 images (6.7) and the null-field robustness (6.4) are distinct properties.
- The `PROTECTED_CODES` invariant (7.4) and the spatial adjacency symmetry (6.2) are distinct.

After reflection, 8 unique properties remain.

---

### Property 1: Leading-zero diameter correction preserves digits

*For any* string matching the pattern `^0\d{2,}$` (a leading zero followed by two or more digits), applying `post_process_text()` SHALL produce a string starting with the diameter symbol followed by the same digit sequence.

**Validates: Requirements 1.1**

---

### Property 2: Aspect-ratio heuristic is correctly gated

*For any* bounding box with positive width and height, the `"8"` to diameter symbol correction SHALL be applied if and only if the text is exactly `"8"`, the width is <= 20 pixels, and the height is >= 20 pixels.

**Validates: Requirements 1.2, 11.1**

---

### Property 3: Radius callout classification and extraction round-trip

*For any* string matching `^R\d+(\.\d+)?$`, `classify()` SHALL return `"radius_callout"` and `extract_parsed("radius_callout", text)["radius"]` SHALL equal the float value of the digit sequence following the `R` prefix.

**Validates: Requirements 2.2, 2.4**

---

### Property 4: Material name classification with fuzzy matching

*For any* string whose uppercase form has edit distance <= 2 from any entry in `MATERIAL_NAMES`, `classify(text, category=2)` SHALL return `"material_name"` (provided the string does not match a higher-priority pattern such as `material_code`).

**Validates: Requirements 4.3, 4.4**

---

### Property 5: Part name classification with compound threshold

*For any* string whose uppercase form has edit distance <= 1 from a single-word entry in `PART_NAMES`, or edit distance <= 2 from a compound entry in `PART_NAMES`, `classify(text, category=2)` SHALL return `"part_name"` (provided the string does not match a higher-priority pattern).

**Validates: Requirements 3.2, 3.3, 3.4**

---

### Property 6: PROTECTED_CODES are never modified by normalise_text

*For any* string whose stripped, uppercased form is a member of `PROTECTED_CODES`, `normalise_text(text)` SHALL return the stripped original string unchanged.

**Validates: Requirements 7.4**

---

### Property 7: Spatial adjacency check is symmetric

*For any* two bounding boxes A and B, the y-centroid proximity predicate used in `reconstruct_bom_rows` SHALL be symmetric: if A and B are grouped into the same row, then B and A are also grouped into the same row.

**Validates: Requirements 6.2**

---

### Property 8: BOM row reconstruction is category-gated

*For any* list of classified entries, `reconstruct_bom_rows(entries, category)` SHALL return an empty list `[]` when `category` is 1 or 3, regardless of the content of `entries`.

**Validates: Requirements 6.7**

---

## Error Handling

### Stage 2 (`vlm_reader.py`)

| Condition | Handling |
|-----------|----------|
| Bounding box width or height is zero | Skip aspect-ratio check; apply text-pattern corrections only (Req 11.1) |
| `post_process_text` receives empty string | Return empty string (existing behavior) |
| EasyOCR returns no results | Return empty list (existing behavior) |

### Stage 3 (`validation.py`)

| Condition | Handling |
|-----------|----------|
| `classify()` receives empty string | Return `"unknown"` (existing behavior) |
| `extract_parsed()` receives non-numeric radius string | Return `{"radius": None}` |
| `reconstruct_bom_rows()` entry has missing/malformed `box` field | Skip that entry, continue processing (Req 11.3) |
| `reconstruct_bom_rows()` produces zero rows for Category 2 | Return `[]`; `bom_rows` field is `[]` in output (Req 11.4) |
| `validate_file()` encounters malformed JSON | Return `None`, print error (existing behavior) |
| Multiple entries match the same role in a BOM row | First (leftmost by x-coordinate) wins; subsequent entries of the same type are ignored |

### Malformed Box Field Handling in `reconstruct_bom_rows`

```python
def _get_box_safe(entry):
    """Return (x, y, w, h) or None if box is missing/malformed."""
    box = entry.get("box")
    if not isinstance(box, list) or len(box) < 4:
        return None
    try:
        return (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    except (TypeError, ValueError):
        return None
```

Entries where `_get_box_safe()` returns `None` are silently skipped.

---

## Testing Strategy

### Dual Testing Approach

Unit tests cover specific examples and edge cases. Property-based tests (using [Hypothesis](https://hypothesis.readthedocs.io/)) verify universal properties across many generated inputs.

### Property-Based Tests

Using Hypothesis with `@settings(max_examples=100)` minimum. Each test is tagged with the design property it validates.

**Test file**: `tests/test_ocr_accuracy_properties.py`

```python
from hypothesis import given, settings
from hypothesis import strategies as st
from src.vlm_reader import post_process_text
from src.validation import classify, extract_parsed, normalise_text, reconstruct_bom_rows
from src.validation import MATERIAL_NAMES, PART_NAMES, PROTECTED_CODES

# Feature: ocr-accuracy-improvements, Property 1: Leading-zero diameter correction
@given(st.from_regex(r'^0[0-9]{2,}$', fullmatch=True))
@settings(max_examples=200)
def test_leading_zero_correction_preserves_digits(s):
    result = post_process_text(s)
    assert result[0] == 'O'  # diameter symbol
    assert result[1:] == s[1:]

# Feature: ocr-accuracy-improvements, Property 2: Aspect-ratio heuristic gating
@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100)
)
@settings(max_examples=200)
def test_aspect_ratio_heuristic_gating(w, h):
    should_correct = (w <= 20 and h >= 20)
    result = _apply_aspect_ratio_check("8", w, h)
    assert result == ("O" if should_correct else "8")

# Feature: ocr-accuracy-improvements, Property 3: Radius callout round-trip
@given(st.from_regex(r'^R[0-9]+(\.[0-9]+)?$', fullmatch=True))
@settings(max_examples=200)
def test_radius_callout_classification_and_extraction(s):
    assert classify(s, category=1) == "radius_callout"
    parsed = extract_parsed("radius_callout", s)
    assert parsed["radius"] == float(s[1:])

# Feature: ocr-accuracy-improvements, Property 4: Material name fuzzy matching
@given(st.sampled_from(sorted(MATERIAL_NAMES)))
@settings(max_examples=100)
def test_material_name_exact_match(name):
    assert classify(name.title(), category=2) == "material_name"

# Feature: ocr-accuracy-improvements, Property 5: Part name compound threshold
@given(st.sampled_from([n for n in PART_NAMES if ' ' in n or '-' in n]))
@settings(max_examples=100)
def test_compound_part_name_exact_match(name):
    assert classify(name.title(), category=2) == "part_name"

# Feature: ocr-accuracy-improvements, Property 6: PROTECTED_CODES invariant
@given(st.sampled_from(sorted(PROTECTED_CODES)))
@settings(max_examples=100)
def test_protected_codes_not_modified(code):
    assert normalise_text(code) == code

# Feature: ocr-accuracy-improvements, Property 7: Spatial adjacency symmetry
@given(
    st.lists(st.integers(min_value=0, max_value=800), min_size=4, max_size=4),
    st.lists(st.integers(min_value=0, max_value=800), min_size=4, max_size=4)
)
@settings(max_examples=200)
def test_spatial_adjacency_symmetry(box_a, box_b):
    # y-centroid proximity is symmetric
    y_a = box_a[1] + box_a[3] / 2
    y_b = box_b[1] + box_b[3] / 2
    assert (abs(y_a - y_b) <= 10) == (abs(y_b - y_a) <= 10)

# Feature: ocr-accuracy-improvements, Property 8: BOM rows category gating
@given(
    st.lists(st.fixed_dictionaries({
        "type": st.sampled_from(["balloon_number", "part_name", "material_code", "quantity"]),
        "box": st.lists(st.integers(0, 800), min_size=4, max_size=4),
        "parsed": st.just({}),
    })),
    st.sampled_from([1, 3])
)
@settings(max_examples=100)
def test_bom_rows_empty_for_non_category2(entries, category):
    assert reconstruct_bom_rows(entries, category) == []
```

### Unit Tests

**Test file**: `tests/test_ocr_accuracy.py`

Key example-based tests:

```python
import pytest
from src.validation import classify, reconstruct_bom_rows

# Requirement 2.5: specific radius callouts from dataset
@pytest.mark.parametrize("text", ["R189", "R78", "R13", "R24"])
def test_radius_callout_dataset_examples(text):
    assert classify(text, category=1) == "radius_callout"

# Requirement 4.5: material_name priority over part_name
def test_brass_classifies_as_material_name():
    assert classify("Brass", category=2) == "material_name"

# Requirement 5.1: new BOM headers
@pytest.mark.parametrize("text", ["Matl", "Mat", "Sl No", "Part Name"])
def test_new_bom_headers(text):
    assert classify(text, category=2) == "bom_header"

# Requirement 3.1: compound part names from dataset
@pytest.mark.parametrize("text", [
    "Cover plate", "Rod end", "Lock nut", "Articulated rod",
    "Master rod bearing", "Piston pin plug"
])
def test_compound_part_names(text):
    assert classify(text, category=2) == "part_name"

# Requirement 1.2: aspect-ratio heuristic edge cases
def test_aspect_ratio_zero_width_skips_check():
    # Zero-width box: should not apply heuristic
    result = _apply_aspect_ratio_check("8", w=0, h=30)
    assert result == "8"

# Requirement 6.4: BOM row with missing fields
def test_bom_row_with_missing_fields():
    entries = [
        {"id": 1, "box": [500, 400, 30, 15], "type": "part_name",
         "parsed": {"name": "Body"}, "confidence": 0.99},
    ]
    rows = reconstruct_bom_rows(entries, category=2)
    assert len(rows) == 1
    assert rows[0]["part_name"] == "Body"
    assert rows[0]["part_no"] is None
    assert rows[0]["material"] is None
    assert rows[0]["qty"] is None

# Requirement 11.3: malformed box field
def test_bom_row_reconstruction_skips_malformed_box():
    entries = [
        {"id": 1, "box": None, "type": "part_name",
         "parsed": {"name": "Body"}, "confidence": 0.99},
        {"id": 2, "box": [500, 400, 30, 15], "type": "balloon_number",
         "parsed": {"number": 1}, "confidence": 0.99},
    ]
    rows = reconstruct_bom_rows(entries, category=2)
    assert isinstance(rows, list)
```

### Regression Tests

The existing 109 tests in `tests/test_validation.py` must continue to pass without modification. The new code is additive:
- New types in `VALID_TYPES` do not break existing `result in VALID_TYPES` checks
- New priority levels (3.5, 8.5) do not affect existing type assignments for existing patterns
- New `bom_rows` field does not break existing schema checks (they only check for presence of existing keys)

---

## Error Handling

### Stage 2 (`vlm_reader.py`)

| Condition | Handling |
|-----------|----------|
| Bounding box width or height is zero | Skip aspect-ratio check; apply text-pattern corrections only (Req 11.1) |
| `post_process_text` receives empty string | Return empty string (existing behavior) |
| EasyOCR returns no results | Return empty list (existing behavior) |

### Stage 3 (`validation.py`)

| Condition | Handling |
|-----------|----------|
| `classify()` receives empty string | Return `"unknown"` (existing behavior) |
| `extract_parsed()` receives non-numeric radius string | Return `{"radius": None}` |
| `reconstruct_bom_rows()` entry has missing/malformed `box` field | Skip that entry, continue processing (Req 11.3) |
| `reconstruct_bom_rows()` produces zero rows for Category 2 | Return `[]`; `bom_rows` field is `[]` in output (Req 11.4) |
| `validate_file()` encounters malformed JSON | Return `None`, print error (existing behavior) |
| Multiple entries match the same role in a BOM row | First (leftmost by x-coordinate) wins; subsequent entries of the same type are ignored |

### Malformed Box Field Handling in `reconstruct_bom_rows`

```python
def _get_box_safe(entry):
    """Return (x, y, w, h) or None if box is missing/malformed."""
    box = entry.get("box")
    if not isinstance(box, list) or len(box) < 4:
        return None
    try:
        return (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    except (TypeError, ValueError):
        return None
```

Entries where `_get_box_safe()` returns `None` are silently skipped.

---

## Testing Strategy

### Dual Testing Approach

Unit tests cover specific examples and edge cases. Property-based tests (using [Hypothesis](https://hypothesis.readthedocs.io/)) verify universal properties across many generated inputs.

### Property-Based Tests

Using Hypothesis with `@settings(max_examples=100)` minimum. Each test is tagged with the design property it validates.

**Test file**: `tests/test_ocr_accuracy_properties.py`

```python
from hypothesis import given, settings
from hypothesis import strategies as st
from src.vlm_reader import post_process_text
from src.validation import classify, extract_parsed, normalise_text, reconstruct_bom_rows
from src.validation import MATERIAL_NAMES, PART_NAMES, PROTECTED_CODES

# Feature: ocr-accuracy-improvements, Property 1: Leading-zero diameter correction
@given(st.from_regex(r'^0[0-9]{2,}$', fullmatch=True))
@settings(max_examples=200)
def test_leading_zero_correction_preserves_digits(s):
    result = post_process_text(s)
    assert result[0] == 'O'  # diameter symbol
    assert result[1:] == s[1:]

# Feature: ocr-accuracy-improvements, Property 2: Aspect-ratio heuristic gating
@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100)
)
@settings(max_examples=200)
def test_aspect_ratio_heuristic_gating(w, h):
    should_correct = (w <= 20 and h >= 20)
    result = _apply_aspect_ratio_check("8", w, h)
    assert result == ("O" if should_correct else "8")

# Feature: ocr-accuracy-improvements, Property 3: Radius callout round-trip
@given(st.from_regex(r'^R[0-9]+(\.[0-9]+)?$', fullmatch=True))
@settings(max_examples=200)
def test_radius_callout_classification_and_extraction(s):
    assert classify(s, category=1) == "radius_callout"
    parsed = extract_parsed("radius_callout", s)
    assert parsed["radius"] == float(s[1:])

# Feature: ocr-accuracy-improvements, Property 4: Material name fuzzy matching
@given(st.sampled_from(sorted(MATERIAL_NAMES)))
@settings(max_examples=100)
def test_material_name_exact_match(name):
    assert classify(name.title(), category=2) == "material_name"

# Feature: ocr-accuracy-improvements, Property 5: Part name compound threshold
@given(st.sampled_from([n for n in PART_NAMES if ' ' in n or '-' in n]))
@settings(max_examples=100)
def test_compound_part_name_exact_match(name):
    assert classify(name.title(), category=2) == "part_name"

# Feature: ocr-accuracy-improvements, Property 6: PROTECTED_CODES invariant
@given(st.sampled_from(sorted(PROTECTED_CODES)))
@settings(max_examples=100)
def test_protected_codes_not_modified(code):
    assert normalise_text(code) == code

# Feature: ocr-accuracy-improvements, Property 7: Spatial adjacency symmetry
@given(
    st.lists(st.integers(min_value=0, max_value=800), min_size=4, max_size=4),
    st.lists(st.integers(min_value=0, max_value=800), min_size=4, max_size=4)
)
@settings(max_examples=200)
def test_spatial_adjacency_symmetry(box_a, box_b):
    # y-centroid proximity is symmetric
    y_a = box_a[1] + box_a[3] / 2
    y_b = box_b[1] + box_b[3] / 2
    assert (abs(y_a - y_b) <= 10) == (abs(y_b - y_a) <= 10)

# Feature: ocr-accuracy-improvements, Property 8: BOM rows category gating
@given(
    st.lists(st.fixed_dictionaries({
        "type": st.sampled_from(["balloon_number", "part_name", "material_code", "quantity"]),
        "box": st.lists(st.integers(0, 800), min_size=4, max_size=4),
        "parsed": st.just({}),
    })),
    st.sampled_from([1, 3])
)
@settings(max_examples=100)
def test_bom_rows_empty_for_non_category2(entries, category):
    assert reconstruct_bom_rows(entries, category) == []
```

### Unit Tests

**Test file**: `tests/test_ocr_accuracy.py`

Key example-based tests:

```python
import pytest
from src.validation import classify, reconstruct_bom_rows

# Requirement 2.5: specific radius callouts from dataset
@pytest.mark.parametrize("text", ["R189", "R78", "R13", "R24"])
def test_radius_callout_dataset_examples(text):
    assert classify(text, category=1) == "radius_callout"

# Requirement 4.5: material_name priority over part_name
def test_brass_classifies_as_material_name():
    assert classify("Brass", category=2) == "material_name"

# Requirement 5.1: new BOM headers
@pytest.mark.parametrize("text", ["Matl", "Mat", "Sl No", "Part Name"])
def test_new_bom_headers(text):
    assert classify(text, category=2) == "bom_header"

# Requirement 3.1: compound part names from dataset
@pytest.mark.parametrize("text", [
    "Cover plate", "Rod end", "Lock nut", "Articulated rod",
    "Master rod bearing", "Piston pin plug"
])
def test_compound_part_names(text):
    assert classify(text, category=2) == "part_name"

# Requirement 1.2: aspect-ratio heuristic edge cases
def test_aspect_ratio_zero_width_skips_check():
    # Zero-width box: should not apply heuristic
    result = _apply_aspect_ratio_check("8", w=0, h=30)
    assert result == "8"

# Requirement 6.4: BOM row with missing fields
def test_bom_row_with_missing_fields():
    entries = [
        {"id": 1, "box": [500, 400, 30, 15], "type": "part_name",
         "parsed": {"name": "Body"}, "confidence": 0.99},
    ]
    rows = reconstruct_bom_rows(entries, category=2)
    assert len(rows) == 1
    assert rows[0]["part_name"] == "Body"
    assert rows[0]["part_no"] is None
    assert rows[0]["material"] is None
    assert rows[0]["qty"] is None

# Requirement 11.3: malformed box field
def test_bom_row_reconstruction_skips_malformed_box():
    entries = [
        {"id": 1, "box": None, "type": "part_name",
         "parsed": {"name": "Body"}, "confidence": 0.99},
        {"id": 2, "box": [500, 400, 30, 15], "type": "balloon_number",
         "parsed": {"number": 1}, "confidence": 0.99},
    ]
    rows = reconstruct_bom_rows(entries, category=2)
    assert isinstance(rows, list)
```

### Regression Tests

The existing 109 tests in `tests/test_validation.py` must continue to pass without modification. The new code is additive:
- New types in `VALID_TYPES` do not break existing `result in VALID_TYPES` checks
- New priority levels (3.5, 8.5) do not affect existing type assignments for existing patterns
- New `bom_rows` field does not break existing schema checks (they only check for presence of existing keys)

---

## Backward Compatibility Analysis

### `VALID_TYPES` Set

Before: 14 types. After: 16 types (adds `radius_callout`, `material_name`).

Impact: Any code that checks `result in VALID_TYPES` continues to pass. No existing type is removed or renamed.

### Classification Priority Chain

| Priority | Type | Status |
|----------|------|--------|
| P1 | hole_callout | Unchanged |
| P2 | thread_spec | Unchanged |
| P3 | diameter_callout | Unchanged |
| **P3.5** | **radius_callout** | **NEW** |
| P4 | dimension_with_note | Unchanged |
| P5 | tolerance | Unchanged |
| P6 | spacing_annotation | Unchanged |
| P7 | bom_header | Expanded set (additive) |
| P8 | material_code | Unchanged |
| **P8.5** | **material_name** | **NEW** |
| P9 | part_name | Expanded set, compound threshold |
| P10 | section_marker | Unchanged |
| P11 | balloon_number | Unchanged |
| P12 | quantity | Unchanged |
| P13 | dimension_value | Unchanged |
| P14 | unknown | Unchanged |

**Risk analysis for new priorities**:

- `radius_callout` at P3.5: Strings like `R189` previously fell through to `unknown` (they do not match `hole_callout`, `thread_spec`, `diameter_callout`, or `dimension_with_note`). No existing test expects `R189` to be anything other than `unknown`, so inserting `radius_callout` here cannot break existing tests.

- `material_name` at P8.5: Strings like `Babbit`, `Ni-Cr steel` previously fell through to `unknown`. No existing test expects these to be `part_name` or any other type. The only potential conflict is `BRASS`, which is in `PROTECTED_CODES` and `MATERIAL_NAMES`. However, `BRASS` is not in `MATERIAL_CODES` (the 2-letter set), so it was previously classified as `part_name` via fuzzy matching. After this change, it will be classified as `material_name`. A search of the existing test file should confirm no test asserts `classify("Brass", ...) == "part_name"`.

### `normalise_text()` Function

No changes to `normalise_text()`. The leading-zero fix is already present. All 4 existing correction steps are preserved.

### `_structured.json` Output Schema

The `bom_rows` field is added at the top level. All existing fields are preserved:
- `source_file` checkmark
- `image_category` checkmark
- `total_detections` checkmark
- `classified` checkmark (each entry's fields are unchanged)
- `summary` checkmark

### `post_process_text()` Function

The leading-zero fix is added as the last step. This means strings like `"061"` that previously passed through unchanged will now be corrected to `"Ø61"` (diameter symbol + digits). This is intentional and correct. No existing test should expect `"061"` to remain as `"061"` after `post_process_text()`.

The aspect-ratio check is applied in `read_full_image()` before `post_process_text()`, so `post_process_text()` itself is unchanged in its interface.
