# Design Document: Stage 3 — Validation & Structuring

## Overview

Stage 3 (`src/validation.py`) is the classification and structuring layer of the CAD Tolerance Stack-Up Analysis pipeline. It sits between Stage 2 (EasyOCR output) and Stage 4 (geometric association), consuming `_fullocr.json` files and producing `_structured.json` files.

Its responsibilities are:
1. **Normalise** known OCR artefacts in the `text` field of each entry
2. **Classify** each normalised text string into one of 14 engineering annotation types
3. **Extract** typed sub-fields (`parsed`) from each classified entry
4. **Write** a well-structured `_structured.json` per image

The module is pure Python (stdlib only: `re`, `json`, `os`, `sys`, `pathlib`) and must run on all 36 images in under 60 seconds on the target hardware.

### Pipeline Position

```
data/category_N/cadN_NNN.png
        │
        ▼
[Stage 1] preprocessing.py      → binary image
        │
        ▼
[Stage 1.5] element_detection.py → bounding boxes
        │
        ▼
[Stage 2] vlm_reader.py          → _fullocr.json
        │
        ▼
[Stage 3] validation.py  ◄── DESIGNING NOW
        │
        ▼
results/batch/cadN_NNN_structured.json
        │
        ▼
[Stage 4] association.py         → NOT BUILT YET
        │
        ▼
[Stage 5] Flask visualisation    → NOT BUILT YET
```

---

## Architecture

The module is structured as a single-file pipeline with four sequential processing stages:

```
validate_file(fullocr_path, output_dir)
    │
    ├─ 1. detect_category(filename)          → int (1, 2, or 3)
    │
    ├─ 2. For each OCR entry:
    │       normalise_text(text)             → str
    │       classify(normalised, category)   → str (type)
    │       extract_parsed(type, normalised) → dict
    │
    └─ 3. build_output(entries, meta)        → dict
           write_json(output_path, result)
```

All state is local to each function call — there are no module-level caches or mutable globals (except the compiled regex patterns, which are module-level constants for performance).

### Design Decisions

**Why a priority-ordered if/elif chain instead of a dispatch table?**
The 14 types have overlapping patterns (e.g., `M30` could naively match `dimension_value` if the `M` is stripped). A strict priority chain makes the precedence explicit and easy to audit. A dispatch table would require sorting by priority anyway.

**Why compile regexes at module level?**
`re.compile` is called once at import time. With 36 files x ~50 entries each, avoiding 1800 repeated compilations is a measurable win on a low-RAM machine.

**Why no BOM spatial heuristic for `quantity`?**
The `quantity` type requires knowing whether an entry falls inside a BOM table region. Without Stage 4's geometric association, the only reliable proxy is: Category 2 image + small integer (1-99) + y-coordinate in the lower half of the image (where BOM tables typically appear in Indian standard assembly drawings). This is a best-effort heuristic; Stage 4 can refine it.

---

## Components and Interfaces

### Public API

```python
def validate_file(fullocr_path: str, output_dir: str) -> dict:
    """
    Process a single _fullocr.json file.

    Args:
        fullocr_path: Absolute or relative path to the _fullocr.json file.
        output_dir:   Directory where _structured.json will be written.

    Returns:
        The structured output dictionary (same content as the written file).

    Raises:
        Does NOT raise. Logs errors to stdout and returns an empty-result dict
        on malformed JSON or missing fields.
    """

def validate_batch(input_dir: str, output_dir: str) -> list:
    """
    Process all _fullocr.json files in input_dir.

    Args:
        input_dir:  Directory containing _fullocr.json files.
        output_dir: Directory where _structured.json files will be written.

    Returns:
        List of structured output dicts (one per successfully processed file).
        Files that fail are skipped; errors are logged to stdout.
    """
```

### Internal Functions

```python
def detect_category(filename: str) -> int:
    """
    Infer drawing category from filename.
    Returns 1, 2, or 3. Returns 0 and prints a warning if unrecognised.
    """

def normalise_text(text: str) -> str:
    """
    Apply OCR artefact corrections to a single text string.
    Respects PROTECTED_CODES — tokens that exactly match a protected code
    are returned unchanged.
    """

def _is_protected(token: str) -> bool:
    """
    Return True if the entire text string is a PROTECTED_CODE token.
    Used by normalise_text to skip correction on protected strings.
    """

def classify(text: str, category: int) -> str:
    """
    Assign one of the 14 type strings to a normalised text string.
    Evaluates patterns in the defined priority order.
    """

def extract_parsed(type_: str, text: str) -> dict:
    """
    Extract typed sub-fields from a classified text string.
    Returns a dict appropriate for the given type.
    """

def _edit_distance(a: str, b: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.
    Used for fuzzy part-name matching (threshold <= 1).
    Pure Python, O(len(a) * len(b)).
    """

def build_structured_output(
    source_file: str,
    category: int,
    classified_entries: list
) -> dict:
    """
    Assemble the final output dict from processed entries.
    Computes total_detections and summary counts.
    """
```

---

## Data Models

### Input: OCR Entry (from `_fullocr.json`)

```python
{
    "id":         int,                    # sequential, 1-based
    "box":        [int, int, int, int],   # [x, y, w, h] in original image pixels
    "text":       str,                    # post-processed text from vlm_reader.py
    "raw_text":   str,                    # verbatim EasyOCR output
    "confidence": float                   # EasyOCR confidence score [0.0, 1.0]
}
```

### Output: Structured Record (element of `classified` array)

```python
{
    "id":         int,                    # preserved from input
    "box":        [int, int, int, int],   # preserved from input
    "text":       str,                    # normalised text (may differ from input)
    "type":       str,                    # one of the 14 classification type strings
    "confidence": float,                  # preserved from input
    "parsed":     dict                    # type-specific sub-fields
}
```

### Output: Top-level `_structured.json`

```python
{
    "source_file":       str,   # basename of the _fullocr.json file
    "image_category":    int,   # 1, 2, 3, or 0 (unknown)
    "total_detections":  int,   # len(classified)
    "classified":        list,  # array of Structured Records
    "summary":           dict   # {type_string: count, ...} for types present
}
```

### Parsed Fields by Type

| Type | `parsed` dict |
|---|---|
| `dimension_value` | `{"value": float}` |
| `thread_spec` | `{"nominal": str, "pitch": float or null}` |
| `tolerance` | `{"tolerance_string": str}` |
| `diameter_callout` | `{"diameter": float}` |
| `hole_callout` | `{"raw": str}` |
| `section_marker` | `{"label": str}` |
| `spacing_annotation` | `{"annotation": "EQUI-SP"}` |
| `material_code` | `{"code": str}` (uppercase) |
| `part_name` | `{"name": str}` (title case) |
| `bom_header` | `{"header": str}` (uppercase) |
| `balloon_number` | `{"number": int}` |
| `quantity` | `{"qty": int}` |
| `dimension_with_note` | `{"raw": str}` |
| `unknown` | `{}` |

---

## Normalisation Pipeline

Normalisation is applied to the `text` field of each OCR entry **before** classification. The `raw_text` field is never modified.

### Step-by-Step Logic

```
normalise_text(text):
    1. Strip leading/trailing whitespace.
    2. If the entire stripped string is in PROTECTED_CODES → return unchanged.
    3. Apply substitutions in this order:
       a. Leading-zero diameter fix:
          regex: ^0(\d{2,})$   →   replace with Ø\1
          e.g. "018" → "Ø18", "0118" → "Ø118"
          NOTE: only applies when the ENTIRE string is 0 + 2+ digits.
          "0" alone, "01" alone, or "0118 DEEP" are NOT matched here
          (compound strings are handled by the classifier's context).
       b. Degree symbol fix:
          regex: (\d{1,2})"   →   \1°
          e.g. "45\"" → "45°", "90\"" → "90°"
       c. THICK typo fix:
          literal: "IHICK" → "THICK", "MICK" → "THICK"
          (case-insensitive substring replacement)
       d. Multiplication symbol fix:
          literal: "*" → "×"
    4. Return the modified string.
```

### PROTECTED_CODES Interaction

The protection check at step 2 is a **whole-string** check. If `text.strip().upper()` is in `PROTECTED_CODES`, the entire string is returned as-is. This prevents:
- `"MS"` being misread as a dimension
- `"THICK"` being re-processed
- `"X-X"` being altered

For compound strings like `"DIA 40×20 IHICK"`, the whole string is NOT in `PROTECTED_CODES`, so normalisation proceeds and corrects `IHICK` → `THICK`.

### Normalisation Examples

| Input `text` | Normalised | Rule applied |
|---|---|---|
| `"018"` | `"Ø18"` | Leading-zero diameter |
| `"0118"` | `"Ø118"` | Leading-zero diameter |
| `"45\""` | `"45°"` | Degree symbol |
| `"12 MICK"` | `"12 THICK"` | THICK typo |
| `"DIA 40*20 IHICK"` | `"DIA 40×20 THICK"` | `*`→`×` + THICK typo |
| `"M30 × 2.5"` | `"M30 × 2.5"` | No change needed |
| `"X-X"` | `"X-X"` | PROTECTED_CODE |
| `"MS"` | `"MS"` | PROTECTED_CODE |
| `"EQUI-SP"` | `"EQUI-SP"` | PROTECTED_CODE |

---

## Classification Engine

### Compiled Regex Patterns (module-level constants)

```python
import re

# Priority 1: hole_callout
RE_HOLE = re.compile(r'HOLE', re.IGNORECASE)
RE_HOLE_NUMERIC = re.compile(r'(\d+|DIA)', re.IGNORECASE)

# Priority 2: thread_spec
RE_THREAD = re.compile(r'^M\d+(\s*[×x]\s*\d+(\.\d+)?)?$', re.IGNORECASE)

# Priority 3: diameter_callout
RE_DIAMETER = re.compile(r'^(Ø\d+(\.\d+)?|DIA\s+\d+(\.\d+)?)$', re.IGNORECASE)

# Priority 4: dimension_with_note
RE_DIM_NOTE = re.compile(
    r'(\d+(\.\d+)?)\s*(THICK|DEEP|LONG|WIDE)'
    r'|^DIA\s+\d+.*\s+(THICK|DEEP|LONG|WIDE)',
    re.IGNORECASE
)

# Priority 5: tolerance
RE_TOLERANCE = re.compile(
    r'^[±]\d'                    # ±0.5
    r'|^\+\d.*\/\s*-\d'         # +0.12/-0.00
    r'|^[A-Z]\d+\/[a-z]\d+'     # H7/h6
    r'|^[A-Z]\d+$',             # H7 alone
    re.IGNORECASE
)

# Priority 6: spacing_annotation
RE_SPACING = re.compile(r'^EQUI[-\s]SP$', re.IGNORECASE)

# Priority 7: bom_header
BOM_HEADERS = {'PARTS LIST', 'NAME', 'MATERIAL', 'QTY', 'NO', 'SL NO', 'PART NO'}

# Priority 8: material_code
MATERIAL_CODES = {'MS', 'CI', 'FS', 'GM', 'CS', 'CR', 'AL', 'BR'}

# Priority 9: part_name (fuzzy, edit distance <= 1)
PART_NAMES = {
    'VALVE', 'SPRING', 'PIN', 'BODY', 'SPINDLE', 'HANDWHEEL',
    'GLAND', 'BONNET', 'SLEEVE', 'COLLAR', 'COVER', 'PLATE',
    'SEAT', 'NUT', 'BOLT', 'WASHER'
}

# Priority 10: section_marker (Cat 1 and 3 only)
RE_SECTION = re.compile(r'^[A-Z]-[A-Z]$|^[A-Z]$')

# Priority 11: balloon_number (Cat 2 and 3 only)
RE_BALLOON = re.compile(r'^[1-9]$')

# Priority 12: quantity (Cat 2 only, spatial heuristic)
RE_QUANTITY = re.compile(r'^[1-9]\d?$')

# Priority 13: dimension_value
RE_DIMENSION = re.compile(r'^\d+(\.\d+)?$')
```

### Classification Decision Flow

```
classify(text, category):

    t = text.strip()
    if not t:
        return "unknown"

    # P1: hole_callout — must contain HOLE + numeric/DIA
    if RE_HOLE.search(t) and RE_HOLE_NUMERIC.search(t):
        return "hole_callout"

    # P2: thread_spec — M followed by digits, optional pitch
    if RE_THREAD.match(t):
        return "thread_spec"

    # P3: diameter_callout — Ø prefix or DIA prefix (no compound note)
    if RE_DIAMETER.match(t):
        return "diameter_callout"

    # P4: dimension_with_note — number + THICK/DEEP/LONG/WIDE
    if RE_DIM_NOTE.search(t):
        return "dimension_with_note"

    # P5: tolerance — ±, +x/-y, H7/h6 patterns
    if RE_TOLERANCE.match(t):
        return "tolerance"

    # P6: spacing_annotation
    if RE_SPACING.match(t):
        return "spacing_annotation"

    # P7: bom_header — exact match against known headers
    if t.upper() in BOM_HEADERS:
        return "bom_header"

    # P8: material_code — exact match against 8 codes
    if t.upper() in MATERIAL_CODES:
        return "material_code"

    # P9: part_name — exact or fuzzy (edit distance <= 1)
    upper_t = t.upper()
    for name in PART_NAMES:
        if upper_t == name or _edit_distance(upper_t, name) <= 1:
            return "part_name"

    # P10: section_marker — only in Cat 1 and Cat 3
    if category in (1, 3) and RE_SECTION.match(t):
        return "section_marker"

    # P11: balloon_number — single digit 1-9, only in Cat 2 and Cat 3
    if category in (2, 3) and RE_BALLOON.match(t):
        return "balloon_number"

    # P12: quantity — 1-2 digit number, Cat 2 only, BOM spatial region
    if category == 2 and RE_QUANTITY.match(t):
        # Heuristic: treat as quantity (Stage 4 will refine)
        return "quantity"

    # P13: dimension_value — bare number
    if RE_DIMENSION.match(t):
        return "dimension_value"

    # P14: fallback
    return "unknown"
```

### Priority Rationale

The ordering resolves the following ambiguities:

| Conflict | Resolution |
|---|---|
| `"HOLE; DIA 21"` could match `diameter_callout` | `hole_callout` checked first (P1 > P3) |
| `"M30"` could match `dimension_value` if M stripped | `thread_spec` checked before `dimension_value` (P2 > P13) |
| `"DIA 40×20 THICK"` could match `diameter_callout` | `dimension_with_note` checked before `diameter_callout` (P4 > P3) |
| `"H7"` could match `section_marker` (single uppercase + digit) | `tolerance` checked before `section_marker` (P5 > P10) |
| `"MS"` could match `section_marker` (two uppercase letters) | `material_code` checked before `section_marker` (P8 > P10) |
| `"8"` in Cat 2 could be `balloon_number` or `dimension_value` | `balloon_number` checked before `dimension_value` (P11 > P13) |
| `"2"` in Cat 2 BOM region could be `quantity` or `balloon_number` | `balloon_number` (single digit) checked before `quantity` (P11 > P12) |

### Category-Gated Rules

Two types are only valid in specific categories:

- `section_marker`: Categories 1 and 3 only. In Category 2 (assembly drawings), single uppercase letters are more likely BOM column headers or noise.
- `balloon_number`: Categories 2 and 3 only. In Category 1 (part drawings), single digits are dimension values.
- `quantity`: Category 2 only. Category 3 has no BOM.

---

## Parsed Field Extraction

```python
def extract_parsed(type_: str, text: str) -> dict:
    if type_ == "dimension_value":
        return {"value": float(text.strip())}

    if type_ == "thread_spec":
        # e.g. "M30 × 2.5" or "M16×1.5" or "M30"
        m = re.match(r'^(M\d+)(?:\s*[×x]\s*(\d+(?:\.\d+)?))?', text.strip(), re.IGNORECASE)
        if m:
            nominal = m.group(1).upper()
            pitch = float(m.group(2)) if m.group(2) else None
            return {"nominal": nominal, "pitch": pitch}
        return {"nominal": text.strip(), "pitch": None}

    if type_ == "tolerance":
        return {"tolerance_string": text.strip()}

    if type_ == "diameter_callout":
        # strip Ø or "DIA " prefix, parse number
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
        return {"number": int(text.strip())}

    if type_ == "quantity":
        return {"qty": int(text.strip())}

    if type_ == "dimension_with_note":
        return {"raw": text.strip()}

    # unknown
    return {}
```

---

## Category Detection

```python
def detect_category(filename: str) -> int:
    """Infer category from filename substring."""
    name = os.path.basename(filename).lower()
    if 'cad1_' in name:
        return 1
    if 'cad2_' in name:
        return 2
    if 'cad3_' in name:
        return 3
    print(f"WARNING: cannot determine category from filename '{filename}', defaulting to 0")
    return 0
```

The category is embedded in the filename by convention (`cad1_NNN`, `cad2_NNN`, `cad3_NNN`). This is the only reliable signal available without reading the image itself.

---

## Output Schema

### File Naming

Input: `results/batch/cad1_001_fullocr.json`
Output: `results/batch/cad1_001_structured.json`

The `_fullocr` suffix is replaced with `_structured`. The output directory is the same as the input directory unless `output_dir` is specified differently.

### Full Example

```json
{
  "source_file": "cad1_001_fullocr.json",
  "image_category": 1,
  "total_detections": 12,
  "classified": [
    {
      "id": 1,
      "box": [337, 18, 37, 16],
      "text": "X-X",
      "type": "section_marker",
      "confidence": 0.967,
      "parsed": { "label": "X-X" }
    },
    {
      "id": 5,
      "box": [419, 98, 60, 15],
      "text": "M30 × 2.5",
      "type": "thread_spec",
      "confidence": 0.570,
      "parsed": { "nominal": "M30", "pitch": 2.5 }
    },
    {
      "id": 33,
      "box": [729, 242, 23, 14],
      "text": "Ø18",
      "type": "diameter_callout",
      "confidence": 0.940,
      "parsed": { "diameter": 18.0 }
    }
  ],
  "summary": {
    "section_marker": 3,
    "thread_spec": 1,
    "dimension_value": 6,
    "spacing_annotation": 1,
    "diameter_callout": 1
  }
}
```

### Encoding and Formatting

- UTF-8 encoding, no BOM
- `json.dump(..., indent=2, ensure_ascii=False)`
- Existing files are overwritten silently

---

## Integration Points

### How `batch_process.py` Calls `validation.py`

The existing `batch_process.py` calls `read_full_image()` per image and saves `_fullocr.json`. Stage 3 should be invoked immediately after, passing the output path:

```python
# In batch_process.py — proposed addition to process_category()
from src.validation import validate_file   # or: from validation import validate_file

# After read_full_image() returns:
fullocr_path = os.path.join(results_dir, f"{basename}_fullocr.json")
structured = validate_file(fullocr_path, output_dir=results_dir)
```

### Import Compatibility

`validation.py` must be importable from both:
- `from validation import validate_file` (when `src/` is on `sys.path`)
- `from src.validation import validate_file` (from project root)

This mirrors the existing pattern in `batch_process.py`:
```python
sys.path.insert(0, 'src')
try:
    from vlm_reader import read_full_image
except ImportError:
    from src.vlm_reader import read_full_image
```

### Standalone CLI Usage

```bash
python src/validation.py results/batch results/batch
```

The `__main__` block calls `validate_batch(input_dir, output_dir)` and prints a per-file summary:

```
[1/22] cad1_001_fullocr.json → 12 entries | section_marker:3 dimension_value:6 thread_spec:1 ...
[2/22] cad1_002_fullocr.json → 18 entries | ...
...
Batch complete: 22 files processed, 0 errors
```

### Error Handling Contract

| Condition | Behaviour |
|---|---|
| Malformed JSON in `_fullocr.json` | Log error, skip file, continue batch |
| Missing `text` field in an entry | Assign `type="unknown"`, `parsed={}` |
| Empty `text` field | Assign `type="unknown"`, `parsed={}` |
| Empty `_fullocr.json` (zero entries) | Write valid output with `total_detections=0`, empty `classified`, empty `summary` |
| Output directory does not exist | Create it with `os.makedirs(..., exist_ok=True)` |
| `_structured.json` already exists | Overwrite silently |
| Unrecognised filename (no `cad1_`/`cad2_`/`cad3_`) | Set `image_category=0`, log warning, continue |

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

This feature is well-suited to property-based testing. The normalisation and classification functions are pure (no I/O, no side effects), operate over a large input space (arbitrary strings), and have universal invariants that should hold for all inputs. The property-based testing library used is **Hypothesis** (already available in the project environment via `pip install hypothesis`; no new library is needed if it is already present, otherwise it is a pure-Python test dependency only).

### Property 1: Normalisation preserves PROTECTED_CODES

*For any* token that is an exact member of PROTECTED_CODES, calling `normalise_text` on that token SHALL return the token unchanged.

**Validates: Requirements 1.6**

### Property 2: Leading-zero diameter correction

*For any* string of the form `"0"` followed by two or more decimal digits (and nothing else), `normalise_text` SHALL return the string with the leading `"0"` replaced by `"Ø"`.

**Validates: Requirements 1.1**

### Property 3: Degree symbol correction

*For any* string containing one or two decimal digits immediately followed by a double-quote character, `normalise_text` SHALL replace the `"` with `°` in the output.

**Validates: Requirements 1.2**

### Property 4: THICK typo correction

*For any* string containing the substring `IHICK` or `MICK` (case-insensitive), `normalise_text` SHALL replace those substrings with `THICK` in the output.

**Validates: Requirements 1.3**

### Property 5: Classifier always returns a valid type

*For any* non-empty string and any category value in {0, 1, 2, 3}, `classify` SHALL return a value that is a member of the set of 14 defined type strings: `{dimension_value, thread_spec, tolerance, diameter_callout, hole_callout, section_marker, spacing_annotation, material_code, part_name, bom_header, balloon_number, quantity, dimension_with_note, unknown}`.

**Validates: Requirements 2.1**

### Property 6: Thread spec classification

*For any* string matching the pattern `M` followed by one or more digits (optionally followed by `×` and a pitch value), `classify` SHALL return `"thread_spec"` regardless of category.

**Validates: Requirements 2.3**

### Property 7: Diameter callout classification

*For any* string of the form `Ø` followed by a positive number, or `DIA ` followed by a positive number (with no trailing note keyword), `classify` SHALL return `"diameter_callout"` regardless of category.

**Validates: Requirements 2.5**

### Property 8: Priority ordering — hole_callout beats diameter_callout

*For any* string that contains the token `HOLE` and also contains a numeric value or `DIA`, `classify` SHALL return `"hole_callout"` and NOT `"diameter_callout"`, regardless of category.

**Validates: Requirements 2.6, 2.16**

### Property 9: Priority ordering — dimension_with_note beats diameter_callout

*For any* string that starts with `DIA` and also contains a note keyword (`THICK`, `DEEP`, `LONG`, or `WIDE`), `classify` SHALL return `"dimension_with_note"` and NOT `"diameter_callout"`, regardless of category.

**Validates: Requirements 2.14, 2.16**

### Property 10: Category-gated section_marker

*For any* string matching `[A-Z]-[A-Z]` or a single uppercase letter, `classify` SHALL return `"section_marker"` when category is 1 or 3, and SHALL NOT return `"section_marker"` when category is 2.

**Validates: Requirements 2.7**

### Property 11: Category-gated balloon_number

*For any* single digit string `"1"` through `"9"`, `classify` SHALL return `"balloon_number"` when category is 2 or 3, and SHALL return `"dimension_value"` when category is 1.

**Validates: Requirements 2.12**

### Property 12: Parsed fields are structurally correct for the assigned type

*For any* string and category, if `classify(text, category)` returns type `T`, then `extract_parsed(T, text)` SHALL return a dict whose keys exactly match the schema defined for type `T` (e.g., `dimension_value` → `{"value": float}`, `thread_spec` → `{"nominal": str, "pitch": float or None}`, `unknown` → `{}`).

**Validates: Requirements 3.1–3.14**

### Property 13: Thread spec parsed fields are correct

*For any* thread spec string of the form `M{n}` or `M{n}×{p}`, `extract_parsed("thread_spec", text)` SHALL return `{"nominal": "M{n}", "pitch": p}` where `p` is the float pitch value if present, or `None` if absent.

**Validates: Requirements 3.2**

### Property 14: total_detections equals classified array length

*For any* valid `_fullocr.json` input with N entries, the output dict produced by `validate_file` SHALL satisfy `output["total_detections"] == len(output["classified"]) == N`.

**Validates: Requirements 4.3**

### Property 15: Summary counts are consistent with classified array

*For any* valid `_fullocr.json` input, the `summary` dict in the output SHALL satisfy: `sum(summary.values()) == total_detections`, and for each type key in `summary`, the count SHALL equal the number of entries in `classified` with that type.

**Validates: Requirements 4.5**

### Property 16: Category detection from filename

*For any* filename containing the substring `cad1_`, `cad2_`, or `cad3_`, `detect_category` SHALL return `1`, `2`, or `3` respectively, regardless of the rest of the filename.

**Validates: Requirements 5.1–5.3**

---

## Error Handling

### Malformed or Missing Input

| Condition | Handling |
|---|---|
| `_fullocr.json` contains invalid JSON | Catch `json.JSONDecodeError`, print error to stdout, return `None`, skip writing output |
| `_fullocr.json` is empty array `[]` | Process normally; output has `total_detections=0`, `classified=[]`, `summary={}` |
| Entry missing `text` key | Treat as empty string; assign `type="unknown"`, `parsed={}` |
| Entry has `text=""` | Assign `type="unknown"`, `parsed={}` |
| Entry missing `id` or `box` | Use `entry.get("id", 0)` and `entry.get("box", [])` as fallbacks |
| Entry has `confidence < 0.6` | Classify normally; confidence does not affect type assignment |
| Output directory does not exist | Create with `os.makedirs(output_dir, exist_ok=True)` |

### Batch Processing Resilience

`validate_batch` wraps each `validate_file` call in a `try/except Exception` block. On failure:
- Print: `ERROR processing {filename}: {exception_message}`
- Append `None` to the results list (or skip, depending on caller needs)
- Continue to the next file

This ensures a single corrupt file (e.g., truncated JSON from a crashed OCR run) does not abort the entire batch.

### Parsed Field Extraction Safety

`extract_parsed` uses `try/except` around numeric conversions (`float()`, `int()`) to handle edge cases where the regex matched but the captured group is not a valid number. In such cases, the field is set to `None` rather than raising.

---

## Testing Strategy

### Dual Testing Approach

Unit tests cover specific examples, edge cases, and error conditions. Property-based tests verify universal invariants across a wide input space. Both are needed: unit tests catch concrete bugs in specific patterns, property tests find edge cases in the regex and normalisation logic.

### Property-Based Testing

**Library**: Hypothesis (`pip install hypothesis`)
**Minimum iterations**: 100 per property (Hypothesis default is 100; set `@settings(max_examples=200)` for the classification properties)
**Tag format**: `# Feature: stage3-validation-structuring, Property {N}: {property_text}`

Each correctness property maps to a single Hypothesis test. Example:

```python
from hypothesis import given, settings, strategies as st

# Feature: stage3-validation-structuring, Property 2: Leading-zero diameter correction
@given(st.from_regex(r'0\d{2,}', fullmatch=True))
@settings(max_examples=200)
def test_leading_zero_diameter(text):
    result = normalise_text(text)
    assert result.startswith('Ø')
    assert result[1:] == text[1:]

# Feature: stage3-validation-structuring, Property 5: Classifier always returns a valid type
VALID_TYPES = {
    'dimension_value', 'thread_spec', 'tolerance', 'diameter_callout',
    'hole_callout', 'section_marker', 'spacing_annotation', 'material_code',
    'part_name', 'bom_header', 'balloon_number', 'quantity',
    'dimension_with_note', 'unknown'
}

@given(st.text(min_size=1), st.integers(min_value=0, max_value=3))
@settings(max_examples=500)
def test_classifier_always_valid_type(text, category):
    result = classify(text, category)
    assert result in VALID_TYPES
```

### Unit Tests

Unit tests should cover:
- Each normalisation rule with concrete examples (including the examples in the Normalisation Examples table)
- Each classification type with at least one positive and one negative example
- Each parsed field extraction with at least one example per type
- Edge cases: empty string, whitespace-only string, very long string, Unicode characters
- Error handling: malformed JSON, missing fields, empty input file
- Integration: `validate_file` end-to-end with a small synthetic `_fullocr.json`

### Test File Location

```
tests/
    test_validation.py          # unit + property tests for validation.py
    fixtures/
        sample_fullocr.json     # small synthetic input for integration tests
        sample_structured.json  # expected output for integration tests
```

### Accuracy Evaluation (Manual)

The 90%/70%/70% accuracy targets (Requirement 7) require a manually labelled ground-truth dataset. This is not automated. The recommended approach:
1. Run `validate_batch` on all 36 images
2. Manually review `_structured.json` outputs
3. Count correct vs. incorrect classifications per category
4. Adjust regex patterns or priority order based on findings
