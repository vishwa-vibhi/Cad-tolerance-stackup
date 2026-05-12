# Implementation Plan: OCR Accuracy Improvements

## Overview

Targeted improvements to two existing modules: `src/vlm_reader.py` (Stage 2 OCR post-processing) and `src/validation.py` (Stage 3 classification and structuring). The changes add diameter-symbol heuristics, radius callout and material name types, expanded BOM header/part name sets, and a BOM row reconstruction function. All 109 existing tests in `tests/test_validation.py` must continue to pass.

## Tasks

- [x] 1. Stage 2 — Add leading-zero diameter fix to `post_process_text()`
  - In `src/vlm_reader.py`, append `text = re.sub(r'^0(\d{2,})$', r'Ø\1', text)` as the **last** correction step in `post_process_text()`, immediately before the final `return text`
  - The fix must run after all existing symbol replacements and whitespace collapsing
  - Must not affect strings that are already protected (the `is_protected()` guard at the top of the function already handles this)
  - _Requirements: 1.1, 1.4_

- [x] 2. Stage 2 — Add aspect-ratio heuristic to `read_full_image()`
  - In `src/vlm_reader.py`, inside the `for i, (bbox, raw_text, conf) in enumerate(results, 1):` loop, compute `w_box_ocr` and `h_box_ocr` from the OCR-scale bounding box **before** calling `post_process_text()`
  - Insert the guard: `if raw_text.strip() == "8" and w_box_ocr > 0 and h_box_ocr > 0 and w_box_ocr <= 20 and h_box_ocr >= 20: raw_text = "Ø"`
  - Use the raw OCR-scale coordinates (`xs`, `ys` from `bbox`) for the width/height check, not the rescaled output coordinates
  - _Requirements: 1.2, 1.5, 11.1_

- [x] 3. Stage 3 — Add constants: `RE_RADIUS`, expanded `BOM_HEADERS`, new `MATERIAL_NAMES`, expanded `PART_NAMES`, updated `VALID_TYPES`
  - [x] 3.1 Add `RE_RADIUS = re.compile(r'^R\d+(\.\d+)?$')` to the compiled regex constants section in `src/validation.py` (no `re.IGNORECASE` flag)
    - Place it after the existing `RE_DIAMETER` constant, with a comment: `# Priority 3.5: radius_callout`
    - _Requirements: 2.1, 2.2_
  - [x] 3.2 Replace the existing `BOM_HEADERS` set with the expanded version containing: `'PARTS LIST', 'NAME', 'MATERIAL', 'QTY', 'NO', 'SL NO', 'PART NO', 'MATL', 'MAT', 'SL.NO', 'SL. NO', 'PART NAME', 'PART NO.', 'NO.'`
    - _Requirements: 5.1, 5.2_
  - [x] 3.3 Add `MATERIAL_NAMES` set constant after `MATERIAL_CODES`: `{'BABBIT', 'BRASS', 'NI-CR STEEL', 'CD-AG', 'CAST IRON', 'MILD STEEL', 'HIGH CARBON STEEL', 'LOW CARBON STEEL', 'STAINLESS STEEL', 'ALUMINUM', 'BRONZE', 'COPPER'}`
    - _Requirements: 4.1, 4.2_
  - [x] 3.4 Expand `PART_NAMES` to add compound names: `'ARTICULATED ROD', 'COVER PLATE', 'ROD END', 'LOCK NUT', 'LINK PIN', 'PISTON PIN', 'PISTON RING', 'ROD BUSH-UPPER', 'ROD BUSH-LOWER', 'MASTER ROD BEARING', 'PISTON PIN PLUG', 'PISTON', 'CONNECTING ROD', 'COTTER PIN'`
    - _Requirements: 3.1_
  - [x] 3.5 Add `'radius_callout'` and `'material_name'` to the `VALID_TYPES` set
    - _Requirements: 2.1, 4.1_

- [x] 4. Stage 3 — Update `classify()` with two new priority levels
  - [x] 4.1 Insert P3.5 radius callout check after the P3 `diameter_callout` block and before the P4 `dimension_with_note` block:
    ```python
    # P3.5: radius_callout
    if RE_RADIUS.match(t):
        return "radius_callout"
    ```
    - _Requirements: 2.2, 2.3, 11.2_
  - [x] 4.2 Insert P8.5 material name check after the P8 `material_code` block and before the P9 `part_name` block:
    ```python
    # P8.5: material_name (exact or fuzzy edit distance <= 2)
    upper_t = t.upper()
    for mat in MATERIAL_NAMES:
        if upper_t == mat or _edit_distance(upper_t, mat) <= 2:
            return "material_name"
    ```
    - _Requirements: 4.3, 4.4, 4.5_
  - [x] 4.3 Update the P9 `part_name` block to use a compound-aware threshold: single-word names keep threshold `<= 1`, compound names (containing a space or hyphen) use threshold `<= 2`
    - Replace the existing `for name in PART_NAMES: if upper_t == name or _edit_distance(upper_t, name) <= 1:` loop with the compound-aware version
    - Note: the `upper_t` variable is already defined in step 4.2; ensure it is defined before the P9 block regardless of whether P8.5 matched
    - _Requirements: 3.2, 3.3, 3.4_

- [x] 5. Stage 3 — Update `extract_parsed()` with two new type cases
  - Add `radius_callout` case: strip the leading `'R'`, convert remainder to `float`, return `{"radius": float_value}`; on `ValueError` or `IndexError` return `{"radius": None}`
  - Add `material_name` case: return `{"name": text.strip().title()}`
  - Insert both cases before the final `# unknown` return, following the existing pattern of the function
  - _Requirements: 2.4, 4.6_

- [x] 6. Stage 3 — Add `_get_box_safe()` helper and `reconstruct_bom_rows()` function
  - [x] 6.1 Add `_get_box_safe(entry)` helper in the internal helpers section of `src/validation.py`:
    - Returns `(x, y, w, h)` as a 4-tuple of ints if `entry["box"]` is a list with at least 4 elements and all values are numeric; returns `None` otherwise
    - Wrap the int conversions in `try/except (TypeError, ValueError)`
    - _Requirements: 11.3_
  - [x] 6.2 Add `reconstruct_bom_rows(classified_entries: list, category: int) -> list` function after `extract_parsed()`:
    - Gate: return `[]` immediately if `category != 2`
    - Filter entries to `BOM_TYPES = {'balloon_number', 'part_name', 'material_code', 'material_name', 'quantity'}`, skipping any entry where `_get_box_safe()` returns `None`
    - Compute y-centroid as `box[1] + box[3] / 2` for each entry
    - Sort by y-centroid ascending
    - Group into rows using `Y_TOLERANCE = 10` pixels: entries within 10px of the current row's first entry's y-centroid belong to the same row
    - Within each row, sort entries by `box[0]` (x-coordinate) ascending
    - Assign roles by type (first match wins per role): `balloon_number` → `part_no`, `part_name` → `part_name`, `material_code` or `material_name` → `material`, `quantity` → `qty`
    - Include rows with missing fields as `null` (Requirement 6.4)
    - Return list of `{"part_no": ..., "part_name": ..., "material": ..., "qty": ...}` dicts
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.7, 10.4, 11.3, 11.4_

- [x] 7. Stage 3 — Update `build_structured_output()` to include `bom_rows` field
  - Call `reconstruct_bom_rows(classified_entries, category)` inside `build_structured_output()`
  - Add `"bom_rows": bom_rows` as a new key in the returned dict, after `"summary"`
  - All existing keys (`source_file`, `image_category`, `total_detections`, `classified`, `summary`) must remain unchanged
  - _Requirements: 6.5, 7.5_

- [x] 8. Regression check — run existing test suite
  - Run `pytest tests/test_validation.py -v` and confirm all 109 tests pass
  - If any test fails, diagnose and fix the regression before proceeding to new tests
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 9. Write unit tests in `tests/test_ocr_accuracy.py`
  - [x] 9.1 Tests for `post_process_text()` leading-zero fix
    - `"061"` → `"Ø61"`, `"085"` → `"Ø85"`, `"0100"` → `"Ø100"`
    - Edge cases: `"0"` unchanged, `"01"` unchanged (only one digit after zero), `"Ø61"` unchanged (already corrected)
    - _Requirements: 1.1_
  - [ ]* 9.2 Tests for aspect-ratio heuristic logic (unit-test the conditional directly)
    - `raw_text="8"`, `w=10`, `h=25` → corrected to `"Ø"`
    - `raw_text="8"`, `w=25`, `h=25` → not corrected (too wide)
    - `raw_text="8"`, `w=0`, `h=25` → not corrected (zero width guard)
    - `raw_text="18"`, `w=10`, `h=25` → not corrected (not exactly `"8"`)
    - _Requirements: 1.2, 11.1_
  - [x] 9.3 Tests for `classify()` — radius callout
    - `"R189"` → `"radius_callout"`, `"R78"` → `"radius_callout"`, `"R13.5"` → `"radius_callout"`
    - `"r189"` → NOT `"radius_callout"` (lowercase `r` must not match)
    - `"R"` alone → NOT `"radius_callout"`
    - _Requirements: 2.2, 2.3_
  - [x] 9.4 Tests for `extract_parsed()` — radius callout
    - `extract_parsed("radius_callout", "R189")` → `{"radius": 189.0}`
    - `extract_parsed("radius_callout", "R13.5")` → `{"radius": 13.5}`
    - `extract_parsed("radius_callout", "R")` → `{"radius": None}`
    - _Requirements: 2.4_
  - [x] 9.5 Tests for `classify()` — material name
    - `"BRASS"` → `"material_name"` (exact match, category=2)
    - `"Brass"` → `"material_name"` (case-insensitive)
    - `"CAST IRON"` → `"material_name"`
    - `"MILD STEEL"` → `"material_name"`
    - `"MS"` → `"material_code"` (must NOT be reclassified as material_name; P8 fires first)
    - _Requirements: 4.3, 4.4, 4.5_
  - [x] 9.6 Tests for `extract_parsed()` — material name
    - `extract_parsed("material_name", "BRASS")` → `{"name": "Brass"}`
    - `extract_parsed("material_name", "CAST IRON")` → `{"name": "Cast Iron"}`
    - _Requirements: 4.6_
  - [x] 9.7 Tests for `classify()` — compound part names
    - `"ARTICULATED ROD"` → `"part_name"`, `"COVER PLATE"` → `"part_name"`, `"ROD END"` → `"part_name"`
    - `"LOCK NUT"` → `"part_name"`, `"PISTON RING"` → `"part_name"`, `"CONNECTING ROD"` → `"part_name"`
    - `"ROD BUSH-UPPER"` → `"part_name"`, `"MASTER ROD BEARING"` → `"part_name"`
    - _Requirements: 3.1, 3.2_
  - [x] 9.8 Tests for expanded `BOM_HEADERS`
    - `"MATL"` → `"bom_header"`, `"MAT"` → `"bom_header"`, `"SL.NO"` → `"bom_header"`
    - `"PART NAME"` → `"bom_header"`, `"PART NO."` → `"bom_header"`, `"NO."` → `"bom_header"`
    - _Requirements: 5.1, 5.2_
  - [ ]* 9.9 Tests for `_get_box_safe()`
    - Valid box `[10, 20, 30, 40]` → `(10, 20, 30, 40)`
    - Missing box key → `None`
    - Box with fewer than 4 elements → `None`
    - Box with non-numeric values → `None`
    - _Requirements: 11.3_
  - [x] 9.10 Tests for `reconstruct_bom_rows()` — basic row grouping
    - Category 1 input → returns `[]`
    - Category 3 input → returns `[]`
    - Category 2 with no BOM-type entries → returns `[]`
    - Category 2 with one complete row (balloon + part_name + material_code + quantity) → returns one dict with all four fields non-null
    - Category 2 with a row missing material → returns one dict with `material: null`
    - _Requirements: 6.1, 6.3, 6.4, 6.7, 11.4_
  - [ ]* 9.11 Tests for `reconstruct_bom_rows()` — y-grouping tolerance
    - Two entries with y-centroids 5px apart → grouped into same row
    - Two entries with y-centroids 15px apart → split into separate rows
    - _Requirements: 6.2_
  - [x] 9.12 Tests for `build_structured_output()` — `bom_rows` field present
    - Category 1 output dict contains `"bom_rows": []`
    - Category 2 output dict contains `"bom_rows"` key (list, may be empty)
    - All existing keys still present: `source_file`, `image_category`, `total_detections`, `classified`, `summary`
    - _Requirements: 6.5, 7.5_

- [x] 10. Checkpoint — Ensure all tests pass
  - Run `pytest tests/test_validation.py tests/test_ocr_accuracy.py -v`
  - All 109 existing tests plus new unit tests must pass; ask the user if any failures arise

- [x] 11. Write property-based tests in `tests/test_ocr_accuracy_properties.py`
  - [ ]* 11.1 Write property test for Property 1: leading-zero correction preserves digits
    - Use `hypothesis.strategies.from_regex(r'0[0-9]{2,}')` to generate matching strings
    - Assert `post_process_text(s)` starts with `'Ø'` and the remaining characters equal `s[1:]`
    - **Property 1: Leading-zero diameter correction preserves digits**
    - **Validates: Requirements 1.1**
  - [ ]* 11.2 Write property test for Property 2: aspect-ratio heuristic is correctly gated
    - Generate `(raw_text, w, h)` triples; test the conditional logic directly (extract it or test via a helper)
    - Assert correction applied iff `raw_text.strip() == "8"` and `0 < w <= 20` and `h >= 20`
    - **Property 2: Aspect-ratio heuristic is correctly gated**
    - **Validates: Requirements 1.2, 11.1**
  - [ ]* 11.3 Write property test for Property 3: radius callout classification and extraction round-trip
    - Use `hypothesis.strategies.from_regex(r'R[0-9]+(\.[0-9]+)?')` to generate valid radius strings
    - Assert `classify(s, category=1)` returns `"radius_callout"`
    - Assert `extract_parsed("radius_callout", s)["radius"]` equals `float(s[1:])`
    - **Property 3: Radius callout classification and extraction round-trip**
    - **Validates: Requirements 2.2, 2.4**
  - [ ]* 11.4 Write property test for Property 4: material name classification with fuzzy matching
    - Draw a string from `MATERIAL_NAMES` and optionally apply 0–2 single-character mutations
    - Assert `classify(mutated, category=2)` returns `"material_name"` (provided no higher-priority pattern matches)
    - **Property 4: Material name classification with fuzzy matching**
    - **Validates: Requirements 4.3, 4.4**
  - [ ]* 11.5 Write property test for Property 5: part name classification with compound threshold
    - Draw a name from `PART_NAMES`; for compound names apply 0–2 mutations, for single-word names apply 0–1 mutations
    - Assert `classify(mutated, category=2)` returns `"part_name"` (provided no higher-priority pattern matches)
    - **Property 5: Part name classification with compound threshold**
    - **Validates: Requirements 3.2, 3.3, 3.4**
  - [ ]* 11.6 Write property test for Property 6: PROTECTED_CODES are never modified by `normalise_text`
    - Draw a string from `PROTECTED_CODES` (with optional surrounding whitespace)
    - Assert `normalise_text(s)` equals `s.strip()`
    - **Property 6: PROTECTED_CODES are never modified by normalise_text**
    - **Validates: Requirements 7.4**
  - [ ]* 11.7 Write property test for Property 7: spatial adjacency symmetry
    - Generate two bounding boxes A and B as `[x, y, w, h]` lists
    - Compute y-centroids and check the `abs(y_center_A - y_center_B) <= 10` predicate
    - Assert the predicate is symmetric: `same_row(A, B) == same_row(B, A)`
    - **Property 7: Spatial adjacency check is symmetric**
    - **Validates: Requirements 6.2**
  - [ ]* 11.8 Write property test for Property 8: BOM row reconstruction is category-gated
    - Generate arbitrary lists of classified entries and a category drawn from `{1, 3}`
    - Assert `reconstruct_bom_rows(entries, category)` returns `[]`
    - **Property 8: BOM row reconstruction is category-gated**
    - **Validates: Requirements 6.7**

- [x] 12. Final checkpoint — Ensure all tests pass
  - Run `pytest tests/test_validation.py tests/test_ocr_accuracy.py tests/test_ocr_accuracy_properties.py -v`
  - All 109 existing tests plus all new tests must pass; ask the user if any failures arise

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Tasks 1–7 are pure implementation; tasks 8–12 are verification
- The `upper_t` variable introduced in task 4.2 (P8.5 loop) must be defined before the P9 block in task 4.3 — ensure the variable is not conditionally scoped
- The aspect-ratio heuristic (task 2) uses OCR-scale coordinates (`xs`, `ys` from `bbox`) before the `inverse_scale` multiplication; this is intentional — the thresholds (w≤20, h≥20) are calibrated to OCR-scale pixels
- Property tests use Hypothesis, which is already present in the project environment (`.hypothesis/` directory exists)
- All 109 existing tests in `tests/test_validation.py` must pass without modification at every checkpoint
