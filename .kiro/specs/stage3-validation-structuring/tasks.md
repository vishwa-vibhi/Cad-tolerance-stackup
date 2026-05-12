# Implementation Plan: Stage 3 — Validation & Structuring

## Overview

Implement `src/validation.py` as a pure Python (stdlib only) module that consumes `_fullocr.json` files produced by Stage 2 and writes `_structured.json` files. The module normalises OCR artefacts, classifies each text entry into one of 14 engineering annotation types using a priority-ordered regex chain, extracts typed parsed fields, and integrates with the existing `batch_process.py` pipeline.

## Tasks

- [x] 1. Scaffold `src/validation.py` — module constants and skeleton
  - Create `src/validation.py` with the module docstring, all `import` statements (`re`, `json`, `os`, `sys`, `pathlib`), and the `PROTECTED_CODES` set copied verbatim from `src/vlm_reader.py`
  - Declare all 11 module-level compiled regex constants: `RE_HOLE`, `RE_HOLE_NUMERIC`, `RE_THREAD`, `RE_DIAMETER`, `RE_DIM_NOTE`, `RE_TOLERANCE`, `RE_SPACING`, `RE_SECTION`, `RE_BALLOON`, `RE_QUANTITY`, `RE_DIMENSION`
  - Declare the `BOM_HEADERS`, `MATERIAL_CODES`, and `PART_NAMES` sets
  - Add empty stub definitions (with `pass`) for all public and internal functions: `validate_file`, `validate_batch`, `detect_category`, `normalise_text`, `_is_protected`, `classify`, `extract_parsed`, `_edit_distance`, `build_structured_output`
  - Add the `if __name__ == "__main__":` block that reads `sys.argv[1]` and `sys.argv[2]` as `input_dir` and `output_dir` and calls `validate_batch`
  - _Requirements: 6.1, 6.2, 6.5, 6.6, 8.2, 8.3_

- [ ] 2. Implement normalisation pipeline
  - [x] 2.1 Implement `_is_protected(token: str) -> bool`
    - Return `True` if `token.strip().upper()` is in `PROTECTED_CODES`; otherwise `False`
    - _Requirements: 1.6_

  - [x] 2.2 Implement `normalise_text(text: str) -> str`
    - Strip leading/trailing whitespace; if the stripped string is protected, return it unchanged
    - Apply the four substitutions in order: (a) leading-zero diameter `^0(\d{2,})$` → `Ø\1`, (b) degree symbol `(\d{1,2})"` → `\1°`, (c) THICK typos `IHICK`/`MICK` → `THICK` (case-insensitive), (d) `*` → `×`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [ ]* 2.3 Write property test — Property 1: Normalisation preserves PROTECTED_CODES
    - **Property 1: For any token that is an exact member of PROTECTED_CODES, `normalise_text` returns the token unchanged**
    - **Validates: Requirements 1.6**

  - [ ]* 2.4 Write property test — Property 2: Leading-zero diameter correction
    - **Property 2: For any string matching `0\d{2,}` (fullmatch), `normalise_text` returns a string starting with `Ø` with the remaining digits preserved**
    - **Validates: Requirements 1.1**

  - [ ]* 2.5 Write property test — Property 3: Degree symbol correction
    - **Property 3: For any string containing one or two decimal digits immediately followed by `"`, `normalise_text` replaces `"` with `°`**
    - **Validates: Requirements 1.2**

  - [ ]* 2.6 Write property test — Property 4: THICK typo correction
    - **Property 4: For any string containing `IHICK` or `MICK` (case-insensitive), `normalise_text` replaces those substrings with `THICK`**
    - **Validates: Requirements 1.3**

- [ ] 3. Implement helper utilities
  - [x] 3.1 Implement `_edit_distance(a: str, b: str) -> int`
    - Pure Python Levenshtein distance, O(len(a) × len(b)), using a 2-row rolling array
    - _Requirements: 2.10_

  - [x] 3.2 Implement `detect_category(filename: str) -> int`
    - Check `os.path.basename(filename).lower()` for substrings `cad1_`, `cad2_`, `cad3_`; return 1, 2, or 3 respectively; print a warning and return 0 if none match
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [ ]* 3.3 Write property test — Property 16: Category detection from filename
    - **Property 16: For any filename containing `cad1_`, `cad2_`, or `cad3_`, `detect_category` returns 1, 2, or 3 respectively**
    - **Validates: Requirements 5.1–5.3**

- [ ] 4. Implement classification engine
  - [x] 4.1 Implement `classify(text: str, category: int) -> str`
    - Implement the 14-step priority if/elif chain exactly as specified in the design: P1 `hole_callout` → P2 `thread_spec` → P3 `diameter_callout` → P4 `dimension_with_note` → P5 `tolerance` → P6 `spacing_annotation` → P7 `bom_header` → P8 `material_code` → P9 `part_name` (with `_edit_distance` fuzzy match ≤ 1) → P10 `section_marker` (Cat 1/3 only) → P11 `balloon_number` (Cat 2/3 only) → P12 `quantity` (Cat 2 only) → P13 `dimension_value` → P14 `unknown`
    - Return `"unknown"` immediately for empty or whitespace-only input
    - _Requirements: 2.1–2.16_

  - [ ]* 4.2 Write property test — Property 5: Classifier always returns a valid type
    - **Property 5: For any non-empty string and any category in {0,1,2,3}, `classify` returns a member of the 14-type set**
    - **Validates: Requirements 2.1**

  - [ ]* 4.3 Write property test — Property 6: Thread spec classification
    - **Property 6: For any string matching `M\d+` (optionally followed by `×` and a pitch), `classify` returns `"thread_spec"` for all categories**
    - **Validates: Requirements 2.3**

  - [ ]* 4.4 Write property test — Property 7: Diameter callout classification
    - **Property 7: For any string of the form `Ø{digits}` or `DIA {digits}` with no trailing note keyword, `classify` returns `"diameter_callout"` for all categories**
    - **Validates: Requirements 2.5**

  - [ ]* 4.5 Write property test — Property 8: Priority — hole_callout beats diameter_callout
    - **Property 8: For any string containing `HOLE` and a numeric value or `DIA`, `classify` returns `"hole_callout"` and not `"diameter_callout"`**
    - **Validates: Requirements 2.6, 2.16**

  - [ ]* 4.6 Write property test — Property 9: Priority — dimension_with_note beats diameter_callout
    - **Property 9: For any string starting with `DIA` and containing `THICK`, `DEEP`, `LONG`, or `WIDE`, `classify` returns `"dimension_with_note"` and not `"diameter_callout"`**
    - **Validates: Requirements 2.14, 2.16**

  - [ ]* 4.7 Write property test — Property 10: Category-gated section_marker
    - **Property 10: For any string matching `[A-Z]-[A-Z]` or a single uppercase letter, `classify` returns `"section_marker"` for category 1 or 3, and does not return `"section_marker"` for category 2**
    - **Validates: Requirements 2.7**

  - [ ]* 4.8 Write property test — Property 11: Category-gated balloon_number
    - **Property 11: For any single-digit string `"1"`–`"9"`, `classify` returns `"balloon_number"` for category 2 or 3, and `"dimension_value"` for category 1**
    - **Validates: Requirements 2.12**

- [x] 5. Checkpoint — verify normalisation and classification
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement parsed field extraction
  - [x] 6.1 Implement `extract_parsed(type_: str, text: str) -> dict`
    - Implement all 14 branches as specified in the design's `extract_parsed` pseudocode
    - Wrap `float()` and `int()` conversions in `try/except` and fall back to `None` on failure
    - `dimension_value` → `{"value": float}`, `thread_spec` → `{"nominal": str, "pitch": float|None}`, `tolerance` → `{"tolerance_string": str}`, `diameter_callout` → `{"diameter": float}`, `hole_callout` → `{"raw": str}`, `section_marker` → `{"label": str}`, `spacing_annotation` → `{"annotation": "EQUI-SP"}`, `material_code` → `{"code": str}` (uppercase), `part_name` → `{"name": str}` (title case), `bom_header` → `{"header": str}` (uppercase), `balloon_number` → `{"number": int}`, `quantity` → `{"qty": int}`, `dimension_with_note` → `{"raw": str}`, `unknown` → `{}`
    - _Requirements: 3.1–3.14_

  - [ ]* 6.2 Write property test — Property 12: Parsed fields are structurally correct for the assigned type
    - **Property 12: For any string and category, if `classify(text, category)` returns type T, then `extract_parsed(T, text)` returns a dict whose keys exactly match the schema for T**
    - **Validates: Requirements 3.1–3.14**

  - [ ]* 6.3 Write property test — Property 13: Thread spec parsed fields are correct
    - **Property 13: For any string matching `M{n}` or `M{n}×{p}`, `extract_parsed("thread_spec", text)` returns `{"nominal": "M{n}", "pitch": p_or_None}` with correct values**
    - **Validates: Requirements 3.2**

- [ ] 7. Implement output generation and file I/O
  - [x] 7.1 Implement `build_structured_output(source_file: str, category: int, classified_entries: list) -> dict`
    - Assemble the top-level dict with keys `source_file`, `image_category`, `total_detections`, `classified`, `summary`
    - Set `total_detections = len(classified_entries)`
    - Build `summary` by counting occurrences of each `type` value in `classified_entries`
    - _Requirements: 4.2, 4.3, 4.5_

  - [x] 7.2 Implement `validate_file(fullocr_path: str, output_dir: str) -> dict`
    - Read and parse the `_fullocr.json` file; catch `json.JSONDecodeError` and log + return an empty-result dict on failure
    - Call `detect_category` on the filename
    - For each entry: get `text` via `entry.get("text", "")`, call `normalise_text`, `classify`, `extract_parsed`; handle missing `id`/`box` with `.get()` fallbacks
    - Call `build_structured_output` and write the result to `{output_dir}/{basename}_structured.json` using `json.dump(..., indent=2, ensure_ascii=False)`
    - Create `output_dir` with `os.makedirs(..., exist_ok=True)` if it does not exist
    - _Requirements: 4.1, 4.2, 4.4, 4.6, 4.7, 9.1–9.4_

  - [ ]* 7.3 Write property test — Property 14: total_detections equals classified array length
    - **Property 14: For any valid `_fullocr.json` input with N entries, `output["total_detections"] == len(output["classified"]) == N`**
    - **Validates: Requirements 4.3**

  - [ ]* 7.4 Write property test — Property 15: Summary counts are consistent with classified array
    - **Property 15: `sum(summary.values()) == total_detections` and each summary count equals the number of `classified` entries with that type**
    - **Validates: Requirements 4.5**

- [ ] 8. Implement `validate_batch` and CLI entry point
  - [x] 8.1 Implement `validate_batch(input_dir: str, output_dir: str) -> list`
    - Glob all `*_fullocr.json` files in `input_dir` using `pathlib.Path(input_dir).glob("*_fullocr.json")`
    - Call `validate_file` for each file inside a `try/except Exception` block; log `ERROR processing {filename}: {e}` on failure and continue
    - Return the list of result dicts (skip `None` entries from failed files)
    - _Requirements: 6.2, 6.3, 6.4_

  - [x] 8.2 Complete the `__main__` block
    - Parse `sys.argv[1]` as `input_dir` and `sys.argv[2]` as `output_dir`; print usage and exit if fewer than 3 args
    - Call `validate_batch` and print a per-file summary line: `[N/total] {filename} → {count} entries | {type}:{count} ...`
    - Print a final line: `Batch complete: {n} files processed, {errors} errors`
    - _Requirements: 6.6_

- [ ] 9. Create test fixtures and unit tests
  - [x] 9.1 Create `tests/fixtures/sample_fullocr.json`
    - Write a synthetic fixture with at least 12 entries covering all 14 classification types (use entries from `results/batch/cad1_001_fullocr.json` and `results/batch/cad2_001_fullocr.json` as a basis, adding synthetic entries for types not naturally present)
    - Include at least one entry with a missing `text` field, one with `confidence < 0.6`, and one with an empty string `text`
    - _Requirements: 9.1, 9.2, 9.3_

  - [x] 9.2 Create `tests/test_validation.py` with unit tests
    - Import `sys` and insert `src` onto `sys.path`; import `normalise_text`, `classify`, `extract_parsed`, `detect_category`, `validate_file`, `validate_batch` from `validation`
    - Write concrete unit tests for each normalisation rule (one positive + one negative per rule, using the examples from the design's Normalisation Examples table)
    - Write concrete unit tests for each of the 14 classification types (at least one positive example per type)
    - Write unit tests for `extract_parsed` covering all 14 type branches
    - Write unit tests for `validate_file` using `tests/fixtures/sample_fullocr.json`: verify output schema, `total_detections`, `summary` consistency, and that the `_structured.json` file is written
    - Write unit tests for error handling: malformed JSON input, empty array input, entry with missing `text`
    - _Requirements: 2.1–2.16, 3.1–3.14, 4.1–4.7, 9.1–9.4_

- [x] 10. Write all 16 Hypothesis property-based tests
  - Add all property-based tests to `tests/test_validation.py` using `@given` and `@settings` decorators from Hypothesis
  - Each test must include the comment `# Feature: stage3-validation-structuring, Property {N}: {property_text}` immediately above the `@given` decorator
  - Use `@settings(max_examples=200)` for classification properties (Properties 5–11) and `@settings(max_examples=500)` for Property 5
  - Properties 1–4 test `normalise_text`; Properties 5–11 test `classify`; Properties 12–13 test `extract_parsed`; Properties 14–15 test `validate_file` output structure; Property 16 tests `detect_category`
  - _Requirements: 1.1–1.6, 2.1–2.16, 3.1–3.14, 4.3, 4.5, 5.1–5.3_

- [x] 11. Checkpoint — run full test suite
  - Ensure all unit tests and property-based tests pass, ask the user if questions arise.

- [x] 12. Integrate Stage 3 into `batch_process.py`
  - Add `from src.validation import validate_file` (with `try/except ImportError` fallback to `from validation import validate_file`) at the top of `batch_process.py`, mirroring the existing `vlm_reader` import pattern
  - In `process_category()`, after the `read_full_image()` call succeeds, derive `basename` from `img_name` (strip extension), construct `fullocr_path = os.path.join(results_dir, f"{basename}_fullocr.json")`, and call `validate_file(fullocr_path, output_dir=results_dir)`
  - Add a `structured_count` field to the per-image `stat` dict (number of classified entries from the returned dict)
  - _Requirements: 6.1, 6.5_

- [x] 13. End-to-end validation run on all 36 images
  - Create a standalone script `scripts/run_stage3.py` (or add a `__main__` invocation note) that calls `validate_batch("results/batch", "results/batch")` and prints the per-file summary
  - Verify that 36 `_structured.json` files are written to `results/batch/` (one per `_fullocr.json`)
  - Verify that no file has `total_detections = 0` unless the corresponding `_fullocr.json` is genuinely empty
  - Verify that the `unknown` type accounts for ≤ 15% of entries in any single Category 1 image (Requirement 7.4)
  - _Requirements: 6.2, 6.3, 7.1–7.4, 8.1_

- [x] 14. Final checkpoint — full pipeline smoke test
  - Ensure all tests pass and all 36 `_structured.json` files are present and valid, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- All 16 Hypothesis properties are grouped in task 10 but individual property sub-tasks (2.3–2.6, 3.3, 4.2–4.8, 6.2–6.3, 7.3–7.4) place each property close to the implementation it validates — implement whichever grouping is more convenient
- `PROTECTED_CODES` must be kept in sync with `src/vlm_reader.py`; the design specifies copying it verbatim
- The `quantity` vs `balloon_number` distinction relies on a spatial heuristic (y-coordinate in lower half of image for Cat 2); Stage 4 will refine this — implement the heuristic as described in the design
- Accuracy targets (Req 7) require manual review of `_structured.json` outputs; they are not automatically verified by the test suite
