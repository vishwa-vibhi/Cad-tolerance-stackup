# Requirements Document

## Introduction

The OCR Accuracy Improvements feature enhances the CAD Tolerance Stack-Up Analysis Tool's ability to correctly read and classify engineering annotations from CAD drawings. The current pipeline (Stages 1–3) successfully processes Category 1 part drawings but struggles with specific OCR patterns: diameter symbols (Ø/φ) are consistently misread as digits or currency symbols, radius dimensions are not classified, compound part names and full material names in BOM tables fall through to `unknown`, and BOM table text is fragmented rather than structured into rows. This feature addresses these issues through targeted improvements to Stage 2 (OCR reading in `src/vlm_reader.py`) and Stage 3 (classification and structuring in `src/validation.py`), raising classification accuracy for Category 2 assembly drawings from ~70% to ≥80% for part names and ≥70% for material names, while maintaining 100% backward compatibility with the existing 109 passing tests.

---

## Glossary

- **OCR_Reader**: The Stage 2 module (`src/vlm_reader.py`) that uses EasyOCR to extract text from CAD drawing images.
- **Validator**: The Stage 3 module (`src/validation.py`) that classifies and structures OCR output.
- **Diameter_Symbol**: The engineering symbol Ø or φ indicating a diameter dimension.
- **Radius_Callout**: A dimension annotation of the form `R` followed by a numeric value (e.g., `R189`, `R78`, `R13`).
- **Compound_Part_Name**: A multi-word part name such as "Articulated rod", "Cover plate", "Rod end", "Lock nut", "Link pin", "Piston pin", "Piston ring", "Rod bush-upper", "Master rod bearing", "Piston pin plug".
- **Material_Name**: A full material name string such as "Babbit", "Ni-Cr steel", "Cd-Ag", "Brass", "Cast iron".
- **Material_Code**: A 2-letter material abbreviation such as "MS", "CI", "FS", "GM", "CS", "CR", "AL", "BR".
- **BOM_Row**: A structured record representing one row of a Bill of Materials table, containing fields `part_no`, `part_name`, `material`, and `qty`.
- **BOM_Header**: A column header in a BOM table such as "Parts list", "Name", "Material", "Qty", "No", "Sl No", "Part No", "Matl".
- **Leading_Zero_Pattern**: An OCR artefact where a diameter symbol Ø is misread as a leading zero followed by digits (e.g., `061` for Ø61, `085` for Ø85).
- **Category_1**: Single-part drawings containing dimensions, tolerances, and section markers; no BOM.
- **Category_2**: Assembly drawings containing balloon numbers, BOM tables, part names, and material codes.
- **Category_3**: Assembly-view-only drawings containing balloon numbers but no dimensions.
- **Unknown_Rate**: The percentage of OCR entries classified as `unknown` in a single image.
- **Spatial_Adjacency**: Two OCR bounding boxes are spatially adjacent if their vertical distance is ≤ 20 pixels and their horizontal overlap or gap is ≤ 50 pixels.

---

## Requirements

### Requirement 1: Diameter Symbol Post-Processing in Stage 2

**User Story:** As a pipeline engineer, I want the OCR Reader to detect and correct diameter symbol misreads immediately after EasyOCR returns results, so that Stage 3 receives cleaner input and classification accuracy improves.

#### Acceptance Criteria

1. WHEN the OCR_Reader processes an EasyOCR result whose `text` field matches the pattern `^0(\d{2,})$` (leading zero followed by 2+ digits), THE OCR_Reader SHALL replace the leading `0` with `Ø` in the `text` field before writing the `_fullocr.json` file.
2. WHEN the OCR_Reader processes an EasyOCR result whose `text` field is exactly `8` and the bounding box width is ≤ 20 pixels and height is ≥ 20 pixels (tall, narrow box characteristic of a diameter symbol), THE OCR_Reader SHALL replace `8` with `Ø` in the `text` field.
3. WHEN the OCR_Reader processes an EasyOCR result whose `text` field contains `€`, `∅`, `ϕ`, `φ`, or `ø`, THE OCR_Reader SHALL replace those characters with `Ø` in the `text` field.
4. THE OCR_Reader SHALL apply diameter symbol corrections in the `post_process_text()` function after all other existing corrections.
5. THE OCR_Reader SHALL preserve the original EasyOCR output in the `raw_text` field so that corrections are traceable.

---

### Requirement 2: Radius Callout Classification in Stage 3

**User Story:** As a pipeline engineer, I want radius dimensions like `R189`, `R78`, `R13` classified as a distinct type, so that Stage 4 can associate them with arc geometry and they no longer fall through to `unknown`.

#### Acceptance Criteria

1. THE Validator SHALL add a new classification type `radius_callout` to the set of valid types.
2. WHEN the Validator classifies a normalised text string matching the pattern `^R\d+(\.\d+)?$` (uppercase `R` followed by one or more digits, optional decimal), THE Validator SHALL assign type `radius_callout`.
3. THE Validator SHALL evaluate the `radius_callout` pattern at priority level 3.5 (after `diameter_callout` and before `dimension_with_note`) in the classification chain.
4. WHEN the Validator assigns type `radius_callout`, THE Validator SHALL populate `parsed` with `{"radius": <float>}` where the float is the numeric value following the `R` prefix.
5. WHEN evaluated against the dataset, THE Validator SHALL classify 100% of radius callouts (e.g., `R189`, `R78`, `R13`, `R24`) as type `radius_callout` rather than `unknown`.

---

### Requirement 3: Compound Part Name Classification in Stage 3

**User Story:** As a pipeline engineer, I want multi-word part names from BOM tables classified as `part_name`, so that BOM row reconstruction can identify part name fields reliably.

#### Acceptance Criteria

1. THE Validator SHALL expand the `PART_NAMES` set to include the following compound names: `ARTICULATED ROD`, `COVER PLATE`, `ROD END`, `LOCK NUT`, `LINK PIN`, `PISTON PIN`, `PISTON RING`, `ROD BUSH-UPPER`, `ROD BUSH-LOWER`, `MASTER ROD BEARING`, `PISTON PIN PLUG`.
2. WHEN the Validator classifies a normalised text string that exactly matches (case-insensitive) a compound part name from the expanded `PART_NAMES` set, THE Validator SHALL assign type `part_name`.
3. WHEN the Validator classifies a normalised text string that has an edit distance ≤ 2 from a compound part name in the expanded `PART_NAMES` set, THE Validator SHALL assign type `part_name`.
4. THE Validator SHALL preserve the existing fuzzy-matching logic (edit distance ≤ 1 for single-word names, ≤ 2 for compound names) to handle OCR typos.
5. WHEN evaluated against Category 2 images, THE Validator SHALL classify at least 80% of compound part names correctly as `part_name`.

---

### Requirement 4: Material Name Classification in Stage 3

**User Story:** As a pipeline engineer, I want full material names like "Babbit", "Ni-Cr steel", "Cd-Ag" classified as a distinct type, so that BOM row reconstruction can distinguish material names from material codes.

#### Acceptance Criteria

1. THE Validator SHALL add a new classification type `material_name` to the set of valid types.
2. THE Validator SHALL create a new set constant `MATERIAL_NAMES` containing the following entries: `BABBIT`, `BRASS`, `NI-CR STEEL`, `CD-AG`, `CAST IRON`, `MILD STEEL`, `HIGH CARBON STEEL`, `LOW CARBON STEEL`, `STAINLESS STEEL`, `ALUMINUM`, `BRONZE`, `COPPER`.
3. WHEN the Validator classifies a normalised text string that exactly matches (case-insensitive) an entry in `MATERIAL_NAMES`, THE Validator SHALL assign type `material_name`.
4. WHEN the Validator classifies a normalised text string that has an edit distance ≤ 2 from an entry in `MATERIAL_NAMES`, THE Validator SHALL assign type `material_name`.
5. THE Validator SHALL evaluate the `material_name` pattern at priority level 8.5 (after `material_code` and before `part_name`) in the classification chain.
6. WHEN the Validator assigns type `material_name`, THE Validator SHALL populate `parsed` with `{"name": <string>}` in title case.
7. WHEN evaluated against Category 2 images, THE Validator SHALL classify at least 70% of full material names correctly as `material_name`.

---

### Requirement 5: BOM Header Expansion in Stage 3

**User Story:** As a pipeline engineer, I want common BOM column header abbreviations like "Matl" recognized, so that BOM table detection is more robust across different drawing standards.

#### Acceptance Criteria

1. THE Validator SHALL expand the `BOM_HEADERS` set to include the following entries: `MATL`, `MAT`, `MATERIAL`, `SL.NO`, `SL NO`, `SL. NO`, `PART NAME`, `PART NO`, `PART NO.`, `PARTS LIST`, `NAME`, `QTY`, `NO`.
2. WHEN the Validator classifies a normalised text string that exactly matches (case-insensitive) an entry in the expanded `BOM_HEADERS` set, THE Validator SHALL assign type `bom_header`.
3. THE Validator SHALL preserve the existing priority level (P7) for `bom_header` classification.

---

### Requirement 6: BOM Row Reconstruction in Stage 3

**User Story:** As a pipeline engineer, I want spatially-adjacent BOM fragments grouped into structured rows, so that downstream stages can consume complete part records instead of isolated text fragments.

#### Acceptance Criteria

1. THE Validator SHALL add a new function `reconstruct_bom_rows(classified_entries: list, category: int) -> list` that accepts a list of classified entries and returns a list of BOM_Row dictionaries.
2. WHEN `reconstruct_bom_rows` is called with a Category 2 image's classified entries, THE Validator SHALL identify all entries of type `balloon_number`, `part_name`, `material_code`, `material_name`, and `quantity` that are spatially adjacent (vertical distance ≤ 20 pixels, horizontal overlap or gap ≤ 50 pixels).
3. WHEN `reconstruct_bom_rows` identifies a spatially-adjacent group of entries, THE Validator SHALL construct a BOM_Row dictionary with fields `part_no` (from `balloon_number`), `part_name` (from `part_name`), `material` (from `material_code` or `material_name`), and `qty` (from `quantity`).
4. IF a BOM_Row is missing any of the four required fields, THE Validator SHALL still include the row in the output with the missing fields set to `null`.
5. THE Validator SHALL add a new top-level field `bom_rows` to the `_structured.json` output schema, containing the list of reconstructed BOM_Row dictionaries.
6. WHEN evaluated against Category 2 images, THE Validator SHALL produce at least one complete BOM_Row (all four fields non-null) for at least 60% of Category 2 images.
7. THE Validator SHALL only invoke `reconstruct_bom_rows` for Category 2 images; for Category 1 and Category 3 images, the `bom_rows` field SHALL be an empty list.

---

### Requirement 7: Backward Compatibility with Existing Tests

**User Story:** As a pipeline engineer, I want all existing Stage 3 tests to continue passing after the improvements, so that I can be confident the changes do not break existing functionality.

#### Acceptance Criteria

1. WHEN the test suite `tests/test_validation.py` is run after implementing the OCR accuracy improvements, THE Validator SHALL pass all 109 existing tests without modification to the test code.
2. THE Validator SHALL preserve the existing classification priority order for all types that existed before this feature (e.g., `hole_callout` before `diameter_callout`, `thread_spec` before `dimension_value`).
3. THE Validator SHALL preserve the existing `normalise_text()` function behavior for all patterns that existed before this feature (leading-zero diameter fix, degree symbol fix, THICK typo fix, multiplication symbol fix).
4. THE Validator SHALL preserve the existing `PROTECTED_CODES` set and SHALL NOT modify any text that matches a protected code during normalisation.
5. THE Validator SHALL preserve the existing output schema for `_structured.json` files, adding only the new `bom_rows` field without removing or renaming any existing fields.

---

### Requirement 8: Accuracy Targets for Category 1 Images

**User Story:** As a student researcher, I want the improvements to maintain or improve accuracy on Category 1 part drawings, so that the pipeline continues to perform well on the primary use case.

#### Acceptance Criteria

1. WHEN evaluated against Category 1 images after implementing the improvements, THE Validator SHALL maintain a correct-classification rate of at least 90% across all OCR entries.
2. WHEN evaluated against Category 1 images after implementing the improvements, THE Validator SHALL assign type `unknown` to no more than 15% of entries in any single Category 1 image.
3. THE Validator SHALL classify 100% of radius callouts in Category 1 images as type `radius_callout` rather than `unknown`.

---

### Requirement 9: Accuracy Targets for Category 2 Images

**User Story:** As a student researcher, I want the improvements to significantly raise accuracy on Category 2 assembly drawings, so that BOM extraction becomes viable for the final project deliverable.

#### Acceptance Criteria

1. WHEN evaluated against Category 2 images after implementing the improvements, THE Validator SHALL classify at least 80% of compound part names correctly as `part_name`.
2. WHEN evaluated against Category 2 images after implementing the improvements, THE Validator SHALL classify at least 70% of full material names correctly as `material_name`.
3. WHEN evaluated against Category 2 images after implementing the improvements, THE Validator SHALL produce at least one complete BOM_Row (all four fields non-null) for at least 60% of Category 2 images.
4. WHEN evaluated against Category 2 images after implementing the improvements, THE Validator SHALL reduce the unknown rate by at least 10 percentage points compared to the baseline (before improvements).

---

### Requirement 10: Performance and Resource Constraints

**User Story:** As a student researcher, I want the improvements to run within the same performance budget as the existing pipeline, so that batch processing remains practical on my laptop.

#### Acceptance Criteria

1. WHEN processing all 36 images in batch mode on a machine with 8 GB RAM and a Windows 11 operating system, THE Validator SHALL complete within 60 seconds of wall-clock time (same as the existing requirement).
2. THE Validator SHALL NOT import any new third-party libraries beyond those already listed in `requirements.txt`.
3. THE Validator SHALL use only Python standard library modules (`re`, `json`, `os`, `sys`, `pathlib`) plus libraries already present in the project virtual environment.
4. THE `reconstruct_bom_rows` function SHALL have time complexity O(n²) or better, where n is the number of classified entries in a single image.

---

### Requirement 11: Robustness to Edge Cases

**User Story:** As a pipeline engineer, I want the improvements to handle edge cases gracefully, so that a single malformed entry does not corrupt the output or crash the pipeline.

#### Acceptance Criteria

1. WHEN the OCR_Reader encounters an EasyOCR result with a bounding box width or height of zero, THE OCR_Reader SHALL skip the diameter symbol aspect-ratio check and apply only the text-pattern-based corrections.
2. WHEN the Validator encounters a normalised text string that matches multiple new patterns (e.g., both `radius_callout` and `dimension_value`), THE Validator SHALL assign the type with the highest priority in the classification chain.
3. WHEN `reconstruct_bom_rows` encounters a classified entry with a missing or malformed `box` field, THE Validator SHALL skip that entry and continue processing the remaining entries without raising an exception.
4. IF `reconstruct_bom_rows` produces zero BOM_Row dictionaries for a Category 2 image, THE Validator SHALL write an empty list `[]` to the `bom_rows` field in the `_structured.json` output.

---

### Requirement 12: Traceability of OCR Corrections

**User Story:** As a student researcher, I want to see which OCR corrections were applied to each entry, so that I can debug classification errors and evaluate the effectiveness of the improvements.

#### Acceptance Criteria

1. THE OCR_Reader SHALL preserve the original EasyOCR output in the `raw_text` field of each entry in the `_fullocr.json` file.
2. THE Validator SHALL preserve the `raw_text` field in the `_structured.json` output so that the original OCR output is traceable through the entire pipeline.
3. WHEN the OCR_Reader applies a diameter symbol correction, THE `text` field SHALL differ from the `raw_text` field, making the correction visible in the output.
4. WHEN the Validator applies a normalisation correction, THE `text` field in the `_structured.json` output SHALL reflect the normalised value, while `raw_text` SHALL remain unchanged.

---

## Special Requirements Guidance

### Parser and Serializer Requirements

This feature does not introduce new parsers or serializers. The existing JSON reading and writing logic in `src/validation.py` and `src/vlm_reader.py` is sufficient and does not require round-trip testing.

### Property-Based Testing Guidance

The following acceptance criteria are suitable for property-based testing:

- **Requirement 1, Criterion 1**: Property: For all strings matching `^0(\d{2,})$`, applying the correction produces a string starting with `Ø` followed by the same digits.
- **Requirement 2, Criterion 4**: Property: For all strings matching `^R\d+(\.\d+)?$`, extracting the radius value and formatting it back produces a string matching the same pattern.
- **Requirement 3, Criterion 4**: Property: For all strings in `PART_NAMES`, computing edit distance to itself returns 0, and computing edit distance to a 1-character mutation returns ≤ 2.
- **Requirement 6, Criterion 2**: Property: For all pairs of bounding boxes, the spatial adjacency check is symmetric (if A is adjacent to B, then B is adjacent to A).

The following acceptance criteria are NOT suitable for property-based testing and should use integration tests with representative examples:

- **Requirement 6, Criterion 6**: BOM row reconstruction success rate (requires real OCR data, not synthetic inputs).
- **Requirement 8, Criterion 1**: Overall classification accuracy (requires ground-truth labels, not generated inputs).
- **Requirement 9, Criteria 1–4**: Category 2 accuracy targets (requires real OCR data and manual labels).
