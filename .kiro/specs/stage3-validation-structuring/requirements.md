# Requirements Document

## Introduction

Stage 3 of the CAD Tolerance Stack-Up Analysis Tool is a Validation & Structuring module (`src/validation.py`). It consumes the raw OCR output produced by Stage 2 (`_fullocr.json` files) and converts every detected text entry into a typed, structured engineering record. The module must classify each text string into one of 14 engineering annotation types using pure Python regex, correct known OCR artefacts, extract typed fields from each classification, and write a `_structured.json` file per image. It must also integrate with the existing `batch_process.py` pipeline and run on all 36 images across three drawing categories without requiring any new third-party libraries.

---

## Glossary

- **Classifier**: The component within the Validation Module responsible for assigning an engineering type to a text entry.
- **OCR Entry**: A single JSON object from a `_fullocr.json` file, containing `id`, `box`, `text`, `raw_text`, and `confidence` fields.
- **Structured Record**: An OCR Entry augmented with a `type` field and a `parsed` field containing typed, extracted sub-fields.
- **Validation Module**: The Python module `src/validation.py` that implements Stage 3.
- **Batch Runner**: The existing `batch_process.py` script that orchestrates the full pipeline.
- **PROTECTED_CODES**: The set of engineering abbreviations defined in `src/vlm_reader.py` that must never be altered during text normalisation.
- **Category 1**: Single-part drawings containing dimensions, tolerances, and section markers; no BOM.
- **Category 2**: Assembly drawings containing balloon numbers, BOM tables, part names, and material codes.
- **Category 3**: Assembly-view-only drawings containing balloon numbers but no dimensions.
- **Balloon Number**: A circled integer in an assembly drawing that references a row in the BOM.
- **BOM**: Bill of Materials — a tabular list of parts in an assembly drawing.
- **OCR Artefact**: A character substitution introduced by EasyOCR, such as `0` for `Ø` or `"` for `°`.
- **Normalised Text**: The `text` field after OCR artefact correction, used as input to the Classifier.
- **Parsed Fields**: A dictionary of typed sub-values extracted from a classified text entry (e.g., `{"nominal": "M30", "pitch": 2.5}`).
- **Summary**: A count of classified entries grouped by type, included in each `_structured.json` output.

---

## Requirements

### Requirement 1: OCR Artefact Normalisation

**User Story:** As a pipeline engineer, I want known OCR character substitutions corrected before classification, so that downstream stages receive clean, consistent text.

#### Acceptance Criteria

1. WHEN the Validation Module processes an OCR Entry whose `text` field begins with `0` followed by two or more digits (e.g., `018`, `0118`), THE Validation Module SHALL replace the leading `0` with `Ø` to produce a diameter callout (e.g., `Ø18`, `Ø118`).
2. WHEN the Validation Module processes an OCR Entry whose `text` field contains `"` immediately following one or two digits (e.g., `45"`), THE Validation Module SHALL replace `"` with `°` to produce an angle annotation (e.g., `45°`).
3. WHEN the Validation Module processes an OCR Entry whose `text` field contains `IHICK` or `MICK`, THE Validation Module SHALL replace those substrings with `THICK`.
4. WHEN the Validation Module processes an OCR Entry whose `text` field contains `*`, THE Validation Module SHALL replace `*` with `×`.
5. THE Validation Module SHALL apply artefact normalisation before invoking the Classifier on any OCR Entry.
6. THE Validation Module SHALL NOT modify any text that exactly matches or contains a token from PROTECTED_CODES during normalisation.

---

### Requirement 2: Text Classification

**User Story:** As a pipeline engineer, I want every OCR Entry assigned a precise engineering type, so that Stage 4 can associate annotations with geometry without ambiguity.

#### Acceptance Criteria

1. WHEN the Classifier receives a normalised text string, THE Classifier SHALL assign exactly one type from the set: `dimension_value`, `thread_spec`, `tolerance`, `diameter_callout`, `hole_callout`, `section_marker`, `spacing_annotation`, `material_code`, `part_name`, `bom_header`, `balloon_number`, `quantity`, `dimension_with_note`, `unknown`.
2. WHEN the normalised text matches the pattern `^\d+(\.\d+)?$` (a bare positive number with optional decimal), THE Classifier SHALL assign type `dimension_value`.
3. WHEN the normalised text matches the pattern `^M\d+(\s*[×x]\s*\d+(\.\d+)?)?$` (ISO metric thread, e.g., `M30`, `M30×2.5`, `M16×1.5`), THE Classifier SHALL assign type `thread_spec`.
4. WHEN the normalised text matches any of the patterns `^[±]\d`, `^\+\d.*\/\s*-\d`, or `^[A-Z]\d+\/[a-z]\d+` (e.g., `±0.5`, `+0.12/-0.00`, `H7/h6`, `H7`), THE Classifier SHALL assign type `tolerance`.
5. WHEN the normalised text matches the pattern `^Ø\d+` or `^DIA\s+\d+` (e.g., `Ø50`, `DIA 21`), THE Classifier SHALL assign type `diameter_callout`.
6. WHEN the normalised text contains the token `HOLE` (case-insensitive) and also contains a numeric value or `DIA` (e.g., `HOLE; DIA 21`, `2 HOLES M B`), THE Classifier SHALL assign type `hole_callout`.
7. WHEN the normalised text matches the pattern `^[A-Z]-[A-Z]$` or is a single uppercase letter `^[A-Z]$` that appears in a Category 1 or Category 3 drawing (e.g., `X-X`, `A-A`, `X`, `A`), THE Classifier SHALL assign type `section_marker`.
8. WHEN the normalised text matches `EQUI-SP` or `EQUI SP` (case-insensitive), THE Classifier SHALL assign type `spacing_annotation`.
9. WHEN the normalised text exactly matches one of the material abbreviations `MS`, `CI`, `FS`, `GM`, `CS`, `CR`, `AL`, `BR` (case-insensitive), THE Classifier SHALL assign type `material_code`.
10. WHEN the normalised text exactly matches or closely matches (edit distance ≤ 1) a known part name from the set `Valve`, `Spring`, `Pin`, `Body`, `Spindle`, `Handwheel`, `Gland`, `Bonnet`, `Sleeve`, `Collar`, `Cover`, `Plate`, `Seat`, `Nut`, `Bolt`, `Washer` (case-insensitive), THE Classifier SHALL assign type `part_name`.
11. WHEN the normalised text matches any BOM column header from the set `PARTS LIST`, `NAME`, `MATERIAL`, `QTY`, `NO`, `SL NO`, `PART NO` (case-insensitive), THE Classifier SHALL assign type `bom_header`.
12. WHEN the normalised text is a single digit `^[1-9]$` and the source image is Category 2 or Category 3, THE Classifier SHALL assign type `balloon_number`.
13. WHEN the normalised text is a small integer `^[1-9]\d?$` and the source image is Category 2 and the entry appears in a spatial region consistent with a BOM table, THE Classifier SHALL assign type `quantity`.
14. WHEN the normalised text contains a numeric value followed by `THICK`, `DEEP`, `LONG`, or `WIDE` (case-insensitive), or begins with `DIA` followed by a compound expression (e.g., `DIA 40×20 THICK`), THE Classifier SHALL assign type `dimension_with_note`.
15. WHEN the normalised text does not match any of the patterns in criteria 2–14, THE Classifier SHALL assign type `unknown`.
16. THE Classifier SHALL evaluate patterns in the priority order: `hole_callout` → `thread_spec` → `diameter_callout` → `dimension_with_note` → `tolerance` → `spacing_annotation` → `bom_header` → `material_code` → `part_name` → `section_marker` → `balloon_number` → `quantity` → `dimension_value` → `unknown`, so that more specific patterns take precedence over more general ones.

---

### Requirement 3: Parsed Field Extraction

**User Story:** As a pipeline engineer, I want structured sub-fields extracted from each classified entry, so that Stage 4 can consume typed values directly without re-parsing text.

#### Acceptance Criteria

1. WHEN the Classifier assigns type `dimension_value`, THE Validation Module SHALL populate `parsed` with `{"value": <float>}`.
2. WHEN the Classifier assigns type `thread_spec`, THE Validation Module SHALL populate `parsed` with `{"nominal": <string>, "pitch": <float or null>}`, where `nominal` is the thread designation (e.g., `"M30"`) and `pitch` is the thread pitch if present (e.g., `2.5`), or `null` if absent.
3. WHEN the Classifier assigns type `tolerance`, THE Validation Module SHALL populate `parsed` with `{"tolerance_string": <string>}` containing the normalised tolerance text.
4. WHEN the Classifier assigns type `diameter_callout`, THE Validation Module SHALL populate `parsed` with `{"diameter": <float>}`.
5. WHEN the Classifier assigns type `hole_callout`, THE Validation Module SHALL populate `parsed` with `{"raw": <string>}` containing the normalised text.
6. WHEN the Classifier assigns type `section_marker`, THE Validation Module SHALL populate `parsed` with `{"label": <string>}`.
7. WHEN the Classifier assigns type `spacing_annotation`, THE Validation Module SHALL populate `parsed` with `{"annotation": "EQUI-SP"}`.
8. WHEN the Classifier assigns type `material_code`, THE Validation Module SHALL populate `parsed` with `{"code": <string>}` in uppercase.
9. WHEN the Classifier assigns type `part_name`, THE Validation Module SHALL populate `parsed` with `{"name": <string>}` in title case.
10. WHEN the Classifier assigns type `bom_header`, THE Validation Module SHALL populate `parsed` with `{"header": <string>}` in uppercase.
11. WHEN the Classifier assigns type `balloon_number`, THE Validation Module SHALL populate `parsed` with `{"number": <int>}`.
12. WHEN the Classifier assigns type `quantity`, THE Validation Module SHALL populate `parsed` with `{"qty": <int>}`.
13. WHEN the Classifier assigns type `dimension_with_note`, THE Validation Module SHALL populate `parsed` with `{"raw": <string>}` containing the normalised text.
14. WHEN the Classifier assigns type `unknown`, THE Validation Module SHALL populate `parsed` with `{}` (an empty dictionary).

---

### Requirement 4: Structured Output Generation

**User Story:** As a pipeline engineer, I want each `_fullocr.json` converted to a `_structured.json` file with a defined schema, so that Stage 4 has a stable, predictable input format.

#### Acceptance Criteria

1. WHEN the Validation Module processes a `_fullocr.json` file, THE Validation Module SHALL write a corresponding `_structured.json` file to `results/batch/` with the filename pattern `{basename}_structured.json`.
2. THE Validation Module SHALL produce a `_structured.json` file whose top-level object contains exactly the fields: `source_file` (string), `image_category` (integer 1, 2, or 3), `total_detections` (integer), `classified` (array), and `summary` (object).
3. THE Validation Module SHALL set `total_detections` to the count of entries in the `classified` array.
4. WHEN writing the `classified` array, THE Validation Module SHALL include for each entry the fields: `id` (integer, preserved from input), `box` (array of 4 integers, preserved from input), `text` (string, normalised), `type` (string), `confidence` (float, preserved from input), and `parsed` (object).
5. THE Validation Module SHALL set `summary` to a dictionary mapping each type string that appears at least once in `classified` to its integer count.
6. THE Validation Module SHALL write `_structured.json` files as UTF-8 encoded JSON with 2-space indentation.
7. IF a `_structured.json` file already exists at the target path, THE Validation Module SHALL overwrite it without prompting.

---

### Requirement 5: Image Category Detection

**User Story:** As a pipeline engineer, I want the Validation Module to determine the drawing category from the filename, so that category-specific classification rules (e.g., balloon numbers in Category 2/3) are applied correctly.

#### Acceptance Criteria

1. WHEN the Validation Module receives a filename containing the substring `cad1_`, THE Validation Module SHALL set `image_category` to `1`.
2. WHEN the Validation Module receives a filename containing the substring `cad2_`, THE Validation Module SHALL set `image_category` to `2`.
3. WHEN the Validation Module receives a filename containing the substring `cad3_`, THE Validation Module SHALL set `image_category` to `3`.
4. IF the Validation Module cannot determine the category from the filename, THE Validation Module SHALL set `image_category` to `0` and log a warning to stdout.

---

### Requirement 6: Batch Processing Integration

**User Story:** As a pipeline engineer, I want Stage 3 to run automatically after Stage 2 in the existing batch pipeline, so that the full pipeline produces structured output without manual intervention.

#### Acceptance Criteria

1. THE Validation Module SHALL expose a function `validate_file(fullocr_path: str, output_dir: str) -> dict` that accepts the path to a `_fullocr.json` file and an output directory, and returns the structured output dictionary.
2. THE Validation Module SHALL expose a function `validate_batch(input_dir: str, output_dir: str) -> list` that processes all `_fullocr.json` files in `input_dir` and returns a list of result dictionaries.
3. WHEN `validate_batch` is called, THE Validation Module SHALL process each `_fullocr.json` file independently so that a failure on one file does not prevent processing of the remaining files.
4. IF an exception occurs while processing a single file, THE Validation Module SHALL log the filename and exception message to stdout and continue to the next file.
5. THE Validation Module SHALL be importable from `batch_process.py` using the import path `from validation import validate_file` or `from src.validation import validate_file` without raising an ImportError.
6. WHEN run as a standalone script (`python src/validation.py <input_dir> <output_dir>`), THE Validation Module SHALL invoke `validate_batch` and print a per-file summary to stdout.

---

### Requirement 7: Classification Accuracy

**User Story:** As a student researcher, I want the classifier to meet minimum accuracy thresholds per drawing category, so that the pipeline produces results suitable for a final-year project evaluation.

#### Acceptance Criteria

1. WHEN evaluated against a manually labelled ground-truth set of Category 1 drawings, THE Classifier SHALL achieve a correct-classification rate of at least 90% across all OCR entries.
2. WHEN evaluated against a manually labelled ground-truth set of Category 2 drawings, THE Classifier SHALL achieve a correct-classification rate of at least 70% across all OCR entries.
3. WHEN evaluated against a manually labelled ground-truth set of Category 3 drawings, THE Classifier SHALL achieve a correct-classification rate of at least 70% across all OCR entries.
4. THE Classifier SHALL assign type `unknown` to no more than 15% of entries in any single Category 1 image.

---

### Requirement 8: Performance and Resource Constraints

**User Story:** As a student researcher, I want Stage 3 to complete quickly on my laptop, so that iterating on the pipeline during development is practical.

#### Acceptance Criteria

1. WHEN processing all 36 images in batch mode on a machine with 8 GB RAM and a Windows 11 operating system, THE Validation Module SHALL complete within 60 seconds of wall-clock time.
2. THE Validation Module SHALL NOT import OpenCV, EasyOCR, PyTorch, TensorFlow, or any library not already listed in `requirements.txt`.
3. THE Validation Module SHALL use only Python standard library modules (`re`, `json`, `os`, `sys`, `pathlib`) plus libraries already present in the project virtual environment.

---

### Requirement 9: Robustness to Low-Confidence OCR Input

**User Story:** As a pipeline engineer, I want the Validation Module to handle low-confidence and malformed OCR entries gracefully, so that a single bad entry does not corrupt the output file.

#### Acceptance Criteria

1. WHEN an OCR Entry has a `confidence` value below 0.6, THE Validation Module SHALL still classify and include it in the output, and SHALL set the `type` field based on the normalised text alone without applying any confidence-based override.
2. IF an OCR Entry is missing the `text` field or the `text` field is an empty string, THE Validation Module SHALL assign type `unknown` and `parsed` `{}` without raising an exception.
3. IF the `_fullocr.json` file is empty (zero entries), THE Validation Module SHALL write a valid `_structured.json` with `total_detections` equal to `0`, an empty `classified` array, and an empty `summary` object.
4. IF the `_fullocr.json` file contains malformed JSON, THE Validation Module SHALL log an error to stdout and skip writing a `_structured.json` for that file.
