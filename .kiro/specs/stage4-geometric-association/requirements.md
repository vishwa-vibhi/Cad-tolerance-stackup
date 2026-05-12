# Requirements Document

## Introduction

Stage 4 (Geometric Association) is the fourth stage of the CAD Tolerance Stack-Up Analysis pipeline. It consumes the classified annotation data produced by Stage 3 (`_structured.json`) and the geometric element data produced by Stage 1.5 (`_elements.json`), then links each annotation to the geometric element it most plausibly describes using pure OpenCV geometry and Python standard library arithmetic. The output is a per-image `_associations.json` file containing each annotation's matched element and distance, plus a `_associations.png` visualization. Stage 4 must run on all 36 images in batch mode and integrate with `batch_process.py` without requiring any re-execution of earlier stages.

---

## Glossary

- **Associator**: The Stage 4 module (`src/association.py`) that performs geometric association.
- **Annotation**: A single classified OCR entry from a `_structured.json` file, identified by its `id`, `box` (x, y, w, h), `text`, and `type` fields.
- **Annotation_Center**: The pixel coordinate (cx, cy) computed as (box_x + box_w / 2, box_y + box_h / 2) for a given Annotation.
- **Geometric_Element**: A single detected geometric primitive from a `_elements.json` file. One of: `line_horizontal`, `line_vertical`, `line_diagonal`, `circle`, `contour`, or `hatching`.
- **Association**: A record pairing one Annotation to one Geometric_Element, including the element type, element data, and distance in pixels.
- **Distance_To_Line**: The perpendicular distance from the Annotation_Center to the infinite line through the segment, clamped to the segment endpoints (i.e., the minimum distance from the point to any point on the finite segment).
- **Distance_To_Circle**: The Euclidean distance from the Annotation_Center to the circle center, minus the circle radius. A value of 0 means the Annotation_Center lies exactly on the circle edge; negative values mean it is inside the circle.
- **Distance_To_Contour**: The minimum distance from the Annotation_Center to any of the four edges of the contour bounding box. Zero if the Annotation_Center is inside the bounding box.
- **MAX_DISTANCE_PX**: The maximum pixel distance within which a Geometric_Element is considered a candidate for association. Default value: 150 pixels.
- **Unassociated**: An Annotation for which no Geometric_Element was found within MAX_DISTANCE_PX, or whose type is `unknown`.
- **Category_1**: Single-part engineering drawings (filenames matching `cad1_NNN`). Contain dimension lines, extension lines, circles for holes/threads, and section cut lines.
- **Category_2**: Assembly drawings (filenames matching `cad2_NNN`). Contain balloon circles, BOM tables, and part outlines.
- **Category_3**: Assembly-view drawings (filenames matching `cad3_NNN`). Contain balloon circles but no BOM tables.
- **Balloon_Circle**: A circle detected in a Category 2 or Category 3 image with radius in the range 15–30 pixels, used to enclose a balloon number annotation.
- **BOM_Region**: The lower portion of a Category 2 image, defined as the area where y > (image_height × 0.6), typically containing the Bill of Materials table.
- **Structured_JSON**: The `_structured.json` file produced by Stage 3 for a given image.
- **Elements_JSON**: The `_elements.json` file produced by Stage 1.5 for a given image.

---

## Requirements

### Requirement 1: Input Loading

**User Story:** As a pipeline engineer, I want the Associator to load both the Structured_JSON and Elements_JSON for each image, so that it has all the data needed to perform geometric association.

#### Acceptance Criteria

1. WHEN the Associator is invoked with a `structured_path` and an `elements_path`, THE Associator SHALL load and parse both JSON files before performing any association.
2. IF the Structured_JSON file does not exist or cannot be parsed, THEN THE Associator SHALL log an error message to stdout and return `None` without raising an exception.
3. IF the Elements_JSON file does not exist or cannot be parsed, THEN THE Associator SHALL log an error message to stdout and return `None` without raising an exception.
4. THE Associator SHALL read the `image_category` field from the Structured_JSON to determine which association rules to apply.
5. THE Associator SHALL read the `classified` list from the Structured_JSON, where each entry contains `id`, `box`, `text`, `type`, and `parsed` fields.
6. THE Associator SHALL read the following keys from the Elements_JSON: `lines_horizontal`, `lines_vertical`, `lines_diagonal`, `circles`, `contours`, and `hatching`, treating any missing key as an empty list.

---

### Requirement 2: Annotation Center Computation

**User Story:** As a pipeline engineer, I want the Associator to compute a consistent reference point for each annotation, so that distance calculations are deterministic and reproducible.

#### Acceptance Criteria

1. THE Associator SHALL compute the Annotation_Center for each Annotation as `(box_x + box_w / 2.0, box_y + box_h / 2.0)`, where `box_x`, `box_y`, `box_w`, `box_h` are the four elements of the `box` field.
2. IF an Annotation's `box` field is missing, has fewer than 4 elements, or contains non-numeric values, THEN THE Associator SHALL mark that Annotation as `unassociated` and continue processing the remaining annotations.
3. THE Associator SHALL use floating-point arithmetic for all center and distance computations.

---

### Requirement 3: Distance Metric — Line Segments

**User Story:** As a pipeline engineer, I want the Associator to compute geometrically correct distances from annotation centers to line segments, so that annotations are matched to the lines they actually annotate rather than distant lines that happen to be nearby.

#### Acceptance Criteria

1. THE Associator SHALL implement a `distance_point_to_segment(px, py, x1, y1, x2, y2)` function that returns the minimum Euclidean distance from point (px, py) to the finite line segment from (x1, y1) to (x2, y2).
2. WHEN the perpendicular foot of the point onto the infinite line falls within the segment, THE `distance_point_to_segment` function SHALL return the perpendicular distance.
3. WHEN the perpendicular foot falls outside the segment endpoints, THE `distance_point_to_segment` function SHALL return the distance to the nearest endpoint.
4. WHEN the segment has zero length (x1 == x2 and y1 == y2), THE `distance_point_to_segment` function SHALL return the Euclidean distance from the point to (x1, y1).
5. FOR ALL valid point and segment inputs, THE `distance_point_to_segment` function SHALL return a non-negative float.

---

### Requirement 4: Distance Metric — Circles

**User Story:** As a pipeline engineer, I want the Associator to compute distances from annotation centers to circle edges, so that balloon numbers and hole callouts are matched to the circle they sit inside or adjacent to.

#### Acceptance Criteria

1. THE Associator SHALL implement a `distance_point_to_circle(px, py, cx, cy, r)` function that returns `sqrt((px - cx)² + (py - cy)²) - r`.
2. WHEN the Annotation_Center is inside the circle (Euclidean distance to center < r), THE `distance_point_to_circle` function SHALL return a negative value.
3. WHEN the Annotation_Center lies exactly on the circle edge, THE `distance_point_to_circle` function SHALL return 0.0.
4. FOR ALL valid inputs, THE `distance_point_to_circle` function SHALL return a float (positive, negative, or zero).

---

### Requirement 5: Distance Metric — Contour Bounding Boxes

**User Story:** As a pipeline engineer, I want the Associator to compute distances from annotation centers to contour bounding boxes, so that BOM labels and unmatched annotations can be associated with the nearest enclosing shape.

#### Acceptance Criteria

1. THE Associator SHALL implement a `distance_point_to_contour(px, py, bx, by, bw, bh)` function that returns the minimum distance from point (px, py) to the nearest edge of the axis-aligned bounding box defined by (bx, by, bw, bh).
2. WHEN the point is inside the bounding box, THE `distance_point_to_contour` function SHALL return 0.0.
3. WHEN the point is outside the bounding box, THE `distance_point_to_contour` function SHALL return the Euclidean distance to the nearest point on the bounding box perimeter.
4. FOR ALL valid inputs, THE `distance_point_to_contour` function SHALL return a non-negative float.

---

### Requirement 6: Association Rules by Annotation Type

**User Story:** As a pipeline engineer, I want the Associator to apply type-specific association rules, so that each annotation is matched to the geometric element that is semantically correct for its annotation type in engineering drawing conventions.

#### Acceptance Criteria

1. WHEN the Annotation type is one of `dimension_value`, `diameter_callout`, `radius_callout`, `thread_spec`, `tolerance`, or `dimension_with_note`, THE Associator SHALL first search for the nearest line (horizontal or vertical combined) within MAX_DISTANCE_PX using Distance_To_Line; if no line is found within MAX_DISTANCE_PX, THE Associator SHALL fall back to the nearest contour within MAX_DISTANCE_PX using Distance_To_Contour; if still no match, THE Associator SHALL mark the Annotation as `unassociated`.

2. WHEN the Annotation type is `hole_callout`, THE Associator SHALL first search for the nearest circle within MAX_DISTANCE_PX using Distance_To_Circle; if no circle is found within MAX_DISTANCE_PX, THE Associator SHALL fall back to the nearest contour within MAX_DISTANCE_PX using Distance_To_Contour; if still no match, THE Associator SHALL mark the Annotation as `unassociated`.

3. WHEN the Annotation type is `balloon_number` and the image category is 2 or 3, THE Associator SHALL search for the nearest circle within MAX_DISTANCE_PX using Distance_To_Circle; if no circle is found within MAX_DISTANCE_PX, THE Associator SHALL mark the Annotation as `unassociated`.

4. WHEN the Annotation type is `section_marker`, THE Associator SHALL first search for the nearest diagonal line within MAX_DISTANCE_PX using Distance_To_Line; if no diagonal line is found within MAX_DISTANCE_PX, THE Associator SHALL fall back to the nearest vertical line within MAX_DISTANCE_PX; if still no match, THE Associator SHALL mark the Annotation as `unassociated`.

5. WHEN the Annotation type is `spacing_annotation`, THE Associator SHALL search for the nearest horizontal line within MAX_DISTANCE_PX using Distance_To_Line; if no horizontal line is found within MAX_DISTANCE_PX, THE Associator SHALL mark the Annotation as `unassociated`.

6. WHEN the Annotation type is one of `material_code`, `material_name`, `part_name`, `bom_header`, or `quantity`, THE Associator SHALL search for the nearest contour within MAX_DISTANCE_PX using Distance_To_Contour; if no contour is found within MAX_DISTANCE_PX, THE Associator SHALL mark the Annotation as `unassociated`.

7. WHEN the Annotation type is `unknown`, THE Associator SHALL mark the Annotation as `unassociated` without performing any distance search.

8. WHEN the Annotation type is `balloon_number` and the image category is 1, THE Associator SHALL apply the same rules as `dimension_value` (nearest line, then contour fallback), since single-digit values in Category 1 are dimension values, not balloons.

---

### Requirement 7: Nearest-Element Selection

**User Story:** As a pipeline engineer, I want the Associator to select the single closest element from the candidate set, so that each annotation receives exactly one association.

#### Acceptance Criteria

1. WHEN multiple Geometric_Elements of the target type are within MAX_DISTANCE_PX, THE Associator SHALL select the one with the minimum computed distance.
2. WHEN two Geometric_Elements have equal minimum distance, THE Associator SHALL select the one that appears first in the Elements_JSON list (stable tie-breaking).
3. THE Associator SHALL produce exactly one Association record per Annotation (either a matched element or `unassociated`).
4. THE Associator SHALL NOT modify the Elements_JSON data or consume elements (multiple annotations may associate to the same element).

---

### Requirement 8: Output Schema — `_associations.json`

**User Story:** As a pipeline engineer, I want the Associator to write a well-structured JSON file, so that Stage 5 (Web Visualization) can consume association data without additional parsing.

#### Acceptance Criteria

1. THE Associator SHALL write a `_associations.json` file for each processed image with the following top-level fields: `source_structured`, `source_elements`, `image_category`, and `associations`.
2. THE `source_structured` field SHALL contain the basename of the Structured_JSON file (e.g., `"cad1_001_structured.json"`).
3. THE `source_elements` field SHALL contain the basename of the Elements_JSON file (e.g., `"cad1_001_elements.json"`).
4. THE `image_category` field SHALL contain the integer category (1, 2, or 3) read from the Structured_JSON.
5. THE `associations` field SHALL contain a list of Association records, one per Annotation in the `classified` list, in the same order.
6. WHEN an Annotation is successfully matched, THE Association record SHALL contain: `annotation_id` (int), `annotation_text` (str), `annotation_type` (str), `annotation_box` (list of 4 ints), and `associated_element` (object with `element_type` (str), `element_data` (list of ints), and `distance_px` (float rounded to 1 decimal place)).
7. WHEN an Annotation is `unassociated`, THE Association record SHALL contain: `annotation_id` (int), `annotation_text` (str), `annotation_type` (str), `annotation_box` (list of 4 ints), and `associated_element` set to `null`.
8. THE Associator SHALL write the JSON file with 2-space indentation and UTF-8 encoding.
9. FOR ALL valid inputs, parsing the written `_associations.json` and re-serializing it SHALL produce an identical file (round-trip property).

---

### Requirement 9: Output — `_associations.png` Visualization

**User Story:** As a student researcher, I want a visual overlay showing which annotation connects to which geometric element, so that I can inspect association quality and debug errors without reading raw JSON.

#### Acceptance Criteria

1. THE Associator SHALL produce a `_associations.png` file for each processed image by drawing on a color copy of the original image.
2. THE Associator SHALL draw a line from each Annotation_Center to the nearest point on its associated Geometric_Element for every successfully matched Annotation.
3. THE Associator SHALL use distinct colors for each element type in the visualization: blue for `line_horizontal`, green for `line_vertical`, cyan for `line_diagonal`, red for `circle`, magenta for `contour`.
4. THE Associator SHALL draw a small filled circle (radius 3 pixels) at each Annotation_Center in yellow.
5. THE Associator SHALL NOT draw any connecting line for `unassociated` Annotations.
6. THE Associator SHALL save the visualization using `cv2.imwrite` to the specified output directory.
7. WHERE the original image cannot be loaded (file not found or corrupt), THE Associator SHALL skip visualization output and log a warning, but SHALL still write the `_associations.json` file.

---

### Requirement 10: Batch Processing Integration

**User Story:** As a pipeline engineer, I want Stage 4 to integrate with `batch_process.py`, so that the full pipeline can be run end-to-end with a single command.

#### Acceptance Criteria

1. THE Associator SHALL expose an `associate_file(structured_path, elements_path, image_path, output_dir)` function that processes a single image and returns the association result dict.
2. THE Associator SHALL expose an `associate_batch(structured_dir, elements_dir, image_dir, output_dir)` function that processes all `_structured.json` files in `structured_dir`, locating the corresponding `_elements.json` by replacing `_structured.json` with `_elements.json` in the same `elements_dir`.
3. WHEN `associate_batch` is called, THE Associator SHALL process files in sorted filename order.
4. IF a corresponding Elements_JSON file does not exist for a given Structured_JSON, THEN THE Associator SHALL log a warning and skip that image without aborting the batch.
5. THE Associator SHALL print a per-image summary line to stdout in the format: `[N/total] <filename> -> <total_annotations> annotations | matched: <matched_count> | unassociated: <unassociated_count>`.
6. WHEN `batch_process.py` calls `associate_file` after `validate_file` for each image, THE Associator SHALL write its outputs to the same `output_dir` used by the other stages.
7. THE Associator SHALL NOT require re-running Stage 1.5; it SHALL read existing `_elements.json` files from disk.

---

### Requirement 11: Performance and Resource Constraints

**User Story:** As a student researcher, I want Stage 4 to complete within the existing batch time budget, so that the full pipeline remains practical on my laptop.

#### Acceptance Criteria

1. WHEN processing all 36 images in batch mode on a machine with 8 GB RAM and a Windows 11 operating system, THE Associator SHALL complete within 30 seconds of wall-clock time.
2. THE Associator SHALL NOT import any third-party libraries beyond `opencv-python` (`cv2`), `numpy`, and the Python standard library modules already present in the project.
3. THE Associator SHALL NOT load or invoke EasyOCR, any ML model, or any GPU-accelerated inference library.
4. THE `associate_file` function SHALL have time complexity O(A × E) or better per image, where A is the number of annotations and E is the total number of geometric elements.

---

### Requirement 12: Robustness to Missing or Malformed Data

**User Story:** As a pipeline engineer, I want Stage 4 to handle incomplete or malformed inputs gracefully, so that a single bad image does not abort the batch.

#### Acceptance Criteria

1. IF an Annotation's `box` field is missing or malformed, THEN THE Associator SHALL mark that Annotation as `unassociated` and continue processing.
2. IF the `classified` list in the Structured_JSON is empty, THEN THE Associator SHALL write a valid `_associations.json` with an empty `associations` list.
3. IF all element lists in the Elements_JSON are empty, THEN THE Associator SHALL mark all Annotations as `unassociated` and write a valid `_associations.json`.
4. IF an element entry in the Elements_JSON contains non-numeric values or has fewer elements than expected (e.g., a line with fewer than 4 values, a circle with fewer than 3), THEN THE Associator SHALL skip that element and continue processing the remaining elements.
5. WHEN any unhandled exception occurs during processing of a single image, THE Associator SHALL catch the exception, log it to stdout, and continue with the next image in the batch.

---

### Requirement 13: Accuracy Targets

**User Story:** As a student researcher, I want Stage 4 to meet minimum accuracy thresholds, so that the association data is useful for tolerance stack-up analysis in the final project deliverable.

#### Acceptance Criteria

1. WHEN evaluated against Category 1 images, THE Associator SHALL correctly associate at least 70% of `dimension_value` annotations to a line or contour (where "correct" means the associated element is within 50 pixels of the annotation center and is a line or contour, not a circle).
2. WHEN evaluated against Category 2 images, THE Associator SHALL correctly associate at least 80% of `balloon_number` annotations to a circle.
3. WHEN evaluated against Category 3 images, THE Associator SHALL correctly associate at least 80% of `balloon_number` annotations to a circle.
4. WHEN evaluated across all 36 images, THE Associator SHALL mark no more than 20% of non-`unknown` annotations as `unassociated`.

---

### Requirement 14: Configuration

**User Story:** As a pipeline engineer, I want the association distance threshold to be configurable, so that I can tune it for different drawing scales without modifying source code.

#### Acceptance Criteria

1. THE Associator SHALL define `MAX_DISTANCE_PX` as a module-level constant with a default value of 150.
2. THE `associate_file` function SHALL accept an optional `max_distance` parameter that overrides `MAX_DISTANCE_PX` for that call.
3. WHEN `max_distance` is not provided, THE Associator SHALL use the module-level `MAX_DISTANCE_PX` value.

---

## Special Requirements Guidance

### Parser and Serializer Requirements

Stage 4 reads JSON produced by Stages 1.5 and 3, and writes JSON consumed by Stage 5. The JSON reading and writing logic must be correct. Requirement 8, Criterion 9 mandates a round-trip property: for all valid inputs, `json.loads(json.dumps(result)) == result`. This should be verified with a property-based test using generated association dicts.

### Property-Based Testing Guidance

The following acceptance criteria are suitable for property-based testing:

- **Requirement 3, Criterion 5**: Property: For all finite segments and points, `distance_point_to_segment` returns a non-negative float.
- **Requirement 4, Criterion 4**: Property: For all circles and points, `distance_point_to_circle` returns a float; for points outside the circle, the value is positive.
- **Requirement 5, Criterion 4**: Property: For all bounding boxes and points, `distance_point_to_contour` returns a non-negative float; for points inside the box, the value is 0.0.
- **Requirement 8, Criterion 9**: Round-trip property: `json.loads(json.dumps(association_record)) == association_record` for all generated association records.

The following acceptance criteria are NOT suitable for property-based testing and should use integration tests with representative examples:

- **Requirement 13, Criteria 1–4**: Accuracy targets require ground-truth labels from real images, not synthetic inputs.
- **Requirement 9**: Visualization correctness requires visual inspection, not automated property checking.
