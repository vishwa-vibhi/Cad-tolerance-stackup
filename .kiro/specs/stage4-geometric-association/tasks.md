# Implementation Plan: Stage 4 — Geometric Association

## Overview

Build `src/association.py` — a pure OpenCV + stdlib module that links each classified annotation from Stage 3 to the nearest geometric element detected by Stage 1.5. Elements are computed on-the-fly (no pre-existing `_elements.json` required). Outputs `_associations.json` and `_associations.png` per image. Integrates with `batch_process.py`.

## Tasks

- [x] 1. Scaffold `src/association.py` — module constants, imports, and function stubs
  - Create `src/association.py` with module docstring, all imports (`cv2`, `os`, `sys`, `json`, `math`, `pathlib`), and the `MAX_DISTANCE_PX = 150` module-level constant
  - Add try/except import block for `preprocessing.preprocess` and `element_detection.detect_all_elements` (same pattern as existing modules)
  - Add empty stub definitions with docstrings for all public and internal functions: `associate_file`, `associate_batch`, `_get_elements`, `distance_point_to_segment`, `distance_point_to_circle`, `distance_point_to_contour`, `_annotation_center`, `_safe_box`, `_find_nearest_line`, `_find_nearest_circle`, `_find_nearest_contour`, `_find_nearest_diagonal`, `_find_nearest_horizontal`, `_associate_annotation`, `_build_output`, `_visualize`
  - Add `if __name__ == "__main__":` block that accepts `image_path` and `structured_path` as CLI args and calls `associate_file`
  - Verify syntax with `python -m py_compile src/association.py`
  - _Requirements: 10.1, 10.2, 14.1_

- [x] 2. Implement distance metric functions
  - [x] 2.1 Implement `distance_point_to_segment(px, py, x1, y1, x2, y2) -> float`
    - Compute `dx = x2-x1`, `dy = y2-y1`, `seg_len_sq = dx*dx + dy*dy`
    - If `seg_len_sq == 0`, return `math.sqrt((px-x1)**2 + (py-y1)**2)` (degenerate segment)
    - Compute `t = ((px-x1)*dx + (py-y1)*dy) / seg_len_sq`, clamp to `[0.0, 1.0]`
    - Return distance from `(px,py)` to `(x1 + t*dx, y1 + t*dy)`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 2.2 Implement `distance_point_to_circle(px, py, cx, cy, r) -> float`
    - Return `math.sqrt((px-cx)**2 + (py-cy)**2) - r`
    - Negative when point is inside circle, zero on edge, positive outside
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 2.3 Implement `distance_point_to_contour(px, py, bx, by, bw, bh) -> float`
    - Clamp `px` to `[bx, bx+bw]` and `py` to `[by, by+bh]`
    - Return `math.sqrt((px - clamped_x)**2 + (py - clamped_y)**2)`
    - Returns 0.0 when point is inside or on the box
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 3. Implement helper utilities
  - [x] 3.1 Implement `_safe_box(box) -> tuple or None`
    - Return `(x, y, w, h)` as ints if `box` is a list/tuple with ≥ 4 numeric elements; return `None` otherwise
    - Wrap conversions in `try/except (TypeError, ValueError)`
    - _Requirements: 2.2, 12.1_

  - [x] 3.2 Implement `_annotation_center(box) -> tuple`
    - Accept a validated 4-element box `(x, y, w, h)`
    - Return `(x + w / 2.0, y + h / 2.0)` as floats
    - _Requirements: 2.1, 2.3_

  - [x] 3.3 Implement `_get_elements(image_path: str) -> dict`
    - Call `preprocess(image_path, save_result=False)` to get binary image
    - Call `detect_all_elements(binary)` to get elements dict
    - Return the elements dict; let exceptions propagate to caller
    - _Requirements: 10.7_

- [x] 4. Implement nearest-element finder functions
  - [x] 4.1 Implement `_find_nearest_line(cx, cy, h_lines, v_lines, max_distance) -> tuple or None`
    - Iterate over all horizontal lines, compute `distance_point_to_segment`, track minimum
    - Iterate over all vertical lines, compute `distance_point_to_segment`, track minimum
    - Return `(element_data, distance, "line_horizontal" or "line_vertical")` for the global minimum if `distance <= max_distance`; else return `None`
    - Tie-break: first element in list order (stable)
    - _Requirements: 6.1, 7.1, 7.2_

  - [x] 4.2 Implement `_find_nearest_circle(cx, cy, circles, max_distance) -> tuple or None`
    - Iterate over all circles `(ecx, ecy, r)`, compute `distance_point_to_circle`
    - Rank by `abs(distance)` so annotations inside a circle rank closest
    - Return `((ecx, ecy, r), distance, "circle")` if `abs(distance) <= max_distance`; else `None`
    - _Requirements: 6.2, 6.3_

  - [x] 4.3 Implement `_find_nearest_contour(cx, cy, contours, max_distance) -> tuple or None`
    - Iterate over all contours `(bx, by, bw, bh)`, compute `distance_point_to_contour`
    - Return `((bx, by, bw, bh), distance, "contour")` for minimum if `distance <= max_distance`; else `None`
    - _Requirements: 6.1, 6.2, 6.6_

  - [x] 4.4 Implement `_find_nearest_diagonal(cx, cy, diagonals, max_distance) -> tuple or None`
    - Same pattern as `_find_nearest_line` but over diagonal lines only
    - Return element type `"line_diagonal"`
    - _Requirements: 6.4_

  - [x] 4.5 Implement `_find_nearest_horizontal(cx, cy, h_lines, max_distance) -> tuple or None`
    - Same pattern but over horizontal lines only
    - Return element type `"line_horizontal"`
    - _Requirements: 6.5_

- [x] 5. Implement core association dispatcher
  - Implement `_associate_annotation(annotation, elements, category, max_distance) -> dict`
  - Extract `ann_type`, validate box with `_safe_box`, compute center with `_annotation_center`
  - Implement the full type-dispatch if/elif chain:
    - `unknown` → unassociated immediately
    - `dimension_value`, `diameter_callout`, `radius_callout`, `thread_spec`, `tolerance`, `dimension_with_note` → `_find_nearest_line` → fallback `_find_nearest_contour`
    - `hole_callout` → `_find_nearest_circle` → fallback `_find_nearest_contour`
    - `balloon_number` with category 2 or 3 → `_find_nearest_circle` (no fallback)
    - `balloon_number` with category 1 → `_find_nearest_line` → fallback `_find_nearest_contour`
    - `section_marker` → `_find_nearest_diagonal` → fallback `_find_nearest_line` (v_lines only)
    - `spacing_annotation` → `_find_nearest_horizontal` (no fallback)
    - `material_code`, `material_name`, `part_name`, `bom_header`, `quantity` → `_find_nearest_contour`
  - Return matched record with `element_type`, `element_data` (list of ints), `distance_px` (rounded to 1 decimal) or unassociated record with `associated_element: null`
  - _Requirements: 6.1–6.8, 7.1–7.4_

- [x] 6. Implement output assembly and file I/O
  - [x] 6.1 Implement `_build_output(structured_data, associations, source_structured) -> dict`
    - Compute `matched = sum(1 for a in associations if a["associated_element"] is not None)`
    - Return dict with keys: `source_structured`, `image_category`, `total_annotations`, `matched`, `unassociated`, `associations`
    - _Requirements: 8.1–8.8_

  - [x] 6.2 Implement `associate_file(image_path, structured_path, output_dir, max_distance) -> dict`
    - Load and parse `structured_path`; on `json.JSONDecodeError` or `OSError`, log error and return `None`
    - Call `_get_elements(image_path)`; on exception, log error and return `None`
    - Extract `category` and `classified` list from structured data
    - For each annotation in `classified`, call `_associate_annotation`
    - Call `_build_output` to assemble result
    - Write `_associations.json` with `json.dump(..., indent=2, ensure_ascii=False)`
    - Call `_visualize(image_path, result["associations"], vis_path)`
    - Print per-image summary: `[filename] -> N annotations | matched: M | unassociated: U`
    - Return result dict
    - _Requirements: 10.1, 8.8, 9.6, 9.7, 12.2–12.5_

- [x] 7. Implement visualization
  - Implement `_visualize(image_path, associations, output_path)`
  - Load image with `cv2.imread`; if `None`, print warning and return (do not crash)
  - Make a color copy with `img.copy()`
  - Define color map: `line_horizontal`=blue `(255,0,0)`, `line_vertical`=green `(0,180,0)`, `line_diagonal`=cyan `(200,200,0)`, `circle`=red `(0,0,255)`, `contour`=magenta `(200,0,200)`
  - For each association: draw yellow filled circle (r=3) at annotation center
  - For matched associations: draw colored line from annotation center to element midpoint/center; draw the element itself in its color (line segment, circle outline, or rectangle)
  - Save with `cv2.imwrite(output_path, vis)`
  - _Requirements: 9.1–9.6_

- [x] 8. Implement `associate_batch`
  - Implement `associate_batch(image_dir, structured_dir, output_dir, max_distance) -> list`
  - Glob all `*_structured.json` files in `structured_dir`, sort by name
  - For each file, locate the corresponding image in `image_dir` by replacing `_structured.json` with `.png` in the basename
  - If image not found, log warning and skip
  - Call `associate_file` inside `try/except Exception`; on failure log error and continue
  - Return list of non-None results
  - _Requirements: 10.2, 10.3, 10.4, 10.5, 12.5_

- [x] 9. Integrate Stage 4 into `batch_process.py`
  - Add import at top of `batch_process.py`:
    ```python
    try:
        from association import associate_file
    except ImportError:
        from src.association import associate_file
    ```
  - In `process_category()`, after `validate_file()` succeeds, derive `structured_path` from `basename` and call `associate_file(img_path, structured_path, results_dir)`
  - Add `association_matched` and `association_unassociated` fields to the per-image `stat` dict
  - _Requirements: 10.6_

- [x] 10. Write unit tests in `tests/test_association.py`
  - [x] 10.1 Tests for `distance_point_to_segment`
    - Point on segment midpoint → distance 0
    - Point perpendicular to segment interior → correct perpendicular distance
    - Point beyond endpoint → distance to endpoint
    - Degenerate segment (zero length) → distance to point
    - _Requirements: 3.1–3.5_

  - [x] 10.2 Tests for `distance_point_to_circle`
    - Point at center → returns `-r` (negative)
    - Point on edge → returns 0.0
    - Point outside → returns positive value
    - _Requirements: 4.1–4.4_

  - [x] 10.3 Tests for `distance_point_to_contour`
    - Point inside box → returns 0.0
    - Point on box edge → returns 0.0
    - Point outside box → returns correct distance
    - _Requirements: 5.1–5.4_

  - [x] 10.4 Tests for `_safe_box`
    - Valid `[10, 20, 30, 40]` → `(10, 20, 30, 40)`
    - `None` → `None`
    - Too short → `None`
    - Non-numeric → `None`

  - [x] 10.5 Tests for `_associate_annotation` — one test per annotation type
    - `dimension_value` with nearby h-line → `line_horizontal`
    - `hole_callout` with nearby circle → `circle`
    - `balloon_number` Cat 2 with nearby circle → `circle`
    - `balloon_number` Cat 1 with nearby line → `line_horizontal`
    - `section_marker` with nearby diagonal → `line_diagonal`
    - `unknown` → `associated_element` is `null`
    - No elements within max_distance → `associated_element` is `null`
    - _Requirements: 6.1–6.8_

  - [x] 10.6 Tests for `_build_output`
    - `total_annotations == matched + unassociated` invariant
    - All required keys present
    - _Requirements: 8.1–8.5_

  - [x] 10.7 End-to-end test for `associate_file` using `data/category_1/cad1_001.png`
    - Result is not `None`
    - `total_annotations > 0`
    - `_associations.json` written to tmp dir
    - `_associations.png` written to tmp dir
    - `total_annotations == matched + unassociated`
    - _Requirements: 8.1–8.9, 9.1, 10.1_

- [x] 11. Write property-based tests in `tests/test_association_properties.py`
  - [ ]* 11.1 Property 1: `distance_point_to_segment` is always non-negative
    - `@given(floats, floats, floats, floats, floats, floats)` → result >= 0
    - **Property 1: distance_point_to_segment returns non-negative float for all inputs**
    - **Validates: Requirement 3.5**

  - [ ]* 11.2 Property 2: `distance_point_to_circle` is negative for interior points
    - Generate circle (cx,cy,r>0) and point strictly inside → result < 0
    - **Property 2: distance_point_to_circle is negative when point is inside circle**
    - **Validates: Requirement 4.2**

  - [ ]* 11.3 Property 3: `distance_point_to_contour` is non-negative
    - `@given(floats, floats, ints, ints, positive_ints, positive_ints)` → result >= 0
    - **Property 3: distance_point_to_contour returns non-negative float for all inputs**
    - **Validates: Requirement 5.4**

  - [ ]* 11.4 Property 4: `distance_point_to_contour` is zero for interior points
    - Generate box and point strictly inside → result == 0.0
    - **Property 4: distance_point_to_contour returns 0.0 for points inside bounding box**
    - **Validates: Requirement 5.2**

  - [ ]* 11.5 Property 5: Output JSON round-trip
    - Generate synthetic association result dict → `json.loads(json.dumps(result)) == result`
    - **Property 5: association output dict survives JSON round-trip unchanged**
    - **Validates: Requirement 8.9**

  - [ ]* 11.6 Property 6: Count invariant
    - Generate list of association records → `total == matched + unassociated`
    - **Property 6: total_annotations == matched + unassociated in all outputs**
    - **Validates: Requirement 8 (count consistency)**

  - [ ]* 11.7 Property 7: One record per annotation
    - Generate N annotations → `len(associations) == N`
    - **Property 7: associations list has exactly one record per input annotation**
    - **Validates: Requirement 7.3**

- [x] 12. Checkpoint — run full test suite
  - Run `pytest tests/test_association.py tests/test_association_properties.py -v`
  - Run `pytest tests/test_validation.py tests/test_ocr_accuracy.py -q` to confirm no regressions
  - All tests must pass before proceeding

- [x] 13. End-to-end batch run on all 36 images
  - Run `python src/association.py` (or via batch_process.py) on all 36 images
  - Verify 36 `_associations.json` files written to `results/batch/`
  - Verify 36 `_associations.png` files written to `results/batch/`
  - Check `matched / total_annotations` ratio across all images — target ≥ 80% overall
  - Inspect `_associations.png` for at least 3 Category 1 images and 2 Category 2 images visually
  - _Requirements: 13.1–13.4_

- [x] 14. Final checkpoint — full pipeline smoke test
  - Run `python batch_process.py` end-to-end (Stages 2 → 3 → 4 in sequence)
  - Confirm all stages complete without errors
  - Confirm `_structured.json` and `_associations.json` are both present for all processed images

## Notes

- Tasks marked `*` are optional property-based tests — implement for final project quality
- `_get_elements` runs Stage 1 + 1.5 on-the-fly; this adds ~0.5–1s per image but avoids a pre-step dependency
- The `distance_point_to_circle` function returns a **signed** value — negative means the annotation is inside the circle (common for balloon numbers). The `distance_px` field in the output stores this signed value; Stage 5 can use the sign to distinguish "inside" from "outside"
- For the final project submission, the `_associations.png` visualizations are the primary evidence of Stage 4 working correctly — make sure they look clean
- All 220 existing tests must continue to pass after Stage 4 is added
