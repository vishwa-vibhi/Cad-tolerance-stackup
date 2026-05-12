# Design Document: Stage 4 — Geometric Association

## Overview

Stage 4 (`src/association.py`) links each classified text annotation from Stage 3 to the geometric element it describes, using pure OpenCV geometry. It sits between Stage 3 (Validation & Structuring) and Stage 5 (Web Visualization).

Key design decision: **elements are computed on-the-fly** by calling `preprocess()` + `detect_all_elements()` from the existing Stage 1.5 code. No `_elements.json` pre-step is required — this simplifies the pipeline and avoids a dependency on files that don't yet exist on disk.

### Pipeline Position

```
[Image] ──► Stage 1  preprocessing.py  ──► binary image
                                              │
                                              ▼
         Stage 1.5  element_detection.py ──► elements dict (computed on-the-fly)
                                              │
[Image] ──► Stage 2  vlm_reader.py      ──► _fullocr.json
                                              │
                                              ▼
         Stage 3  validation.py         ──► _structured.json
                                              │
                                              ▼
         Stage 4  association.py  ◄── DESIGNING NOW
                                              │
                                              ▼
         _associations.json + _associations.png
                                              │
                                              ▼
         Stage 5  Flask visualization   (NOT BUILT YET)
```

---

## Module Structure

### Public API

```python
def associate_file(
    image_path: str,
    structured_path: str,
    output_dir: str,
    max_distance: int = MAX_DISTANCE_PX
) -> dict:
    """
    Process a single image: run element detection, associate annotations,
    write _associations.json and _associations.png.

    Args:
        image_path:      Path to the original PNG image.
        structured_path: Path to the _structured.json from Stage 3.
        output_dir:      Directory to write outputs.
        max_distance:    Max pixel distance for association (default 150).

    Returns:
        The association result dict, or None on unrecoverable error.
    """

def associate_batch(
    image_dir: str,
    structured_dir: str,
    output_dir: str,
    max_distance: int = MAX_DISTANCE_PX
) -> list:
    """
    Process all _structured.json files in structured_dir.
    Locates images by matching basename in image_dir.

    Returns:
        List of result dicts (one per successfully processed image).
    """
```

### Internal Functions

```python
# Module-level constant
MAX_DISTANCE_PX = 150

# Distance metrics
def distance_point_to_segment(px, py, x1, y1, x2, y2) -> float
def distance_point_to_circle(px, py, cx, cy, r) -> float
def distance_point_to_contour(px, py, bx, by, bw, bh) -> float

# Element detection wrapper
def _get_elements(image_path: str) -> dict

# Annotation helpers
def _annotation_center(box: list) -> tuple  # (cx, cy)
def _safe_box(box) -> tuple or None         # validates box, returns (x,y,w,h) or None

# Nearest-element finders (return (element_data, distance) or None)
def _find_nearest_line(cx, cy, h_lines, v_lines, max_distance) -> tuple or None
def _find_nearest_circle(cx, cy, circles, max_distance) -> tuple or None
def _find_nearest_contour(cx, cy, contours, max_distance) -> tuple or None
def _find_nearest_diagonal(cx, cy, diagonals, max_distance) -> tuple or None
def _find_nearest_horizontal(cx, cy, h_lines, max_distance) -> tuple or None

# Core association
def _associate_annotation(annotation: dict, elements: dict, category: int, max_distance: int) -> dict

# Output assembly
def _build_output(structured_data: dict, associations: list, source_structured: str) -> dict

# Visualization
def _visualize(image_path: str, associations: list, output_path: str)
```

---

## Data Models

### Elements Dict (from `detect_all_elements`)

```python
{
    "lines": {
        "horizontal": [(x1,y1,x2,y2), ...],
        "vertical":   [(x1,y1,x2,y2), ...],
        "diagonal":   [(x1,y1,x2,y2), ...],
        "total": int
    },
    "circles":      [(cx,cy,r), ...],
    "contours":     [(x,y,w,h), ...],
    "text_regions": [(x,y,w,h), ...],
    "hatching":     [(x1,y1,x2,y2), ...]
}
```

### Association Record

```python
# Matched
{
    "annotation_id":   int,
    "annotation_text": str,
    "annotation_type": str,
    "annotation_box":  [int, int, int, int],
    "associated_element": {
        "element_type": str,   # "line_horizontal" | "line_vertical" | "line_diagonal"
                               # | "circle" | "contour"
        "element_data": list,  # [x1,y1,x2,y2] for lines, [cx,cy,r] for circles,
                               # [x,y,w,h] for contours
        "distance_px":  float  # rounded to 1 decimal place
    }
}

# Unassociated
{
    "annotation_id":   int,
    "annotation_text": str,
    "annotation_type": str,
    "annotation_box":  [int, int, int, int],
    "associated_element": null
}
```

### Output `_associations.json`

```python
{
    "source_structured": str,   # basename of _structured.json
    "image_category":    int,   # 1, 2, or 3
    "total_annotations": int,   # len(associations)
    "matched":           int,   # count where associated_element is not null
    "unassociated":      int,   # count where associated_element is null
    "associations":      list   # list of Association Records
}
```

---

## Distance Metric Implementations

### `distance_point_to_segment(px, py, x1, y1, x2, y2) -> float`

```
dx = x2 - x1
dy = y2 - y1
seg_len_sq = dx*dx + dy*dy

if seg_len_sq == 0:
    return sqrt((px-x1)**2 + (py-y1)**2)   # degenerate segment

# Parameter t: projection of point onto line, clamped to [0,1]
t = ((px-x1)*dx + (py-y1)*dy) / seg_len_sq
t = max(0.0, min(1.0, t))

# Nearest point on segment
nearest_x = x1 + t * dx
nearest_y = y1 + t * dy

return sqrt((px - nearest_x)**2 + (py - nearest_y)**2)
```

### `distance_point_to_circle(px, py, cx, cy, r) -> float`

```
return sqrt((px-cx)**2 + (py-cy)**2) - r
# Negative if point is inside circle
# Zero if point is on circle edge
# Positive if point is outside circle
```

### `distance_point_to_contour(px, py, bx, by, bw, bh) -> float`

```
# Clamp point to bounding box
clamped_x = max(bx, min(bx+bw, px))
clamped_y = max(by, min(by+bh, py))

return sqrt((px - clamped_x)**2 + (py - clamped_y)**2)
# Returns 0.0 if point is inside or on the box
```

---

## Association Dispatch Logic

### `_associate_annotation(annotation, elements, category, max_distance)`

```python
ann_type = annotation.get("type", "unknown")
box = annotation.get("box")

# Validate box
safe = _safe_box(box)
if safe is None:
    return _unassociated_record(annotation)

cx, cy = _annotation_center(safe)

h_lines  = elements["lines"]["horizontal"]
v_lines  = elements["lines"]["vertical"]
d_lines  = elements["lines"]["diagonal"]
circles  = elements["circles"]
contours = elements["contours"]

# --- Dispatch by type ---

if ann_type == "unknown":
    return _unassociated_record(annotation)

elif ann_type in ("dimension_value", "diameter_callout", "radius_callout",
                  "thread_spec", "tolerance", "dimension_with_note"):
    result = _find_nearest_line(cx, cy, h_lines, v_lines, max_distance)
    if result is None:
        result = _find_nearest_contour(cx, cy, contours, max_distance)

elif ann_type == "hole_callout":
    result = _find_nearest_circle(cx, cy, circles, max_distance)
    if result is None:
        result = _find_nearest_contour(cx, cy, contours, max_distance)

elif ann_type == "balloon_number":
    if category in (2, 3):
        result = _find_nearest_circle(cx, cy, circles, max_distance)
        # No fallback for balloons — if no circle, unassociated
    else:
        # Category 1: single digits are dimension values
        result = _find_nearest_line(cx, cy, h_lines, v_lines, max_distance)
        if result is None:
            result = _find_nearest_contour(cx, cy, contours, max_distance)

elif ann_type == "section_marker":
    result = _find_nearest_diagonal(cx, cy, d_lines, max_distance)
    if result is None:
        result = _find_nearest_line(cx, cy, [], v_lines, max_distance)

elif ann_type == "spacing_annotation":
    result = _find_nearest_horizontal(cx, cy, h_lines, max_distance)
    # No fallback

elif ann_type in ("material_code", "material_name", "part_name",
                  "bom_header", "quantity"):
    result = _find_nearest_contour(cx, cy, contours, max_distance)

else:
    result = _unassociated_record(annotation)
    return result

if result is None:
    return _unassociated_record(annotation)

element_data, distance, element_type = result
return {
    "annotation_id":   annotation["id"],
    "annotation_text": annotation.get("text", ""),
    "annotation_type": ann_type,
    "annotation_box":  list(safe),
    "associated_element": {
        "element_type": element_type,
        "element_data": [int(v) for v in element_data],
        "distance_px":  round(float(distance), 1)
    }
}
```

### `_find_nearest_line(cx, cy, h_lines, v_lines, max_distance)`

```python
best_dist = max_distance + 1
best_data = None
best_type = None

for line in h_lines:
    x1,y1,x2,y2 = line
    d = distance_point_to_segment(cx, cy, x1, y1, x2, y2)
    if d < best_dist:
        best_dist = d
        best_data = line
        best_type = "line_horizontal"

for line in v_lines:
    x1,y1,x2,y2 = line
    d = distance_point_to_segment(cx, cy, x1, y1, x2, y2)
    if d < best_dist:
        best_dist = d
        best_data = line
        best_type = "line_vertical"

if best_data is None:
    return None
return (best_data, best_dist, best_type)
```

Similar pattern for `_find_nearest_circle`, `_find_nearest_contour`, `_find_nearest_diagonal`, `_find_nearest_horizontal`.

For circles, use `distance_point_to_circle` — but compare using `abs(d)` for ranking (so annotations inside a circle rank closer than those outside):

```python
for (cx_c, cy_c, r) in circles:
    d = distance_point_to_circle(cx, cy, cx_c, cy_c, r)
    rank = abs(d)   # inside circle (negative) still ranks well
    if rank < best_rank:
        best_rank = rank
        best_dist = d
        best_data = (cx_c, cy_c, r)
```

The stored `distance_px` in the output is the raw signed value (can be negative for annotations inside circles).

---

## Element Detection Wrapper

### `_get_elements(image_path: str) -> dict`

```python
def _get_elements(image_path):
    try:
        from preprocessing import preprocess
        from element_detection import detect_all_elements
    except ImportError:
        from src.preprocessing import preprocess
        from src.element_detection import detect_all_elements

    binary = preprocess(image_path, save_result=False)
    return detect_all_elements(binary)
```

This runs Stage 1 + Stage 1.5 on-the-fly. Since `preprocess()` and `detect_all_elements()` are already implemented and tested, this is safe. The binary image is not saved to disk.

**Performance note**: For 36 images, element detection takes ~0.5–1s per image (Hough lines + contours). Total overhead: ~20–36s, within the 30s budget.

---

## Output Assembly

### `_build_output(structured_data, associations, source_structured)`

```python
matched = sum(1 for a in associations if a["associated_element"] is not None)
return {
    "source_structured": source_structured,
    "image_category":    structured_data.get("image_category", 0),
    "total_annotations": len(associations),
    "matched":           matched,
    "unassociated":      len(associations) - matched,
    "associations":      associations
}
```

### File Naming

Input:  `results/batch/cad1_001_structured.json`
Output: `results/batch/cad1_001_associations.json`
        `results/batch/cad1_001_associations.png`

The `_structured` suffix is replaced with `_associations`.

---

## Visualization Design

### `_visualize(image_path, associations, output_path)`

```python
img = cv2.imread(image_path)
if img is None:
    print(f"WARNING: cannot load image for visualization: {image_path}")
    return

vis = img.copy()

COLORS = {
    "line_horizontal": (255, 0, 0),    # Blue
    "line_vertical":   (0, 180, 0),    # Green
    "line_diagonal":   (200, 200, 0),  # Cyan
    "circle":          (0, 0, 255),    # Red
    "contour":         (200, 0, 200),  # Magenta
}

for assoc in associations:
    box = assoc["annotation_box"]
    cx = int(box[0] + box[2] / 2)
    cy = int(box[1] + box[3] / 2)

    # Yellow dot at annotation center
    cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1)

    elem = assoc.get("associated_element")
    if elem is None:
        continue

    color = COLORS.get(elem["element_type"], (128, 128, 128))
    data = elem["element_data"]

    if elem["element_type"] in ("line_horizontal", "line_vertical", "line_diagonal"):
        x1, y1, x2, y2 = data
        # Draw line to midpoint of the segment
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.line(vis, (cx, cy), (mx, my), color, 1)
        cv2.line(vis, (x1, y1), (x2, y2), color, 2)  # highlight the element

    elif elem["element_type"] == "circle":
        ecx, ecy, r = data
        cv2.line(vis, (cx, cy), (ecx, ecy), color, 1)
        cv2.circle(vis, (ecx, ecy), r, color, 2)

    elif elem["element_type"] == "contour":
        bx, by, bw, bh = data
        cv2.line(vis, (cx, cy), (bx + bw//2, by + bh//2), color, 1)
        cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), color, 1)

cv2.imwrite(output_path, vis)
```

---

## Integration with `batch_process.py`

Add after `validate_file()` call:

```python
try:
    from association import associate_file
except ImportError:
    from src.association import associate_file

# In process_category(), after validate_file():
structured_path = os.path.join(results_dir, f"{basename}_structured.json")
if os.path.exists(structured_path):
    assoc_result = associate_file(img_path, structured_path, results_dir)
    if assoc_result:
        stat["association_matched"] = assoc_result.get("matched", 0)
        stat["association_unassociated"] = assoc_result.get("unassociated", 0)
```

---

## Correctness Properties

### Property 1: `distance_point_to_segment` is non-negative

*For any* point (px, py) and segment (x1,y1,x2,y2), `distance_point_to_segment` SHALL return a non-negative float.

**Validates: Requirement 3.5**

### Property 2: `distance_point_to_circle` is negative for interior points

*For any* circle (cx,cy,r) with r > 0 and point (px,py) where `sqrt((px-cx)²+(py-cy)²) < r`, `distance_point_to_circle` SHALL return a negative value.

**Validates: Requirement 4.2**

### Property 3: `distance_point_to_contour` is non-negative

*For any* bounding box (bx,by,bw,bh) and point (px,py), `distance_point_to_contour` SHALL return a non-negative float.

**Validates: Requirement 5.4**

### Property 4: `distance_point_to_contour` is zero for interior points

*For any* bounding box (bx,by,bw,bh) and point (px,py) where `bx <= px <= bx+bw` and `by <= py <= by+bh`, `distance_point_to_contour` SHALL return 0.0.

**Validates: Requirement 5.2**

### Property 5: Output JSON round-trip

*For any* valid association result dict, `json.loads(json.dumps(result)) == result`.

**Validates: Requirement 8.9**

### Property 6: Count invariant

*For any* valid output dict, `total_annotations == matched + unassociated`.

**Validates: Requirement 8 (implicit)**

### Property 7: One record per annotation

*For any* valid output dict with N entries in `classified`, the `associations` list SHALL have exactly N records.

**Validates: Requirement 7.3**

---

## Error Handling

| Condition | Handling |
|-----------|----------|
| `image_path` not found | Log warning, skip visualization, still write JSON |
| `structured_path` not found or malformed JSON | Log error, return None |
| `preprocess()` raises exception | Log error, return None |
| `detect_all_elements()` raises exception | Log error, return None |
| Annotation `box` missing or malformed | Mark annotation as unassociated, continue |
| Element entry has wrong number of values | Skip that element, continue |
| All element lists empty | All annotations marked unassociated, write valid JSON |
| `classified` list empty | Write valid JSON with empty `associations` list |
| Any unhandled exception in `associate_file` | Log error, return None |
| Any unhandled exception in `associate_batch` per-file | Log error, skip file, continue batch |

---

## Testing Strategy

### Unit Tests (`tests/test_association.py`)

- `distance_point_to_segment`: concrete examples (point on segment, at endpoint, perpendicular foot outside segment, degenerate segment)
- `distance_point_to_circle`: point inside, on edge, outside
- `distance_point_to_contour`: point inside, on edge, outside
- `_associate_annotation`: one test per annotation type with synthetic elements
- `_build_output`: count invariant, field presence
- `associate_file`: end-to-end with a real image from `data/category_1/cad1_001.png`

### Property-Based Tests (`tests/test_association_properties.py`)

All 7 correctness properties using Hypothesis.

### Accuracy Evaluation (Manual)

Run `associate_batch` on all 36 images, inspect `_associations.png` outputs visually, count correct associations per category.
