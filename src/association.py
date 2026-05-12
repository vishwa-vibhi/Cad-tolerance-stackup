"""
Stage 4: Geometric Association for the CAD Tolerance Stack-Up Analysis pipeline.

Links each classified annotation from Stage 3 (_structured.json) to the nearest
geometric element detected by Stage 1.5 (run on-the-fly via element_detection.py).

Pipeline position:
    Stage 3 _structured.json  ──►  association.py  ──►  _associations.json
    Original image             ──►                  ──►  _associations.png

Usage (CLI):
    python src/association.py <image_path> <structured_path> <output_dir>

Usage (API):
    from src.association import associate_file, associate_batch
"""

import cv2
import os
import sys
import json
import math
import pathlib

try:
    from preprocessing import preprocess
    from element_detection import detect_all_elements
except ImportError:
    from src.preprocessing import preprocess
    from src.element_detection import detect_all_elements


# ============================================================
# Configuration
# ============================================================

MAX_DISTANCE_PX = 150  # Maximum pixel distance for association


# ============================================================
# Visualization color map
# ============================================================

ELEMENT_COLORS = {
    "line_horizontal": (255, 0,   0),    # Blue
    "line_vertical":   (0,   180, 0),    # Green
    "line_diagonal":   (200, 200, 0),    # Cyan
    "circle":          (0,   0,   255),  # Red
    "contour":         (200, 0,   200),  # Magenta
}


# ============================================================
# Distance metric functions
# ============================================================

def distance_point_to_segment(px, py, x1, y1, x2, y2):
    """
    Minimum distance from point (px, py) to finite line segment (x1,y1)-(x2,y2).

    Uses clamped projection: the perpendicular foot is clamped to the segment
    endpoints, so the result is always the true minimum distance to the segment.

    Args:
        px, py: Query point coordinates.
        x1, y1, x2, y2: Segment endpoint coordinates.

    Returns:
        Non-negative float distance in pixels.
    """
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq == 0:
        # Degenerate segment: both endpoints are the same point
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    # Parameter t: projection of point onto infinite line, clamped to [0, 1]
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))

    # Nearest point on segment
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy

    return math.sqrt((px - nearest_x) ** 2 + (py - nearest_y) ** 2)


def distance_point_to_circle(px, py, cx, cy, r):
    """
    Signed distance from point (px, py) to circle edge.

    Returns:
        Negative if point is inside the circle (distance to center < r).
        Zero if point is exactly on the circle edge.
        Positive if point is outside the circle.
    """
    return math.sqrt((px - cx) ** 2 + (py - cy) ** 2) - r


def distance_point_to_contour(px, py, bx, by, bw, bh):
    """
    Minimum distance from point (px, py) to axis-aligned bounding box.

    Returns:
        0.0 if the point is inside or on the bounding box.
        Euclidean distance to the nearest point on the box perimeter otherwise.
    """
    # Clamp point to box
    clamped_x = max(float(bx), min(float(bx + bw), float(px)))
    clamped_y = max(float(by), min(float(by + bh), float(py)))
    return math.sqrt((px - clamped_x) ** 2 + (py - clamped_y) ** 2)


# ============================================================
# Helper utilities
# ============================================================

def _safe_box(box):
    """
    Validate and convert a box field to a 4-tuple of ints.

    Args:
        box: Expected to be a list/tuple of 4 numeric values [x, y, w, h].

    Returns:
        (x, y, w, h) as ints, or None if invalid.
    """
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        return (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    except (TypeError, ValueError):
        return None


def _annotation_center(box):
    """
    Compute the center of an annotation bounding box.

    Args:
        box: Validated 4-tuple (x, y, w, h).

    Returns:
        (cx, cy) as floats.
    """
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def _get_elements(image_path):
    """
    Run Stage 1 preprocessing + Stage 1.5 element detection on an image.

    Args:
        image_path: Path to the original PNG image.

    Returns:
        Elements dict from detect_all_elements().

    Raises:
        Exception if preprocessing or detection fails.
    """
    binary = preprocess(image_path, save_result=False)
    return detect_all_elements(binary)


# ============================================================
# Nearest-element finder functions
# ============================================================

def _find_nearest_line(cx, cy, h_lines, v_lines, max_distance):
    """
    Find the nearest horizontal or vertical line to point (cx, cy).

    Returns:
        (element_data, distance, element_type) or None if nothing within max_distance.
    """
    best_dist = max_distance + 1.0
    best_data = None
    best_type = None

    for line in h_lines:
        if len(line) < 4:
            continue
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        d = distance_point_to_segment(cx, cy, x1, y1, x2, y2)
        if d < best_dist:
            best_dist = d
            best_data = [int(x1), int(y1), int(x2), int(y2)]
            best_type = "line_horizontal"

    for line in v_lines:
        if len(line) < 4:
            continue
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        d = distance_point_to_segment(cx, cy, x1, y1, x2, y2)
        if d < best_dist:
            best_dist = d
            best_data = [int(x1), int(y1), int(x2), int(y2)]
            best_type = "line_vertical"

    if best_data is None:
        return None
    return (best_data, best_dist, best_type)


def _find_nearest_circle(cx, cy, circles, max_distance):
    """
    Find the nearest circle to point (cx, cy).

    Ranks by abs(distance_to_circle) so annotations inside a circle rank closest.

    Returns:
        (element_data, signed_distance, "circle") or None.
    """
    best_rank = max_distance + 1.0
    best_dist = None
    best_data = None

    for circle in circles:
        if len(circle) < 3:
            continue
        ecx, ecy, r = circle[0], circle[1], circle[2]
        d = distance_point_to_circle(cx, cy, ecx, ecy, r)
        rank = abs(d)
        if rank < best_rank:
            best_rank = rank
            best_dist = d
            best_data = [int(ecx), int(ecy), int(r)]

    if best_data is None:
        return None
    return (best_data, best_dist, "circle")


def _find_nearest_contour(cx, cy, contours, max_distance):
    """
    Find the nearest contour bounding box to point (cx, cy).

    Returns:
        (element_data, distance, "contour") or None.
    """
    best_dist = max_distance + 1.0
    best_data = None

    for contour in contours:
        if len(contour) < 4:
            continue
        bx, by, bw, bh = contour[0], contour[1], contour[2], contour[3]
        d = distance_point_to_contour(cx, cy, bx, by, bw, bh)
        if d < best_dist:
            best_dist = d
            best_data = [int(bx), int(by), int(bw), int(bh)]

    if best_data is None:
        return None
    return (best_data, best_dist, "contour")


def _find_nearest_diagonal(cx, cy, diagonals, max_distance):
    """
    Find the nearest diagonal line to point (cx, cy).

    Returns:
        (element_data, distance, "line_diagonal") or None.
    """
    best_dist = max_distance + 1.0
    best_data = None

    for line in diagonals:
        if len(line) < 4:
            continue
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        d = distance_point_to_segment(cx, cy, x1, y1, x2, y2)
        if d < best_dist:
            best_dist = d
            best_data = [int(x1), int(y1), int(x2), int(y2)]

    if best_data is None:
        return None
    return (best_data, best_dist, "line_diagonal")


def _find_nearest_horizontal(cx, cy, h_lines, max_distance):
    """
    Find the nearest horizontal line to point (cx, cy).

    Returns:
        (element_data, distance, "line_horizontal") or None.
    """
    best_dist = max_distance + 1.0
    best_data = None

    for line in h_lines:
        if len(line) < 4:
            continue
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        d = distance_point_to_segment(cx, cy, x1, y1, x2, y2)
        if d < best_dist:
            best_dist = d
            best_data = [int(x1), int(y1), int(x2), int(y2)]

    if best_data is None:
        return None
    return (best_data, best_dist, "line_horizontal")


def _find_extension_line_target(cx, cy, h_lines, v_lines, contours, max_distance):
    """
    Trace extension lines to find the exact feature a dimension annotates.

    In engineering drawings, dimension lines have two short perpendicular
    extension lines at their ends that touch the feature being measured.
    This function looks for a short line near the annotation that is
    perpendicular to the nearest dimension line, then follows it to find
    the feature contour.

    Args:
        cx, cy:       Annotation center.
        h_lines:      Horizontal lines from element detection.
        v_lines:      Vertical lines from element detection.
        contours:     Part contours from element detection.
        max_distance: Max search distance.

    Returns:
        (element_data, distance, element_type) or None.
    """
    # Step 1: Find the nearest line (this is the dimension line)
    nearest = _find_nearest_line(cx, cy, h_lines, v_lines, max_distance)
    if nearest is None:
        return None

    dim_line_data, dim_dist, dim_type = nearest

    # Step 2: Determine if dimension line is horizontal or vertical
    x1, y1, x2, y2 = dim_line_data
    is_horizontal = (dim_type == "line_horizontal")

    # Step 3: Look for extension lines — short lines perpendicular to the dim line
    # near the annotation endpoints
    ext_search_dist = min(max_distance, 80)

    if is_horizontal:
        # Extension lines are vertical, near x=cx
        best_ext = None
        best_ext_dist = ext_search_dist + 1
        for line in v_lines:
            if len(line) < 4:
                continue
            lx1, ly1, lx2, ly2 = line
            # Extension line should be near the annotation x-position
            line_cx = (lx1 + lx2) / 2.0
            if abs(line_cx - cx) > 30:
                continue
            d = distance_point_to_segment(cx, cy, lx1, ly1, lx2, ly2)
            if d < best_ext_dist:
                best_ext_dist = d
                best_ext = line
    else:
        # Extension lines are horizontal, near y=cy
        best_ext = None
        best_ext_dist = ext_search_dist + 1
        for line in h_lines:
            if len(line) < 4:
                continue
            lx1, ly1, lx2, ly2 = line
            line_cy = (ly1 + ly2) / 2.0
            if abs(line_cy - cy) > 30:
                continue
            d = distance_point_to_segment(cx, cy, lx1, ly1, lx2, ly2)
            if d < best_ext_dist:
                best_ext_dist = d
                best_ext = line

    # Step 4: If extension line found, find the contour at its far end
    if best_ext is not None:
        lx1, ly1, lx2, ly2 = best_ext
        # The far end of the extension line (away from annotation) points to the feature
        # Pick the endpoint farther from the annotation
        d1 = math.sqrt((lx1 - cx) ** 2 + (ly1 - cy) ** 2)
        d2 = math.sqrt((lx2 - cx) ** 2 + (ly2 - cy) ** 2)
        if d1 > d2:
            target_x, target_y = lx1, ly1
        else:
            target_x, target_y = lx2, ly2

        # Find contour at the target point
        contour_result = _find_nearest_contour(target_x, target_y, contours, 50)
        if contour_result is not None:
            return contour_result

    # Fallback: return the dimension line itself
    return nearest




def _unassociated_record(annotation):
    """Build an unassociated record for an annotation."""
    box = _safe_box(annotation.get("box")) or []
    return {
        "annotation_id":      annotation.get("id", 0),
        "annotation_text":    annotation.get("text", ""),
        "annotation_type":    annotation.get("type", "unknown"),
        "annotation_box":     list(box) if box else annotation.get("box", []),
        "associated_element": None,
    }


def _associate_annotation(annotation, elements, category, max_distance):
    """
    Dispatch association logic based on annotation type.

    Args:
        annotation: Single classified entry dict from _structured.json.
        elements:   Elements dict from detect_all_elements().
        category:   Image category (1, 2, or 3).
        max_distance: Maximum pixel distance for association.

    Returns:
        Association record dict with associated_element or null.
    """
    ann_type = annotation.get("type", "unknown")
    box = _safe_box(annotation.get("box"))

    if box is None:
        return _unassociated_record(annotation)

    cx, cy = _annotation_center(box)

    h_lines  = elements.get("lines", {}).get("horizontal", [])
    v_lines  = elements.get("lines", {}).get("vertical", [])
    d_lines  = elements.get("lines", {}).get("diagonal", [])
    circles  = elements.get("circles", [])
    contours = elements.get("contours", [])

    result = None

    if ann_type == "unknown":
        return _unassociated_record(annotation)

    elif ann_type in ("dimension_value", "diameter_callout", "radius_callout",
                      "thread_spec", "tolerance", "dimension_with_note"):
        # For Category 1 part drawings, use extension line tracing for better accuracy
        if category == 1:
            result = _find_extension_line_target(cx, cy, h_lines, v_lines, contours, max_distance)
        else:
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
            # No fallback for balloons in assembly drawings
        else:
            # Category 1: single digits are dimension values
            result = _find_nearest_line(cx, cy, h_lines, v_lines, max_distance)
            if result is None:
                result = _find_nearest_contour(cx, cy, contours, max_distance)

    elif ann_type == "section_marker":
        result = _find_nearest_diagonal(cx, cy, d_lines, max_distance)
        if result is None:
            # Fallback: nearest vertical line
            result = _find_nearest_line(cx, cy, [], v_lines, max_distance)

    elif ann_type == "spacing_annotation":
        result = _find_nearest_horizontal(cx, cy, h_lines, max_distance)
        # No fallback

    elif ann_type in ("material_code", "material_name", "part_name",
                      "bom_header", "quantity"):
        result = _find_nearest_contour(cx, cy, contours, max_distance)

    else:
        # Any unrecognised type → unassociated
        return _unassociated_record(annotation)

    if result is None:
        return _unassociated_record(annotation)

    element_data, distance, element_type = result
    return {
        "annotation_id":   annotation.get("id", 0),
        "annotation_text": annotation.get("text", ""),
        "annotation_type": ann_type,
        "annotation_box":  list(box),
        "associated_element": {
            "element_type": element_type,
            "element_data": [int(v) for v in element_data],
            "distance_px":  round(float(distance), 1),
        }
    }


# ============================================================
# Output assembly
# ============================================================

def _build_output(structured_data, associations, source_structured):
    """
    Assemble the final _associations.json output dict.

    Args:
        structured_data:   Parsed _structured.json dict.
        associations:      List of association record dicts.
        source_structured: Basename of the _structured.json file.

    Returns:
        Output dict with summary counts and associations list.
    """
    matched = sum(1 for a in associations if a["associated_element"] is not None)
    return {
        "source_structured": source_structured,
        "image_category":    structured_data.get("image_category", 0),
        "total_annotations": len(associations),
        "matched":           matched,
        "unassociated":      len(associations) - matched,
        "associations":      associations,
    }


# ============================================================
# Visualization
# ============================================================

def _visualize(image_path, associations, output_path):
    """
    Draw rich association visualization on the original image.

    - Colored bounding boxes around each annotation (color = type)
    - Text label showing the annotation text
    - Connecting line from annotation to associated element
    - Legend in corner
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  WARNING: cannot load image for visualization: {image_path}")
        return

    vis = img.copy()
    h_img, w_img = vis.shape[:2]

    # Type → color mapping for annotation boxes
    TYPE_COLORS = {
        "dimension_value":     (255, 100,   0),   # orange
        "diameter_callout":    (  0, 200, 100),   # green
        "radius_callout":      (  0, 180,  80),   # green
        "thread_spec":         (180,   0, 255),   # purple
        "tolerance":           (  0,   0, 255),   # red
        "hole_callout":        (255, 150,   0),   # amber
        "dimension_with_note": (255, 100,   0),   # orange
        "balloon_number":      (  0, 255, 200),   # cyan
        "part_name":           (  0, 180, 255),   # sky blue
        "material_code":       (200,   0, 200),   # magenta
        "material_name":       (200,   0, 200),   # magenta
        "bom_header":          (100, 200, 255),   # light blue
        "section_marker":      (  0, 255,  80),   # lime
        "spacing_annotation":  (  0, 255,  80),   # lime
        "quantity":            (150, 255, 100),   # yellow-green
        "unknown":             ( 80,  80,  80),   # gray
    }

    ASSOC_COLORS = {
        "line_horizontal": (255,   0,   0),   # blue
        "line_vertical":   (  0, 180,   0),   # green
        "line_diagonal":   (200, 200,   0),   # cyan
        "circle":          (  0,   0, 255),   # red
        "contour":         (200,   0, 200),   # magenta
    }

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    thickness = 1

    for assoc in associations:
        box = _safe_box(assoc.get("annotation_box"))
        if box is None:
            continue

        ann_type = assoc.get("annotation_type", "unknown")
        ann_text = assoc.get("annotation_text", "")
        cx, cy = _annotation_center(box)
        icx, icy = int(cx), int(cy)
        x, y, w, h = box

        color = TYPE_COLORS.get(ann_type, (128, 128, 128))

        # Skip unknown in visualization to reduce clutter
        if ann_type == "unknown":
            continue

        # Draw annotation bounding box
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)

        # Draw text label above the box
        label = ann_text[:18] if ann_text else ann_type[:12]
        label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        lx = max(0, x)
        ly = max(label_size[1] + 2, y - 2)
        # Small background for readability
        cv2.rectangle(vis, (lx, ly - label_size[1] - 2), (lx + label_size[0] + 2, ly + 1),
                      (0, 0, 0), -1)
        cv2.putText(vis, label, (lx + 1, ly - 1), font, font_scale, color, thickness)

        # Draw association connection
        elem = assoc.get("associated_element")
        if elem is None:
            continue

        assoc_color = ASSOC_COLORS.get(elem["element_type"], (128, 128, 128))
        data = elem["element_data"]

        if elem["element_type"] in ("line_horizontal", "line_vertical", "line_diagonal"):
            if len(data) >= 4:
                x1, y1, x2, y2 = data[0], data[1], data[2], data[3]
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                # Thin dashed-style line from annotation to element midpoint
                cv2.line(vis, (icx, icy), (mx, my), assoc_color, 1, cv2.LINE_AA)
                # Highlight the associated line segment
                cv2.line(vis, (x1, y1), (x2, y2), assoc_color, 2)

        elif elem["element_type"] == "circle":
            if len(data) >= 3:
                ecx, ecy, r = data[0], data[1], data[2]
                cv2.line(vis, (icx, icy), (ecx, ecy), assoc_color, 1, cv2.LINE_AA)
                cv2.circle(vis, (ecx, ecy), r, assoc_color, 2)

        elif elem["element_type"] == "contour":
            if len(data) >= 4:
                bx2, by2, bw2, bh2 = data[0], data[1], data[2], data[3]
                cv2.line(vis, (icx, icy), (bx2 + bw2 // 2, by2 + bh2 // 2),
                         assoc_color, 1, cv2.LINE_AA)
                cv2.rectangle(vis, (bx2, by2), (bx2 + bw2, by2 + bh2), assoc_color, 1)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        ("Dimension",   TYPE_COLORS["dimension_value"]),
        ("Diameter/R",  TYPE_COLORS["diameter_callout"]),
        ("Thread",      TYPE_COLORS["thread_spec"]),
        ("Hole",        TYPE_COLORS["hole_callout"]),
        ("Balloon",     TYPE_COLORS["balloon_number"]),
        ("Part name",   TYPE_COLORS["part_name"]),
        ("BOM header",  TYPE_COLORS["bom_header"]),
    ]
    lx0, ly0 = 6, 6
    pad = 2
    row_h = 14
    legend_w = 90
    legend_h = len(legend_items) * row_h + pad * 2
    # Semi-transparent background
    overlay = vis.copy()
    cv2.rectangle(overlay, (lx0, ly0), (lx0 + legend_w, ly0 + legend_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
    for i, (label, color) in enumerate(legend_items):
        ry = ly0 + pad + i * row_h + row_h // 2
        cv2.rectangle(vis, (lx0 + pad, ry - 4), (lx0 + pad + 10, ry + 4), color, -1)
        cv2.putText(vis, label, (lx0 + pad + 14, ry + 4),
                    font, 0.32, (220, 220, 220), 1)

    cv2.imwrite(output_path, vis)



# ============================================================
# Public API
# ============================================================

def associate_file(image_path, structured_path, output_dir, max_distance=MAX_DISTANCE_PX):
    """
    Process a single image: detect elements, associate annotations, write outputs.

    Args:
        image_path:      Path to the original PNG image.
        structured_path: Path to the _structured.json from Stage 3.
        output_dir:      Directory to write _associations.json and _associations.png.
        max_distance:    Max pixel distance for association (default MAX_DISTANCE_PX).

    Returns:
        The association result dict, or None on unrecoverable error.
    """
    structured_path = str(structured_path)
    image_path = str(image_path)
    source_structured = os.path.basename(structured_path)

    # Load structured JSON
    try:
        with open(structured_path, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ERROR: cannot load structured JSON '{structured_path}': {e}")
        return None

    # Run element detection on-the-fly
    try:
        elements = _get_elements(image_path)
    except Exception as e:
        print(f"  ERROR: element detection failed for '{image_path}': {e}")
        return None

    category = structured_data.get("image_category", 0)
    classified = structured_data.get("classified", [])

    # Associate each annotation
    associations = []
    for annotation in classified:
        try:
            record = _associate_annotation(annotation, elements, category, max_distance)
            associations.append(record)
        except Exception as e:
            print(f"  WARNING: failed to associate annotation {annotation.get('id')}: {e}")
            associations.append(_unassociated_record(annotation))

    result = _build_output(structured_data, associations, source_structured)

    # Write JSON output
    os.makedirs(output_dir, exist_ok=True)
    stem = source_structured
    if stem.endswith("_structured.json"):
        stem = stem[: -len("_structured.json")]
    json_path = os.path.join(output_dir, f"{stem}_associations.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Write visualization
    vis_path = os.path.join(output_dir, f"{stem}_associations.png")
    _visualize(image_path, associations, vis_path)

    matched = result["matched"]
    total = result["total_annotations"]
    unassoc = result["unassociated"]
    print(f"  {source_structured} -> {total} annotations | matched: {matched} | unassociated: {unassoc}")

    return result


def associate_batch(image_dir, structured_dir, output_dir, max_distance=MAX_DISTANCE_PX):
    """
    Process all _structured.json files in structured_dir.

    Locates the corresponding image by replacing _structured.json with .png
    in the same image_dir.

    Args:
        image_dir:      Directory containing original PNG images.
        structured_dir: Directory containing _structured.json files.
        output_dir:     Directory to write outputs.
        max_distance:   Max pixel distance for association.

    Returns:
        List of result dicts (one per successfully processed image).
    """
    structured_files = sorted(pathlib.Path(structured_dir).glob("*_structured.json"))
    total = len(structured_files)
    results = []

    print(f"\n=== STAGE 4: Geometric Association ===")
    print(f"Processing {total} structured files from: {structured_dir}")
    print("-" * 60)

    for idx, structured_path in enumerate(structured_files, 1):
        basename = structured_path.name
        image_name = basename.replace("_structured.json", ".png")
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"  [{idx}/{total}] WARNING: image not found: {image_path} — skipping")
            continue

        print(f"  [{idx}/{total}] ", end="", flush=True)
        try:
            result = associate_file(str(image_path), str(structured_path), output_dir, max_distance)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"  ERROR processing {basename}: {e}")

    total_ann = sum(r["total_annotations"] for r in results)
    total_matched = sum(r["matched"] for r in results)
    match_pct = (total_matched / total_ann * 100) if total_ann > 0 else 0

    print("-" * 60)
    print(f"Batch complete: {len(results)}/{total} images processed")
    print(f"Total annotations: {total_ann} | Matched: {total_matched} ({match_pct:.1f}%)")

    return results


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python src/association.py <image_path> <structured_path> <output_dir>")
        sys.exit(1)
    image_path = sys.argv[1]
    structured_path = sys.argv[2]
    output_dir = sys.argv[3]
    result = associate_file(image_path, structured_path, output_dir)
    if result:
        print(f"\nDone. Matched {result['matched']}/{result['total_annotations']} annotations.")
    else:
        print("Association failed.")
        sys.exit(1)
