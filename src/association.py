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


# ============================================================
# Core association dispatcher
# ============================================================

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
    Draw association connections on a copy of the original image.

    Yellow dots mark annotation centers. Colored lines connect each annotation
    to its associated element. Unassociated annotations get a dot only.

    Args:
        image_path:  Path to the original PNG image.
        associations: List of association record dicts.
        output_path: Path to write the visualization PNG.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  WARNING: cannot load image for visualization: {image_path}")
        return

    vis = img.copy()

    for assoc in associations:
        box = _safe_box(assoc.get("annotation_box"))
        if box is None:
            continue

        cx, cy = _annotation_center(box)
        icx, icy = int(cx), int(cy)

        # Yellow filled dot at annotation center
        cv2.circle(vis, (icx, icy), 3, (0, 255, 255), -1)

        elem = assoc.get("associated_element")
        if elem is None:
            continue

        color = ELEMENT_COLORS.get(elem["element_type"], (128, 128, 128))
        data = elem["element_data"]

        if elem["element_type"] in ("line_horizontal", "line_vertical", "line_diagonal"):
            if len(data) >= 4:
                x1, y1, x2, y2 = data[0], data[1], data[2], data[3]
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.line(vis, (icx, icy), (mx, my), color, 1)
                cv2.line(vis, (x1, y1), (x2, y2), color, 2)

        elif elem["element_type"] == "circle":
            if len(data) >= 3:
                ecx, ecy, r = data[0], data[1], data[2]
                cv2.line(vis, (icx, icy), (ecx, ecy), color, 1)
                cv2.circle(vis, (ecx, ecy), r, color, 2)

        elif elem["element_type"] == "contour":
            if len(data) >= 4:
                bx, by, bw, bh = data[0], data[1], data[2], data[3]
                cv2.line(vis, (icx, icy), (bx + bw // 2, by + bh // 2), color, 1)
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, 1)

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
