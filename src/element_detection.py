"""
Stage 1.5: CV-based Element Detection
Detects all geometric elements in an engineering drawing using pure OpenCV.
Output: structured dict of detected elements + visualization image.
"""

import cv2
import numpy as np
import os
import sys
import json

try:
    from preprocessing import preprocess
except ImportError:
    from .preprocessing import preprocess


def detect_lines(binary, min_length=20, max_gap=5):
    """
    Use Probabilistic Hough Transform to detect line segments.
    Returns list of (x1, y1, x2, y2) tuples.
    """
    inverted = 255 - binary
    edges = cv2.Canny(inverted, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_length,
        maxLineGap=max_gap
    )
    if lines is None:
        return []
    return [tuple(line[0]) for line in lines]


def classify_lines(lines):
    """
    Separate lines into horizontal, vertical, diagonal.
    """
    h_lines, v_lines, d_lines = [], [], []
    for x1, y1, x2, y2 in lines:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 10 or angle > 170:
            h_lines.append((x1, y1, x2, y2))
        elif 80 < angle < 100:
            v_lines.append((x1, y1, x2, y2))
        else:
            d_lines.append((x1, y1, x2, y2))
    return h_lines, v_lines, d_lines


def detect_circles(binary, min_radius=8, max_radius=100):
    """
    Detect circles using contour analysis with morphological gap-closing.
    """
    inverted = 255 - binary

    # close tiny gaps so circles become continuous contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    detected = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0 or area < 50:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.6:  # slightly looser for partial circles
            continue

        (cx, cy), r = cv2.minEnclosingCircle(c)
        r = int(r)
        if r < min_radius or r > max_radius:
            continue

        detected.append((int(cx), int(cy), r))

    # deduplicate concentric ones
    filtered = []
    for c in detected:
        cx, cy, r = c
        too_close = False
        for fc in filtered:
            fcx, fcy, fr = fc
            dist = np.sqrt((cx - fcx)**2 + (cy - fcy)**2)
            if dist < 10 and abs(r - fr) < 8:
                too_close = True
                break
        if not too_close:
            filtered.append(c)
    return filtered

def detect_contours(binary, min_area=100):
    """
    Find closed contours (part outlines, frame boxes).
    Returns list of bounding rectangles (x, y, w, h).
    """
    inverted = 255 - binary
    contours, _ = cv2.findContours(
        inverted,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))
    return boxes


def detect_text_regions(binary):
    """
    Detect text regions by morphological dilation - groups nearby characters
    into single text strings instead of individual letter fragments.
    """
    inverted = 255 - binary

    # Dilate horizontally to merge characters into words/strings
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(inverted, kernel_h, iterations=1)

    # Find connected components of merged regions
    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # filter: text strings have specific size range
        if h < 8 or h > 35:
            continue
        if w < 10:
            continue
        # text aspect ratio: wide and not too tall
        if w / h < 0.8:
            continue
        boxes.append((x, y, w, h))

    return boxes

def merge_overlapping_boxes(boxes, overlap_threshold=0.3):
    """Merge boxes that overlap significantly."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [boxes[0]]
    for box in boxes[1:]:
        x, y, w, h = box
        last_x, last_y, last_w, last_h = merged[-1]
        if x < last_x + last_w + 5 and abs(y - last_y) < 10:
            new_x = min(x, last_x)
            new_y = min(y, last_y)
            new_w = max(x + w, last_x + last_w) - new_x
            new_h = max(y + h, last_y + last_h) - new_y
            merged[-1] = (new_x, new_y, new_w, new_h)
        else:
            merged.append(box)
    return merged


def detect_hatching(binary):
    """
    Detect diagonal hatch lines (~45 degrees) in section views.
    """
    lines = detect_lines(binary, min_length=10, max_gap=3)
    _, _, diagonal_lines = classify_lines(lines)

    hatch_lines = []
    for x1, y1, x2, y2 in diagonal_lines:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if 30 < angle < 60 or 120 < angle < 150:
            hatch_lines.append((x1, y1, x2, y2))
    return hatch_lines


def detect_all_elements(binary):
    """
    Run all detection methods and return a structured dict.
    """
    print("\nDetecting elements...")
    print("-" * 50)

    all_lines = detect_lines(binary)
    h_lines, v_lines, d_lines = classify_lines(all_lines)
    circles = detect_circles(binary)
    contours = detect_contours(binary)
    text_regions = detect_text_regions(binary)
    hatching = detect_hatching(binary)

    elements = {
        "lines": {
            "horizontal": h_lines,
            "vertical": v_lines,
            "diagonal": d_lines,
            "total": len(all_lines)
        },
        "circles": circles,
        "contours": contours,
        "text_regions": text_regions,
        "hatching": hatching
    }

    print(f"  Lines:        {len(all_lines)} (H:{len(h_lines)} V:{len(v_lines)} D:{len(d_lines)})")
    print(f"  Circles:      {len(circles)}")
    print(f"  Contours:     {len(contours)}")
    print(f"  Text regions: {len(text_regions)}")
    print(f"  Hatch lines:  {len(hatching)}")
    print("-" * 50)

    return elements


def visualize_elements(binary, elements, output_path):
    """
    Draw all detected elements on a color image.
    Lighter background so the overlay is clear.
    """
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    vis = cv2.addWeighted(vis, 0.4, np.full_like(vis, 255), 0.6, 0)

    # horizontal lines = BLUE
    for x1, y1, x2, y2 in elements["lines"]["horizontal"]:
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # vertical lines = GREEN
    for x1, y1, x2, y2 in elements["lines"]["vertical"]:
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 180, 0), 2)

    # diagonal lines = CYAN (for hatching, etc.)
    for x1, y1, x2, y2 in elements["lines"]["diagonal"]:
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 0), 1)

    # circles = RED
    for cx, cy, r in elements["circles"]:
        cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 0, 255), 2)

    # text regions = MAGENTA boxes
    for x, y, w, h in elements["text_regions"]:
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (200, 0, 200), 1)

    cv2.imwrite(output_path, vis)
    print(f"  Visualization saved: {output_path}")


def to_int_list(items):
    """Convert numpy ints to python ints for JSON serialization."""
    return [[int(x) for x in item] for item in items]


def detect(image_path, output_dir="results"):
    """Main entry point."""
    binary = preprocess(image_path, save_result=False)
    elements = detect_all_elements(binary)

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(image_path))[0]

    vis_path = os.path.join(output_dir, f"{filename}_detected.png")
    visualize_elements(binary, elements, vis_path)

    json_path = os.path.join(output_dir, f"{filename}_elements.json")
    json_data = {
        "lines_horizontal": to_int_list(elements["lines"]["horizontal"]),
        "lines_vertical":   to_int_list(elements["lines"]["vertical"]),
        "lines_diagonal":   to_int_list(elements["lines"]["diagonal"]),
        "circles":          to_int_list(elements["circles"]),
        "contours":         to_int_list(elements["contours"]),
        "text_regions":     to_int_list(elements["text_regions"]),
        "hatching":         to_int_list(elements["hatching"])
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  JSON saved:          {json_path}")

    return elements


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/element_detection.py <image_path>")
    else:
        detect(sys.argv[1])
        print("\nElement detection complete.")