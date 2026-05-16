"""
Semantic Dimension Labeller — Stage 4.6

Assigns a human-readable semantic label to each dimension annotation
based on:
  1. The annotation type (diameter_callout, thread_spec, etc.)
  2. The associated geometric element type (line_horizontal, circle, etc.)
  3. The annotation text content (THICK, DEEP, HOLE, etc.)
  4. Spatial position relative to the part (top/bottom/left/right)

Output labels (what each dimension MEANS):
  length          — horizontal linear dimension (width of a feature)
  height          — vertical linear dimension
  depth           — dimension into the page / step depth
  thickness       — material thickness
  bore_diameter   — internal circular feature diameter
  shaft_diameter  — external circular feature diameter
  hole_diameter   — drilled/bored hole diameter
  thread_size     — threaded feature (M16×2 etc.)
  radius          — fillet or corner radius
  chamfer         — angled feature (45° etc.)
  pitch_circle    — PCD / bolt circle diameter
  spacing         — equally spaced feature pitch
  section_cut     — section view marker (not a dimension)
  unknown         — cannot determine

Usage (API):
    from src.semantic_labeller import label_file, label_batch

Usage (CLI):
    python src/semantic_labeller.py <associations_path> <output_dir>
"""

import os
import sys
import json
import math
import pathlib


# ============================================================
# Semantic label definitions
# ============================================================

# Maps (annotation_type, element_type) → semantic label
# More specific rules take priority over general ones
LABEL_RULES = [
    # ── Diameter / circle features ────────────────────────────────────────
    ("diameter_callout",  "circle",          "bore_diameter"),
    ("diameter_callout",  "line_horizontal", "shaft_diameter"),
    ("diameter_callout",  "line_vertical",   "shaft_diameter"),
    ("diameter_callout",  "contour",         "shaft_diameter"),
    ("hole_callout",      "circle",          "hole_diameter"),
    ("hole_callout",      "contour",         "hole_diameter"),
    ("hole_callout",      "line_horizontal", "hole_diameter"),

    # ── Thread features ───────────────────────────────────────────────────
    ("thread_spec",       "circle",          "thread_size"),
    ("thread_spec",       "line_horizontal", "thread_size"),
    ("thread_spec",       "line_vertical",   "thread_size"),
    ("thread_spec",       "contour",         "thread_size"),

    # ── Radius ────────────────────────────────────────────────────────────
    ("radius_callout",    "circle",          "radius"),
    ("radius_callout",    "contour",         "radius"),
    ("radius_callout",    "line_diagonal",   "radius"),

    # ── Section markers ───────────────────────────────────────────────────
    ("section_marker",    "line_diagonal",   "section_cut"),
    ("section_marker",    "line_vertical",   "section_cut"),

    # ── Spacing ───────────────────────────────────────────────────────────
    ("spacing_annotation","line_horizontal", "spacing"),

    # ── Linear dimensions — direction from element type ───────────────────
    ("dimension_value",   "line_horizontal", "length"),
    ("dimension_value",   "line_vertical",   "height"),
    ("dimension_value",   "line_diagonal",   "chamfer"),
    ("dimension_value",   "circle",          "bore_diameter"),
    ("dimension_value",   "contour",         "length"),

    # ── Dimension with note ───────────────────────────────────────────────
    ("dimension_with_note","line_horizontal","thickness"),
    ("dimension_with_note","line_vertical",  "depth"),
    ("dimension_with_note","contour",        "thickness"),
]

# Text-content overrides — checked BEFORE element-type rules
# If the annotation text contains these keywords, use this label
TEXT_OVERRIDES = [
    ("THICK",   "thickness"),
    ("DEEP",    "depth"),
    ("LONG",    "length"),
    ("WIDE",    "width"),
    ("HEIGHT",  "height"),
    ("PCD",     "pitch_circle"),
    ("GROOVE",  "groove_depth"),
    ("KEY",     "keyway"),
    ("HOLE",    "hole_diameter"),
    ("HOLES",   "hole_diameter"),
    ("TURNS",   "coil_spec"),
    ("TEETH",   "gear_spec"),
    ("MODULE",  "gear_module"),
    ("SPIRAL",  "coil_spec"),
    ("EQUI",    "spacing"),
    ("CHAMFER", "chamfer"),
]

# Human-readable descriptions for each label
LABEL_DESCRIPTIONS = {
    "length":        "Length / Width",
    "height":        "Height",
    "depth":         "Depth",
    "thickness":     "Thickness",
    "bore_diameter": "Bore Diameter",
    "shaft_diameter":"Shaft Diameter",
    "hole_diameter": "Hole Diameter",
    "thread_size":   "Thread Size",
    "radius":        "Fillet Radius",
    "chamfer":       "Chamfer",
    "pitch_circle":  "Pitch Circle Dia.",
    "spacing":       "Equal Spacing",
    "section_cut":   "Section Marker",
    "groove_depth":  "Groove Depth",
    "keyway":        "Keyway",
    "coil_spec":     "Coil Specification",
    "gear_spec":     "Gear Specification",
    "gear_module":   "Gear Module",
    "width":         "Width",
    "unknown":       "Unknown",
}


# ============================================================
# Core labelling logic
# ============================================================

def _get_semantic_label(ann_type, element_type, text):
    """
    Determine the semantic label for one annotation.

    Priority order:
    1. Text content overrides (THICK, DEEP, PCD, etc.)
    2. (annotation_type, element_type) rule table
    3. Annotation type alone (fallback)
    4. "unknown"
    """
    text_upper = (text or "").upper()

    # Priority 1: text content overrides
    for keyword, label in TEXT_OVERRIDES:
        if keyword in text_upper:
            return label

    # Priority 2: (type, element) rule table
    for rule_ann, rule_elem, label in LABEL_RULES:
        if ann_type == rule_ann and element_type == rule_elem:
            return label

    # Priority 3: annotation type alone (no element match)
    type_fallbacks = {
        "diameter_callout":   "shaft_diameter",
        "hole_callout":       "hole_diameter",
        "thread_spec":        "thread_size",
        "radius_callout":     "radius",
        "section_marker":     "section_cut",
        "spacing_annotation": "spacing",
        "dimension_with_note":"thickness",
        "dimension_value":    "length",
        "tolerance":          "unknown",
        "balloon_number":     "unknown",
        "part_name":          "unknown",
        "material_code":      "unknown",
        "material_name":      "unknown",
        "bom_header":         "unknown",
        "quantity":           "unknown",
    }
    if ann_type in type_fallbacks:
        return type_fallbacks[ann_type]

    return "unknown"


def _get_direction(element_type, element_data):
    """
    Determine measurement direction from the associated geometric element.

    Returns: "horizontal", "vertical", "diagonal", "radial", or None
    """
    if element_type == "line_horizontal":
        return "horizontal"
    if element_type == "line_vertical":
        return "vertical"
    if element_type == "line_diagonal":
        # Compute actual angle
        if element_data and len(element_data) >= 4:
            x1, y1, x2, y2 = element_data[:4]
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if angle < 10 or angle > 170:
                return "horizontal"
            elif 80 < angle < 100:
                return "vertical"
            else:
                return "diagonal"
        return "diagonal"
    if element_type == "circle":
        return "radial"
    if element_type == "contour":
        return "horizontal"  # bounding box default
    return None


def label_annotations(associations):
    """
    Add semantic labels to a list of association records.

    Args:
        associations: List of association dicts from _associations.json

    Returns:
        List of labelled annotation dicts, each with added fields:
            semantic_label, semantic_description, direction
    """
    labelled = []
    for assoc in associations:
        ann_type = assoc.get("annotation_type", "unknown")
        text     = assoc.get("annotation_text", "")
        elem     = assoc.get("associated_element")

        element_type = elem.get("element_type") if elem else None
        element_data = elem.get("element_data") if elem else None

        label       = _get_semantic_label(ann_type, element_type, text)
        description = LABEL_DESCRIPTIONS.get(label, label)
        direction   = _get_direction(element_type, element_data) if elem else None

        record = dict(assoc)
        record["semantic_label"]       = label
        record["semantic_description"] = description
        record["direction"]            = direction
        labelled.append(record)

    return labelled


# ============================================================
# Summary helpers
# ============================================================

def _build_label_summary(labelled):
    """Count how many of each semantic label were found."""
    counts = {}
    for a in labelled:
        lbl = a.get("semantic_label", "unknown")
        counts[lbl] = counts.get(lbl, 0) + 1
    # Sort by count descending
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def _build_dimension_table(labelled):
    """
    Build a clean dimension table — only meaningful dimension annotations.

    Returns list of dicts: text, label, description, direction, confidence
    """
    DIMENSION_LABELS = {
        "length", "height", "depth", "thickness", "bore_diameter",
        "shaft_diameter", "hole_diameter", "thread_size", "radius",
        "chamfer", "pitch_circle", "spacing", "groove_depth",
        "keyway", "width", "gear_module", "gear_spec", "coil_spec",
    }
    table = []
    for a in labelled:
        if a.get("semantic_label") not in DIMENSION_LABELS:
            continue
        table.append({
            "id":          a.get("annotation_id"),
            "text":        a.get("annotation_text", ""),
            "type":        a.get("annotation_type", ""),
            "label":       a.get("semantic_label", ""),
            "description": a.get("semantic_description", ""),
            "direction":   a.get("direction"),
            "confidence":  a.get("associated_element", {}).get("distance_px") if a.get("associated_element") else None,
        })
    return table


# ============================================================
# Public API
# ============================================================

def label_file(associations_path, output_dir):
    """
    Add semantic labels to all annotations in an _associations.json file.

    Args:
        associations_path: Path to _associations.json from Stage 4.
        output_dir:        Directory to write _labelled.json.

    Returns:
        Labelled result dict, or None on error.
    """
    associations_path = str(associations_path)
    source_name = os.path.basename(associations_path)

    try:
        with open(associations_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ERROR: cannot load '{associations_path}': {e}")
        return None

    associations = data.get("associations", [])
    labelled     = label_annotations(associations)
    summary      = _build_label_summary(labelled)
    dim_table    = _build_dimension_table(labelled)

    # Count labelled vs unknown
    total     = len(labelled)
    n_unknown = sum(1 for a in labelled if a.get("semantic_label") == "unknown")
    n_labelled = total - n_unknown

    result = {
        "source_associations": source_name,
        "image_category":      data.get("image_category", 0),
        "total_annotations":   total,
        "labelled_count":      n_labelled,
        "unknown_count":       n_unknown,
        "labelled_rate_pct":   round(n_labelled / total * 100, 1) if total else 0,
        "label_summary":       summary,
        "dimension_table":     dim_table,
        "annotations":         labelled,
    }

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    stem = source_name
    if stem.endswith("_associations.json"):
        stem = stem[: -len("_associations.json")]
    out_path = os.path.join(output_dir, f"{stem}_labelled.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  {source_name} -> {n_labelled}/{total} labelled "
          f"({result['labelled_rate_pct']}%) | "
          f"labels: {dict(list(summary.items())[:5])}")

    return result


def label_batch(associations_dir, output_dir):
    """
    Process all _associations.json files in associations_dir.

    Returns list of result dicts.
    """
    files = sorted(pathlib.Path(associations_dir).glob("*_associations.json"))
    total = len(files)
    results = []

    print(f"\n=== SEMANTIC LABELLING ===")
    print(f"Processing {total} association files from: {associations_dir}")
    print("-" * 60)

    for idx, path in enumerate(files, 1):
        print(f"  [{idx}/{total}] ", end="", flush=True)
        try:
            result = label_file(str(path), output_dir)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  ERROR: {path.name}: {e}")

    # Aggregate stats
    total_ann     = sum(r["total_annotations"] for r in results)
    total_labelled = sum(r["labelled_count"] for r in results)
    total_unknown  = sum(r["unknown_count"] for r in results)
    rate = round(total_labelled / total_ann * 100, 1) if total_ann else 0

    # Aggregate label distribution
    all_labels = {}
    for r in results:
        for lbl, cnt in r.get("label_summary", {}).items():
            all_labels[lbl] = all_labels.get(lbl, 0) + cnt
    all_labels = dict(sorted(all_labels.items(), key=lambda x: -x[1]))

    print("-" * 60)
    print(f"Batch complete: {len(results)}/{total} files processed")
    print(f"Total: {total_ann} annotations | Labelled: {total_labelled} ({rate}%) | "
          f"Unknown: {total_unknown}")
    print(f"Label distribution: {dict(list(all_labels.items())[:8])}")

    return results, all_labels


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/semantic_labeller.py <associations_path_or_dir> <output_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    if os.path.isdir(input_path):
        results, dist = label_batch(input_path, output_dir)
        print(f"\nFull label distribution:")
        for lbl, cnt in dist.items():
            desc = LABEL_DESCRIPTIONS.get(lbl, lbl)
            bar  = "█" * min(cnt, 40)
            print(f"  {desc:<22} {cnt:4d}  {bar}")
    else:
        result = label_file(input_path, output_dir)
        if result:
            print(f"\nDimension table:")
            print(f"  {'Text':<20} {'Label':<18} {'Direction'}")
            print(f"  {'─'*20} {'─'*18} {'─'*12}")
            for d in result["dimension_table"]:
                print(f"  {d['text']:<20} {d['description']:<18} {d['direction'] or '—'}")
