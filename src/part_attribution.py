"""
Part Attribution — links dimension annotations to named parts.

For Category 2 assembly drawings:
  - Builds a BOM lookup table from bom_rows in _structured.json
  - For each dimension annotation, finds the nearest balloon number
  - Looks up that balloon's part_no in the BOM to get part_name + material
  - Outputs an enriched _attributed.json with part_attribution per annotation

For Category 1 (single-part drawings):
  - All dimensions belong to the single part shown
  - Part name is inferred from the drawing filename or set to "Part"

For Category 3 (assembly views, no BOM):
  - Balloon numbers are linked to circles but no BOM exists
  - Attribution is limited to balloon number only

Usage (CLI):
    python src/part_attribution.py <structured_path> <output_dir>

Usage (API):
    from src.part_attribution import attribute_file, attribute_batch
"""

import os
import sys
import json
import math
import pathlib


# ============================================================
# Constants
# ============================================================

MAX_BALLOON_SEARCH_PX = 300   # max distance to search for a nearby balloon
DIMENSION_TYPES = {
    'dimension_value', 'diameter_callout', 'radius_callout',
    'thread_spec', 'tolerance', 'dimension_with_note', 'hole_callout'
}


# ============================================================
# Helpers
# ============================================================

def _center(box):
    """Return (cx, cy) for a [x, y, w, h] box."""
    if not box or len(box) < 4:
        return None
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)


def _dist(p1, p2):
    """Euclidean distance between two (x, y) points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _build_bom_lookup(bom_rows):
    """
    Build a dict mapping part_no → {part_name, material, qty} from bom_rows.
    When multiple rows have the same part_no, prefer the one with more fields filled.
    """
    lookup = {}
    for row in bom_rows:
        pno = row.get("part_no")
        if pno is None:
            continue
        existing = lookup.get(pno)
        # Score = number of non-null fields
        score = sum(1 for v in row.values() if v is not None)
        if existing is None or score > existing.get("_score", 0):
            lookup[pno] = {
                "part_no":   pno,
                "part_name": row.get("part_name"),
                "material":  row.get("material"),
                "qty":       row.get("qty"),
                "_score":    score,
            }
    # Remove internal score field
    for v in lookup.values():
        v.pop("_score", None)
    return lookup


def _find_nearest_balloon(ann_center, balloon_entries, max_distance):
    """
    Find the nearest balloon_number annotation to ann_center.

    Args:
        ann_center:       (cx, cy) of the dimension annotation
        balloon_entries:  list of classified entries with type == 'balloon_number'
        max_distance:     max search radius in pixels

    Returns:
        (balloon_entry, distance) or (None, None)
    """
    best_entry = None
    best_dist = max_distance + 1.0

    for entry in balloon_entries:
        c = _center(entry.get("box"))
        if c is None:
            continue
        d = _dist(ann_center, c)
        if d < best_dist:
            best_dist = d
            best_entry = entry

    if best_entry is None:
        return None, None
    return best_entry, best_dist


# ============================================================
# Core attribution logic
# ============================================================

def _attribute_cat1(classified, filename):
    """
    Category 1: all dimensions belong to the single part shown.
    Part name is derived from the filename (e.g., cad1_001 → 'Part 001').
    """
    # Extract part identifier from filename
    basename = os.path.splitext(os.path.basename(filename))[0]
    # e.g. cad1_001_structured → 001
    parts = basename.replace("_structured", "").split("_")
    part_id = parts[-1] if parts else "001"
    part_name = f"Part {part_id}"

    attributions = []
    for entry in classified:
        ann_type = entry.get("type", "unknown")
        attribution = None

        if ann_type in DIMENSION_TYPES:
            attribution = {
                "part_no":    None,
                "part_name":  part_name,
                "material":   None,
                "confidence": "high",   # single-part drawing — trivially correct
                "method":     "single_part",
                "balloon_distance_px": None,
            }

        attributions.append({
            "annotation_id":   entry.get("id"),
            "annotation_text": entry.get("text", ""),
            "annotation_type": ann_type,
            "annotation_box":  entry.get("box", []),
            "part_attribution": attribution,
        })

    return attributions


def _attribute_cat2(classified, bom_rows, max_distance):
    """
    Category 2: link each dimension to the nearest balloon, then look up BOM.
    Multi-strategy fallback chain for robust attribution.
    """
    bom_lookup = _build_bom_lookup(bom_rows)

    # Build direct text → BOM row lookup for when balloon lookup fails
    # Key: normalised part_name text → bom row
    text_to_bom = {}
    for row in bom_rows:
        if row.get("part_name"):
            key = row["part_name"].upper().strip()
            if key not in text_to_bom:
                text_to_bom[key] = row

    balloon_entries    = [e for e in classified if e.get("type") == "balloon_number"]
    part_name_entries  = [e for e in classified if e.get("type") == "part_name"]
    material_entries   = [e for e in classified
                          if e.get("type") in ("material_code", "material_name")]

    attributions = []
    for entry in classified:
        ann_type = entry.get("type", "unknown")
        attribution = None

        if ann_type in DIMENSION_TYPES:
            ann_center = _center(entry.get("box"))
            if ann_center is not None:

                # ── Strategy 1: nearest balloon → BOM lookup ─────────────
                nearest_balloon, balloon_dist = _find_nearest_balloon(
                    ann_center, balloon_entries, max_distance
                )
                if nearest_balloon is not None:
                    part_no   = nearest_balloon.get("parsed", {}).get("number")
                    bom_entry = bom_lookup.get(part_no, {})
                    has_name  = bool(bom_entry.get("part_name"))

                    if balloon_dist < 50 and has_name:
                        confidence = "high"
                    elif balloon_dist < 150 and has_name:
                        confidence = "medium"
                    elif has_name:
                        confidence = "low"
                    else:
                        confidence = "balloon_only"

                    attribution = {
                        "part_no":             part_no,
                        "part_name":           bom_entry.get("part_name"),
                        "material":            bom_entry.get("material"),
                        "qty":                 bom_entry.get("qty"),
                        "confidence":          confidence,
                        "method":              "nearest_balloon_bom",
                        "balloon_distance_px": round(balloon_dist, 1),
                    }

                # ── Strategy 2: balloon found but BOM has no name ─────────
                # Try nearest part_name OCR entry and match to BOM
                if attribution is not None and not attribution.get("part_name"):
                    nearest_pname, pname_dist = _find_nearest_balloon(
                        ann_center, part_name_entries, max_distance * 2
                    )
                    if nearest_pname is not None:
                        raw_name  = nearest_pname.get("text", "").rstrip(":").strip()
                        bom_match = text_to_bom.get(raw_name.upper(), {})
                        attribution.update({
                            "part_name":  raw_name or attribution.get("part_name"),
                            "material":   bom_match.get("material") or attribution.get("material"),
                            "qty":        bom_match.get("qty") or attribution.get("qty"),
                            "confidence": "medium" if pname_dist < 100 else "low",
                            "method":     "balloon_bom_with_name_fallback",
                        })

                # ── Strategy 3: no balloon → nearest part_name entry ──────
                if attribution is None:
                    nearest_pname, pname_dist = _find_nearest_balloon(
                        ann_center, part_name_entries, max_distance * 2
                    )
                    if nearest_pname is not None:
                        raw_name  = nearest_pname.get("text", "").rstrip(":").strip()
                        bom_match = text_to_bom.get(raw_name.upper(), {})
                        # Also look for nearest material entry
                        nearest_mat, _ = _find_nearest_balloon(
                            ann_center, material_entries, max_distance * 2
                        )
                        mat_text = None
                        if nearest_mat:
                            mat_text = (nearest_mat.get("parsed", {}).get("code") or
                                        nearest_mat.get("parsed", {}).get("name") or
                                        nearest_mat.get("text", ""))
                        attribution = {
                            "part_no":             bom_match.get("part_no"),
                            "part_name":           raw_name,
                            "material":            bom_match.get("material") or mat_text,
                            "qty":                 bom_match.get("qty"),
                            "confidence":          "medium" if pname_dist < 100 else "low",
                            "method":              "nearest_part_name",
                            "balloon_distance_px": round(pname_dist, 1),
                        }
                    else:
                        attribution = {
                            "part_no":             None,
                            "part_name":           None,
                            "material":            None,
                            "qty":                 None,
                            "confidence":          "none",
                            "method":              "no_match",
                            "balloon_distance_px": None,
                        }

        attributions.append({
            "annotation_id":    entry.get("id"),
            "annotation_text":  entry.get("text", ""),
            "annotation_type":  ann_type,
            "annotation_box":   entry.get("box", []),
            "part_attribution": attribution,
        })

    return attributions


def _attribute_cat3(classified, max_distance):
    """
    Category 3: assembly views, no BOM.
    Link dimensions to nearest balloon. Also try nearest part_name OCR entry.
    """
    balloon_entries   = [e for e in classified if e.get("type") == "balloon_number"]
    part_name_entries = [e for e in classified if e.get("type") == "part_name"]
    material_entries  = [e for e in classified
                         if e.get("type") in ("material_code", "material_name")]

    attributions = []
    for entry in classified:
        ann_type = entry.get("type", "unknown")
        attribution = None

        if ann_type in DIMENSION_TYPES:
            ann_center = _center(entry.get("box"))
            if ann_center is not None:

                # Strategy 1: nearest balloon
                nearest_balloon, balloon_dist = _find_nearest_balloon(
                    ann_center, balloon_entries, max_distance
                )
                if nearest_balloon is not None:
                    part_no = nearest_balloon.get("parsed", {}).get("number")
                    attribution = {
                        "part_no":             part_no,
                        "part_name":           f"Part {part_no}" if part_no else None,
                        "material":            None,
                        "qty":                 None,
                        "confidence":          "balloon_only",
                        "method":              "nearest_balloon_no_bom",
                        "balloon_distance_px": round(balloon_dist, 1),
                    }

                # Strategy 2: also try nearest part_name OCR text
                nearest_pname, pname_dist = _find_nearest_balloon(
                    ann_center, part_name_entries, max_distance * 2
                )
                if nearest_pname is not None:
                    raw_name = nearest_pname.get("text", "").rstrip(":").strip()
                    nearest_mat, _ = _find_nearest_balloon(
                        ann_center, material_entries, max_distance * 2
                    )
                    mat_text = None
                    if nearest_mat:
                        mat_text = (nearest_mat.get("parsed", {}).get("code") or
                                    nearest_mat.get("parsed", {}).get("name") or
                                    nearest_mat.get("text", ""))
                    if attribution is not None:
                        # Enrich existing balloon attribution with name
                        attribution["part_name"] = raw_name
                        attribution["material"]  = mat_text
                        attribution["confidence"] = "medium" if pname_dist < 100 else "low"
                        attribution["method"]    = "balloon_with_part_name"
                    else:
                        attribution = {
                            "part_no":             None,
                            "part_name":           raw_name,
                            "material":            mat_text,
                            "qty":                 None,
                            "confidence":          "low",
                            "method":              "nearest_part_name_only",
                            "balloon_distance_px": round(pname_dist, 1),
                        }

        attributions.append({
            "annotation_id":    entry.get("id"),
            "annotation_text":  entry.get("text", ""),
            "annotation_type":  ann_type,
            "annotation_box":   entry.get("box", []),
            "part_attribution": attribution,
        })

    return attributions


# ============================================================
# Summary helpers
# ============================================================

def _build_summary(attributions, bom_lookup):
    """Build a human-readable summary of attributed dimensions per part."""
    per_part = {}

    for a in attributions:
        attr = a.get("part_attribution")
        if attr is None:
            continue
        ann_type = a.get("annotation_type", "unknown")
        if ann_type not in DIMENSION_TYPES:
            continue

        part_no   = attr.get("part_no")
        part_name = attr.get("part_name") or (f"Part {part_no}" if part_no else "Unknown")
        material  = attr.get("material")
        qty       = attr.get("qty")
        key       = part_no if part_no is not None else f"unnamed_{part_name}"

        if key not in per_part:
            per_part[key] = {
                "part_no":   part_no,
                "part_name": part_name,
                "material":  material,
                "qty":       qty,
                "dimensions": [],
            }
        else:
            # Update with better data if available
            if per_part[key]["part_name"] is None and part_name:
                per_part[key]["part_name"] = part_name
            if per_part[key]["material"] is None and material:
                per_part[key]["material"] = material
            if per_part[key]["qty"] is None and qty:
                per_part[key]["qty"] = qty

        per_part[key]["dimensions"].append({
            "text":       a.get("annotation_text", ""),
            "type":       ann_type,
            "confidence": attr.get("confidence", "none"),
        })

    return list(per_part.values())


# ============================================================
# Public API
# ============================================================

def attribute_file(structured_path, output_dir,
                   max_distance=MAX_BALLOON_SEARCH_PX):
    """
    Attribute dimension annotations to parts for a single image.

    Args:
        structured_path: Path to _structured.json from Stage 3.
        output_dir:      Directory to write _attributed.json.
        max_distance:    Max pixel distance to search for a nearby balloon.

    Returns:
        Attribution result dict, or None on error.
    """
    structured_path = str(structured_path)
    source_name = os.path.basename(structured_path)

    try:
        with open(structured_path, 'r', encoding='utf-8') as f:
            structured = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ERROR: cannot load '{structured_path}': {e}")
        return None

    category   = structured.get("image_category", 0)
    classified = structured.get("classified", [])
    bom_rows   = structured.get("bom_rows", [])

    # Run attribution by category
    if category == 1:
        attributions = _attribute_cat1(classified, source_name)
    elif category == 2:
        attributions = _attribute_cat2(classified, bom_rows, max_distance)
    elif category == 3:
        attributions = _attribute_cat3(classified, max_distance)
    else:
        attributions = []

    # Build BOM lookup for summary
    bom_lookup = _build_bom_lookup(bom_rows) if category == 2 else {}
    summary = _build_summary(attributions, bom_lookup)

    # Count attribution quality
    dim_attributions = [
        a for a in attributions
        if a.get("annotation_type") in DIMENSION_TYPES
        and a.get("part_attribution") is not None
    ]
    high_conf   = sum(1 for a in dim_attributions
                      if a["part_attribution"].get("confidence") == "high")
    medium_conf = sum(1 for a in dim_attributions
                      if a["part_attribution"].get("confidence") == "medium")
    named       = sum(1 for a in dim_attributions
                      if a["part_attribution"].get("part_name") is not None)

    result = {
        "source_structured":    source_name,
        "image_category":       category,
        "total_annotations":    len(attributions),
        "dimension_annotations": len(dim_attributions),
        "named_attributions":   named,
        "confidence_counts": {
            "high":         high_conf,
            "medium":       medium_conf,
            "low":          sum(1 for a in dim_attributions
                               if a["part_attribution"].get("confidence") == "low"),
            "balloon_only": sum(1 for a in dim_attributions
                               if a["part_attribution"].get("confidence") == "balloon_only"),
            "none":         sum(1 for a in dim_attributions
                               if a["part_attribution"].get("confidence") == "none"),
        },
        "parts_summary":  summary,
        "attributions":   attributions,
    }

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    stem = source_name
    if stem.endswith("_structured.json"):
        stem = stem[: -len("_structured.json")]
    out_path = os.path.join(output_dir, f"{stem}_attributed.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    named_pct = round(named / len(dim_attributions) * 100, 1) if dim_attributions else 0
    print(f"  {source_name} -> {len(dim_attributions)} dims | "
          f"named: {named}/{len(dim_attributions)} ({named_pct}%) | "
          f"high-conf: {high_conf}")

    return result


def attribute_batch(structured_dir, output_dir,
                    max_distance=MAX_BALLOON_SEARCH_PX):
    """
    Process all _structured.json files in structured_dir.

    Returns:
        List of result dicts.
    """
    structured_files = sorted(pathlib.Path(structured_dir).glob("*_structured.json"))
    total = len(structured_files)
    results = []

    print(f"\n=== PART ATTRIBUTION ===")
    print(f"Processing {total} structured files from: {structured_dir}")
    print("-" * 60)

    for idx, path in enumerate(structured_files, 1):
        print(f"  [{idx}/{total}] ", end="", flush=True)
        try:
            result = attribute_file(str(path), output_dir, max_distance)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"  ERROR processing {path.name}: {e}")

    # Aggregate stats
    total_dims   = sum(r["dimension_annotations"] for r in results)
    total_named  = sum(r["named_attributions"] for r in results)
    total_high   = sum(r["confidence_counts"]["high"] for r in results)
    named_pct    = round(total_named / total_dims * 100, 1) if total_dims else 0

    print("-" * 60)
    print(f"Batch complete: {len(results)}/{total} images processed")
    print(f"Total dimensions: {total_dims} | Named: {total_named} ({named_pct}%) | "
          f"High-confidence: {total_high}")

    return results


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/part_attribution.py <structured_path_or_dir> <output_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    if os.path.isdir(input_path):
        attribute_batch(input_path, output_dir)
    else:
        result = attribute_file(input_path, output_dir)
        if result:
            print(f"\nParts found:")
            for part in result["parts_summary"]:
                dims = [d["text"] for d in part["dimensions"]]
                print(f"  Part {part['part_no']}: {part['part_name']} "
                      f"(material: {part['material']}) — "
                      f"dimensions: {', '.join(dims)}")
