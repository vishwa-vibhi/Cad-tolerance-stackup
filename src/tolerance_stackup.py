"""
Tolerance Stack-Up Analysis — Stage 4.5

Parses structured engineering annotations into numerical tolerance data
and computes basic tolerance stack-up chains.

This is the core deliverable of the project name:
"CAD Tolerance Stack-Up Analysis Tool"

Inputs:  _structured.json from Stage 3
Outputs: _stackup.json with:
  - parsed tolerances (nominal, upper, lower)
  - fit specifications (H7/h6 → numerical limits)
  - dimension chains (linked dimensions)
  - cumulative tolerance estimate

Usage:
    python src/tolerance_stackup.py <structured_path> <output_dir>
    from src.tolerance_stackup import analyse_file, analyse_batch
"""

import os
import sys
import re
import json
import math
import pathlib


# ============================================================
# ISO Tolerance Tables (simplified subset for common fits)
# Based on ISO 286-1 fundamental deviations
# ============================================================

# Fundamental deviations (microns) for common hole tolerances (uppercase)
# Format: {letter: {IT_grade: (lower_dev_um, upper_dev_um)}}
# Positive = above zero line, negative = below
HOLE_DEVIATIONS = {
    # H series (zero lower deviation — most common)
    'H': {
        5:  (0, 11),   6:  (0, 16),   7:  (0, 25),
        8:  (0, 39),   9:  (0, 62),   10: (0, 100),
        11: (0, 160),  12: (0, 250),
    },
    # F series
    'F': {
        7:  (20, 45),  8:  (20, 59),
    },
    # G series
    'G': {
        7:  (12, 37),
    },
    # JS series (symmetric)
    'JS': {
        7:  (-12, 12),
    },
    # K series
    'K': {
        7:  (-18, 7),
    },
    # N series
    'N': {
        7:  (-29, -4),
    },
    # P series
    'P': {
        7:  (-42, -17),
    },
}

# Shaft deviations (microns) for common shaft tolerances (lowercase)
SHAFT_DEVIATIONS = {
    # h series (zero upper deviation)
    'h': {
        5:  (-11, 0),  6:  (-16, 0),  7:  (-25, 0),
        8:  (-39, 0),  9:  (-62, 0),  10: (-100, 0),
        11: (-160, 0), 12: (-250, 0),
    },
    # f series
    'f': {
        7:  (-45, -20), 8:  (-59, -20),
    },
    # g series
    'g': {
        7:  (-37, -12),
    },
    # js series (symmetric)
    'js': {
        7:  (-12, 12),
    },
    # k series
    'k': {
        7:  (-7, 18),
    },
    # n series
    'n': {
        7:  (4, 29),
    },
    # p series
    'p': {
        7:  (17, 42),
    },
    # e series
    'e': {
        8:  (-89, -50),
    },
    # d series
    'd': {
        9:  (-142, -80),
    },
}

# IT grade tolerances (microns) for nominal sizes 18-30mm (most common range)
# In practice these vary by size range; this is a simplified table
IT_GRADES_18_30 = {
    5: 9, 6: 13, 7: 21, 8: 33, 9: 52, 10: 84, 11: 130, 12: 210
}


# ============================================================
# Regex patterns for tolerance parsing
# ============================================================

RE_PLUS_MINUS = re.compile(
    r'[±]\s*(\d+(?:\.\d+)?)',
    re.IGNORECASE
)

RE_BILATERAL = re.compile(
    r'\+\s*(\d+(?:\.\d+)?)\s*/\s*-\s*(\d+(?:\.\d+)?)',
    re.IGNORECASE
)

RE_UNILATERAL_PLUS = re.compile(
    r'\+\s*(\d+(?:\.\d+)?)\s*/\s*0',
    re.IGNORECASE
)

RE_UNILATERAL_MINUS = re.compile(
    r'0\s*/\s*-\s*(\d+(?:\.\d+)?)',
    re.IGNORECASE
)

RE_FIT = re.compile(
    r'([A-Z]{1,2})(\d+)\s*/\s*([a-z]{1,2})(\d+)',
)

RE_FIT_HOLE_ONLY = re.compile(
    r'^([A-Z]{1,2})(\d+)$'
)

RE_FIT_SHAFT_ONLY = re.compile(
    r'^([a-z]{1,2})(\d+)$'
)

RE_DIMENSION = re.compile(
    r'^(\d+(?:\.\d+)?)$'
)

RE_DIAMETER_DIM = re.compile(
    r'^Ø\s*(\d+(?:\.\d+)?)'
)

RE_RADIUS_DIM = re.compile(
    r'^R\s*(\d+(?:\.\d+)?)'
)

RE_THREAD = re.compile(
    r'^M(\d+(?:\.\d+)?)\s*[×x]\s*(\d+(?:\.\d+)?)'
)


# ============================================================
# Tolerance parsing functions
# ============================================================

def parse_tolerance(text):
    """
    Parse a tolerance string into numerical upper/lower deviations.

    Args:
        text: Tolerance string e.g. "±0.5", "+0.12/-0.00", "H7/h6", "H7"

    Returns:
        dict with keys: type, upper_dev, lower_dev, nominal_dev, fit_type
        or None if not parseable.
    """
    t = text.strip()

    # ±0.5 symmetric tolerance
    m = RE_PLUS_MINUS.match(t)
    if m:
        val = float(m.group(1))
        return {
            "type": "symmetric",
            "upper_dev": val,
            "lower_dev": -val,
            "nominal_dev": 0.0,
            "fit_type": None,
        }

    # +0.12/-0.00 bilateral
    m = RE_BILATERAL.match(t)
    if m:
        upper = float(m.group(1))
        lower = -float(m.group(2))
        return {
            "type": "bilateral",
            "upper_dev": upper,
            "lower_dev": lower,
            "nominal_dev": (upper + lower) / 2,
            "fit_type": None,
        }

    # +0.05/0 unilateral plus
    m = RE_UNILATERAL_PLUS.match(t)
    if m:
        upper = float(m.group(1))
        return {
            "type": "unilateral_plus",
            "upper_dev": upper,
            "lower_dev": 0.0,
            "nominal_dev": upper / 2,
            "fit_type": None,
        }

    # 0/-0.05 unilateral minus
    m = RE_UNILATERAL_MINUS.match(t)
    if m:
        lower = -float(m.group(1))
        return {
            "type": "unilateral_minus",
            "upper_dev": 0.0,
            "lower_dev": lower,
            "nominal_dev": lower / 2,
            "fit_type": None,
        }

    # H7/h6 fit specification
    m = RE_FIT.match(t)
    if m:
        hole_letter = m.group(1)
        hole_grade  = int(m.group(2))
        shaft_letter = m.group(3)
        shaft_grade  = int(m.group(4))
        hole_dev  = _lookup_deviation(hole_letter,  hole_grade,  HOLE_DEVIATIONS)
        shaft_dev = _lookup_deviation(shaft_letter, shaft_grade, SHAFT_DEVIATIONS)
        return {
            "type": "fit",
            "fit_type": f"{hole_letter}{hole_grade}/{shaft_letter}{shaft_grade}",
            "hole": {
                "letter": hole_letter, "grade": hole_grade,
                "upper_dev_um": hole_dev[1] if hole_dev else None,
                "lower_dev_um": hole_dev[0] if hole_dev else None,
            },
            "shaft": {
                "letter": shaft_letter, "grade": shaft_grade,
                "upper_dev_um": shaft_dev[1] if shaft_dev else None,
                "lower_dev_um": shaft_dev[0] if shaft_dev else None,
            },
            "upper_dev": hole_dev[1] / 1000.0 if hole_dev else None,
            "lower_dev": shaft_dev[0] / 1000.0 if shaft_dev else None,
            "nominal_dev": 0.0,
        }

    # H7 hole-only
    m = RE_FIT_HOLE_ONLY.match(t)
    if m:
        letter = m.group(1)
        grade  = int(m.group(2))
        dev = _lookup_deviation(letter, grade, HOLE_DEVIATIONS)
        if dev:
            return {
                "type": "hole_fit",
                "fit_type": f"{letter}{grade}",
                "upper_dev": dev[1] / 1000.0,
                "lower_dev": dev[0] / 1000.0,
                "nominal_dev": 0.0,
            }

    # h6 shaft-only
    m = RE_FIT_SHAFT_ONLY.match(t)
    if m:
        letter = m.group(1)
        grade  = int(m.group(2))
        dev = _lookup_deviation(letter, grade, SHAFT_DEVIATIONS)
        if dev:
            return {
                "type": "shaft_fit",
                "fit_type": f"{letter}{grade}",
                "upper_dev": dev[1] / 1000.0,
                "lower_dev": dev[0] / 1000.0,
                "nominal_dev": 0.0,
            }

    return None


def _lookup_deviation(letter, grade, table):
    """Look up deviation from ISO table. Returns (lower_um, upper_um) or None."""
    if letter in table and grade in table[letter]:
        return table[letter][grade]
    return None


def parse_dimension(text):
    """
    Parse a dimension annotation into a numerical value with type.

    Returns dict with: value, unit, dim_type (linear/diameter/radius/thread)
    or None if not parseable.
    """
    t = text.strip()

    m = RE_DIAMETER_DIM.match(t)
    if m:
        return {"value": float(m.group(1)), "unit": "mm", "dim_type": "diameter"}

    m = RE_RADIUS_DIM.match(t)
    if m:
        return {"value": float(m.group(1)), "unit": "mm", "dim_type": "radius"}

    m = RE_THREAD.match(t)
    if m:
        return {
            "value": float(m.group(1)),
            "pitch": float(m.group(2)),
            "unit": "mm",
            "dim_type": "thread",
        }

    m = RE_DIMENSION.match(t)
    if m:
        return {"value": float(m.group(1)), "unit": "mm", "dim_type": "linear"}

    return None


# ============================================================
# Tolerance stack-up computation
# ============================================================

def compute_stackup(dimension_chain):
    """
    Compute worst-case and RSS tolerance stack-up for a chain of dimensions.

    Args:
        dimension_chain: List of dicts, each with:
            - nominal: float (nominal dimension value in mm)
            - upper_dev: float (upper deviation in mm, positive)
            - lower_dev: float (lower deviation in mm, negative)

    Returns:
        dict with:
            - total_nominal: sum of nominal dimensions
            - worst_case_upper: worst-case upper limit
            - worst_case_lower: worst-case lower limit
            - worst_case_tolerance: total worst-case tolerance band
            - rss_tolerance: root-sum-square tolerance (statistical)
            - n_dimensions: number of dimensions in chain
    """
    if not dimension_chain:
        return None

    total_nominal = sum(d.get("nominal", 0) for d in dimension_chain)
    total_upper = sum(d.get("upper_dev", 0) for d in dimension_chain)
    total_lower = sum(d.get("lower_dev", 0) for d in dimension_chain)

    # Worst-case: add all tolerances
    wc_tolerance = total_upper - total_lower

    # RSS: root sum of squares of individual tolerance bands
    rss = math.sqrt(sum(
        ((d.get("upper_dev", 0) - d.get("lower_dev", 0)) ** 2)
        for d in dimension_chain
    ))

    return {
        "total_nominal_mm":      round(total_nominal, 4),
        "worst_case_upper_mm":   round(total_nominal + total_upper, 4),
        "worst_case_lower_mm":   round(total_nominal + total_lower, 4),
        "worst_case_tolerance_mm": round(wc_tolerance, 4),
        "rss_tolerance_mm":      round(rss, 4),
        "n_dimensions":          len(dimension_chain),
    }


# ============================================================
# Main analysis function
# ============================================================

def analyse_file(structured_path, output_dir):
    """
    Analyse a _structured.json file for tolerance stack-up.

    Parses all dimension and tolerance annotations, computes stack-up
    for dimension chains, and writes _stackup.json.

    Args:
        structured_path: Path to _structured.json from Stage 3.
        output_dir:      Directory to write _stackup.json.

    Returns:
        Stack-up result dict, or None on error.
    """
    structured_path = str(structured_path)
    source_name = os.path.basename(structured_path)

    try:
        with open(structured_path, 'r', encoding='utf-8') as f:
            structured = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ERROR: cannot load '{structured_path}': {e}")
        return None

    classified = structured.get("classified", [])
    category   = structured.get("image_category", 0)

    # Parse all dimension and tolerance annotations
    parsed_dims = []
    parsed_tols = []
    parsed_fits = []

    for entry in classified:
        ann_type = entry.get("type", "unknown")
        text     = entry.get("text", "")
        conf     = entry.get("confidence", 0)
        box      = entry.get("box", [])

        if ann_type in ("dimension_value", "diameter_callout", "radius_callout"):
            dim = parse_dimension(text)
            if dim:
                parsed_dims.append({
                    "id":         entry.get("id"),
                    "text":       text,
                    "type":       ann_type,
                    "box":        box,
                    "confidence": conf,
                    "nominal":    dim["value"],
                    "dim_type":   dim.get("dim_type", "linear"),
                    "upper_dev":  0.0,   # no tolerance yet — will be linked
                    "lower_dev":  0.0,
                })

        elif ann_type == "thread_spec":
            dim = parse_dimension(text)
            if dim:
                parsed_dims.append({
                    "id":         entry.get("id"),
                    "text":       text,
                    "type":       ann_type,
                    "box":        box,
                    "confidence": conf,
                    "nominal":    dim["value"],
                    "dim_type":   "thread",
                    "pitch":      dim.get("pitch"),
                    "upper_dev":  0.0,
                    "lower_dev":  0.0,
                })

        elif ann_type == "tolerance":
            tol = parse_tolerance(text)
            if tol:
                parsed_tols.append({
                    "id":         entry.get("id"),
                    "text":       text,
                    "box":        box,
                    "confidence": conf,
                    **tol,
                })
                if tol.get("type") == "fit":
                    parsed_fits.append({
                        "id":       entry.get("id"),
                        "text":     text,
                        "fit_type": tol.get("fit_type"),
                        "hole":     tol.get("hole"),
                        "shaft":    tol.get("shaft"),
                    })

    # Link tolerances to nearest dimension (by spatial proximity)
    def _box_center(box):
        if not box or len(box) < 4:
            return None
        return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)

    def _dist(c1, c2):
        if c1 is None or c2 is None:
            return float('inf')
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    for tol in parsed_tols:
        tol_center = _box_center(tol.get("box"))
        best_dim = None
        best_dist = 200.0  # max linking distance in pixels

        for dim in parsed_dims:
            d = _dist(tol_center, _box_center(dim.get("box")))
            if d < best_dist:
                best_dist = d
                best_dim = dim

        if best_dim is not None:
            best_dim["upper_dev"] = tol.get("upper_dev", 0) or 0
            best_dim["lower_dev"] = tol.get("lower_dev", 0) or 0
            best_dim["linked_tolerance"] = tol.get("text")
            best_dim["tolerance_type"]   = tol.get("type")

    # Compute stack-up for all linear dimensions (simple chain)
    linear_dims = [d for d in parsed_dims if d.get("dim_type") == "linear"]
    diameter_dims = [d for d in parsed_dims if d.get("dim_type") == "diameter"]

    stackup_linear   = compute_stackup(linear_dims)   if linear_dims   else None
    stackup_diameter = compute_stackup(diameter_dims) if diameter_dims else None

    # Summary statistics
    tol_values = [abs(t.get("upper_dev", 0)) for t in parsed_tols if t.get("upper_dev")]
    fit_types  = list({f["fit_type"] for f in parsed_fits if f.get("fit_type")})

    result = {
        "source_structured":    source_name,
        "image_category":       category,
        "summary": {
            "total_dimensions":  len(parsed_dims),
            "total_tolerances":  len(parsed_tols),
            "total_fits":        len(parsed_fits),
            "fit_types_found":   fit_types,
            "mean_tolerance_mm": round(sum(tol_values) / len(tol_values), 4) if tol_values else None,
            "max_tolerance_mm":  round(max(tol_values), 4) if tol_values else None,
            "min_tolerance_mm":  round(min(tol_values), 4) if tol_values else None,
        },
        "dimensions":           parsed_dims,
        "tolerances":           parsed_tols,
        "fit_specifications":   parsed_fits,
        "stackup_linear":       stackup_linear,
        "stackup_diameter":     stackup_diameter,
    }

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    stem = source_name
    if stem.endswith("_structured.json"):
        stem = stem[: -len("_structured.json")]
    out_path = os.path.join(output_dir, f"{stem}_stackup.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    n_dims = len(parsed_dims)
    n_tols = len(parsed_tols)
    n_fits = len(parsed_fits)
    wc = stackup_linear.get("worst_case_tolerance_mm") if stackup_linear else "N/A"
    print(f"  {source_name} -> {n_dims} dims | {n_tols} tols | {n_fits} fits | "
          f"WC stack-up: {wc}mm")

    return result


def analyse_batch(structured_dir, output_dir):
    """
    Process all _structured.json files in structured_dir.

    Returns list of result dicts.
    """
    files = sorted(pathlib.Path(structured_dir).glob("*_structured.json"))
    total = len(files)
    results = []

    print(f"\n=== TOLERANCE STACK-UP ANALYSIS ===")
    print(f"Processing {total} structured files from: {structured_dir}")
    print("-" * 60)

    total_dims = total_tols = total_fits = 0
    for idx, path in enumerate(files, 1):
        print(f"  [{idx}/{total}] ", end="", flush=True)
        try:
            result = analyse_file(str(path), output_dir)
            if result:
                results.append(result)
                total_dims += result["summary"]["total_dimensions"]
                total_tols += result["summary"]["total_tolerances"]
                total_fits += result["summary"]["total_fits"]
        except Exception as e:
            print(f"  ERROR: {path.name}: {e}")

    print("-" * 60)
    print(f"Batch complete: {len(results)}/{total} images")
    print(f"Total: {total_dims} dimensions | {total_tols} tolerances | {total_fits} fits")

    return results


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/tolerance_stackup.py <structured_path_or_dir> <output_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    if os.path.isdir(input_path):
        analyse_batch(input_path, output_dir)
    else:
        result = analyse_file(input_path, output_dir)
        if result:
            s = result["summary"]
            print(f"\nResults:")
            print(f"  Dimensions:  {s['total_dimensions']}")
            print(f"  Tolerances:  {s['total_tolerances']}")
            print(f"  Fit specs:   {s['total_fits']} {s['fit_types_found']}")
            if result["stackup_linear"]:
                sk = result["stackup_linear"]
                print(f"\n  Linear stack-up ({sk['n_dimensions']} dims):")
                print(f"    Nominal:     {sk['total_nominal_mm']} mm")
                print(f"    Worst-case:  ±{sk['worst_case_tolerance_mm']/2:.4f} mm")
                print(f"    RSS:         ±{sk['rss_tolerance_mm']/2:.4f} mm")
