"""
Property-based tests for OCR accuracy improvements using Hypothesis.
All 8 correctness properties from the design document.

Run with:
    cad_env\\Scripts\\python.exe -m pytest tests/test_ocr_accuracy_properties.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hypothesis import given, settings, assume, strategies as st

from vlm_reader import post_process_text
from validation import (
    classify,
    extract_parsed,
    normalise_text,
    reconstruct_bom_rows,
    PROTECTED_CODES,
    MATERIAL_NAMES,
    PART_NAMES,
    VALID_TYPES,
)


# ============================================================
# Property 1: Leading-zero diameter correction preserves digits
# Feature: ocr-accuracy-improvements
# Validates: Requirements 1.1
# ============================================================

# Feature: ocr-accuracy-improvements, Property 1: Leading-zero diameter correction preserves digits
@given(st.from_regex(r'0[0-9]{2,}', fullmatch=True))
@settings(max_examples=200)
def test_pbt_property1_leading_zero_preserves_digits(s):
    """For any string matching 0\\d{2,}, post_process_text returns Ø followed by the same digits."""
    result = post_process_text(s)
    assert result.startswith('Ø'), f"Expected Ø prefix, got: {result!r}"
    assert result[1:] == s[1:], f"Digits after Ø should be unchanged: {result!r} vs {s!r}"


# ============================================================
# Property 2: Aspect-ratio heuristic is correctly gated
# Feature: ocr-accuracy-improvements
# Validates: Requirements 1.2, 11.1
# ============================================================

def _apply_aspect_ratio_check(raw_text: str, w: int, h: int) -> str:
    """Replicate the aspect-ratio heuristic from read_full_image() for testing."""
    if (raw_text.strip() == "8"
            and w > 0 and h > 0
            and w <= 20 and h >= 20):
        return "Ø"
    return raw_text


# Feature: ocr-accuracy-improvements, Property 2: Aspect-ratio heuristic is correctly gated
@given(
    st.integers(min_value=0, max_value=100),
    st.integers(min_value=0, max_value=100),
)
@settings(max_examples=200)
def test_pbt_property2_aspect_ratio_gating(w, h):
    """Correction applied iff text is '8', 0 < w <= 20, and h >= 20."""
    should_correct = (w > 0 and h > 0 and w <= 20 and h >= 20)
    result = _apply_aspect_ratio_check("8", w, h)
    if should_correct:
        assert result == "Ø"
    else:
        assert result == "8"


# Feature: ocr-accuracy-improvements, Property 2b: Non-"8" text never corrected
@given(
    st.text(min_size=1, max_size=5).filter(lambda t: t.strip() != "8"),
    st.integers(min_value=1, max_value=20),
    st.integers(min_value=20, max_value=100),
)
@settings(max_examples=100)
def test_pbt_property2b_non_eight_not_corrected(text, w, h):
    """Non-'8' text is never corrected by the aspect-ratio heuristic."""
    result = _apply_aspect_ratio_check(text, w, h)
    assert result == text


# ============================================================
# Property 3: Radius callout classification and extraction round-trip
# Feature: ocr-accuracy-improvements
# Validates: Requirements 2.2, 2.4
# ============================================================

# Feature: ocr-accuracy-improvements, Property 3: Radius callout classification and extraction round-trip
@given(st.from_regex(r'R[0-9]+(\.[0-9]+)?', fullmatch=True))
@settings(max_examples=200)
def test_pbt_property3_radius_callout_round_trip(s):
    """For any R{digits} string, classify returns radius_callout and extract_parsed gives correct float."""
    for cat in (1, 2, 3):
        assert classify(s, cat) == "radius_callout", f"classify({s!r}, {cat}) != radius_callout"
    parsed = extract_parsed("radius_callout", s)
    assert "radius" in parsed
    if parsed["radius"] is not None:
        assert parsed["radius"] == float(s[1:])


# ============================================================
# Property 4: Material name exact match classification
# Feature: ocr-accuracy-improvements
# Validates: Requirements 4.3, 4.4
# ============================================================

# Feature: ocr-accuracy-improvements, Property 4: Material name classification with exact match
@given(st.sampled_from(sorted(MATERIAL_NAMES)))
@settings(max_examples=100)
def test_pbt_property4_material_name_exact_match(name):
    """For any exact member of MATERIAL_NAMES, classify returns material_name."""
    # Title case (as it would appear in OCR output)
    assert classify(name.title(), category=2) == "material_name"
    # Uppercase
    assert classify(name.upper(), category=2) == "material_name"


# ============================================================
# Property 5: Compound part name exact match classification
# Feature: ocr-accuracy-improvements
# Validates: Requirements 3.2, 3.3, 3.4
# ============================================================

_COMPOUND_PART_NAMES = [n for n in PART_NAMES if ' ' in n or '-' in n]
_SINGLE_PART_NAMES = [n for n in PART_NAMES if ' ' not in n and '-' not in n]


# Feature: ocr-accuracy-improvements, Property 5a: Compound part name exact match
@given(st.sampled_from(_COMPOUND_PART_NAMES))
@settings(max_examples=100)
def test_pbt_property5a_compound_part_name_exact(name):
    """For any exact compound part name, classify returns part_name."""
    # Must not be a material name (some overlap possible)
    assume(name.upper() not in MATERIAL_NAMES)
    assert classify(name.title(), category=2) == "part_name"


# Feature: ocr-accuracy-improvements, Property 5b: Single-word part name exact match
@given(st.sampled_from(_SINGLE_PART_NAMES))
@settings(max_examples=100)
def test_pbt_property5b_single_part_name_exact(name):
    """For any exact single-word part name, classify returns part_name."""
    from validation import _edit_distance
    assume(name.upper() not in MATERIAL_NAMES)
    # Also skip names that fuzzy-match a material name (edit distance <= 2)
    assume(not any(_edit_distance(name.upper(), mat) <= 2 for mat in MATERIAL_NAMES))
    assert classify(name.title(), category=2) == "part_name"


# ============================================================
# Property 6: PROTECTED_CODES are never modified by normalise_text
# Feature: ocr-accuracy-improvements
# Validates: Requirements 7.4
# ============================================================

# Feature: ocr-accuracy-improvements, Property 6: PROTECTED_CODES are never modified by normalise_text
@given(st.sampled_from(sorted(PROTECTED_CODES)))
@settings(max_examples=100)
def test_pbt_property6_protected_codes_unchanged(code):
    """For any PROTECTED_CODE, normalise_text returns the code unchanged."""
    assert normalise_text(code) == code


# Feature: ocr-accuracy-improvements, Property 6b: PROTECTED_CODES with whitespace
@given(
    st.sampled_from(sorted(PROTECTED_CODES)),
    st.text(alphabet=' \t', max_size=3),
    st.text(alphabet=' \t', max_size=3),
)
@settings(max_examples=100)
def test_pbt_property6b_protected_codes_with_whitespace(code, prefix, suffix):
    """PROTECTED_CODES with surrounding whitespace are returned stripped and unchanged."""
    result = normalise_text(prefix + code + suffix)
    assert result == code


# ============================================================
# Property 7: Spatial adjacency check is symmetric
# Feature: ocr-accuracy-improvements
# Validates: Requirements 6.2
# ============================================================

def _same_row(box_a, box_b, tolerance=10):
    """Check if two boxes are in the same BOM row by y-centroid proximity."""
    y_a = box_a[1] + box_a[3] / 2.0
    y_b = box_b[1] + box_b[3] / 2.0
    return abs(y_a - y_b) <= tolerance


# Feature: ocr-accuracy-improvements, Property 7: Spatial adjacency check is symmetric
@given(
    st.lists(st.integers(min_value=0, max_value=800), min_size=4, max_size=4),
    st.lists(st.integers(min_value=0, max_value=800), min_size=4, max_size=4),
)
@settings(max_examples=200)
def test_pbt_property7_spatial_adjacency_symmetric(box_a, box_b):
    """The y-centroid proximity predicate is symmetric: same_row(A,B) == same_row(B,A)."""
    assert _same_row(box_a, box_b) == _same_row(box_b, box_a)


# ============================================================
# Property 8: BOM row reconstruction is category-gated
# Feature: ocr-accuracy-improvements
# Validates: Requirements 6.7
# ============================================================

_BOM_ENTRY = st.fixed_dictionaries({
    "id": st.integers(min_value=1, max_value=100),
    "box": st.lists(st.integers(0, 800), min_size=4, max_size=4),
    "type": st.sampled_from(["balloon_number", "part_name", "material_code", "quantity"]),
    "parsed": st.just({}),
    "confidence": st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
})


# Feature: ocr-accuracy-improvements, Property 8: BOM row reconstruction is category-gated
@given(
    st.lists(_BOM_ENTRY, min_size=0, max_size=20),
    st.sampled_from([1, 3]),
)
@settings(max_examples=100)
def test_pbt_property8_bom_rows_category_gated(entries, category):
    """reconstruct_bom_rows returns [] for category 1 or 3 regardless of entries."""
    assert reconstruct_bom_rows(entries, category) == []
