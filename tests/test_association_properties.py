"""
Property-based tests for src/association.py — Stage 4: Geometric Association.
All 7 correctness properties from the design document.

Run with:
    cad_env\\Scripts\\python.exe -m pytest tests/test_association_properties.py -v
"""

import sys
import os
import json
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hypothesis import given, settings, assume, strategies as st

from association import (
    distance_point_to_segment,
    distance_point_to_circle,
    distance_point_to_contour,
    _build_output,
    _associate_annotation,
)


# Reusable strategies
_floats = st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
_pos_floats = st.floats(min_value=0.1, max_value=500, allow_nan=False, allow_infinity=False)
_ints = st.integers(min_value=0, max_value=800)
_pos_ints = st.integers(min_value=1, max_value=200)


# ============================================================
# Property 1: distance_point_to_segment is always non-negative
# Feature: stage4-geometric-association
# Validates: Requirement 3.5
# ============================================================

# Feature: stage4-geometric-association, Property 1: distance_point_to_segment returns non-negative float
@given(_floats, _floats, _floats, _floats, _floats, _floats)
@settings(max_examples=500)
def test_pbt_property1_segment_distance_non_negative(px, py, x1, y1, x2, y2):
    """For any point and segment, distance_point_to_segment returns a non-negative float."""
    d = distance_point_to_segment(px, py, x1, y1, x2, y2)
    assert d >= 0.0
    assert isinstance(d, float)


# ============================================================
# Property 2: distance_point_to_circle is negative for interior points
# Feature: stage4-geometric-association
# Validates: Requirement 4.2
# ============================================================

# Feature: stage4-geometric-association, Property 2: distance_point_to_circle is negative for interior points
@given(
    _floats, _floats,   # circle center cx, cy
    _pos_floats,        # radius r > 0
    st.floats(min_value=0.0, max_value=0.99, allow_nan=False),  # fraction of radius
    st.floats(min_value=0.0, max_value=2 * math.pi, allow_nan=False),  # angle
)
@settings(max_examples=300)
def test_pbt_property2_circle_distance_negative_inside(cx, cy, r, frac, angle):
    """For any point strictly inside a circle, distance_point_to_circle returns a negative value."""
    # Generate point strictly inside circle using polar coordinates
    dist = frac * r  # dist < r always
    px = cx + dist * math.cos(angle)
    py = cy + dist * math.sin(angle)
    d = distance_point_to_circle(px, py, cx, cy, r)
    assert d <= 0.0  # <= 0 (exactly 0 when frac==0, i.e. at center)


# ============================================================
# Property 3: distance_point_to_contour is always non-negative
# Feature: stage4-geometric-association
# Validates: Requirement 5.4
# ============================================================

# Feature: stage4-geometric-association, Property 3: distance_point_to_contour returns non-negative float
@given(_floats, _floats, _floats, _floats, _pos_floats, _pos_floats)
@settings(max_examples=500)
def test_pbt_property3_contour_distance_non_negative(px, py, bx, by, bw, bh):
    """For any point and bounding box, distance_point_to_contour returns a non-negative float."""
    d = distance_point_to_contour(px, py, bx, by, bw, bh)
    assert d >= 0.0
    assert isinstance(d, float)


# ============================================================
# Property 4: distance_point_to_contour is zero for interior points
# Feature: stage4-geometric-association
# Validates: Requirement 5.2
# ============================================================

# Feature: stage4-geometric-association, Property 4: distance_point_to_contour is 0.0 for interior points
@given(
    _floats, _floats,   # box origin bx, by
    _pos_floats,        # box width bw
    _pos_floats,        # box height bh
)
@settings(max_examples=300)
def test_pbt_property4_contour_distance_zero_inside(bx, by, bw, bh):
    """For any point strictly inside a bounding box, distance_point_to_contour returns 0.0."""
    # Generate a point strictly inside the box
    px = bx + bw / 2.0
    py = by + bh / 2.0
    d = distance_point_to_contour(px, py, bx, by, bw, bh)
    assert d == pytest.approx(0.0, abs=1e-9)


# ============================================================
# Property 5: Output JSON round-trip
# Feature: stage4-geometric-association
# Validates: Requirement 8.9
# ============================================================

_assoc_record = st.one_of(
    # Matched record
    st.fixed_dictionaries({
        "annotation_id":   st.integers(1, 100),
        "annotation_text": st.text(max_size=20),
        "annotation_type": st.sampled_from(["dimension_value", "thread_spec", "circle"]),
        "annotation_box":  st.lists(st.integers(0, 800), min_size=4, max_size=4),
        "associated_element": st.fixed_dictionaries({
            "element_type": st.sampled_from(["line_horizontal", "circle", "contour"]),
            "element_data": st.lists(st.integers(0, 800), min_size=3, max_size=4),
            "distance_px":  st.floats(min_value=-50, max_value=200,
                                      allow_nan=False, allow_infinity=False),
        }),
    }),
    # Unassociated record
    st.fixed_dictionaries({
        "annotation_id":   st.integers(1, 100),
        "annotation_text": st.text(max_size=20),
        "annotation_type": st.just("unknown"),
        "annotation_box":  st.lists(st.integers(0, 800), min_size=4, max_size=4),
        "associated_element": st.none(),
    }),
)


# Feature: stage4-geometric-association, Property 5: association output dict survives JSON round-trip
@given(st.lists(_assoc_record, min_size=0, max_size=10))
@settings(max_examples=100)
def test_pbt_property5_json_round_trip(associations):
    """For any list of association records, json.loads(json.dumps(result)) == result."""
    structured = {"image_category": 1, "classified": associations}
    result = _build_output(structured, associations, "test_structured.json")
    serialised = json.dumps(result, ensure_ascii=False)
    deserialised = json.loads(serialised)
    assert deserialised == result


# ============================================================
# Property 6: Count invariant — total == matched + unassociated
# Feature: stage4-geometric-association
# Validates: Requirement 8 (count consistency)
# ============================================================

# Feature: stage4-geometric-association, Property 6: total_annotations == matched + unassociated
@given(st.lists(_assoc_record, min_size=0, max_size=20))
@settings(max_examples=200)
def test_pbt_property6_count_invariant(associations):
    """total_annotations == matched + unassociated for any list of association records."""
    structured = {"image_category": 1, "classified": associations}
    result = _build_output(structured, associations, "test_structured.json")
    assert result["total_annotations"] == result["matched"] + result["unassociated"]
    assert result["total_annotations"] == len(associations)


# ============================================================
# Property 7: One record per annotation
# Feature: stage4-geometric-association
# Validates: Requirement 7.3
# ============================================================

def _make_annotation(id_, type_, box):
    return {"id": id_, "type": type_, "text": f"t{id_}", "box": box, "parsed": {}}


def _make_elements_empty():
    return {
        "lines": {"horizontal": [], "vertical": [], "diagonal": [], "total": 0},
        "circles": [], "contours": [], "text_regions": [], "hatching": [],
    }


# Feature: stage4-geometric-association, Property 7: associations list has exactly one record per annotation
@given(st.integers(min_value=0, max_value=20))
@settings(max_examples=100)
def test_pbt_property7_one_record_per_annotation(n):
    """For N input annotations, the associations list has exactly N records."""
    annotations = [
        _make_annotation(i, "dimension_value", [i * 10, 50, 20, 10])
        for i in range(n)
    ]
    elements = _make_elements_empty()
    associations = [
        _associate_annotation(ann, elements, category=1, max_distance=150)
        for ann in annotations
    ]
    assert len(associations) == n
