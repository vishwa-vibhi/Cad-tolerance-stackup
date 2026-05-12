"""
Unit tests for src/association.py — Stage 4: Geometric Association.

Run with:
    cad_env\\Scripts\\python.exe -m pytest tests/test_association.py -v
"""

import sys
import os
import json
import math
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from association import (
    distance_point_to_segment,
    distance_point_to_circle,
    distance_point_to_contour,
    _safe_box,
    _annotation_center,
    _associate_annotation,
    _build_output,
    associate_file,
    MAX_DISTANCE_PX,
)


# ============================================================
# Task 10.1 — distance_point_to_segment
# ============================================================

class TestDistancePointToSegment:

    def test_point_on_midpoint(self):
        # Point exactly on the midpoint of a horizontal segment
        d = distance_point_to_segment(5, 0, 0, 0, 10, 0)
        assert d == pytest.approx(0.0)

    def test_point_perpendicular_to_interior(self):
        # Point directly above midpoint of horizontal segment
        d = distance_point_to_segment(5, 3, 0, 0, 10, 0)
        assert d == pytest.approx(3.0)

    def test_point_beyond_left_endpoint(self):
        # Point to the left of segment — should return distance to left endpoint
        d = distance_point_to_segment(-3, 0, 0, 0, 10, 0)
        assert d == pytest.approx(3.0)

    def test_point_beyond_right_endpoint(self):
        # Point to the right of segment — should return distance to right endpoint
        d = distance_point_to_segment(13, 0, 0, 0, 10, 0)
        assert d == pytest.approx(3.0)

    def test_degenerate_segment(self):
        # Both endpoints are the same point
        d = distance_point_to_segment(3, 4, 0, 0, 0, 0)
        assert d == pytest.approx(5.0)

    def test_vertical_segment(self):
        # Point perpendicular to a vertical segment
        d = distance_point_to_segment(3, 5, 0, 0, 0, 10)
        assert d == pytest.approx(3.0)

    def test_always_non_negative(self):
        # Distance is always >= 0
        d = distance_point_to_segment(100, 200, 0, 0, 50, 50)
        assert d >= 0.0

    def test_diagonal_segment(self):
        # Point at (0,0), segment from (1,0) to (0,1) — nearest point is midpoint (0.5, 0.5)
        d = distance_point_to_segment(0, 0, 1, 0, 0, 1)
        assert d == pytest.approx(math.sqrt(0.5), rel=1e-5)


# ============================================================
# Task 10.2 — distance_point_to_circle
# ============================================================

class TestDistancePointToCircle:

    def test_point_at_center(self):
        # Point at center → returns -r
        d = distance_point_to_circle(0, 0, 0, 0, 10)
        assert d == pytest.approx(-10.0)

    def test_point_on_edge(self):
        # Point exactly on circle edge → returns 0
        d = distance_point_to_circle(10, 0, 0, 0, 10)
        assert d == pytest.approx(0.0)

    def test_point_outside(self):
        # Point outside circle → positive
        d = distance_point_to_circle(20, 0, 0, 0, 10)
        assert d == pytest.approx(10.0)
        assert d > 0

    def test_point_inside(self):
        # Point inside circle → negative
        d = distance_point_to_circle(5, 0, 0, 0, 10)
        assert d < 0

    def test_returns_float(self):
        d = distance_point_to_circle(3, 4, 0, 0, 5)
        assert isinstance(d, float)


# ============================================================
# Task 10.3 — distance_point_to_contour
# ============================================================

class TestDistancePointToContour:

    def test_point_inside_box(self):
        # Point inside bounding box → 0.0
        d = distance_point_to_contour(50, 50, 0, 0, 100, 100)
        assert d == pytest.approx(0.0)

    def test_point_on_edge(self):
        # Point on the right edge → 0.0
        d = distance_point_to_contour(100, 50, 0, 0, 100, 100)
        assert d == pytest.approx(0.0)

    def test_point_outside_right(self):
        # Point to the right of box
        d = distance_point_to_contour(110, 50, 0, 0, 100, 100)
        assert d == pytest.approx(10.0)

    def test_point_outside_corner(self):
        # Point at corner (110, 110) — nearest corner is (100, 100)
        d = distance_point_to_contour(110, 110, 0, 0, 100, 100)
        assert d == pytest.approx(math.sqrt(200), rel=1e-5)

    def test_always_non_negative(self):
        d = distance_point_to_contour(-50, -50, 0, 0, 100, 100)
        assert d >= 0.0


# ============================================================
# Task 10.4 — _safe_box
# ============================================================

class TestSafeBox:

    def test_valid_box(self):
        assert _safe_box([10, 20, 30, 40]) == (10, 20, 30, 40)

    def test_none_returns_none(self):
        assert _safe_box(None) is None

    def test_too_short(self):
        assert _safe_box([1, 2, 3]) is None

    def test_non_numeric(self):
        assert _safe_box(["a", "b", "c", "d"]) is None

    def test_float_values_converted(self):
        result = _safe_box([1.5, 2.5, 3.5, 4.5])
        assert result == (1, 2, 3, 4)

    def test_tuple_input(self):
        assert _safe_box((5, 10, 15, 20)) == (5, 10, 15, 20)


# ============================================================
# Task 10.5 — _associate_annotation per type
# ============================================================

def _make_annotation(id_, type_, box):
    return {"id": id_, "type": type_, "text": f"text_{id_}", "box": box, "parsed": {}}


def _make_elements(h_lines=None, v_lines=None, d_lines=None, circles=None, contours=None):
    return {
        "lines": {
            "horizontal": h_lines or [],
            "vertical":   v_lines or [],
            "diagonal":   d_lines or [],
            "total": 0,
        },
        "circles":  circles or [],
        "contours": contours or [],
        "text_regions": [],
        "hatching": [],
    }


class TestAssociateAnnotation:

    def test_dimension_value_nearest_hline(self):
        ann = _make_annotation(1, "dimension_value", [100, 50, 20, 10])
        # Horizontal line at y=60 (10px below annotation center y=55)
        elements = _make_elements(h_lines=[[50, 60, 200, 60]])
        result = _associate_annotation(ann, elements, category=1, max_distance=150)
        assert result["associated_element"] is not None
        assert result["associated_element"]["element_type"] == "line_horizontal"

    def test_dimension_value_fallback_to_contour(self):
        ann = _make_annotation(1, "dimension_value", [100, 50, 20, 10])
        # No lines, but a nearby contour
        elements = _make_elements(contours=[[80, 40, 60, 40]])
        result = _associate_annotation(ann, elements, category=1, max_distance=150)
        assert result["associated_element"] is not None
        assert result["associated_element"]["element_type"] == "contour"

    def test_hole_callout_nearest_circle(self):
        ann = _make_annotation(2, "hole_callout", [100, 100, 20, 10])
        # Circle centered at (110, 105) with radius 15
        elements = _make_elements(circles=[[110, 105, 15]])
        result = _associate_annotation(ann, elements, category=1, max_distance=150)
        assert result["associated_element"] is not None
        assert result["associated_element"]["element_type"] == "circle"

    def test_balloon_number_cat2_nearest_circle(self):
        ann = _make_annotation(3, "balloon_number", [100, 100, 10, 10])
        elements = _make_elements(circles=[[105, 105, 12]])
        result = _associate_annotation(ann, elements, category=2, max_distance=150)
        assert result["associated_element"] is not None
        assert result["associated_element"]["element_type"] == "circle"

    def test_balloon_number_cat2_no_circle_unassociated(self):
        ann = _make_annotation(3, "balloon_number", [100, 100, 10, 10])
        elements = _make_elements()  # no circles
        result = _associate_annotation(ann, elements, category=2, max_distance=150)
        assert result["associated_element"] is None

    def test_balloon_number_cat1_nearest_line(self):
        ann = _make_annotation(4, "balloon_number", [100, 50, 10, 10])
        elements = _make_elements(h_lines=[[50, 60, 200, 60]])
        result = _associate_annotation(ann, elements, category=1, max_distance=150)
        assert result["associated_element"] is not None
        assert result["associated_element"]["element_type"] == "line_horizontal"

    def test_section_marker_nearest_diagonal(self):
        ann = _make_annotation(5, "section_marker", [100, 100, 20, 10])
        elements = _make_elements(d_lines=[[80, 80, 130, 130]])
        result = _associate_annotation(ann, elements, category=1, max_distance=150)
        assert result["associated_element"] is not None
        assert result["associated_element"]["element_type"] == "line_diagonal"

    def test_spacing_annotation_nearest_hline(self):
        ann = _make_annotation(6, "spacing_annotation", [100, 50, 30, 10])
        elements = _make_elements(h_lines=[[50, 60, 200, 60]])
        result = _associate_annotation(ann, elements, category=1, max_distance=150)
        assert result["associated_element"] is not None
        assert result["associated_element"]["element_type"] == "line_horizontal"

    def test_material_code_nearest_contour(self):
        ann = _make_annotation(7, "material_code", [500, 400, 20, 12])
        elements = _make_elements(contours=[[480, 390, 60, 30]])
        result = _associate_annotation(ann, elements, category=2, max_distance=150)
        assert result["associated_element"] is not None
        assert result["associated_element"]["element_type"] == "contour"

    def test_unknown_always_unassociated(self):
        ann = _make_annotation(8, "unknown", [100, 100, 20, 10])
        elements = _make_elements(
            h_lines=[[50, 105, 200, 105]],
            circles=[[110, 105, 10]],
            contours=[[80, 90, 60, 30]],
        )
        result = _associate_annotation(ann, elements, category=1, max_distance=150)
        assert result["associated_element"] is None

    def test_nothing_within_max_distance(self):
        ann = _make_annotation(9, "dimension_value", [100, 100, 20, 10])
        # Line is 200px away — beyond max_distance=150
        elements = _make_elements(h_lines=[[0, 300, 50, 300]])
        result = _associate_annotation(ann, elements, category=1, max_distance=150)
        assert result["associated_element"] is None

    def test_malformed_box_unassociated(self):
        ann = {"id": 10, "type": "dimension_value", "text": "50", "box": None, "parsed": {}}
        elements = _make_elements(h_lines=[[50, 60, 200, 60]])
        result = _associate_annotation(ann, elements, category=1, max_distance=150)
        assert result["associated_element"] is None

    def test_distance_px_rounded_to_1_decimal(self):
        ann = _make_annotation(11, "dimension_value", [100, 50, 20, 10])
        elements = _make_elements(h_lines=[[50, 60, 200, 60]])
        result = _associate_annotation(ann, elements, category=1, max_distance=150)
        if result["associated_element"]:
            d = result["associated_element"]["distance_px"]
            assert d == round(d, 1)


# ============================================================
# Task 10.6 — _build_output
# ============================================================

class TestBuildOutput:

    def _make_assoc(self, matched=True):
        elem = {"element_type": "line_horizontal", "element_data": [0, 0, 100, 0], "distance_px": 5.0}
        return {
            "annotation_id": 1, "annotation_text": "50",
            "annotation_type": "dimension_value", "annotation_box": [10, 10, 20, 10],
            "associated_element": elem if matched else None,
        }

    def test_count_invariant(self):
        assocs = [self._make_assoc(True), self._make_assoc(False), self._make_assoc(True)]
        structured = {"image_category": 1, "classified": assocs}
        result = _build_output(structured, assocs, "cad1_001_structured.json")
        assert result["total_annotations"] == result["matched"] + result["unassociated"]
        assert result["matched"] == 2
        assert result["unassociated"] == 1

    def test_required_keys_present(self):
        assocs = [self._make_assoc()]
        structured = {"image_category": 1, "classified": assocs}
        result = _build_output(structured, assocs, "cad1_001_structured.json")
        for key in ("source_structured", "image_category", "total_annotations",
                    "matched", "unassociated", "associations"):
            assert key in result

    def test_source_structured_basename(self):
        assocs = []
        structured = {"image_category": 2, "classified": []}
        result = _build_output(structured, assocs, "cad2_001_structured.json")
        assert result["source_structured"] == "cad2_001_structured.json"

    def test_empty_classified(self):
        structured = {"image_category": 1, "classified": []}
        result = _build_output(structured, [], "cad1_001_structured.json")
        assert result["total_annotations"] == 0
        assert result["matched"] == 0
        assert result["unassociated"] == 0
        assert result["associations"] == []


# ============================================================
# Task 10.7 — End-to-end test with real image
# ============================================================

REAL_IMAGE = "data/category_1/cad1_001.png"
REAL_STRUCTURED = "results/batch/cad1_001_structured.json"


@pytest.mark.skipif(
    not os.path.exists(REAL_IMAGE) or not os.path.exists(REAL_STRUCTURED),
    reason="Real image or structured JSON not available"
)
class TestAssociateFileEndToEnd:

    def test_result_not_none(self, tmp_path):
        result = associate_file(REAL_IMAGE, REAL_STRUCTURED, str(tmp_path))
        assert result is not None

    def test_total_annotations_positive(self, tmp_path):
        result = associate_file(REAL_IMAGE, REAL_STRUCTURED, str(tmp_path))
        assert result["total_annotations"] > 0

    def test_json_written(self, tmp_path):
        associate_file(REAL_IMAGE, REAL_STRUCTURED, str(tmp_path))
        json_file = tmp_path / "cad1_001_associations.json"
        assert json_file.exists()

    def test_png_written(self, tmp_path):
        associate_file(REAL_IMAGE, REAL_STRUCTURED, str(tmp_path))
        png_file = tmp_path / "cad1_001_associations.png"
        assert png_file.exists()

    def test_count_invariant(self, tmp_path):
        result = associate_file(REAL_IMAGE, REAL_STRUCTURED, str(tmp_path))
        assert result["total_annotations"] == result["matched"] + result["unassociated"]

    def test_json_round_trip(self, tmp_path):
        result = associate_file(REAL_IMAGE, REAL_STRUCTURED, str(tmp_path))
        # JSON round-trip: serialise and deserialise should give identical result
        serialised = json.dumps(result, ensure_ascii=False)
        deserialised = json.loads(serialised)
        assert deserialised == result

    def test_one_record_per_annotation(self, tmp_path):
        result = associate_file(REAL_IMAGE, REAL_STRUCTURED, str(tmp_path))
        # Load structured to get expected count
        with open(REAL_STRUCTURED, encoding='utf-8') as f:
            structured = json.load(f)
        expected = len(structured.get("classified", []))
        assert len(result["associations"]) == expected
