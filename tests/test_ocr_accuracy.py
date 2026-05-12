"""
Unit tests for OCR accuracy improvements.
Tests for: post_process_text leading-zero fix, radius_callout, material_name,
compound part names, expanded BOM headers, reconstruct_bom_rows, bom_rows field.

Run with:
    cad_env\\Scripts\\python.exe -m pytest tests/test_ocr_accuracy.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vlm_reader import post_process_text
from validation import (
    classify,
    extract_parsed,
    reconstruct_bom_rows,
    build_structured_output,
    _get_box_safe,
    VALID_TYPES,
)


# ============================================================
# Task 9.1 — post_process_text leading-zero fix
# ============================================================

class TestPostProcessLeadingZero:

    def test_061_to_phi61(self):
        assert post_process_text("061") == "Ø61"

    def test_085_to_phi85(self):
        assert post_process_text("085") == "Ø85"

    def test_0100_to_phi100(self):
        assert post_process_text("0100") == "Ø100"

    def test_0118_to_phi118(self):
        assert post_process_text("0118") == "Ø118"

    def test_bare_zero_unchanged(self):
        assert post_process_text("0") == "0"

    def test_01_unchanged(self):
        # Only one digit after zero — should NOT be converted
        assert post_process_text("01") == "01"

    def test_already_phi_unchanged(self):
        # Already has Ø prefix — no double conversion
        result = post_process_text("Ø61")
        assert result == "Ø61"

    def test_protected_code_unchanged(self):
        # Protected codes must not be touched
        assert post_process_text("MS") == "MS"


# ============================================================
# Task 9.3 — classify() radius callout
# ============================================================

class TestClassifyRadiusCallout:

    @pytest.mark.parametrize("text", ["R189", "R78", "R13", "R24", "R5"])
    def test_radius_callout_dataset_examples(self, text):
        assert classify(text, category=1) == "radius_callout"

    def test_radius_with_decimal(self):
        assert classify("R13.5", category=1) == "radius_callout"

    def test_radius_all_categories(self):
        for cat in (1, 2, 3):
            assert classify("R50", cat) == "radius_callout"

    def test_lowercase_r_not_radius(self):
        # Lowercase r must NOT match — avoids false positives in part names
        result = classify("r189", category=1)
        assert result != "radius_callout"

    def test_bare_r_not_radius(self):
        result = classify("R", category=1)
        assert result != "radius_callout"

    def test_radius_in_valid_types(self):
        assert "radius_callout" in VALID_TYPES


# ============================================================
# Task 9.4 — extract_parsed() radius callout
# ============================================================

class TestExtractParsedRadiusCallout:

    def test_r189(self):
        assert extract_parsed("radius_callout", "R189") == {"radius": 189.0}

    def test_r78(self):
        assert extract_parsed("radius_callout", "R78") == {"radius": 78.0}

    def test_r13_5(self):
        assert extract_parsed("radius_callout", "R13.5") == {"radius": 13.5}

    def test_bare_r_returns_none(self):
        result = extract_parsed("radius_callout", "R")
        assert result == {"radius": None}

    def test_r_with_spaces(self):
        result = extract_parsed("radius_callout", " R24 ")
        assert result == {"radius": 24.0}


# ============================================================
# Task 9.5 — classify() material name
# ============================================================

class TestClassifyMaterialName:

    def test_brass_exact(self):
        assert classify("BRASS", category=2) == "material_name"

    def test_brass_title_case(self):
        assert classify("Brass", category=2) == "material_name"

    def test_cast_iron(self):
        assert classify("CAST IRON", category=2) == "material_name"

    def test_mild_steel(self):
        assert classify("MILD STEEL", category=2) == "material_name"

    def test_babbit(self):
        assert classify("Babbit", category=2) == "material_name"

    def test_ni_cr_steel(self):
        assert classify("Ni-Cr steel", category=2) == "material_name"

    def test_ms_stays_material_code(self):
        # P8 (material_code) fires before P8.5 (material_name)
        assert classify("MS", category=2) == "material_code"

    def test_ci_stays_material_code(self):
        assert classify("CI", category=2) == "material_code"

    def test_material_name_in_valid_types(self):
        assert "material_name" in VALID_TYPES


# ============================================================
# Task 9.6 — extract_parsed() material name
# ============================================================

class TestExtractParsedMaterialName:

    def test_brass(self):
        assert extract_parsed("material_name", "BRASS") == {"name": "Brass"}

    def test_cast_iron(self):
        assert extract_parsed("material_name", "CAST IRON") == {"name": "Cast Iron"}

    def test_babbit(self):
        assert extract_parsed("material_name", "Babbit") == {"name": "Babbit"}

    def test_ni_cr_steel(self):
        result = extract_parsed("material_name", "Ni-Cr steel")
        assert result == {"name": "Ni-Cr Steel"}


# ============================================================
# Task 9.7 — classify() compound part names
# ============================================================

class TestClassifyCompoundPartNames:

    @pytest.mark.parametrize("text", [
        "ARTICULATED ROD", "Articulated rod",
        "COVER PLATE", "Cover plate",
        "ROD END", "Rod end",
        "LOCK NUT", "Lock nut",
        "PISTON RING", "Piston ring",
        "CONNECTING ROD", "Connecting rod",
        "LINK PIN", "Link pin",
        "PISTON PIN", "Piston pin",
    ])
    def test_compound_part_name(self, text):
        assert classify(text, category=2) == "part_name"

    @pytest.mark.parametrize("text", [
        "ROD BUSH-UPPER", "Rod bush-upper",
        "MASTER ROD BEARING", "Master rod bearing",
        "PISTON PIN PLUG", "Piston pin plug",
    ])
    def test_longer_compound_part_name(self, text):
        assert classify(text, category=2) == "part_name"

    def test_piston_single_word(self):
        assert classify("Piston", category=2) == "part_name"


# ============================================================
# Task 9.8 — expanded BOM headers
# ============================================================

class TestExpandedBomHeaders:

    @pytest.mark.parametrize("text", [
        "MATL", "Matl", "matl",
        "MAT", "Mat",
        "SL.NO", "Sl.No",
        "SL. NO", "Sl. No",
        "PART NAME", "Part Name",
        "PART NO.", "Part No.",
        "NO.", "No.",
    ])
    def test_new_bom_header(self, text):
        assert classify(text, category=2) == "bom_header"

    @pytest.mark.parametrize("text", [
        "PARTS LIST", "NAME", "MATERIAL", "QTY", "NO", "SL NO", "PART NO",
    ])
    def test_existing_bom_headers_still_work(self, text):
        assert classify(text, category=2) == "bom_header"


# ============================================================
# Task 9.9 — _get_box_safe()
# ============================================================

class TestGetBoxSafe:

    def test_valid_box(self):
        entry = {"box": [10, 20, 30, 40]}
        assert _get_box_safe(entry) == (10, 20, 30, 40)

    def test_missing_box_key(self):
        assert _get_box_safe({}) is None

    def test_none_box(self):
        assert _get_box_safe({"box": None}) is None

    def test_too_short(self):
        assert _get_box_safe({"box": [1, 2, 3]}) is None

    def test_non_numeric(self):
        assert _get_box_safe({"box": ["a", "b", "c", "d"]}) is None

    def test_float_values_converted(self):
        result = _get_box_safe({"box": [1.5, 2.5, 3.5, 4.5]})
        assert result == (1, 2, 3, 4)


# ============================================================
# Task 9.10 — reconstruct_bom_rows() basic grouping
# ============================================================

def _make_entry(id_, x, y, w, h, type_, parsed):
    return {"id": id_, "box": [x, y, w, h], "type": type_, "parsed": parsed, "confidence": 0.9}


class TestReconstructBomRows:

    def test_category1_returns_empty(self):
        entries = [_make_entry(1, 100, 400, 30, 15, "part_name", {"name": "Body"})]
        assert reconstruct_bom_rows(entries, category=1) == []

    def test_category3_returns_empty(self):
        entries = [_make_entry(1, 100, 400, 30, 15, "part_name", {"name": "Body"})]
        assert reconstruct_bom_rows(entries, category=3) == []

    def test_no_bom_entries_returns_empty(self):
        entries = [_make_entry(1, 100, 100, 30, 15, "dimension_value", {"value": 50.0})]
        assert reconstruct_bom_rows(entries, category=2) == []

    def test_complete_row(self):
        entries = [
            _make_entry(1, 300, 400, 10, 14, "balloon_number", {"number": 1}),
            _make_entry(2, 380, 402, 40, 14, "part_name", {"name": "Body"}),
            _make_entry(3, 500, 401, 20, 14, "material_code", {"code": "MS"}),
            _make_entry(4, 600, 400, 10, 14, "quantity", {"qty": 1}),
        ]
        rows = reconstruct_bom_rows(entries, category=2)
        assert len(rows) == 1
        assert rows[0]["part_no"] == 1
        assert rows[0]["part_name"] == "Body"
        assert rows[0]["material"] == "MS"
        assert rows[0]["qty"] == 1

    def test_row_missing_material(self):
        entries = [
            _make_entry(1, 300, 400, 10, 14, "balloon_number", {"number": 2}),
            _make_entry(2, 380, 402, 40, 14, "part_name", {"name": "Bolt"}),
            _make_entry(3, 600, 400, 10, 14, "quantity", {"qty": 4}),
        ]
        rows = reconstruct_bom_rows(entries, category=2)
        assert len(rows) == 1
        assert rows[0]["material"] is None
        assert rows[0]["part_name"] == "Bolt"
        assert rows[0]["qty"] == 4

    def test_malformed_box_skipped(self):
        entries = [
            {"id": 1, "box": None, "type": "part_name", "parsed": {"name": "Body"}, "confidence": 0.9},
            _make_entry(2, 300, 400, 10, 14, "balloon_number", {"number": 1}),
        ]
        rows = reconstruct_bom_rows(entries, category=2)
        # Should not crash; balloon_number entry processed, part_name skipped
        assert isinstance(rows, list)

    def test_material_name_used_as_material(self):
        entries = [
            _make_entry(1, 300, 400, 10, 14, "balloon_number", {"number": 4}),
            _make_entry(2, 380, 402, 50, 14, "part_name", {"name": "Brasses"}),
            _make_entry(3, 500, 401, 30, 14, "material_name", {"name": "Brass"}),
            _make_entry(4, 600, 400, 10, 14, "quantity", {"qty": 2}),
        ]
        rows = reconstruct_bom_rows(entries, category=2)
        assert len(rows) == 1
        assert rows[0]["material"] == "Brass"

    def test_two_rows_separated(self):
        # Row 1 at y=400, Row 2 at y=420 (20px apart — should be separate rows)
        entries = [
            _make_entry(1, 300, 400, 10, 14, "balloon_number", {"number": 1}),
            _make_entry(2, 380, 400, 40, 14, "part_name", {"name": "Body"}),
            _make_entry(3, 300, 420, 10, 14, "balloon_number", {"number": 2}),
            _make_entry(4, 380, 420, 40, 14, "part_name", {"name": "Bolt"}),
        ]
        rows = reconstruct_bom_rows(entries, category=2)
        assert len(rows) == 2


# ============================================================
# Task 9.11 — y-grouping tolerance
# ============================================================

class TestYGroupingTolerance:

    def test_entries_5px_apart_same_row(self):
        entries = [
            _make_entry(1, 300, 400, 10, 14, "balloon_number", {"number": 1}),
            _make_entry(2, 380, 405, 40, 14, "part_name", {"name": "Body"}),
        ]
        rows = reconstruct_bom_rows(entries, category=2)
        assert len(rows) == 1

    def test_entries_15px_apart_different_rows(self):
        entries = [
            _make_entry(1, 300, 400, 10, 14, "balloon_number", {"number": 1}),
            _make_entry(2, 380, 415, 40, 14, "part_name", {"name": "Body"}),
        ]
        rows = reconstruct_bom_rows(entries, category=2)
        assert len(rows) == 2


# ============================================================
# Task 9.12 — build_structured_output() bom_rows field
# ============================================================

class TestBuildStructuredOutputBomRows:

    def _make_classified(self, type_, text, box=None):
        return {
            "id": 1, "box": box or [100, 400, 30, 14],
            "text": text, "type": type_,
            "confidence": 0.9, "parsed": {}
        }

    def test_category1_has_empty_bom_rows(self):
        entries = [self._make_classified("dimension_value", "50")]
        result = build_structured_output("cad1_001_fullocr.json", 1, entries)
        assert "bom_rows" in result
        assert result["bom_rows"] == []

    def test_category2_has_bom_rows_key(self):
        entries = [self._make_classified("dimension_value", "50")]
        result = build_structured_output("cad2_001_fullocr.json", 2, entries)
        assert "bom_rows" in result
        assert isinstance(result["bom_rows"], list)

    def test_existing_keys_preserved(self):
        entries = [self._make_classified("dimension_value", "50")]
        result = build_structured_output("cad1_001_fullocr.json", 1, entries)
        for key in ("source_file", "image_category", "total_detections", "classified", "summary"):
            assert key in result
