"""
Unit and property-based tests for src/validation.py — Stage 3.

Run with:
    cad_env\\Scripts\\python.exe -m pytest tests/test_validation.py -v
    cad_env\\Scripts\\python.exe -m pytest tests/test_validation.py -v -k "not hypothesis"
"""

import sys
import os
import json
import tempfile
import pathlib

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from validation import (
    normalise_text,
    classify,
    extract_parsed,
    detect_category,
    validate_file,
    validate_batch,
    PROTECTED_CODES,
    VALID_TYPES,
)

# ============================================================
# Normalisation unit tests
# ============================================================

class TestNormaliseText:

    # --- Leading-zero diameter fix ---
    def test_leading_zero_three_digits(self):
        assert normalise_text("018") == "Ø18"

    def test_leading_zero_four_digits(self):
        assert normalise_text("0118") == "Ø118"

    def test_leading_zero_two_digits_not_matched(self):
        # "01" has only 1 digit after 0 — should NOT be converted
        result = normalise_text("01")
        assert result == "01"

    def test_bare_zero_not_converted(self):
        assert normalise_text("0") == "0"

    # --- Degree symbol fix ---
    def test_degree_two_digits(self):
        assert normalise_text('45"') == "45°"

    def test_degree_one_digit(self):
        assert normalise_text('9"') == "9°"

    def test_no_degree_fix_on_plain_text(self):
        result = normalise_text("hello")
        assert '"' not in result or "°" not in result

    # --- THICK typo fix ---
    def test_ihick_replaced(self):
        assert normalise_text("12 IHICK") == "12 THICK"

    def test_mick_replaced(self):
        assert normalise_text("12 MICK") == "12 THICK"

    def test_ihick_case_insensitive(self):
        assert normalise_text("12 ihick") == "12 THICK"

    def test_compound_dia_ihick(self):
        result = normalise_text("DIA 40*20 IHICK")
        assert "THICK" in result
        assert "×" in result

    # --- Multiplication symbol fix ---
    def test_star_to_times(self):
        assert normalise_text("M30 * 2.5") == "M30 × 2.5"

    # --- PROTECTED_CODES preserved ---
    def test_protected_ms(self):
        assert normalise_text("MS") == "MS"

    def test_protected_equi_sp(self):
        assert normalise_text("EQUI-SP") == "EQUI-SP"

    def test_protected_x_x(self):
        assert normalise_text("X-X") == "X-X"

    def test_protected_fs(self):
        assert normalise_text("FS") == "FS"

    def test_protected_hole(self):
        assert normalise_text("HOLE") == "HOLE"

    # --- Edge cases ---
    def test_empty_string(self):
        assert normalise_text("") == ""

    def test_whitespace_only(self):
        assert normalise_text("   ") == ""

    def test_strips_whitespace(self):
        assert normalise_text("  50  ") == "50"


# ============================================================
# Classification unit tests
# ============================================================

class TestClassify:

    # --- hole_callout (P1) ---
    def test_hole_callout(self):
        assert classify("HOLE; DIA 21", 2) == "hole_callout"

    def test_hole_callout_cat1(self):
        assert classify("2 HOLES M B", 1) == "hole_callout"

    # --- thread_spec (P2) ---
    def test_thread_spec_with_pitch(self):
        assert classify("M30 × 2.5", 1) == "thread_spec"

    def test_thread_spec_no_pitch(self):
        assert classify("M16", 1) == "thread_spec"

    def test_thread_spec_all_categories(self):
        for cat in (1, 2, 3):
            assert classify("M30×2.5", cat) == "thread_spec"

    # --- diameter_callout (P3) ---
    def test_diameter_callout_phi(self):
        assert classify("Ø50", 1) == "diameter_callout"

    def test_diameter_callout_dia(self):
        assert classify("DIA 21", 2) == "diameter_callout"

    def test_diameter_after_normalise(self):
        # "018" normalises to "Ø18"
        assert classify(normalise_text("018"), 2) == "diameter_callout"

    # --- dimension_with_note (P4) beats diameter_callout ---
    def test_dim_with_note_thick(self):
        assert classify("12 THICK", 1) == "dimension_with_note"

    def test_dim_with_note_deep(self):
        assert classify("5 DEEP", 1) == "dimension_with_note"

    def test_dia_with_note_beats_diameter(self):
        assert classify("DIA 40×20 THICK", 2) == "dimension_with_note"

    # --- tolerance (P5) ---
    def test_tolerance_plus_minus(self):
        assert classify("±0.5", 1) == "tolerance"

    def test_tolerance_fit(self):
        assert classify("H7/h6", 1) == "tolerance"

    def test_tolerance_upper_lower(self):
        assert classify("+0.12/-0.00", 1) == "tolerance"

    # --- spacing_annotation (P6) ---
    def test_spacing_annotation(self):
        assert classify("EQUI-SP", 1) == "spacing_annotation"

    # --- bom_header (P7) ---
    def test_bom_header_name(self):
        assert classify("NAME", 2) == "bom_header"

    def test_bom_header_qty(self):
        assert classify("QTY", 2) == "bom_header"

    def test_bom_header_parts_list(self):
        assert classify("PARTS LIST", 2) == "bom_header"

    # --- material_code (P8) ---
    def test_material_ms(self):
        assert classify("MS", 2) == "material_code"

    def test_material_ci(self):
        assert classify("CI", 1) == "material_code"

    def test_material_fs(self):
        assert classify("FS", 2) == "material_code"

    # --- part_name (P9) ---
    def test_part_name_valve(self):
        assert classify("Valve", 2) == "part_name"

    def test_part_name_pin(self):
        assert classify("Pin", 2) == "part_name"

    def test_part_name_body(self):
        assert classify("Body", 2) == "part_name"

    # --- section_marker (P10) — Cat 1/3 only ---
    def test_section_marker_xx(self):
        assert classify("X-X", 1) == "section_marker"

    def test_section_marker_single(self):
        assert classify("X", 1) == "section_marker"

    def test_section_marker_cat3(self):
        assert classify("A-A", 3) == "section_marker"

    def test_section_marker_not_cat2(self):
        # In Cat 2, single letter should NOT be section_marker
        result = classify("X", 2)
        assert result != "section_marker"

    # --- balloon_number (P11) — Cat 2/3 only ---
    def test_balloon_cat2(self):
        assert classify("1", 2) == "balloon_number"

    def test_balloon_cat3(self):
        assert classify("3", 3) == "balloon_number"

    def test_balloon_not_cat1(self):
        # In Cat 1, single digit is dimension_value
        assert classify("5", 1) == "dimension_value"

    # --- quantity (P12) — Cat 2 only, 2-digit numbers (single digits go to balloon_number first) ---
    def test_quantity_cat2(self):
        # Single digit in Cat 2 → balloon_number (P11 fires before P12)
        assert classify("2", 2) == "balloon_number"

    def test_two_digit_quantity_cat2(self):
        assert classify("12", 2) == "quantity"

    # --- dimension_value (P13) ---
    def test_dimension_value_integer(self):
        assert classify("75", 1) == "dimension_value"

    def test_dimension_value_decimal(self):
        assert classify("12.5", 1) == "dimension_value"

    # --- unknown (P14) ---
    def test_unknown_empty(self):
        assert classify("", 1) == "unknown"

    def test_unknown_garbage(self):
        assert classify("Ozw", 1) == "unknown"


# ============================================================
# extract_parsed unit tests
# ============================================================

class TestExtractParsed:

    def test_dimension_value(self):
        r = extract_parsed("dimension_value", "50")
        assert r == {"value": 50.0}

    def test_thread_spec_with_pitch(self):
        r = extract_parsed("thread_spec", "M30 × 2.5")
        assert r["nominal"] == "M30"
        assert r["pitch"] == 2.5

    def test_thread_spec_no_pitch(self):
        r = extract_parsed("thread_spec", "M16")
        assert r["nominal"] == "M16"
        assert r["pitch"] is None

    def test_tolerance(self):
        r = extract_parsed("tolerance", "±0.5")
        assert r == {"tolerance_string": "±0.5"}

    def test_diameter_callout_phi(self):
        r = extract_parsed("diameter_callout", "Ø50")
        assert r == {"diameter": 50.0}

    def test_diameter_callout_dia(self):
        r = extract_parsed("diameter_callout", "DIA 21")
        assert r == {"diameter": 21.0}

    def test_hole_callout(self):
        r = extract_parsed("hole_callout", "HOLE; DIA 21")
        assert r == {"raw": "HOLE; DIA 21"}

    def test_section_marker(self):
        r = extract_parsed("section_marker", "X-X")
        assert r == {"label": "X-X"}

    def test_spacing_annotation(self):
        r = extract_parsed("spacing_annotation", "EQUI-SP")
        assert r == {"annotation": "EQUI-SP"}

    def test_material_code(self):
        r = extract_parsed("material_code", "ms")
        assert r == {"code": "MS"}

    def test_part_name(self):
        r = extract_parsed("part_name", "valve")
        assert r == {"name": "Valve"}

    def test_bom_header(self):
        r = extract_parsed("bom_header", "name")
        assert r == {"header": "NAME"}

    def test_balloon_number(self):
        r = extract_parsed("balloon_number", "3")
        assert r == {"number": 3}

    def test_quantity(self):
        r = extract_parsed("quantity", "2")
        assert r == {"qty": 2}

    def test_dimension_with_note(self):
        r = extract_parsed("dimension_with_note", "12 THICK")
        assert r == {"raw": "12 THICK"}

    def test_unknown(self):
        r = extract_parsed("unknown", "garbage")
        assert r == {}


# ============================================================
# detect_category unit tests
# ============================================================

class TestDetectCategory:

    def test_cat1(self):
        assert detect_category("cad1_001_fullocr.json") == 1

    def test_cat2(self):
        assert detect_category("cad2_005_fullocr.json") == 2

    def test_cat3(self):
        assert detect_category("cad3_001_fullocr.json") == 3

    def test_full_path_cat1(self):
        assert detect_category("results/batch/cad1_010_fullocr.json") == 1

    def test_unknown_returns_zero(self, capsys):
        result = detect_category("unknown_image.json")
        assert result == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.out


# ============================================================
# validate_file integration tests
# ============================================================

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample_fullocr.json")


class TestValidateFile:

    def test_output_schema_keys(self, tmp_path):
        result = validate_file(FIXTURE_PATH, str(tmp_path))
        assert result is not None
        assert "source_file" in result
        assert "image_category" in result
        assert "total_detections" in result
        assert "classified" in result
        assert "summary" in result

    def test_total_detections_matches_classified(self, tmp_path):
        result = validate_file(FIXTURE_PATH, str(tmp_path))
        assert result["total_detections"] == len(result["classified"])

    def test_summary_counts_consistent(self, tmp_path):
        result = validate_file(FIXTURE_PATH, str(tmp_path))
        summary = result["summary"]
        classified = result["classified"]
        assert sum(summary.values()) == result["total_detections"]
        for type_key, count in summary.items():
            actual = sum(1 for e in classified if e["type"] == type_key)
            assert actual == count

    def test_structured_json_written(self, tmp_path):
        validate_file(FIXTURE_PATH, str(tmp_path))
        output_file = tmp_path / "sample_structured.json"
        assert output_file.exists()

    def test_all_entries_have_required_fields(self, tmp_path):
        result = validate_file(FIXTURE_PATH, str(tmp_path))
        for entry in result["classified"]:
            assert "id" in entry
            assert "box" in entry
            assert "text" in entry
            assert "type" in entry
            assert "confidence" in entry
            assert "parsed" in entry

    def test_all_types_are_valid(self, tmp_path):
        result = validate_file(FIXTURE_PATH, str(tmp_path))
        for entry in result["classified"]:
            assert entry["type"] in VALID_TYPES

    def test_missing_text_field_gives_unknown(self, tmp_path):
        # Entry 14 in fixture has no "text" field
        result = validate_file(FIXTURE_PATH, str(tmp_path))
        entry_14 = next(e for e in result["classified"] if e["id"] == 14)
        assert entry_14["type"] == "unknown"

    def test_empty_text_gives_unknown(self, tmp_path):
        # Entry 13 has text=""
        result = validate_file(FIXTURE_PATH, str(tmp_path))
        entry_13 = next(e for e in result["classified"] if e["id"] == 13)
        assert entry_13["type"] == "unknown"

    def test_malformed_json_returns_none(self, tmp_path):
        bad_file = tmp_path / "bad_fullocr.json"
        bad_file.write_text("{not valid json", encoding="utf-8")
        result = validate_file(str(bad_file), str(tmp_path))
        assert result is None

    def test_empty_array_gives_zero_detections(self, tmp_path):
        empty_file = tmp_path / "cad1_empty_fullocr.json"
        empty_file.write_text("[]", encoding="utf-8")
        result = validate_file(str(empty_file), str(tmp_path))
        assert result is not None
        assert result["total_detections"] == 0
        assert result["classified"] == []
        assert result["summary"] == {}

    def test_overwrite_existing_output(self, tmp_path):
        # Run twice — should not raise
        validate_file(FIXTURE_PATH, str(tmp_path))
        validate_file(FIXTURE_PATH, str(tmp_path))
        output_file = tmp_path / "sample_structured.json"
        assert output_file.exists()

    def test_normalisation_applied_018(self, tmp_path):
        # Entry 5 has text="018" → should normalise to "Ø18"
        result = validate_file(FIXTURE_PATH, str(tmp_path))
        entry_5 = next(e for e in result["classified"] if e["id"] == 5)
        assert entry_5["text"] == "Ø18"
        assert entry_5["type"] == "diameter_callout"


# ============================================================
# validate_batch integration tests
# ============================================================

class TestValidateBatch:

    def test_processes_multiple_files(self, tmp_path):
        # Copy fixture twice with different names
        fixture_data = pathlib.Path(FIXTURE_PATH).read_text(encoding="utf-8")
        (tmp_path / "cad1_001_fullocr.json").write_text(fixture_data, encoding="utf-8")
        (tmp_path / "cad2_001_fullocr.json").write_text(fixture_data, encoding="utf-8")
        results = validate_batch(str(tmp_path), str(tmp_path))
        assert len(results) == 2

    def test_skips_failed_files(self, tmp_path):
        fixture_data = pathlib.Path(FIXTURE_PATH).read_text(encoding="utf-8")
        (tmp_path / "cad1_001_fullocr.json").write_text(fixture_data, encoding="utf-8")
        (tmp_path / "cad1_bad_fullocr.json").write_text("{bad json", encoding="utf-8")
        results = validate_batch(str(tmp_path), str(tmp_path))
        # Only the valid file should be in results
        assert len(results) == 1

    def test_empty_dir_returns_empty_list(self, tmp_path):
        results = validate_batch(str(tmp_path), str(tmp_path))
        assert results == []


# ============================================================
# Property-based tests (Hypothesis) — all 16 properties
# ============================================================

try:
    from hypothesis import given, settings, assume, strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

import pytest

if HYPOTHESIS_AVAILABLE:

    # Feature: stage3-validation-structuring, Property 1: Normalisation preserves PROTECTED_CODES
    @given(st.sampled_from(sorted(PROTECTED_CODES)))
    @settings(max_examples=100)
    def test_pbt_property1_protected_codes_preserved(token):
        """For any token that is an exact member of PROTECTED_CODES, normalise_text returns it unchanged."""
        assert normalise_text(token) == token

    # Feature: stage3-validation-structuring, Property 2: Leading-zero diameter correction
    @given(st.from_regex(r'0\d{2,}', fullmatch=True))
    @settings(max_examples=200)
    def test_pbt_property2_leading_zero_diameter(text):
        """For any string matching 0\\d{2,} (fullmatch), normalise_text returns a string starting with Ø."""
        result = normalise_text(text)
        assert result.startswith('Ø')
        assert result[1:] == text[1:]

    # Feature: stage3-validation-structuring, Property 3: Degree symbol correction
    @given(st.from_regex(r'\d{1,2}"', fullmatch=True))
    @settings(max_examples=200)
    def test_pbt_property3_degree_symbol(text):
        """For any string of 1-2 digits followed by a double-quote, normalise_text replaces \" with °."""
        result = normalise_text(text)
        assert '°' in result
        assert '"' not in result

    # Feature: stage3-validation-structuring, Property 4: THICK typo correction
    @given(st.one_of(
        st.from_regex(r'[A-Za-z0-9 ]*IHICK[A-Za-z0-9 ]*', fullmatch=True),
        st.from_regex(r'[A-Za-z0-9 ]*MICK[A-Za-z0-9 ]*', fullmatch=True),
    ))
    @settings(max_examples=200)
    def test_pbt_property4_thick_typo(text):
        """For any string containing IHICK or MICK (case-insensitive), normalise_text replaces with THICK."""
        assume(not _is_protected_helper(text))
        result = normalise_text(text)
        assert 'THICK' in result.upper()

    # Feature: stage3-validation-structuring, Property 5: Classifier always returns a valid type
    @given(st.text(min_size=1), st.integers(min_value=0, max_value=3))
    @settings(max_examples=500)
    def test_pbt_property5_classifier_always_valid_type(text, category):
        """For any non-empty string and any category in {0,1,2,3}, classify returns a member of VALID_TYPES."""
        result = classify(text, category)
        assert result in VALID_TYPES

    # Feature: stage3-validation-structuring, Property 6: Thread spec classification
    @given(st.from_regex(r'M\d+(\s*[×x]\s*\d+(\.\d+)?)?', fullmatch=True))
    @settings(max_examples=200)
    def test_pbt_property6_thread_spec(text):
        """For any string matching M\\d+ (optionally with pitch), classify returns 'thread_spec' for all categories."""
        for cat in (1, 2, 3):
            assert classify(text, cat) == "thread_spec"

    # Feature: stage3-validation-structuring, Property 7: Diameter callout classification
    @given(st.one_of(
        st.from_regex(r'Ø\d+(\.\d+)?', fullmatch=True),
        st.from_regex(r'DIA \d+(\.\d+)?', fullmatch=True),
    ))
    @settings(max_examples=200)
    def test_pbt_property7_diameter_callout(text):
        """For any Ø{digits} or DIA {digits} string with no note keyword, classify returns 'diameter_callout'."""
        for cat in (1, 2, 3):
            assert classify(text, cat) == "diameter_callout"

    # Feature: stage3-validation-structuring, Property 8: Priority — hole_callout beats diameter_callout
    @given(st.from_regex(r'HOLE.*DIA \d+', fullmatch=True))
    @settings(max_examples=200)
    def test_pbt_property8_hole_beats_diameter(text):
        """For any string containing HOLE and DIA+number, classify returns 'hole_callout' not 'diameter_callout'."""
        for cat in (1, 2, 3):
            result = classify(text, cat)
            assert result == "hole_callout"
            assert result != "diameter_callout"

    # Feature: stage3-validation-structuring, Property 9: Priority — dimension_with_note beats diameter_callout
    @given(st.one_of(
        st.from_regex(r'DIA \d+ THICK', fullmatch=True),
        st.from_regex(r'DIA \d+ DEEP', fullmatch=True),
    ))
    @settings(max_examples=200)
    def test_pbt_property9_dim_note_beats_diameter(text):
        """For any DIA string containing THICK/DEEP, classify returns 'dimension_with_note' not 'diameter_callout'."""
        for cat in (1, 2, 3):
            result = classify(text, cat)
            assert result == "dimension_with_note"
            assert result != "diameter_callout"

    # Feature: stage3-validation-structuring, Property 10: Category-gated section_marker
    # Use known safe section markers that won't collide with material codes, BOM headers, or part names
    _SAFE_SECTION_MARKERS = ['X-X', 'A-A', 'B-B', 'Y-Y', 'Z-Z', 'C-C', 'D-D', 'E-E', 'F-F']

    @given(st.sampled_from(_SAFE_SECTION_MARKERS))
    @settings(max_examples=100)
    def test_pbt_property10_section_marker_gated(text):
        """section_marker is returned for Cat 1/3 but not Cat 2 for known [A-Z]-[A-Z] markers."""
        for cat in (1, 3):
            assert classify(text, cat) == "section_marker"
        # In Cat 2, these are not section markers
        assert classify(text, 2) != "section_marker"

    # Feature: stage3-validation-structuring, Property 11: Category-gated balloon_number
    @given(st.sampled_from(list('123456789')))
    @settings(max_examples=100)
    def test_pbt_property11_balloon_number_gated(digit):
        """Single digit 1-9: balloon_number for Cat 2/3, dimension_value for Cat 1."""
        assert classify(digit, 2) == "balloon_number"
        assert classify(digit, 3) == "balloon_number"
        assert classify(digit, 1) == "dimension_value"

    # Feature: stage3-validation-structuring, Property 12: Parsed fields structurally correct for assigned type
    PARSED_SCHEMAS = {
        'dimension_value':    {'value'},
        'thread_spec':        {'nominal', 'pitch'},
        'tolerance':          {'tolerance_string'},
        'diameter_callout':   {'diameter'},
        'hole_callout':       {'raw'},
        'section_marker':     {'label'},
        'spacing_annotation': {'annotation'},
        'material_code':      {'code'},
        'part_name':          {'name'},
        'bom_header':         {'header'},
        'balloon_number':     {'number'},
        'quantity':           {'qty'},
        'dimension_with_note':{'raw'},
        'unknown':            set(),
    }

    @given(st.text(min_size=1), st.integers(min_value=0, max_value=3))
    @settings(max_examples=200)
    def test_pbt_property12_parsed_fields_schema(text, category):
        """For any text/category, extract_parsed(classify(text,cat), text) has keys matching the schema."""
        type_ = classify(text, category)
        parsed = extract_parsed(type_, text)
        expected_keys = PARSED_SCHEMAS[type_]
        assert set(parsed.keys()) == expected_keys

    # Feature: stage3-validation-structuring, Property 13: Thread spec parsed fields correct
    @given(st.from_regex(r'M\d+', fullmatch=True))
    @settings(max_examples=200)
    def test_pbt_property13_thread_spec_parsed(text):
        """For M{n} strings, extract_parsed returns nominal=M{n} and pitch=None."""
        parsed = extract_parsed("thread_spec", text)
        assert parsed["nominal"] == text.upper()
        assert parsed["pitch"] is None

    # Feature: stage3-validation-structuring, Property 14: total_detections equals classified array length
    @given(st.lists(
        st.fixed_dictionaries({
            "id": st.integers(min_value=1, max_value=100),
            "box": st.lists(st.integers(0, 1000), min_size=4, max_size=4),
            "text": st.text(min_size=0, max_size=30),
            "raw_text": st.text(min_size=0, max_size=30),
            "confidence": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        }),
        min_size=0, max_size=20
    ))
    @settings(max_examples=100, deadline=None)
    def test_pbt_property14_total_detections_consistent(entries):
        """total_detections == len(classified) for any valid input list."""
        import tempfile, json as _json
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "cad1_test_fullocr.json")
            with open(fpath, 'w', encoding='utf-8') as f:
                _json.dump(entries, f)
            result = validate_file(fpath, tmpdir)
            assert result is not None
            assert result["total_detections"] == len(result["classified"])
            assert result["total_detections"] == len(entries)

    # Feature: stage3-validation-structuring, Property 15: Summary counts consistent with classified array
    @given(st.lists(
        st.fixed_dictionaries({
            "id": st.integers(min_value=1, max_value=100),
            "box": st.lists(st.integers(0, 1000), min_size=4, max_size=4),
            "text": st.text(min_size=0, max_size=30),
            "raw_text": st.text(min_size=0, max_size=30),
            "confidence": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        }),
        min_size=0, max_size=20
    ))
    @settings(max_examples=100, deadline=None)
    def test_pbt_property15_summary_counts_consistent(entries):
        """sum(summary.values()) == total_detections and each count matches classified array."""
        import tempfile, json as _json
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "cad1_test_fullocr.json")
            with open(fpath, 'w', encoding='utf-8') as f:
                _json.dump(entries, f)
            result = validate_file(fpath, tmpdir)
            assert result is not None
            summary = result["summary"]
            classified = result["classified"]
            assert sum(summary.values()) == result["total_detections"]
            for type_key, count in summary.items():
                actual = sum(1 for e in classified if e["type"] == type_key)
                assert actual == count

    # Feature: stage3-validation-structuring, Property 16: Category detection from filename
    @given(
        st.integers(min_value=1, max_value=3),
        st.from_regex(r'[a-z0-9_]+', fullmatch=True),
    )
    @settings(max_examples=200)
    def test_pbt_property16_category_detection(cat_num, suffix):
        """For any filename containing cad1_, cad2_, or cad3_, detect_category returns 1, 2, or 3."""
        filename = f"cad{cat_num}_{suffix}_fullocr.json"
        assert detect_category(filename) == cat_num


# Helper used in property 4 to skip protected tokens
def _is_protected_helper(text):
    return text.strip().upper() in PROTECTED_CODES
