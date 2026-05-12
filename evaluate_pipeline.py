"""
Pipeline Evaluation Metrics — CAD Tolerance Stack-Up Analysis Tool
Computes all evaluation metrics from existing JSON outputs (no manual labelling needed).

Metrics computed:
  Stage 2 (OCR):        confidence distribution, detection counts, OCR correction rate
  Stage 3 (Classifier): type distribution, unknown rate, BOM completeness
  Stage 4 (Association): match rate, per-type match rate, distance distribution
  Overall:              throughput, per-category summary, CV component ratio

Usage:
    python evaluate_pipeline.py
    python evaluate_pipeline.py --results results/batch --output metrics_report.json
"""

import os
import sys
import json
import glob
import argparse
import math
from collections import defaultdict, Counter


# ============================================================
# Config
# ============================================================

RESULTS_DIR = "results/batch"
VALID_TYPES = {
    'dimension_value', 'thread_spec', 'tolerance', 'diameter_callout',
    'hole_callout', 'section_marker', 'spacing_annotation', 'material_code',
    'part_name', 'bom_header', 'balloon_number', 'quantity',
    'dimension_with_note', 'unknown', 'radius_callout', 'material_name',
}

# Types that are "meaningful" (not noise/unknown)
MEANINGFUL_TYPES = VALID_TYPES - {'unknown'}

# Types that represent geometric dimensions (for Stage 4 accuracy)
DIMENSION_TYPES = {
    'dimension_value', 'diameter_callout', 'radius_callout',
    'thread_spec', 'tolerance', 'dimension_with_note', 'hole_callout'
}


# ============================================================
# Helpers
# ============================================================

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def pct(num, denom):
    return round(num / denom * 100, 1) if denom > 0 else 0.0


def mean(values):
    return sum(values) / len(values) if values else 0.0


def median(values):
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def stdev(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


# ============================================================
# Stage 2 Metrics — OCR
# ============================================================

def compute_ocr_metrics(results_dir):
    """Compute OCR quality metrics from _fullocr.json files."""
    fullocr_files = sorted(glob.glob(os.path.join(results_dir, "*_fullocr.json")))

    total_detections = 0
    high_conf = 0   # > 0.9
    med_conf  = 0   # 0.7 - 0.9
    low_conf  = 0   # < 0.7
    corrections_applied = 0   # text != raw_text
    all_confidences = []
    per_category = defaultdict(lambda: {"images": 0, "detections": 0, "high": 0, "med": 0, "low": 0})

    for path in fullocr_files:
        data = load_json(path)
        if not data or not isinstance(data, list):
            continue

        basename = os.path.basename(path)
        cat = 1 if 'cad1_' in basename else (2 if 'cad2_' in basename else 3)
        per_category[cat]["images"] += 1

        for entry in data:
            conf = entry.get("confidence", 0)
            text = entry.get("text", "")
            raw  = entry.get("raw_text", text)

            total_detections += 1
            all_confidences.append(conf)
            per_category[cat]["detections"] += 1

            if conf > 0.9:
                high_conf += 1
                per_category[cat]["high"] += 1
            elif conf >= 0.7:
                med_conf += 1
                per_category[cat]["med"] += 1
            else:
                low_conf += 1
                per_category[cat]["low"] += 1

            if text != raw:
                corrections_applied += 1

    return {
        "total_images":         len(fullocr_files),
        "total_detections":     total_detections,
        "avg_detections_per_image": round(mean([
            len(load_json(p) or []) for p in fullocr_files
        ]), 1),
        "confidence": {
            "high_pct":   pct(high_conf, total_detections),
            "medium_pct": pct(med_conf, total_detections),
            "low_pct":    pct(low_conf, total_detections),
            "mean":       round(mean(all_confidences), 3),
            "median":     round(median(all_confidences), 3),
        },
        "ocr_correction_rate_pct": pct(corrections_applied, total_detections),
        "corrections_applied":     corrections_applied,
        "per_category": {
            f"cat{k}": {
                "images":     v["images"],
                "detections": v["detections"],
                "avg_per_image": round(v["detections"] / v["images"], 1) if v["images"] else 0,
                "high_conf_pct": pct(v["high"], v["detections"]),
            }
            for k, v in sorted(per_category.items())
        }
    }


# ============================================================
# Stage 3 Metrics — Classification
# ============================================================

def compute_classification_metrics(results_dir):
    """Compute classification quality metrics from _structured.json files."""
    structured_files = sorted(glob.glob(os.path.join(results_dir, "*_structured.json")))

    total_entries = 0
    type_counts = Counter()
    unknown_counts_per_image = []
    bom_complete_images = 0
    bom_total_cat2 = 0
    bom_rows_total = 0
    bom_rows_complete = 0
    per_category = defaultdict(lambda: {
        "images": 0, "entries": 0, "unknown": 0, "types": Counter()
    })

    for path in structured_files:
        data = load_json(path)
        if not data:
            continue

        cat = data.get("image_category", 0)
        classified = data.get("classified", [])
        bom_rows = data.get("bom_rows", [])

        per_category[cat]["images"] += 1
        per_category[cat]["entries"] += len(classified)

        image_unknown = 0
        for entry in classified:
            t = entry.get("type", "unknown")
            type_counts[t] += 1
            total_entries += 1
            per_category[cat]["types"][t] += 1
            if t == "unknown":
                image_unknown += 1
                per_category[cat]["unknown"] += 1

        unknown_counts_per_image.append(
            pct(image_unknown, len(classified)) if classified else 0
        )

        # BOM metrics (Category 2 only)
        if cat == 2:
            bom_total_cat2 += 1
            bom_rows_total += len(bom_rows)
            complete = sum(
                1 for r in bom_rows
                if r.get("part_no") is not None
                and r.get("part_name") is not None
                and r.get("material") is not None
                and r.get("qty") is not None
            )
            bom_rows_complete += complete
            if complete > 0:
                bom_complete_images += 1

    unknown_total = type_counts.get("unknown", 0)
    meaningful_total = total_entries - unknown_total

    return {
        "total_images":   len(structured_files),
        "total_entries":  total_entries,
        "unknown_rate_pct": pct(unknown_total, total_entries),
        "meaningful_rate_pct": pct(meaningful_total, total_entries),
        "type_distribution": {
            t: {"count": type_counts[t], "pct": pct(type_counts[t], total_entries)}
            for t in sorted(type_counts, key=lambda x: -type_counts[x])
        },
        "unknown_rate_per_image": {
            "mean_pct":   round(mean(unknown_counts_per_image), 1),
            "median_pct": round(median(unknown_counts_per_image), 1),
            "max_pct":    round(max(unknown_counts_per_image), 1) if unknown_counts_per_image else 0,
            "images_over_15pct": sum(1 for x in unknown_counts_per_image if x > 15),
        },
        "bom_metrics": {
            "cat2_images_with_bom_rows": bom_complete_images,
            "cat2_total_images":         bom_total_cat2,
            "bom_completeness_pct":      pct(bom_complete_images, bom_total_cat2),
            "total_bom_rows":            bom_rows_total,
            "complete_bom_rows":         bom_rows_complete,
            "complete_row_rate_pct":     pct(bom_rows_complete, bom_rows_total),
        },
        "per_category": {
            f"cat{k}": {
                "images":          v["images"],
                "entries":         v["entries"],
                "unknown_pct":     pct(v["unknown"], v["entries"]),
                "top_types":       dict(v["types"].most_common(5)),
            }
            for k, v in sorted(per_category.items()) if k > 0
        }
    }


# ============================================================
# Stage 4 Metrics — Geometric Association
# ============================================================

def compute_association_metrics(results_dir):
    """Compute association quality metrics from _associations.json files."""
    assoc_files = sorted(glob.glob(os.path.join(results_dir, "*_associations.json")))

    total_annotations = 0
    total_matched = 0
    total_unassociated = 0

    per_type_matched  = defaultdict(int)
    per_type_total    = defaultdict(int)
    per_etype_count   = Counter()   # element types used

    all_distances = []
    distances_by_etype = defaultdict(list)

    per_category = defaultdict(lambda: {
        "images": 0, "annotations": 0, "matched": 0, "unassociated": 0
    })

    for path in assoc_files:
        data = load_json(path)
        if not data:
            continue

        cat = data.get("image_category", 0)
        per_category[cat]["images"] += 1
        per_category[cat]["annotations"] += data.get("total_annotations", 0)
        per_category[cat]["matched"]     += data.get("matched", 0)
        per_category[cat]["unassociated"] += data.get("unassociated", 0)

        total_annotations  += data.get("total_annotations", 0)
        total_matched      += data.get("matched", 0)
        total_unassociated += data.get("unassociated", 0)

        for assoc in data.get("associations", []):
            ann_type = assoc.get("annotation_type", "unknown")
            elem     = assoc.get("associated_element")

            per_type_total[ann_type] += 1

            if elem is not None:
                per_type_matched[ann_type] += 1
                etype = elem.get("element_type", "unknown")
                dist  = elem.get("distance_px", 0)
                per_etype_count[etype] += 1
                all_distances.append(dist)
                distances_by_etype[etype].append(dist)

    # Per-type match rates
    per_type_stats = {}
    for t in sorted(per_type_total, key=lambda x: -per_type_total[x]):
        total = per_type_total[t]
        matched = per_type_matched.get(t, 0)
        per_type_stats[t] = {
            "total":      total,
            "matched":    matched,
            "match_pct":  pct(matched, total),
        }

    # Distance statistics (exclude negative values from inside-circle matches)
    pos_distances = [d for d in all_distances if d >= 0]

    return {
        "total_images":      len(assoc_files),
        "total_annotations": total_annotations,
        "total_matched":     total_matched,
        "total_unassociated": total_unassociated,
        "overall_match_pct": pct(total_matched, total_annotations),
        "unassociated_pct":  pct(total_unassociated, total_annotations),
        "distance_stats": {
            "mean_px":   round(mean(pos_distances), 1),
            "median_px": round(median(pos_distances), 1),
            "stdev_px":  round(stdev(pos_distances), 1),
            "min_px":    round(min(pos_distances), 1) if pos_distances else 0,
            "max_px":    round(max(pos_distances), 1) if pos_distances else 0,
            "pct_under_50px":  pct(sum(1 for d in pos_distances if d < 50), len(pos_distances)),
            "pct_under_100px": pct(sum(1 for d in pos_distances if d < 100), len(pos_distances)),
        },
        "element_type_usage": dict(per_etype_count.most_common()),
        "per_annotation_type": per_type_stats,
        "per_category": {
            f"cat{k}": {
                "images":       v["images"],
                "annotations":  v["annotations"],
                "match_pct":    pct(v["matched"], v["annotations"]),
                "unassoc_pct":  pct(v["unassociated"], v["annotations"]),
            }
            for k, v in sorted(per_category.items()) if k > 0
        }
    }


# ============================================================
# Overall Pipeline Metrics
# ============================================================

def compute_overall_metrics(ocr, clf, assoc, batch_summary_path):
    """Compute overall pipeline metrics."""
    batch = load_json(batch_summary_path) or {}

    # Timing from batch_summary.json
    all_times = []
    for cat_stats in batch.values():
        for img in cat_stats:
            if "time_seconds" in img:
                all_times.append(img["time_seconds"])

    # CV component ratio (from project design)
    # Stage 1 (preprocessing): pure CV
    # Stage 1.5 (element detection): pure CV
    # Stage 2 (EasyOCR): CNN-based (CRAFT+CRNN) — counts as CV
    # Stage 3 (validation): pure Python regex — NOT CV
    # Stage 4 (association): pure CV geometry
    # Stage 5 (Flask): utility — NOT CV
    cv_stages = ["Stage 1 (Preprocessing)", "Stage 1.5 (Element Detection)",
                 "Stage 2 (EasyOCR CRAFT+CRNN)", "Stage 4 (Geometric Association)"]
    non_cv_stages = ["Stage 3 (Validation/Regex)", "Stage 5 (Flask)"]
    cv_ratio = len(cv_stages) / (len(cv_stages) + len(non_cv_stages))

    return {
        "pipeline_stages_complete": 4,
        "pipeline_stages_total":    5,
        "cv_component_ratio_pct":   round(cv_ratio * 100, 1),
        "cv_stages":     cv_stages,
        "non_cv_stages": non_cv_stages,
        "timing": {
            "total_images_timed": len(all_times),
            "mean_ocr_time_sec":  round(mean(all_times), 1),
            "median_ocr_time_sec": round(median(all_times), 1),
            "total_ocr_time_sec": round(sum(all_times), 1),
            "note": "OCR (Stage 2) dominates timing; Stages 3+4 add ~1-2s per image"
        },
        "dataset": {
            "total_images": 36,
            "category_1":   23,
            "category_2":   11,
            "category_3":    2,
            "source":       "K.L. Narayana Machine Drawing textbook (3rd ed.)"
        }
    }


# ============================================================
# Print Report
# ============================================================

def print_report(metrics):
    """Print a formatted evaluation report to stdout."""
    sep = "=" * 70

    print(f"\n{sep}")
    print("  CAD TOLERANCE STACK-UP PIPELINE — EVALUATION REPORT")
    print(sep)

    # Overall
    ov = metrics["overall"]
    print(f"\n{'─'*70}")
    print("  OVERALL PIPELINE")
    print(f"{'─'*70}")
    print(f"  Stages complete:     {ov['pipeline_stages_complete']}/{ov['pipeline_stages_total']}")
    print(f"  CV component ratio:  {ov['cv_component_ratio_pct']}%  (teacher requirement: ≥70%)")
    print(f"  Dataset:             {ov['dataset']['total_images']} images "
          f"(Cat1={ov['dataset']['category_1']}, Cat2={ov['dataset']['category_2']}, "
          f"Cat3={ov['dataset']['category_3']})")
    print(f"  Mean OCR time/image: {ov['timing']['mean_ocr_time_sec']}s")

    # Stage 2
    ocr = metrics["stage2_ocr"]
    print(f"\n{'─'*70}")
    print("  STAGE 2 — OCR (EasyOCR CRAFT+CRNN)")
    print(f"{'─'*70}")
    print(f"  Images processed:    {ocr['total_images']}")
    print(f"  Total detections:    {ocr['total_detections']}")
    print(f"  Avg per image:       {ocr['avg_detections_per_image']}")
    c = ocr["confidence"]
    print(f"  Confidence:          High(>0.9)={c['high_pct']}%  "
          f"Med(0.7-0.9)={c['medium_pct']}%  Low(<0.7)={c['low_pct']}%")
    print(f"  Mean confidence:     {c['mean']}  (median: {c['median']})")
    print(f"  OCR corrections:     {ocr['corrections_applied']} entries "
          f"({ocr['ocr_correction_rate_pct']}%) — text≠raw_text")
    print(f"\n  Per-category:")
    for cat, v in ocr["per_category"].items():
        print(f"    {cat}: {v['images']} images | {v['detections']} detections "
              f"| avg {v['avg_per_image']}/img | high-conf {v['high_conf_pct']}%")

    # Stage 3
    clf = metrics["stage3_classification"]
    print(f"\n{'─'*70}")
    print("  STAGE 3 — CLASSIFICATION (Regex-based, 16 types)")
    print(f"{'─'*70}")
    print(f"  Total entries:       {clf['total_entries']}")
    print(f"  Meaningful rate:     {clf['meaningful_rate_pct']}%  (non-unknown)")
    print(f"  Unknown rate:        {clf['unknown_rate_pct']}%")
    u = clf["unknown_rate_per_image"]
    print(f"  Unknown per image:   mean={u['mean_pct']}%  median={u['median_pct']}%  "
          f"max={u['max_pct']}%  images>15%: {u['images_over_15pct']}")
    print(f"\n  Type distribution (top 8):")
    for i, (t, v) in enumerate(clf["type_distribution"].items()):
        if i >= 8:
            break
        bar = "█" * int(v["pct"] / 2)
        print(f"    {t:<22} {v['count']:5d}  {v['pct']:5.1f}%  {bar}")
    bom = clf["bom_metrics"]
    print(f"\n  BOM extraction (Cat 2):")
    print(f"    Images with ≥1 complete BOM row: {bom['cat2_images_with_bom_rows']}"
          f"/{bom['cat2_total_images']} ({bom['bom_completeness_pct']}%)")
    print(f"    Total BOM rows reconstructed:    {bom['total_bom_rows']}")
    print(f"    Complete rows (all 4 fields):    {bom['complete_bom_rows']} "
          f"({bom['complete_row_rate_pct']}%)")
    print(f"\n  Per-category:")
    for cat, v in clf["per_category"].items():
        print(f"    {cat}: {v['images']} images | {v['entries']} entries | "
              f"unknown={v['unknown_pct']}%")

    # Stage 4
    assoc = metrics["stage4_association"]
    print(f"\n{'─'*70}")
    print("  STAGE 4 — GEOMETRIC ASSOCIATION (OpenCV geometry)")
    print(f"{'─'*70}")
    print(f"  Images processed:    {assoc['total_images']}")
    print(f"  Total annotations:   {assoc['total_annotations']}")
    print(f"  Overall match rate:  {assoc['overall_match_pct']}%  "
          f"(target: ≥80%)")
    print(f"  Unassociated rate:   {assoc['unassociated_pct']}%  "
          f"(target: ≤20%)")
    d = assoc["distance_stats"]
    print(f"  Distance stats:      mean={d['mean_px']}px  median={d['median_px']}px  "
          f"stdev={d['stdev_px']}px")
    print(f"  Within 50px:         {d['pct_under_50px']}%  |  "
          f"Within 100px: {d['pct_under_100px']}%")
    print(f"\n  Element type usage:")
    for etype, count in assoc["element_type_usage"].items():
        print(f"    {etype:<20} {count:5d}")
    print(f"\n  Per-annotation-type match rates:")
    for t, v in assoc["per_annotation_type"].items():
        if v["total"] < 3:
            continue
        bar = "█" * int(v["match_pct"] / 5)
        print(f"    {t:<22} {v['matched']:4d}/{v['total']:4d}  "
              f"{v['match_pct']:5.1f}%  {bar}")
    print(f"\n  Per-category:")
    for cat, v in assoc["per_category"].items():
        print(f"    {cat}: {v['images']} images | match={v['match_pct']}% | "
              f"unassoc={v['unassoc_pct']}%")

    print(f"\n{sep}")
    print("  END OF REPORT")
    print(f"{sep}\n")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate CAD pipeline metrics")
    parser.add_argument("--results", default=RESULTS_DIR,
                        help="Directory containing pipeline output JSON files")
    parser.add_argument("--output", default=None,
                        help="Optional path to save metrics as JSON")
    args = parser.parse_args()

    results_dir = args.results
    batch_summary = os.path.join(results_dir, "batch_summary.json")

    print(f"Computing metrics from: {results_dir}")

    ocr_metrics   = compute_ocr_metrics(results_dir)
    clf_metrics   = compute_classification_metrics(results_dir)
    assoc_metrics = compute_association_metrics(results_dir)
    overall       = compute_overall_metrics(ocr_metrics, clf_metrics, assoc_metrics, batch_summary)

    metrics = {
        "overall":               overall,
        "stage2_ocr":            ocr_metrics,
        "stage3_classification": clf_metrics,
        "stage4_association":    assoc_metrics,
    }

    print_report(metrics)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to: {args.output}")

    return metrics


if __name__ == "__main__":
    main()
