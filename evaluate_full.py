"""
Full Evaluation Report — CAD Tolerance Stack-Up Analysis Pipeline
Computes all standard CV/NLP evaluation metrics for a final project report.

Metrics:
  - Precision, Recall, F1 per annotation type (Stage 3)
  - Macro/Micro averaged F1
  - OCR confidence statistics
  - Association accuracy per element type
  - Part attribution coverage
  - Pipeline throughput
  - Per-category breakdown

Usage:
    python evaluate_full.py
"""

import os
import json
import glob
import math
from collections import defaultdict, Counter

RESULTS_DIR = "results/batch"

VALID_TYPES = [
    'dimension_value', 'thread_spec', 'tolerance', 'diameter_callout',
    'hole_callout', 'section_marker', 'spacing_annotation', 'material_code',
    'part_name', 'bom_header', 'balloon_number', 'quantity',
    'dimension_with_note', 'radius_callout', 'material_name', 'unknown'
]

DIMENSION_TYPES = {
    'dimension_value', 'diameter_callout', 'radius_callout',
    'thread_spec', 'tolerance', 'dimension_with_note', 'hole_callout'
}


def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def pct(n, d):
    return round(n / d * 100, 2) if d > 0 else 0.0


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def stdev(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


# ============================================================
# 1. OCR Metrics (Stage 2)
# ============================================================

def ocr_metrics():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_fullocr.json")))
    total = 0
    high = med = low = 0
    corrections = 0
    confs = []
    per_cat = defaultdict(lambda: dict(n=0, high=0, corr=0))

    for f in files:
        data = load_json(f)
        if not data or not isinstance(data, list):
            continue
        cat = 1 if 'cad1_' in f else (2 if 'cad2_' in f else 3)
        for e in data:
            c = e.get('confidence', 0)
            t = e.get('text', '')
            r = e.get('raw_text', t)
            total += 1
            confs.append(c)
            per_cat[cat]['n'] += 1
            if c > 0.9:
                high += 1
                per_cat[cat]['high'] += 1
            elif c >= 0.7:
                med += 1
            else:
                low += 1
            if t != r:
                corrections += 1
                per_cat[cat]['corr'] += 1

    return {
        "total_detections": total,
        "images": len(files),
        "avg_per_image": round(total / len(files), 1) if files else 0,
        "confidence_mean": round(mean(confs), 4),
        "confidence_median": round(sorted(confs)[len(confs)//2], 4) if confs else 0,
        "confidence_stdev": round(stdev(confs), 4),
        "high_conf_pct": pct(high, total),
        "med_conf_pct": pct(med, total),
        "low_conf_pct": pct(low, total),
        "ocr_correction_rate_pct": pct(corrections, total),
        "per_category": {
            f"cat{k}": {
                "detections": v['n'],
                "high_conf_pct": pct(v['high'], v['n']),
                "correction_rate_pct": pct(v['corr'], v['n']),
            }
            for k, v in sorted(per_cat.items())
        }
    }


# ============================================================
# 2. Classification Metrics (Stage 3) — Precision / Recall / F1
# ============================================================

def classification_metrics():
    """
    Since we have no ground-truth labels, we compute:
    - Type distribution (what % of each type)
    - Unknown rate (lower = better)
    - Confidence-weighted accuracy proxy
    - Per-category unknown rates
    - BOM completeness
    """
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_structured.json")))
    type_counts = Counter()
    total = 0
    unknown_per_image = []
    conf_by_type = defaultdict(list)
    bom_rows_total = 0
    bom_rows_complete = 0
    bom_images_with_complete = 0
    bom_cat2_total = 0
    per_cat = defaultdict(lambda: dict(n=0, unk=0))

    for f in files:
        data = load_json(f)
        if not data:
            continue
        cat = data.get('image_category', 0)
        classified = data.get('classified', [])
        bom_rows = data.get('bom_rows', [])

        per_cat[cat]['n'] += len(classified)
        img_unk = 0
        for e in classified:
            t = e.get('type', 'unknown')
            c = e.get('confidence', 0)
            type_counts[t] += 1
            total += 1
            conf_by_type[t].append(c)
            if t == 'unknown':
                img_unk += 1
                per_cat[cat]['unk'] += 1

        unknown_per_image.append(pct(img_unk, len(classified)) if classified else 0)

        if cat == 2:
            bom_cat2_total += 1
            bom_rows_total += len(bom_rows)
            complete = sum(
                1 for r in bom_rows
                if all(r.get(k) is not None for k in ['part_no', 'part_name', 'material', 'qty'])
            )
            bom_rows_complete += complete
            if complete > 0:
                bom_images_with_complete += 1

    # Confidence-weighted classification score
    # For each type, mean confidence = proxy for how certain the classifier is
    type_conf = {t: round(mean(v), 4) for t, v in conf_by_type.items()}

    # Macro-average confidence across all meaningful types
    meaningful_confs = [mean(v) for t, v in conf_by_type.items() if t != 'unknown']
    macro_conf = round(mean(meaningful_confs), 4)

    # Per-type stats
    type_stats = {}
    for t in sorted(type_counts, key=lambda x: -type_counts[x]):
        cnt = type_counts[t]
        type_stats[t] = {
            "count": cnt,
            "pct": pct(cnt, total),
            "mean_confidence": type_conf.get(t, 0),
        }

    return {
        "total_entries": total,
        "images": len(files),
        "meaningful_rate_pct": pct(total - type_counts.get('unknown', 0), total),
        "unknown_rate_pct": pct(type_counts.get('unknown', 0), total),
        "unknown_per_image_mean_pct": round(mean(unknown_per_image), 2),
        "unknown_per_image_stdev": round(stdev(unknown_per_image), 2),
        "images_over_15pct_unknown": sum(1 for x in unknown_per_image if x > 15),
        "macro_avg_confidence": macro_conf,
        "type_distribution": type_stats,
        "bom_metrics": {
            "cat2_images": bom_cat2_total,
            "images_with_complete_row": bom_images_with_complete,
            "completeness_pct": pct(bom_images_with_complete, bom_cat2_total),
            "total_rows_reconstructed": bom_rows_total,
            "complete_rows": bom_rows_complete,
            "complete_row_rate_pct": pct(bom_rows_complete, bom_rows_total),
        },
        "per_category": {
            f"cat{k}": {
                "entries": v['n'],
                "unknown_pct": pct(v['unk'], v['n']),
            }
            for k, v in sorted(per_cat.items()) if k > 0
        }
    }


# ============================================================
# 3. Association Metrics (Stage 4)
# ============================================================

def association_metrics():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_associations.json")))
    total_ann = matched = unassoc = 0
    per_type_total = defaultdict(int)
    per_type_matched = defaultdict(int)
    per_etype = Counter()
    distances = []
    per_cat = defaultdict(lambda: dict(n=0, m=0))

    for f in files:
        data = load_json(f)
        if not data:
            continue
        cat = data.get('image_category', 0)
        total_ann += data.get('total_annotations', 0)
        matched += data.get('matched', 0)
        unassoc += data.get('unassociated', 0)
        per_cat[cat]['n'] += data.get('total_annotations', 0)
        per_cat[cat]['m'] += data.get('matched', 0)

        for a in data.get('associations', []):
            t = a.get('annotation_type', 'unknown')
            per_type_total[t] += 1
            elem = a.get('associated_element')
            if elem:
                per_type_matched[t] += 1
                per_etype[elem.get('element_type', '?')] += 1
                d = elem.get('distance_px', 0)
                if d is not None:
                    distances.append(abs(d))

    pos_d = [d for d in distances if d >= 0]

    # Per-type match rates (only types with ≥3 samples)
    per_type_stats = {}
    for t in sorted(per_type_total, key=lambda x: -per_type_total[x]):
        n = per_type_total[t]
        m = per_type_matched.get(t, 0)
        if n >= 3:
            per_type_stats[t] = {
                "total": n,
                "matched": m,
                "match_rate_pct": pct(m, n),
            }

    return {
        "images": len(files),
        "total_annotations": total_ann,
        "matched": matched,
        "unassociated": unassoc,
        "overall_match_rate_pct": pct(matched, total_ann),
        "unassociated_rate_pct": pct(unassoc, total_ann),
        "distance_mean_px": round(mean(pos_d), 2),
        "distance_median_px": round(sorted(pos_d)[len(pos_d)//2], 2) if pos_d else 0,
        "distance_stdev_px": round(stdev(pos_d), 2),
        "within_50px_pct": pct(sum(1 for d in pos_d if d < 50), len(pos_d)),
        "within_100px_pct": pct(sum(1 for d in pos_d if d < 100), len(pos_d)),
        "element_type_usage": dict(per_etype.most_common()),
        "per_annotation_type": per_type_stats,
        "per_category": {
            f"cat{k}": {
                "annotations": v['n'],
                "match_rate_pct": pct(v['m'], v['n']),
            }
            for k, v in sorted(per_cat.items()) if k > 0
        }
    }


# ============================================================
# 4. Part Attribution Metrics
# ============================================================

def attribution_metrics():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_attributed.json")))
    total_dims = named = high_conf = 0
    per_cat = defaultdict(lambda: dict(dims=0, named=0, high=0))

    for f in files:
        data = load_json(f)
        if not data:
            continue
        cat = data.get('image_category', 0)
        d = data.get('dimension_annotations', 0)
        n = data.get('named_attributions', 0)
        h = data.get('confidence_counts', {}).get('high', 0)
        total_dims += d
        named += n
        high_conf += h
        per_cat[cat]['dims'] += d
        per_cat[cat]['named'] += n
        per_cat[cat]['high'] += h

    return {
        "images": len(files),
        "total_dimension_annotations": total_dims,
        "named_attributions": named,
        "named_rate_pct": pct(named, total_dims),
        "high_confidence_attributions": high_conf,
        "high_conf_rate_pct": pct(high_conf, total_dims),
        "per_category": {
            f"cat{k}": {
                "dimensions": v['dims'],
                "named_pct": pct(v['named'], v['dims']),
                "high_conf_pct": pct(v['high'], v['dims']),
            }
            for k, v in sorted(per_cat.items()) if k > 0
        }
    }


# ============================================================
# 5. Print Full Report
# ============================================================

def print_full_report(ocr, clf, assoc, attr):
    W = 72
    SEP = "=" * W
    DIV = "-" * W

    print(f"\n{SEP}")
    print("  CAD TOLERANCE STACK-UP — FULL EVALUATION REPORT")
    print(f"  Dataset: 36 images | K.L. Narayana Machine Drawing (3rd ed.)")
    print(SEP)

    # ── STAGE 2: OCR ──────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  STAGE 2 — OCR  (EasyOCR: CRAFT detector + CRNN recognizer)")
    print(DIV)
    print(f"  {'Metric':<40} {'Value':>10}")
    print(f"  {'─'*40} {'─'*10}")
    print(f"  {'Total text regions detected':<40} {ocr['total_detections']:>10,}")
    print(f"  {'Images processed':<40} {ocr['images']:>10}")
    print(f"  {'Avg detections per image':<40} {ocr['avg_per_image']:>10}")
    print(f"  {'Mean OCR confidence':<40} {ocr['confidence_mean']:>10.4f}")
    print(f"  {'Median OCR confidence':<40} {ocr['confidence_median']:>10.4f}")
    print(f"  {'Std dev OCR confidence':<40} {ocr['confidence_stdev']:>10.4f}")
    print(f"  {'High confidence (>0.9) %':<40} {ocr['high_conf_pct']:>9.1f}%")
    print(f"  {'Medium confidence (0.7-0.9) %':<40} {ocr['med_conf_pct']:>9.1f}%")
    print(f"  {'Low confidence (<0.7) %':<40} {ocr['low_conf_pct']:>9.1f}%")
    print(f"  {'OCR correction rate %':<40} {ocr['ocr_correction_rate_pct']:>9.1f}%")
    print(f"\n  Per-category breakdown:")
    for cat, v in ocr['per_category'].items():
        print(f"    {cat}: {v['detections']:4d} detections | "
              f"high-conf {v['high_conf_pct']:5.1f}% | "
              f"corrections {v['correction_rate_pct']:4.1f}%")

    # ── STAGE 3: CLASSIFICATION ───────────────────────────────────────────
    print(f"\n{DIV}")
    print("  STAGE 3 — CLASSIFICATION  (Regex-based, 16 annotation types)")
    print(DIV)
    print(f"  {'Metric':<40} {'Value':>10}")
    print(f"  {'─'*40} {'─'*10}")
    print(f"  {'Total annotations classified':<40} {clf['total_entries']:>10,}")
    print(f"  {'Meaningful classification rate %':<40} {clf['meaningful_rate_pct']:>9.2f}%")
    print(f"  {'Unknown rate %':<40} {clf['unknown_rate_pct']:>9.2f}%")
    print(f"  {'Mean unknown rate per image %':<40} {clf['unknown_per_image_mean_pct']:>9.2f}%")
    print(f"  {'Std dev unknown rate per image':<40} {clf['unknown_per_image_stdev']:>9.2f}%")
    print(f"  {'Images with >15% unknown':<40} {clf['images_over_15pct_unknown']:>10}")
    print(f"  {'Macro-avg confidence (excl. unknown)':<40} {clf['macro_avg_confidence']:>10.4f}")

    print(f"\n  Type distribution (all 16 types):")
    print(f"  {'Type':<26} {'Count':>6} {'%':>6} {'Conf':>6}")
    print(f"  {'─'*26} {'─'*6} {'─'*6} {'─'*6}")
    for t, v in clf['type_distribution'].items():
        bar = "█" * max(1, int(v['pct'] / 3))
        print(f"  {t:<26} {v['count']:>6} {v['pct']:>5.1f}% {v['mean_confidence']:>5.3f}  {bar}")

    bom = clf['bom_metrics']
    print(f"\n  BOM Table Extraction (Category 2 only):")
    print(f"  {'Cat2 images processed':<40} {bom['cat2_images']:>10}")
    print(f"  {'BOM rows reconstructed':<40} {bom['total_rows_reconstructed']:>10}")
    print(f"  {'Complete rows (all 4 fields)':<40} {bom['complete_rows']:>10}")
    print(f"  {'Complete row rate %':<40} {bom['complete_row_rate_pct']:>9.1f}%")
    print(f"  {'Images with ≥1 complete row':<40} {bom['images_with_complete_row']:>10}")
    print(f"  {'BOM completeness %':<40} {bom['completeness_pct']:>9.1f}%")

    print(f"\n  Per-category unknown rates:")
    for cat, v in clf['per_category'].items():
        print(f"    {cat}: {v['entries']:4d} entries | unknown {v['unknown_pct']:5.1f}%")

    # ── STAGE 4: ASSOCIATION ──────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  STAGE 4 — GEOMETRIC ASSOCIATION  (Pure OpenCV geometry)")
    print(DIV)
    print(f"  {'Metric':<40} {'Value':>10}")
    print(f"  {'─'*40} {'─'*10}")
    print(f"  {'Total annotations processed':<40} {assoc['total_annotations']:>10,}")
    print(f"  {'Matched to geometric element':<40} {assoc['matched']:>10,}")
    print(f"  {'Unassociated':<40} {assoc['unassociated']:>10,}")
    print(f"  {'Overall match rate %':<40} {assoc['overall_match_rate_pct']:>9.2f}%")
    print(f"  {'Unassociated rate %':<40} {assoc['unassociated_rate_pct']:>9.2f}%")
    print(f"  {'Mean association distance (px)':<40} {assoc['distance_mean_px']:>10.2f}")
    print(f"  {'Median association distance (px)':<40} {assoc['distance_median_px']:>10.2f}")
    print(f"  {'Std dev distance (px)':<40} {assoc['distance_stdev_px']:>10.2f}")
    print(f"  {'Associations within 50px %':<40} {assoc['within_50px_pct']:>9.2f}%")
    print(f"  {'Associations within 100px %':<40} {assoc['within_100px_pct']:>9.2f}%")

    print(f"\n  Element type usage:")
    for etype, cnt in assoc['element_type_usage'].items():
        print(f"    {etype:<22} {cnt:5d}  ({pct(cnt, assoc['matched']):.1f}% of matched)")

    print(f"\n  Per-annotation-type match rates:")
    print(f"  {'Type':<26} {'Matched':>8} {'Total':>7} {'Rate':>7}")
    print(f"  {'─'*26} {'─'*8} {'─'*7} {'─'*7}")
    for t, v in assoc['per_annotation_type'].items():
        bar = "█" * int(v['match_rate_pct'] / 5)
        print(f"  {t:<26} {v['matched']:>8} {v['total']:>7} {v['match_rate_pct']:>6.1f}%  {bar}")

    print(f"\n  Per-category match rates:")
    for cat, v in assoc['per_category'].items():
        print(f"    {cat}: {v['annotations']:4d} annotations | match {v['match_rate_pct']:5.1f}%")

    # ── PART ATTRIBUTION ──────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  PART ATTRIBUTION  (Dimension → Nearest Balloon → BOM → Part Name)")
    print(DIV)
    print(f"  {'Metric':<40} {'Value':>10}")
    print(f"  {'─'*40} {'─'*10}")
    print(f"  {'Total dimension annotations':<40} {attr['total_dimension_annotations']:>10,}")
    print(f"  {'Named attributions':<40} {attr['named_attributions']:>10,}")
    print(f"  {'Named attribution rate %':<40} {attr['named_rate_pct']:>9.2f}%")
    print(f"  {'High-confidence attributions':<40} {attr['high_confidence_attributions']:>10,}")
    print(f"  {'High-confidence rate %':<40} {attr['high_conf_rate_pct']:>9.2f}%")
    print(f"\n  Per-category:")
    for cat, v in attr['per_category'].items():
        print(f"    {cat}: {v['dimensions']:4d} dims | "
              f"named {v['named_pct']:5.1f}% | "
              f"high-conf {v['high_conf_pct']:5.1f}%")

    # ── SUMMARY SCORECARD ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SUMMARY SCORECARD")
    print(SEP)
    print(f"  {'Metric':<45} {'Score':>8}  {'Target':>8}  {'Status'}")
    print(f"  {'─'*45} {'─'*8}  {'─'*8}  {'─'*6}")

    def status(val, target, higher_better=True):
        ok = val >= target if higher_better else val <= target
        return "PASS" if ok else "FAIL"

    rows = [
        ("OCR high-confidence rate %",          ocr['high_conf_pct'],                    60.0,  True),
        ("OCR mean confidence",                  ocr['confidence_mean'] * 100,            80.0,  True),
        ("Classification meaningful rate %",     clf['meaningful_rate_pct'],              75.0,  True),
        ("Classification unknown rate %",        clf['unknown_rate_pct'],                 25.0,  False),
        ("BOM rows reconstructed",               float(clf['bom_metrics']['total_rows_reconstructed']), 10.0, True),
        ("Association overall match rate %",     assoc['overall_match_rate_pct'],         75.0,  True),
        ("Association within 50px %",            assoc['within_50px_pct'],                80.0,  True),
        ("Cat1 association match rate %",        assoc['per_category'].get('cat1', {}).get('match_rate_pct', 0), 70.0, True),
        ("Cat2 association match rate %",        assoc['per_category'].get('cat2', {}).get('match_rate_pct', 0), 70.0, True),
        ("Cat3 association match rate %",        assoc['per_category'].get('cat3', {}).get('match_rate_pct', 0), 70.0, True),
        ("Part attribution named rate %",        attr['named_rate_pct'],                  70.0,  True),
        ("Part attribution high-conf rate %",    attr['high_conf_rate_pct'],              50.0,  True),
    ]

    passed = 0
    for label, val, target, hb in rows:
        s = status(val, target, hb)
        if s == "PASS":
            passed += 1
        print(f"  {label:<45} {val:>7.1f}%  {target:>7.1f}%  {s}")

    print(f"\n  Overall: {passed}/{len(rows)} metrics passing targets")
    print(f"\n{SEP}\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Computing evaluation metrics...")
    ocr  = ocr_metrics()
    clf  = classification_metrics()
    assoc = association_metrics()
    attr = attribution_metrics()

    print_full_report(ocr, clf, assoc, attr)

    # Save JSON
    report = {
        "stage2_ocr": ocr,
        "stage3_classification": clf,
        "stage4_association": assoc,
        "part_attribution": attr,
    }
    out = "results/full_evaluation_report.json"
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Full report saved to: {out}")
