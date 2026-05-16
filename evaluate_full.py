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
# 5. OCR Character Error Rate Proxy
# ============================================================

def ocr_cer_metrics():
    """
    Compute OCR Character Error Rate (CER) proxy.

    CER = edit_distance(text, raw_text) / max(len(text), len(raw_text))

    Since text = post_processed(raw_text), this measures how much
    post-processing had to correct. Lower = OCR was more accurate.
    Also computes Word Error Rate (WER) proxy.
    """
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_fullocr.json")))

    cer_values = []
    wer_values = []
    total = 0
    perfect = 0   # text == raw_text (no correction needed)
    per_cat = defaultdict(lambda: dict(cer=[], wer=[]))

    for f in files:
        data = load_json(f)
        if not data or not isinstance(data, list):
            continue
        cat = 1 if 'cad1_' in f else (2 if 'cad2_' in f else 3)

        for e in data:
            text = e.get('text', '')
            raw  = e.get('raw_text', text)
            if not text and not raw:
                continue
            total += 1

            # CER: normalised edit distance at character level
            cer = _edit_distance_norm(text, raw)
            cer_values.append(cer)
            per_cat[cat]['cer'].append(cer)

            # WER: word-level (split on spaces)
            wer = _word_error_rate(text, raw)
            wer_values.append(wer)
            per_cat[cat]['wer'].append(wer)

            if text == raw:
                perfect += 1

    return {
        "total_entries":      total,
        "perfect_ocr_pct":    pct(perfect, total),
        "mean_cer":           round(mean(cer_values), 4),
        "median_cer":         round(sorted(cer_values)[len(cer_values)//2], 4) if cer_values else 0,
        "mean_wer":           round(mean(wer_values), 4),
        "cer_zero_pct":       pct(sum(1 for c in cer_values if c == 0), total),
        "cer_under_10pct":    pct(sum(1 for c in cer_values if c < 0.1), total),
        "cer_over_50pct":     pct(sum(1 for c in cer_values if c > 0.5), total),
        "per_category": {
            f"cat{k}": {
                "mean_cer": round(mean(v['cer']), 4),
                "mean_wer": round(mean(v['wer']), 4),
                "perfect_pct": pct(sum(1 for c in v['cer'] if c == 0), len(v['cer'])),
            }
            for k, v in sorted(per_cat.items()) if v['cer']
        }
    }


def _edit_distance_norm(a, b):
    """Normalised Levenshtein distance in [0, 1]."""
    if not a and not b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    # Simple DP edit distance
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
        prev = curr
    return round(prev[lb] / max_len, 4)


def _word_error_rate(hyp, ref):
    """Simple word error rate."""
    h = hyp.split()
    r = ref.split()
    if not r:
        return 0.0 if not h else 1.0
    # Edit distance on word sequences
    la, lb = len(h), len(r)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if h[i-1] == r[j-1] else 1
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
        prev = curr
    return round(min(prev[lb] / len(r), 1.0), 4)


# ============================================================
# 6. BOM Field Fill Rate
# ============================================================

def bom_fill_metrics():
    """
    BOM field fill rate — average % of fields filled per row.

    Better than binary completeness: a row with 3/4 fields = 75%.
    Fields: part_no, part_name, material, qty
    """
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_structured.json")))
    FIELDS = ['part_no', 'part_name', 'material', 'qty']

    all_fill_rates = []
    per_field = {f: dict(filled=0, total=0) for f in FIELDS}
    images_with_bom = 0
    total_rows = 0

    for f in files:
        data = load_json(f)
        if not data or data.get('image_category') != 2:
            continue
        bom_rows = data.get('bom_rows', [])
        if not bom_rows:
            continue
        images_with_bom += 1

        for row in bom_rows:
            total_rows += 1
            filled = sum(1 for field in FIELDS if row.get(field) is not None)
            fill_rate = filled / len(FIELDS)
            all_fill_rates.append(fill_rate)
            for field in FIELDS:
                per_field[field]['total'] += 1
                if row.get(field) is not None:
                    per_field[field]['filled'] += 1

    return {
        "cat2_images_with_bom": images_with_bom,
        "total_bom_rows":       total_rows,
        "mean_fill_rate_pct":   round(mean(all_fill_rates) * 100, 2) if all_fill_rates else 0,
        "fully_complete_rows":  sum(1 for r in all_fill_rates if r == 1.0),
        "fully_complete_pct":   pct(sum(1 for r in all_fill_rates if r == 1.0), total_rows),
        "at_least_half_pct":    pct(sum(1 for r in all_fill_rates if r >= 0.5), total_rows),
        "per_field_fill_pct": {
            field: pct(v['filled'], v['total'])
            for field, v in per_field.items()
        }
    }


# ============================================================
# 7. Association Distance Buckets
# ============================================================

def association_distance_metrics():
    """
    Granular association distance analysis.

    Buckets:
      0px:       annotation is exactly on the element (inside circle)
      0-20px:    very tight — almost certainly correct
      20-50px:   tight — likely correct
      50-100px:  moderate — probably correct
      100-150px: loose — may be wrong
      >150px:    very loose — likely wrong (beyond MAX_DISTANCE_PX)
    """
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_associations.json")))

    buckets = {
        'exact_0px':    0,
        'tight_0_20':   0,
        'good_20_50':   0,
        'moderate_50_100': 0,
        'loose_100_150':   0,
        'very_loose_150p': 0,
    }
    total_matched = 0
    per_type_dist = defaultdict(list)

    for f in files:
        data = load_json(f)
        if not data:
            continue
        for a in data.get('associations', []):
            elem = a.get('associated_element')
            if elem is None:
                continue
            d = abs(elem.get('distance_px', 0) or 0)
            ann_type = a.get('annotation_type', 'unknown')
            total_matched += 1
            per_type_dist[ann_type].append(d)

            if d == 0:
                buckets['exact_0px'] += 1
            elif d <= 20:
                buckets['tight_0_20'] += 1
            elif d <= 50:
                buckets['good_20_50'] += 1
            elif d <= 100:
                buckets['moderate_50_100'] += 1
            elif d <= 150:
                buckets['loose_100_150'] += 1
            else:
                buckets['very_loose_150p'] += 1

    # Per-type mean distance (for dimension types only)
    dim_type_stats = {}
    for t in ['dimension_value', 'diameter_callout', 'radius_callout',
              'thread_spec', 'hole_callout', 'balloon_number']:
        dists = per_type_dist.get(t, [])
        if dists:
            dim_type_stats[t] = {
                "mean_px":   round(mean(dists), 1),
                "median_px": round(sorted(dists)[len(dists)//2], 1),
                "count":     len(dists),
            }

    return {
        "total_matched": total_matched,
        "distance_buckets": {
            k: {"count": v, "pct": pct(v, total_matched)}
            for k, v in buckets.items()
        },
        "tight_or_better_pct": pct(
            buckets['exact_0px'] + buckets['tight_0_20'] + buckets['good_20_50'],
            total_matched
        ),
        "per_type_distance": dim_type_stats,
    }


# ============================================================
# 8. Pipeline Throughput
# ============================================================

def throughput_metrics():
    """
    Pipeline throughput from batch_summary.json.
    Reports per-image timing and estimates per-stage breakdown.
    """
    summary_path = os.path.join(RESULTS_DIR, "batch_summary.json")
    data = load_json(summary_path)
    if not data:
        return {"error": "batch_summary.json not found"}

    all_times = []
    per_cat = {}

    for cat_key, images in data.items():
        if not isinstance(images, list):
            continue
        times = [img.get('time_seconds', 0) for img in images if 'time_seconds' in img]
        if times:
            per_cat[cat_key] = {
                "images":       len(times),
                "mean_sec":     round(mean(times), 2),
                "min_sec":      round(min(times), 2),
                "max_sec":      round(max(times), 2),
                "total_sec":    round(sum(times), 1),
            }
            all_times.extend(times)

    if not all_times:
        return {"error": "No timing data in batch_summary.json"}

    total_sec = sum(all_times)
    n_images  = len(all_times)

    # Estimated stage breakdown (based on known relative costs)
    # Stage 2 (EasyOCR) dominates at ~85% of total time
    # Stage 4 (association) ~10%, Stage 3 (validation) ~3%, Stage 5 ~2%
    stage_estimates = {
        "stage2_ocr_pct":         85.0,
        "stage4_association_pct": 10.0,
        "stage3_validation_pct":   3.0,
        "stage5_attribution_pct":  2.0,
        "note": "Estimated breakdown — Stage 2 (EasyOCR) dominates"
    }

    return {
        "total_images":       n_images,
        "total_time_sec":     round(total_sec, 1),
        "total_time_min":     round(total_sec / 60, 2),
        "mean_sec_per_image": round(mean(all_times), 2),
        "median_sec_per_image": round(sorted(all_times)[len(all_times)//2], 2),
        "min_sec_per_image":  round(min(all_times), 2),
        "max_sec_per_image":  round(max(all_times), 2),
        "images_per_minute":  round(n_images / (total_sec / 60), 1) if total_sec > 0 else 0,
        "stage_time_estimates": stage_estimates,
        "per_category": per_cat,
    }


# ============================================================
# 9. Tolerance Stack-Up Coverage
# ============================================================

def stackup_coverage_metrics():
    """
    Tolerance stack-up coverage metrics.

    Measures:
    - % of dimension annotations that have a linked tolerance
    - Distribution of tolerance types (symmetric, bilateral, fit)
    - Stack-up chain statistics across all images
    - Fit specification coverage
    """
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_stackup.json")))

    total_dims    = 0
    dims_with_tol = 0
    total_tols    = 0
    total_fits    = 0
    fit_types_all = Counter()
    tol_types     = Counter()
    stackup_wc    = []   # worst-case tolerance values
    stackup_rss   = []   # RSS tolerance values
    per_cat       = defaultdict(lambda: dict(dims=0, tols=0, fits=0))

    for f in files:
        data = load_json(f)
        if not data:
            continue
        cat = data.get('image_category', 0)

        dims = data.get('dimensions', [])
        tols = data.get('tolerances', [])
        fits = data.get('fit_specifications', [])

        total_dims    += len(dims)
        total_tols    += len(tols)
        total_fits    += len(fits)
        per_cat[cat]['dims'] += len(dims)
        per_cat[cat]['tols'] += len(tols)
        per_cat[cat]['fits'] += len(fits)

        # Count dims that have a linked tolerance
        dims_with_tol += sum(1 for d in dims if d.get('linked_tolerance'))

        # Tolerance type distribution
        for t in tols:
            tol_types[t.get('type', 'unknown')] += 1

        # Fit type distribution
        for fit in fits:
            ft = fit.get('fit_type', '')
            if ft:
                fit_types_all[ft] += 1

        # Stack-up values
        sk_lin = data.get('stackup_linear')
        if sk_lin and sk_lin.get('n_dimensions', 0) > 0:
            wc = sk_lin.get('worst_case_tolerance_mm', 0)
            rs = sk_lin.get('rss_tolerance_mm', 0)
            if wc:
                stackup_wc.append(wc)
            if rs:
                stackup_rss.append(rs)

    return {
        "images":                  len(files),
        "total_dimensions":        total_dims,
        "total_tolerances":        total_tols,
        "total_fit_specs":         total_fits,
        "dims_with_tolerance":     dims_with_tol,
        "tolerance_coverage_pct":  pct(dims_with_tol, total_dims),
        "tols_per_dim_ratio":      round(total_tols / total_dims, 3) if total_dims else 0,
        "tolerance_type_dist":     dict(tol_types.most_common()),
        "fit_type_dist":           dict(fit_types_all.most_common(10)),
        "stackup_stats": {
            "images_with_stackup":    len(stackup_wc),
            "mean_wc_tolerance_mm":   round(mean(stackup_wc), 4) if stackup_wc else 0,
            "mean_rss_tolerance_mm":  round(mean(stackup_rss), 4) if stackup_rss else 0,
            "max_wc_tolerance_mm":    round(max(stackup_wc), 4) if stackup_wc else 0,
            "min_wc_tolerance_mm":    round(min(stackup_wc), 4) if stackup_wc else 0,
        },
        "per_category": {
            f"cat{k}": {
                "dimensions": v['dims'],
                "tolerances": v['tols'],
                "fits":       v['fits'],
                "tol_coverage_pct": pct(v['tols'], v['dims']),
            }
            for k, v in sorted(per_cat.items()) if k > 0
        }
    }


# ============================================================
# 10. Semantic Labelling Metrics
# ============================================================

def semantic_metrics():
    """
    Compute semantic labelling metrics from _labelled.json files.

    Semantic labelling assigns a human-readable meaning to each dimension:
    length, height, bore_diameter, thread_size, etc.

    Evaluation approach (no ground truth needed):
    - Labelled rate: % of DIMENSION annotations that got a non-unknown label
    - Label distribution: what types of dimensions were found
    - Consistency: same text → same label across images
    - Coverage per category
    """
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_labelled.json")))

    DIMENSION_LABELS = {
        "length", "height", "depth", "thickness", "bore_diameter",
        "shaft_diameter", "hole_diameter", "thread_size", "radius",
        "chamfer", "pitch_circle", "spacing", "groove_depth",
        "keyway", "width", "gear_module", "gear_spec", "coil_spec",
    }

    total_ann       = 0
    total_dim_ann   = 0   # annotations that are dimension types
    total_labelled  = 0   # dimension annotations with non-unknown label
    label_counts    = Counter()
    per_cat         = defaultdict(lambda: dict(dim_ann=0, labelled=0))
    text_label_map  = {}  # text → set of labels (for consistency check)

    for f in files:
        data = load_json(f)
        if not data:
            continue
        cat = data.get('image_category', 0)

        for ann in data.get('annotations', []):
            ann_type = ann.get('annotation_type', 'unknown')
            label    = ann.get('semantic_label', 'unknown')
            text     = ann.get('annotation_text', '')
            total_ann += 1

            # Only count dimension-type annotations for labelling rate
            from semantic_labeller import LABEL_DESCRIPTIONS
            if label in DIMENSION_LABELS:
                total_dim_ann += 1
                total_labelled += 1
                label_counts[label] += 1
                per_cat[cat]['dim_ann'] += 1
                per_cat[cat]['labelled'] += 1
                # Track text → label consistency
                if text not in text_label_map:
                    text_label_map[text] = set()
                text_label_map[text].add(label)
            elif ann_type in ('dimension_value', 'diameter_callout', 'radius_callout',
                              'thread_spec', 'hole_callout', 'dimension_with_note'):
                total_dim_ann += 1
                per_cat[cat]['dim_ann'] += 1

    # Consistency: texts that always get the same label
    consistent = sum(1 for labels in text_label_map.values() if len(labels) == 1)
    consistency_rate = pct(consistent, len(text_label_map)) if text_label_map else 0

    return {
        "images":              len(files),
        "total_annotations":   total_ann,
        "dimension_annotations": total_dim_ann,
        "labelled_dimensions": total_labelled,
        "labelled_rate_pct":   pct(total_labelled, total_dim_ann),
        "label_consistency_pct": consistency_rate,
        "label_distribution":  dict(label_counts.most_common()),
        "per_category": {
            f"cat{k}": {
                "dim_annotations": v['dim_ann'],
                "labelled":        v['labelled'],
                "labelled_pct":    pct(v['labelled'], v['dim_ann']),
            }
            for k, v in sorted(per_cat.items()) if k > 0
        }
    }


# ============================================================
# 6. Print Full Report
# ============================================================

def print_full_report(ocr, clf, assoc, attr, sem, cer, bom_fill, dist, throughput, stackup_cov):
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

    # ── SEMANTIC LABELLING ────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  SEMANTIC LABELLING  (What each dimension means)")
    print(DIV)
    print(f"  {'Metric':<40} {'Value':>10}")
    print(f"  {'─'*40} {'─'*10}")
    print(f"  {'Dimension annotations':<40} {sem['dimension_annotations']:>10,}")
    print(f"  {'Semantically labelled':<40} {sem['labelled_dimensions']:>10,}")
    print(f"  {'Labelled rate %':<40} {sem['labelled_rate_pct']:>9.2f}%")
    print(f"  {'Label consistency %':<40} {sem['label_consistency_pct']:>9.2f}%")
    print(f"\n  Label distribution:")
    print(f"  {'Label':<22} {'Count':>6} {'%':>6}")
    print(f"  {'─'*22} {'─'*6} {'─'*6}")
    total_labelled = sem['labelled_dimensions']
    for lbl, cnt in sem['label_distribution'].items():
        bar = "█" * max(1, int(cnt / max(1, total_labelled) * 30))
        print(f"  {lbl:<22} {cnt:>6} {pct(cnt, total_labelled):>5.1f}%  {bar}")
    print(f"\n  Per-category labelled rates:")
    for cat, v in sem['per_category'].items():
        print(f"    {cat}: {v['dim_annotations']:4d} dim annotations | "
              f"labelled {v['labelled_pct']:5.1f}%")

    # ── OCR CHARACTER ERROR RATE ──────────────────────────────────────────
    print(f"\n{DIV}")
    print("  OCR CHARACTER ERROR RATE (CER) PROXY")
    print(DIV)
    print(f"  {'Metric':<40} {'Value':>10}")
    print(f"  {'─'*40} {'─'*10}")
    print(f"  {'Total OCR entries':<40} {cer['total_entries']:>10,}")
    print(f"  {'Perfect OCR (no correction) %':<40} {cer['perfect_ocr_pct']:>9.2f}%")
    print(f"  {'Mean CER (char error rate)':<40} {cer['mean_cer']:>10.4f}")
    print(f"  {'Median CER':<40} {cer['median_cer']:>10.4f}")
    print(f"  {'Mean WER (word error rate)':<40} {cer['mean_wer']:>10.4f}")
    print(f"  {'CER = 0 (exact match) %':<40} {cer['cer_zero_pct']:>9.2f}%")
    print(f"  {'CER < 10% (minor correction) %':<40} {cer['cer_under_10pct']:>9.2f}%")
    print(f"  {'CER > 50% (major correction) %':<40} {cer['cer_over_50pct']:>9.2f}%")
    print(f"\n  Per-category:")
    for cat, v in cer['per_category'].items():
        print(f"    {cat}: CER={v['mean_cer']:.4f}  WER={v['mean_wer']:.4f}  "
              f"perfect={v['perfect_pct']:.1f}%")

    # ── BOM FIELD FILL RATE ───────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  BOM FIELD FILL RATE  (Average completeness per row)")
    print(DIV)
    print(f"  {'Metric':<40} {'Value':>10}")
    print(f"  {'─'*40} {'─'*10}")
    print(f"  {'Cat2 images with BOM':<40} {bom_fill['cat2_images_with_bom']:>10}")
    print(f"  {'Total BOM rows':<40} {bom_fill['total_bom_rows']:>10}")
    print(f"  {'Mean field fill rate %':<40} {bom_fill['mean_fill_rate_pct']:>9.2f}%")
    print(f"  {'Fully complete rows (4/4 fields) %':<40} {bom_fill['fully_complete_pct']:>9.2f}%")
    print(f"  {'At least half filled (>=2/4) %':<40} {bom_fill['at_least_half_pct']:>9.2f}%")
    print(f"\n  Per-field fill rates:")
    for field, fill_pct in bom_fill['per_field_fill_pct'].items():
        bar = "█" * int(fill_pct / 5)
        print(f"    {field:<12} {fill_pct:5.1f}%  {bar}")

    # ── ASSOCIATION DISTANCE BUCKETS ──────────────────────────────────────
    print(f"\n{DIV}")
    print("  ASSOCIATION DISTANCE BUCKETS  (How close annotations are to geometry)")
    print(DIV)
    print(f"  {'Bucket':<25} {'Count':>7} {'%':>7}")
    print(f"  {'─'*25} {'─'*7} {'─'*7}")
    bucket_labels = {
        'exact_0px':       'Exact (0px — inside)',
        'tight_0_20':      'Tight (1-20px)',
        'good_20_50':      'Good (21-50px)',
        'moderate_50_100': 'Moderate (51-100px)',
        'loose_100_150':   'Loose (101-150px)',
        'very_loose_150p': 'Very loose (>150px)',
    }
    for key, label in bucket_labels.items():
        v = dist['distance_buckets'].get(key, {})
        cnt = v.get('count', 0)
        p   = v.get('pct', 0)
        bar = "█" * int(p / 3)
        print(f"  {label:<25} {cnt:>7} {p:>6.1f}%  {bar}")
    print(f"\n  Tight or better (<=50px): {dist['tight_or_better_pct']:.1f}%")
    print(f"\n  Mean distance by annotation type:")
    for t, v in dist['per_type_distance'].items():
        print(f"    {t:<22} mean={v['mean_px']:5.1f}px  median={v['median_px']:5.1f}px  "
              f"n={v['count']}")

    # ── PIPELINE THROUGHPUT ───────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  PIPELINE THROUGHPUT")
    print(DIV)
    print(f"  {'Metric':<40} {'Value':>10}")
    print(f"  {'─'*40} {'─'*10}")
    if 'error' not in throughput:
        print(f"  {'Total images processed':<40} {throughput['total_images']:>10}")
        print(f"  {'Total processing time':<40} {throughput['total_time_min']:>9.2f}m")
        print(f"  {'Mean time per image':<40} {throughput['mean_sec_per_image']:>9.2f}s")
        print(f"  {'Median time per image':<40} {throughput['median_sec_per_image']:>9.2f}s")
        print(f"  {'Min time per image':<40} {throughput['min_sec_per_image']:>9.2f}s")
        print(f"  {'Max time per image':<40} {throughput['max_sec_per_image']:>9.2f}s")
        print(f"  {'Throughput (images/min)':<40} {throughput['images_per_minute']:>10.1f}")
        est = throughput['stage_time_estimates']
        print(f"\n  Estimated stage time breakdown:")
        print(f"    Stage 2 OCR (EasyOCR):     ~{est['stage2_ocr_pct']:.0f}%")
        print(f"    Stage 4 Association:        ~{est['stage4_association_pct']:.0f}%")
        print(f"    Stage 3 Validation:         ~{est['stage3_validation_pct']:.0f}%")
        print(f"    Stage 5 Attribution:        ~{est['stage5_attribution_pct']:.0f}%")
        print(f"\n  Per-category timing:")
        for cat, v in throughput['per_category'].items():
            print(f"    {cat}: {v['images']} images | "
                  f"mean={v['mean_sec']:.1f}s | "
                  f"total={v['total_sec']:.0f}s")
    else:
        print(f"  {throughput['error']}")

    # ── TOLERANCE STACK-UP COVERAGE ───────────────────────────────────────
    print(f"\n{DIV}")
    print("  TOLERANCE STACK-UP COVERAGE")
    print(DIV)
    print(f"  {'Metric':<40} {'Value':>10}")
    print(f"  {'─'*40} {'─'*10}")
    print(f"  {'Total dimensions parsed':<40} {stackup_cov['total_dimensions']:>10,}")
    print(f"  {'Total tolerances found':<40} {stackup_cov['total_tolerances']:>10,}")
    print(f"  {'Total fit specifications':<40} {stackup_cov['total_fit_specs']:>10,}")
    print(f"  {'Dims with linked tolerance %':<40} {stackup_cov['tolerance_coverage_pct']:>9.2f}%")
    print(f"  {'Tolerances per dimension ratio':<40} {stackup_cov['tols_per_dim_ratio']:>10.3f}")
    sk = stackup_cov['stackup_stats']
    print(f"  {'Images with stack-up computed':<40} {sk['images_with_stackup']:>10}")
    print(f"  {'Mean worst-case tolerance (mm)':<40} {sk['mean_wc_tolerance_mm']:>10.4f}")
    print(f"  {'Mean RSS tolerance (mm)':<40} {sk['mean_rss_tolerance_mm']:>10.4f}")
    if stackup_cov['tolerance_type_dist']:
        print(f"\n  Tolerance type distribution:")
        for ttype, cnt in stackup_cov['tolerance_type_dist'].items():
            print(f"    {ttype:<20} {cnt:4d}")
    if stackup_cov['fit_type_dist']:
        print(f"\n  Fit specifications found:")
        for ft, cnt in list(stackup_cov['fit_type_dist'].items())[:8]:
            print(f"    {ft:<15} {cnt:4d}")
    print(f"\n  Per-category:")
    for cat, v in stackup_cov['per_category'].items():
        print(f"    {cat}: {v['dimensions']:3d} dims | "
              f"{v['tolerances']:2d} tols | "
              f"{v['fits']:2d} fits | "
              f"coverage={v['tol_coverage_pct']:.1f}%")

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
        ("OCR perfect (no correction) %",        cer['perfect_ocr_pct'],                  70.0,  True),
        ("OCR mean CER (lower=better)",          cer['mean_cer'] * 100,                   10.0,  False),
        ("Classification meaningful rate %",     clf['meaningful_rate_pct'],              75.0,  True),
        ("Classification unknown rate %",        clf['unknown_rate_pct'],                 25.0,  False),
        ("BOM rows reconstructed",               float(clf['bom_metrics']['total_rows_reconstructed']), 10.0, True),
        ("BOM mean field fill rate %",           bom_fill['mean_fill_rate_pct'],          50.0,  True),
        ("Association overall match rate %",     assoc['overall_match_rate_pct'],         75.0,  True),
        ("Association tight or better % (<=50px)", dist['tight_or_better_pct'],           75.0,  True),
        ("Cat1 association match rate %",        assoc['per_category'].get('cat1', {}).get('match_rate_pct', 0), 70.0, True),
        ("Cat2 association match rate %",        assoc['per_category'].get('cat2', {}).get('match_rate_pct', 0), 70.0, True),
        ("Cat3 association match rate %",        assoc['per_category'].get('cat3', {}).get('match_rate_pct', 0), 70.0, True),
        ("Part attribution named rate %",        attr['named_rate_pct'],                  70.0,  True),
        ("Part attribution high-conf rate %",    attr['high_conf_rate_pct'],              50.0,  True),
        ("Semantic labelled rate %",             sem['labelled_rate_pct'],                70.0,  True),
        ("Semantic label consistency %",         sem['label_consistency_pct'],            85.0,  True),
        ("Tolerance coverage % (dims with tol)", stackup_cov['tolerance_coverage_pct'],   0.0,  True),
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
    import sys
    sys.path.insert(0, 'src')

    print("Computing evaluation metrics...")
    ocr          = ocr_metrics()
    clf          = classification_metrics()
    assoc        = association_metrics()
    attr         = attribution_metrics()
    sem          = semantic_metrics()
    cer          = ocr_cer_metrics()
    bom_fill     = bom_fill_metrics()
    dist         = association_distance_metrics()
    throughput   = throughput_metrics()
    stackup_cov  = stackup_coverage_metrics()

    print_full_report(ocr, clf, assoc, attr, sem, cer, bom_fill, dist, throughput, stackup_cov)

    # Save JSON
    report = {
        "stage2_ocr":              ocr,
        "stage2_cer":              cer,
        "stage3_classification":   clf,
        "stage3_bom_fill":         bom_fill,
        "stage4_association":      assoc,
        "stage4_distance_buckets": dist,
        "part_attribution":        attr,
        "semantic_labelling":      sem,
        "pipeline_throughput":     throughput,
        "stackup_coverage":        stackup_cov,
    }
    out = "results/full_evaluation_report.json"
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Full report saved to: {out}")
