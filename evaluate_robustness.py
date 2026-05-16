"""
Robustness Testing — tests pipeline on degraded inputs.
Applies synthetic degradations and measures performance drop.

Conditions tested:
1. Clean (original) — baseline
2. Gaussian blur (simulates out-of-focus scan)
3. Rotation 5 degrees (simulates skewed scan)
4. Low contrast (simulates faded print)
5. Salt & pepper noise (simulates dirty scan)
6. Downscaled 50% (simulates low-resolution input)

Usage:
    python evaluate_robustness.py
"""
import sys, os, json, time
import cv2
import numpy as np
sys.path.insert(0, 'src')

from vlm_reader import read_full_image
from validation import validate_file
from association import associate_file

# Test on a representative subset (1 per category)
TEST_IMAGES = [
    "data/category_1/cad1_001.png",
    "data/category_2/cad2_001.png",
    "data/category_3/cad3_001.png",
]

TEMP_DIR = "results/_robustness_temp"


def apply_blur(img, ksize=7):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def apply_rotation(img, angle=5):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))


def apply_low_contrast(img, factor=0.5):
    return cv2.convertScaleAbs(img, alpha=factor, beta=128 * (1 - factor))


def apply_noise(img, amount=0.02):
    noisy = img.copy()
    h, w = img.shape[:2]
    n_salt = int(amount * h * w)
    # Salt
    coords = [np.random.randint(0, i, n_salt) for i in [h, w]]
    noisy[coords[0], coords[1]] = 255
    # Pepper
    coords = [np.random.randint(0, i, n_salt) for i in [h, w]]
    noisy[coords[0], coords[1]] = 0
    return noisy


def apply_downscale(img, factor=0.5):
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w*factor), int(h*factor)))
    return cv2.resize(small, (w, h))  # scale back up (quality lost)


def run_pipeline_on_image(image_path, output_dir):
    """Run OCR + validation + association on one image, return metrics."""
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    try:
        t0 = time.time()
        ocr_results = read_full_image(image_path, output_dir=output_dir)
        elapsed_ocr = time.time() - t0

        fullocr_path = os.path.join(output_dir, f"{basename}_fullocr.json")
        structured = validate_file(fullocr_path, output_dir=output_dir)

        structured_path = os.path.join(output_dir, f"{basename}_structured.json")
        assoc = associate_file(image_path, structured_path, output_dir=output_dir)

        total = structured.get('total_detections', 0) if structured else 0
        unknown = structured.get('summary', {}).get('unknown', 0) if structured else 0
        matched = assoc.get('matched', 0) if assoc else 0
        total_ann = assoc.get('total_annotations', 0) if assoc else 0

        return {
            "detections": len(ocr_results),
            "meaningful_pct": round((total - unknown) / max(total, 1) * 100, 1),
            "assoc_match_pct": round(matched / max(total_ann, 1) * 100, 1),
            "time_sec": round(elapsed_ocr, 2),
            "high_conf_pct": round(sum(1 for r in ocr_results if r['confidence'] > 0.9) /
                                    max(len(ocr_results), 1) * 100, 1),
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    import shutil

    print("=" * 70)
    print("  ROBUSTNESS TESTING")
    print("  Testing pipeline on degraded inputs")
    print("=" * 70)

    conditions = [
        ("Clean (original)", None),
        ("Gaussian blur (k=7)", lambda img: apply_blur(img, 7)),
        ("Rotation 5 deg", lambda img: apply_rotation(img, 5)),
        ("Low contrast (0.5x)", lambda img: apply_low_contrast(img, 0.5)),
        ("Salt & pepper noise (2%)", lambda img: apply_noise(img, 0.02)),
        ("Downscaled 50%", lambda img: apply_downscale(img, 0.5)),
    ]

    all_results = {}

    for cond_name, transform_fn in conditions:
        print(f"\n  Testing: {cond_name}...")
        cond_metrics = []

        for img_path in TEST_IMAGES:
            if not os.path.exists(img_path):
                continue

            out_dir = os.path.join(TEMP_DIR, cond_name.replace(" ", "_").replace("(", "").replace(")", ""))
            os.makedirs(out_dir, exist_ok=True)

            if transform_fn is None:
                # Clean — use original
                test_path = img_path
            else:
                # Apply degradation
                img = cv2.imread(img_path)
                degraded = transform_fn(img)
                test_path = os.path.join(out_dir, os.path.basename(img_path))
                cv2.imwrite(test_path, degraded)

            metrics = run_pipeline_on_image(test_path, out_dir)
            if 'error' not in metrics:
                cond_metrics.append(metrics)
                print(f"    {os.path.basename(img_path)}: "
                      f"det={metrics['detections']} "
                      f"meaningful={metrics['meaningful_pct']}% "
                      f"assoc={metrics['assoc_match_pct']}%")

        if cond_metrics:
            avg = {
                "detections": round(sum(m['detections'] for m in cond_metrics) / len(cond_metrics), 1),
                "meaningful_pct": round(sum(m['meaningful_pct'] for m in cond_metrics) / len(cond_metrics), 1),
                "assoc_match_pct": round(sum(m['assoc_match_pct'] for m in cond_metrics) / len(cond_metrics), 1),
                "high_conf_pct": round(sum(m['high_conf_pct'] for m in cond_metrics) / len(cond_metrics), 1),
                "time_sec": round(sum(m['time_sec'] for m in cond_metrics) / len(cond_metrics), 2),
            }
            all_results[cond_name] = avg

    # Summary table
    print("\n\n  ROBUSTNESS RESULTS SUMMARY:")
    print("  " + "=" * 70)
    print(f"  {'Condition':<30} {'Detections':>11} {'Meaningful%':>12} {'Assoc%':>8} {'HighConf%':>10}")
    print(f"  {'-'*30} {'-'*11} {'-'*12} {'-'*8} {'-'*10}")

    baseline = all_results.get("Clean (original)", {})
    for cond, m in all_results.items():
        det_drop = m['detections'] - baseline.get('detections', 0)
        det_str  = f"{m['detections']:.0f}" + (f" ({det_drop:+.0f})" if cond != "Clean (original)" else "")
        print(f"  {cond:<30} {det_str:>11} {m['meaningful_pct']:>11.1f}% "
              f"{m['assoc_match_pct']:>7.1f}% {m['high_conf_pct']:>9.1f}%")

    # Cleanup
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    # Save
    out = "results/robustness_report.json"
    json.dump(all_results, open(out, 'w'), indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
