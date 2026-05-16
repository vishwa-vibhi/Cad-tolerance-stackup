"""
Runtime Evaluation — per-stage timing breakdown.
Profiles each pipeline stage independently.

Usage:
    python evaluate_runtime.py
"""
import sys, os, json, time, glob
sys.path.insert(0, 'src')

RESULTS_DIR = "results/batch"
TEST_IMAGE  = "data/category_1/cad1_001.png"
TEMP_DIR    = "results/_runtime_temp"


def profile_stages():
    """Profile each stage independently on a single image."""
    import cv2
    from preprocessing import preprocess, preprocess_for_ocr
    from element_detection import detect_all_elements
    from vlm_reader import read_full_image
    from validation import validate_file
    from association import associate_file
    from part_attribution import attribute_file
    from semantic_labeller import label_file

    os.makedirs(TEMP_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(TEST_IMAGE))[0]

    timings = {}

    # Stage 1: Preprocessing
    t0 = time.time()
    binary = preprocess(TEST_IMAGE, save_result=False)
    timings["Stage 1: Preprocessing"] = round(time.time() - t0, 3)

    # Stage 2: Element Detection (Segmentation)
    t0 = time.time()
    elements = detect_all_elements(binary)
    timings["Stage 2: Segmentation"] = round(time.time() - t0, 3)

    # Stage 3: OCR (Dimension Extraction)
    t0 = time.time()
    ocr_results = read_full_image(TEST_IMAGE, output_dir=TEMP_DIR)
    timings["Stage 3: OCR (EasyOCR)"] = round(time.time() - t0, 3)

    # Stage 4: Classification
    fullocr_path = os.path.join(TEMP_DIR, f"{basename}_fullocr.json")
    t0 = time.time()
    structured = validate_file(fullocr_path, output_dir=TEMP_DIR)
    timings["Stage 4: Classification"] = round(time.time() - t0, 3)

    # Stage 5: Association
    structured_path = os.path.join(TEMP_DIR, f"{basename}_structured.json")
    t0 = time.time()
    assoc = associate_file(TEST_IMAGE, structured_path, output_dir=TEMP_DIR)
    timings["Stage 5: Association"] = round(time.time() - t0, 3)

    # Stage 6: Semantic Labelling
    assoc_path = os.path.join(TEMP_DIR, f"{basename}_associations.json")
    t0 = time.time()
    if os.path.exists(assoc_path):
        label_file(assoc_path, output_dir=TEMP_DIR)
    timings["Stage 6: Semantic Labelling"] = round(time.time() - t0, 3)

    # Stage 7: Part Attribution
    t0 = time.time()
    attribute_file(structured_path, output_dir=TEMP_DIR)
    timings["Stage 7: Part Attribution"] = round(time.time() - t0, 3)

    return timings


def batch_timing():
    """Get timing from batch_summary.json."""
    summary_path = os.path.join(RESULTS_DIR, "batch_summary.json")
    if not os.path.exists(summary_path):
        return None
    data = json.load(open(summary_path))
    all_times = []
    for cat_key, images in data.items():
        if isinstance(images, list):
            for img in images:
                if 'time_seconds' in img:
                    all_times.append(img['time_seconds'])
    if not all_times:
        return None
    return {
        "total_images": len(all_times),
        "total_time_sec": round(sum(all_times), 1),
        "mean_per_image": round(sum(all_times) / len(all_times), 2),
        "min_per_image": round(min(all_times), 2),
        "max_per_image": round(max(all_times), 2),
        "images_per_minute": round(len(all_times) / (sum(all_times) / 60), 1),
    }


def main():
    import shutil, platform

    print("=" * 70)
    print("  RUNTIME EVALUATION")
    print("=" * 70)

    # System info
    print(f"\n  System: {platform.system()} {platform.release()}")
    print(f"  Python: {platform.python_version()}")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  GPU: None (CPU only)")
    except ImportError:
        print("  GPU: PyTorch not available")

    # Per-stage profiling
    print(f"\n  Profiling on: {TEST_IMAGE}")
    print("-" * 70)

    timings = profile_stages()
    total = sum(timings.values())

    print(f"\n  PER-STAGE TIMING:")
    print(f"  {'Stage':<35} {'Time (s)':>10} {'%':>6}")
    print(f"  {'-'*35} {'-'*10} {'-'*6}")
    for stage, t in timings.items():
        pct = round(t / total * 100, 1)
        bar = "#" * int(pct / 2)
        print(f"  {stage:<35} {t:>10.3f} {pct:>5.1f}%  {bar}")
    print(f"  {'TOTAL':<35} {total:>10.3f}")

    # Batch timing
    print(f"\n\n  BATCH TIMING (36 images):")
    print("-" * 70)
    batch = batch_timing()
    if batch:
        print(f"  Total images:      {batch['total_images']}")
        print(f"  Total time:        {batch['total_time_sec']}s ({batch['total_time_sec']/60:.1f} min)")
        print(f"  Mean per image:    {batch['mean_per_image']}s")
        print(f"  Min per image:     {batch['min_per_image']}s")
        print(f"  Max per image:     {batch['max_per_image']}s")
        print(f"  Throughput:        {batch['images_per_minute']} images/min")
    else:
        print("  No batch timing data (run batch_process.py first)")

    # Bottleneck analysis
    print(f"\n\n  BOTTLENECK ANALYSIS:")
    print(f"  OCR (EasyOCR) dominates at {timings.get('Stage 3: OCR (EasyOCR)', 0)/total*100:.0f}% of total time")
    print(f"  All other stages combined: {(total - timings.get('Stage 3: OCR (EasyOCR)', 0)):.3f}s")
    print(f"  Pipeline is I/O-bound on OCR inference")

    # Cleanup
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    # Save
    out = "results/runtime_report.json"
    json.dump({"per_stage": timings, "total_sec": total, "batch": batch},
              open(out, 'w'), indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
