"""
End-to-end pipeline wrapper for the CAD tolerance stack-up project.
Runs OCR, structuring, and association in sequence for one image.
"""

import os
import sys

try:
    from vlm_reader import read_full_image
    from validation import structure_ocr_output
    from association import associate_image
    from preprocessing import preprocess
    from gdt_detector import detect_gdt, merge_detections
except ImportError:
    from .vlm_reader import read_full_image
    from .validation import structure_ocr_output
    from .association import associate_image
    from .preprocessing import preprocess
    from .gdt_detector import detect_gdt, merge_detections

GDT_MODEL_PATH = "models/gdt_yolov8.pt"


def run_pipeline(image_path, output_dir="results", min_confidence=0.5,
                 use_gdt_detector=True):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== RUN PIPELINE ===")
    print(f"Image: {image_path}")
    print(f"Output directory: {output_dir}")
    print(f"GD&T detector: {'enabled' if use_gdt_detector else 'disabled'}")

    filename = os.path.splitext(os.path.basename(image_path))[0]

    # Stage 2: OCR
    ocr_results = read_full_image(image_path, output_dir=output_dir,
                                  min_confidence=min_confidence)

    # Stage 3: Structuring (regex-based classification)
    fullocr_path = os.path.join(output_dir, f"{filename}_fullocr.json")
    structured = structure_ocr_output(fullocr_path, output_dir=output_dir)

    # Stage 3-A: GD&T Symbol Detection (YOLO) — merges into structured output
    if use_gdt_detector:
        structured_path = os.path.join(output_dir, f"{filename}_structured.json")
        _apply_gdt_detection(image_path, structured_path, GDT_MODEL_PATH)

    # Stage 4: Association
    binary = preprocess(image_path, save_result=False)
    associations = associate_image(
        image_path,
        ocr_json_path=fullocr_path,
        output_dir=output_dir,
        preprocessed_binary=binary
    )

    print(f"\nPipeline complete for {image_path}")
    return {
        "ocr": ocr_results,
        "structured": structured,
        "associations": associations,
    }


def _apply_gdt_detection(image_path, structured_path, model_path):
    """
    Run YOLO GD&T detection and merge results into the structured JSON in-place.

    Reads the existing _structured.json, merges YOLO detections, and
    overwrites the file with the enriched output.
    """
    if not os.path.exists(structured_path):
        print(f"  WARNING: structured JSON not found at {structured_path}, skipping GD&T merge")
        return

    # Load existing structured output
    try:
        with open(structured_path, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: cannot load structured JSON: {e}")
        return

    # Run YOLO detection
    yolo_detections = detect_gdt(image_path, model_path=model_path)
    if not yolo_detections:
        return  # nothing to merge

    # Merge into classified entries
    ocr_entries = structured_data.get("classified", [])
    merged = merge_detections(ocr_entries, yolo_detections)

    # Update summary counts
    summary = {}
    for entry in merged:
        t = entry.get("type", "unknown")
        summary[t] = summary.get(t, 0) + 1

    structured_data["classified"] = merged
    structured_data["total_detections"] = len(merged)
    structured_data["summary"] = summary
    structured_data["gdt_symbols_detected"] = len(yolo_detections)

    # Overwrite structured JSON with merged output
    with open(structured_path, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)

    print(f"  GD&T merge complete: {len(yolo_detections)} symbols merged into {structured_path}")


def run_dataset(image_dir, output_dir="results/batch", min_confidence=0.5):
    image_names = sorted([f for f in os.listdir(image_dir)
                          if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"\nRunning pipeline on dataset: {image_dir}")
    print(f"Found {len(image_names)} images")

    os.makedirs(output_dir, exist_ok=True)
    stats = []
    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        try:
            result = run_pipeline(image_path, output_dir=output_dir, min_confidence=min_confidence)
            stats.append({
                "image": image_name,
                "regions": len(result["ocr"]),
                "structured": len(result["structured"]),
                "associations": len(result["associations"]["associations"]),
            })
        except Exception as exc:
            stats.append({"image": image_name, "error": str(exc)})
    return stats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/pipeline.py <image_path> [output_dir]")
        print("       python src/pipeline.py --dataset <image_folder> [output_dir]")
        sys.exit(1)

    if sys.argv[1] == "--dataset":
        image_folder = sys.argv[2] if len(sys.argv) >= 3 else None
        output_folder = sys.argv[3] if len(sys.argv) >= 4 else "results/batch"
        if not image_folder:
            print("Missing image folder")
            sys.exit(1)
        run_dataset(image_folder, output_dir=output_folder)
    else:
        image_file = sys.argv[1]
        output_folder = sys.argv[2] if len(sys.argv) >= 3 else "results"
        run_pipeline(image_file, output_dir=output_folder)
