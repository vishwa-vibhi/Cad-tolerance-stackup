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
except ImportError:
    from .vlm_reader import read_full_image
    from .validation import structure_ocr_output
    from .association import associate_image
    from .preprocessing import preprocess


def run_pipeline(image_path, output_dir="results", min_confidence=0.5):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== RUN PIPELINE ===")
    print(f"Image: {image_path}")
    print(f"Output directory: {output_dir}")

    # Stage 2: OCR
    ocr_results = read_full_image(image_path, output_dir=output_dir, min_confidence=min_confidence)

    # Stage 3: Structuring
    filename = os.path.splitext(os.path.basename(image_path))[0]
    fullocr_path = os.path.join(output_dir, f"{filename}_fullocr.json")
    structured = structure_ocr_output(fullocr_path, output_dir=output_dir)

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
