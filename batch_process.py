"""
Batch-process all images in data/ folder.
Runs full pipeline on each image and saves results.
Generates summary statistics.
"""

import sys
import os
import json
import time
sys.path.insert(0, 'src')

try:
    from vlm_reader import read_full_image
except ImportError:
    from src.vlm_reader import read_full_image

try:
    from validation import validate_file
except ImportError:
    from src.validation import validate_file

try:
    from association import associate_file
except ImportError:
    from src.association import associate_file

try:
    from part_attribution import attribute_file as attribute_parts
except ImportError:
    from src.part_attribution import attribute_file as attribute_parts

try:
    from semantic_labeller import label_file as label_semantics
except ImportError:
    from src.semantic_labeller import label_file as label_semantics



def process_category(category_dir, results_dir):
    """Run pipeline on every image in a category folder."""
    images = sorted([f for f in os.listdir(category_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"\n{'='*70}")
    print(f"CATEGORY: {category_dir}")
    print(f"Images found: {len(images)}")
    print(f"{'='*70}\n")

    stats = []
    for i, img_name in enumerate(images, 1):
        img_path = os.path.join(category_dir, img_name)
        print(f"\n[{i}/{len(images)}] Processing: {img_name}")
        print("-" * 60)

        start = time.time()
        try:
            results = read_full_image(img_path, output_dir=results_dir)
            elapsed = time.time() - start

            # Stage 3: Validation & Structuring
            basename = os.path.splitext(img_name)[0]
            fullocr_path = os.path.join(results_dir, f"{basename}_fullocr.json")
            structured = None
            if os.path.exists(fullocr_path):
                structured = validate_file(fullocr_path, output_dir=results_dir)

            # Stage 4: Geometric Association
            structured_path = os.path.join(results_dir, f"{basename}_structured.json")
            assoc_result = None
            if os.path.exists(structured_path):
                assoc_result = associate_file(img_path, structured_path, results_dir)

            # Part Attribution
            attr_result = None
            if os.path.exists(structured_path):
                try:
                    attr_result = attribute_parts(structured_path, results_dir)
                except Exception as e:
                    print(f"  Attribution WARNING: {e}")

            # Semantic Labelling
            assoc_path = os.path.join(results_dir, f"{basename}_associations.json")
            sem_result = None
            if os.path.exists(assoc_path):
                try:
                    sem_result = label_semantics(assoc_path, results_dir)
                except Exception as e:
                    print(f"  Semantic labelling WARNING: {e}")

            # confidence stats
            high_conf = sum(1 for r in results if r['confidence'] > 0.9)
            med_conf  = sum(1 for r in results if 0.7 <= r['confidence'] <= 0.9)
            low_conf  = sum(1 for r in results if r['confidence'] < 0.7)
            structured_count = structured.get("total_detections", 0) if structured else 0
            assoc_matched    = assoc_result.get("matched", 0) if assoc_result else 0
            assoc_unassoc    = assoc_result.get("unassociated", 0) if assoc_result else 0
            named_dims       = attr_result.get("named_attributions", 0) if attr_result else 0

            stat = {
                "image": img_name,
                "total_regions": len(results),
                "high_confidence": high_conf,
                "medium_confidence": med_conf,
                "low_confidence": low_conf,
                "time_seconds": round(elapsed, 1),
                "structured_count": structured_count,
                "association_matched": assoc_matched,
                "association_unassociated": assoc_unassoc,
                "named_dimensions": named_dims,
            }
            stats.append(stat)
            print(f"  TIME: {elapsed:.1f}s | HIGH: {high_conf} | MED: {med_conf} | LOW: {low_conf} | STRUCTURED: {structured_count} | ASSOC: {assoc_matched}/{assoc_matched+assoc_unassoc} | NAMED: {named_dims}")

        except Exception as e:
            print(f"  ERROR: {e}")
            stats.append({"image": img_name, "error": str(e)})

    return stats


def print_summary(all_stats):
    """Print aggregate statistics."""
    print(f"\n\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*70}\n")

    for category, stats in all_stats.items():
        valid = [s for s in stats if 'total_regions' in s]
        if not valid:
            continue

        total_imgs = len(valid)
        total_regions = sum(s['total_regions'] for s in valid)
        total_high = sum(s['high_confidence'] for s in valid)
        total_med = sum(s['medium_confidence'] for s in valid)
        total_low = sum(s['low_confidence'] for s in valid)
        avg_time = sum(s['time_seconds'] for s in valid) / total_imgs

        print(f"--- {category.upper()} ---")
        print(f"  Images processed:    {total_imgs}")
        print(f"  Total text regions:  {total_regions}")
        print(f"  Avg per image:       {total_regions/total_imgs:.1f}")
        print(f"  High confidence:     {total_high} ({total_high*100//max(total_regions,1)}%)")
        print(f"  Medium confidence:   {total_med} ({total_med*100//max(total_regions,1)}%)")
        print(f"  Low confidence:      {total_low} ({total_low*100//max(total_regions,1)}%)")
        print(f"  Avg time per image:  {avg_time:.1f}s")
        print()


def main():
    output_dir = "results/batch"
    os.makedirs(output_dir, exist_ok=True)

    categories = {
        "category_1": "data/category_1",
        "category_2": "data/category_2",
        "category_3": "data/category_3",
    }

    all_stats = {}
    overall_start = time.time()

    for cat_name, cat_path in categories.items():
        if not os.path.exists(cat_path):
            print(f"Skipping {cat_path} - does not exist")
            continue
        stats = process_category(cat_path, output_dir)
        all_stats[cat_name] = stats

    total_time = time.time() - overall_start

    # save aggregate stats
    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_stats, f, indent=2)

    print_summary(all_stats)
    print(f"\nTotal batch time: {total_time/60:.1f} minutes")
    print(f"Stats saved to: {summary_path}")


if __name__ == "__main__":
    main()