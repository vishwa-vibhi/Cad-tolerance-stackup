"""
Confusion Matrix + Per-Class Metrics for the text classifier.
Generates a confusion matrix heatmap and per-class precision/recall/F1.

Usage:
    python evaluate_confusion.py
"""
import sys, os, json, glob
import numpy as np
from collections import Counter
sys.path.insert(0, 'src')

RESULTS_DIR = "results/batch"

TRAINABLE_TYPES = [
    'dimension_value', 'balloon_number', 'quantity', 'part_name',
    'material_code', 'diameter_callout', 'section_marker',
    'radius_callout', 'hole_callout', 'bom_header', 'material_name',
    'dimension_with_note', 'thread_spec', 'tolerance',
]


def load_data():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_structured.json")))
    texts, labels, cats = [], [], []
    for f in files:
        data = json.load(open(f, encoding='utf-8'))
        cat = data.get('image_category', 0)
        for e in data.get('classified', []):
            t = e.get('type', 'unknown')
            text = e.get('text', '').strip()
            if text and t in TRAINABLE_TYPES:
                texts.append(text)
                labels.append(t)
                cats.append(cat)
    return texts, labels, cats


def main():
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score, f1_score)
    from text_classifier import predict_batch, load_model, MODEL_PATH

    print("=" * 70)
    print("  CONFUSION MATRIX & PER-CLASS METRICS")
    print("=" * 70)

    texts, labels, cats = load_data()
    print(f"\n  Samples: {len(texts)}")

    bundle = load_model(MODEL_PATH)
    if bundle is None:
        print("  ERROR: Model not found. Run: python src/text_classifier.py --train")
        return

    preds = predict_batch(texts, cats, MODEL_PATH)
    pred_types = [p[0] for p in preds]

    # Per-class report
    print("\n  PER-CLASS METRICS:")
    print("-" * 70)
    report = classification_report(
        labels, pred_types,
        labels=TRAINABLE_TYPES,
        target_names=TRAINABLE_TYPES,
        zero_division=0,
        output_dict=True,
    )
    print(f"  {'Class':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print(f"  {'-'*22} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for cls in TRAINABLE_TYPES:
        r = report.get(cls, {})
        print(f"  {cls:<22} {r.get('precision',0):>10.3f} {r.get('recall',0):>8.3f} "
              f"{r.get('f1-score',0):>8.3f} {r.get('support',0):>8.0f}")

    print(f"\n  {'MACRO AVG':<22} {report['macro avg']['precision']:>10.3f} "
          f"{report['macro avg']['recall']:>8.3f} {report['macro avg']['f1-score']:>8.3f}")
    print(f"  {'WEIGHTED AVG':<22} {report['weighted avg']['precision']:>10.3f} "
          f"{report['weighted avg']['recall']:>8.3f} {report['weighted avg']['f1-score']:>8.3f}")

    # Confusion matrix
    print("\n\n  CONFUSION MATRIX (top confusions):")
    print("-" * 70)
    cm = confusion_matrix(labels, pred_types, labels=TRAINABLE_TYPES)

    # Find top confusions (off-diagonal)
    confusions = []
    for i in range(len(TRAINABLE_TYPES)):
        for j in range(len(TRAINABLE_TYPES)):
            if i != j and cm[i][j] > 0:
                confusions.append((cm[i][j], TRAINABLE_TYPES[i], TRAINABLE_TYPES[j]))
    confusions.sort(reverse=True)

    print(f"  {'True Class':<22} {'Predicted As':<22} {'Count':>6}")
    print(f"  {'-'*22} {'-'*22} {'-'*6}")
    for cnt, true_cls, pred_cls in confusions[:15]:
        print(f"  {true_cls:<22} {pred_cls:<22} {cnt:>6}")

    # Overall
    acc = accuracy_score(labels, pred_types)
    f1m = f1_score(labels, pred_types, average='macro', zero_division=0)
    f1w = f1_score(labels, pred_types, average='weighted', zero_division=0)
    print(f"\n  OVERALL:")
    print(f"    Accuracy:     {acc:.4f}")
    print(f"    F1 (macro):   {f1m:.4f}")
    print(f"    F1 (weighted):{f1w:.4f}")

    # Save
    out = "results/confusion_matrix_report.json"
    json.dump({
        "per_class": report,
        "confusion_matrix": [[int(x) for x in row] for row in cm],
        "class_names": TRAINABLE_TYPES,
        "top_confusions": [(int(c), t, p) for c, t, p in confusions[:20]],
        "accuracy": float(acc),
        "f1_macro": float(f1m),
        "f1_weighted": float(f1w),
    }, open(out, 'w'), indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
