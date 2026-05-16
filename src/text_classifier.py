"""
Stage 3-ML: Trained Text Classifier for Engineering Drawing Annotations.

Replaces/augments the regex-based classifier in validation.py with a
trained scikit-learn model (TF-IDF char n-grams + SVM).

Training data: all _structured.json files in results/batch/
  - Uses regex-classified labels as training signal
  - Excludes "unknown" from training (those are the hard cases we want to fix)
  - Adds engineered features: has_digit, has_letter, starts_with_symbol, etc.

The trained model is saved to models/text_classifier.pkl and loaded
automatically when classify() is called.

Usage:
    # Train
    python src/text_classifier.py --train --results results/batch

    # Evaluate
    python src/text_classifier.py --evaluate --results results/batch

    # Predict single text
    python src/text_classifier.py --predict "M30x2.5"
"""

import os
import re
import sys
import json
import glob
import math
import pickle
import pathlib
import argparse
import numpy as np
from collections import Counter

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_PATH = "models/text_classifier.pkl"

# ── Classes the model can predict ────────────────────────────────────────────
# We exclude "unknown" from training — the model predicts known types only.
# If confidence is low, we fall back to "unknown".
TRAINABLE_TYPES = [
    'dimension_value',
    'diameter_callout',
    'radius_callout',
    'thread_spec',
    'tolerance',
    'hole_callout',
    'dimension_with_note',
    'section_marker',
    'spacing_annotation',
    'material_code',
    'material_name',
    'part_name',
    'bom_header',
    'balloon_number',
    'quantity',
]

# Minimum confidence to accept ML prediction (else return "unknown")
MIN_CONFIDENCE = 0.45

# ── Lazy-loaded model ─────────────────────────────────────────────────────────
_model = None
_vectorizer = None
_label_encoder = None


# ============================================================
# Feature engineering
# ============================================================

def _extract_features(text):
    """
    Extract a rich feature dict from a single text string.

    Features:
      - Character-level: length, digit count, letter count, symbol count
      - Pattern flags: starts_with_phi, starts_with_r, starts_with_m,
                       has_x_symbol, has_plus_minus, has_slash, has_degree,
                       has_hole, has_thick, has_deep, has_pcd, has_equi
      - Structural: is_single_digit, is_pure_number, is_pure_alpha,
                    is_mixed, has_decimal
    """
    t = text.strip()
    upper = t.upper()

    return {
        # Length features
        'len':           len(t),
        'len_bin':       min(len(t) // 3, 5),   # 0-5 bucket

        # Character composition
        'n_digits':      sum(c.isdigit() for c in t),
        'n_letters':     sum(c.isalpha() for c in t),
        'n_symbols':     sum(not c.isalnum() and not c.isspace() for c in t),
        'digit_ratio':   sum(c.isdigit() for c in t) / max(len(t), 1),
        'letter_ratio':  sum(c.isalpha() for c in t) / max(len(t), 1),

        # Boolean pattern flags (0/1)
        'starts_phi':    int(t.startswith('Ø') or t.startswith('O') and len(t) > 2 and t[1].isdigit()),
        'starts_r':      int(t.startswith('R') and len(t) > 1 and t[1].isdigit()),
        'starts_m':      int(t.upper().startswith('M') and len(t) > 1 and t[1].isdigit()),
        'has_x':         int('×' in t or 'x' in t.lower() and any(c.isdigit() for c in t)),
        'has_pm':        int('±' in t or ('+' in t and '-' in t)),
        'has_slash':     int('/' in t),
        'has_degree':    int('°' in t),
        'has_hole':      int('HOLE' in upper),
        'has_thick':     int('THICK' in upper),
        'has_deep':      int('DEEP' in upper),
        'has_pcd':       int('PCD' in upper),
        'has_equi':      int('EQUI' in upper),
        'has_dia':       int('DIA' in upper),
        'has_module':    int('MODULE' in upper),
        'has_groove':    int('GROOVE' in upper),
        'has_colon':     int(':' in t),
        'has_semicolon': int(';' in t),

        # Structural
        'is_single_digit':  int(len(t) == 1 and t.isdigit()),
        'is_two_digit':     int(len(t) == 2 and t.isdigit()),
        'is_pure_number':   int(bool(re.match(r'^\d+(\.\d+)?$', t))),
        'is_pure_alpha':    int(t.isalpha()),
        'is_mixed':         int(any(c.isdigit() for c in t) and any(c.isalpha() for c in t)),
        'has_decimal':      int('.' in t),
        'starts_digit':     int(len(t) > 0 and t[0].isdigit()),
        'starts_upper':     int(len(t) > 0 and t[0].isupper()),
        'all_upper':        int(t.isupper() and t.isalpha()),
        'all_lower':        int(t.islower() and t.isalpha()),
        'is_short_code':    int(len(t) <= 3 and t.isalpha() and t.isupper()),
    }


def _features_to_vector(features_dict, feature_names):
    """Convert feature dict to numpy array in consistent order."""
    return np.array([features_dict.get(k, 0) for k in feature_names], dtype=float)


# ============================================================
# Data loading
# ============================================================

def load_training_data(results_dir="results/batch"):
    """
    Load all classified entries from _structured.json files.

    Returns:
        texts:    list of text strings
        labels:   list of type strings
        category: list of image categories (1, 2, 3)
    """
    files = sorted(glob.glob(os.path.join(results_dir, "*_structured.json")))
    texts, labels, categories = [], [], []

    for f in files:
        try:
            data = json.load(open(f, encoding='utf-8'))
        except Exception:
            continue
        cat = data.get("image_category", 0)
        for entry in data.get("classified", []):
            t = entry.get("type", "unknown")
            text = entry.get("text", "").strip()
            if not text:
                continue
            # Only train on known types (not "unknown")
            if t in TRAINABLE_TYPES:
                texts.append(text)
                labels.append(t)
                categories.append(cat)

    print(f"  Loaded {len(texts)} training samples from {len(files)} files")
    return texts, labels, categories


# ============================================================
# Model training
# ============================================================

def train(results_dir="results/batch", model_path=MODEL_PATH):
    """
    Train a TF-IDF + SVM classifier on the structured JSON data.

    Pipeline:
      1. Extract char n-gram TF-IDF features (captures subword patterns)
      2. Extract engineered numeric features
      3. Concatenate both feature sets
      4. Train LinearSVC with class balancing
      5. Wrap in CalibratedClassifierCV for probability estimates
      6. Evaluate with stratified 5-fold cross-validation
      7. Save model to disk

    Returns:
        Trained model pipeline.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import classification_report
    from sklearn.utils import class_weight
    import scipy.sparse as sp

    print("\n=== TRAINING TEXT CLASSIFIER ===")
    texts, labels, _ = load_training_data(results_dir)

    if len(texts) < 50:
        print(f"  ERROR: only {len(texts)} samples — need at least 50 to train")
        return None

    # ── Feature extraction ────────────────────────────────────────────────
    print(f"  Extracting features from {len(texts)} samples...")

    # TF-IDF on character n-grams (1-4 chars) — captures patterns like
    # "Ø", "M3", "×2", "DIA", "HOLE", "THICK" etc.
    tfidf = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(1, 4),
        min_df=1,
        max_features=2000,
        sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(texts)

    # Engineered numeric features
    feature_names = sorted(_extract_features("test").keys())
    X_numeric = np.array([
        _features_to_vector(_extract_features(t), feature_names)
        for t in texts
    ])
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)

    # Combine: sparse TF-IDF + dense numeric
    X = sp.hstack([X_tfidf, sp.csr_matrix(X_numeric_scaled)])

    # ── Label encoding ────────────────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Drop classes with fewer than 5 samples (need at least 5 for 5-fold CV)
    class_counts = Counter(labels)
    min_samples  = 5
    valid_mask   = np.array([class_counts[labels[i]] >= min_samples
                             for i in range(len(labels))])
    texts_f  = [texts[i]  for i in range(len(texts))  if valid_mask[i]]
    labels_f = [labels[i] for i in range(len(labels)) if valid_mask[i]]
    dropped  = [c for c, n in class_counts.items() if n < min_samples]
    if dropped:
        print(f"  Dropped classes with <{min_samples} samples: {dropped}")
        print(f"  (These will be handled by regex fallback)")

    le = LabelEncoder()
    y  = le.fit_transform(labels_f)

    # Rebuild feature matrix for filtered data
    X_tfidf   = tfidf.fit_transform(texts_f)
    X_numeric = np.array([
        _features_to_vector(_extract_features(t), feature_names)
        for t in texts_f
    ])
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    X = sp.hstack([X_tfidf, sp.csr_matrix(X_numeric_scaled)])

    print(f"  Classes ({len(le.classes_)}): {list(le.classes_)}")
    print(f"  Feature dimensions: TF-IDF={X_tfidf.shape[1]}, numeric={len(feature_names)}")

    # ── Cross-validation ──────────────────────────────────────────────────
    print(f"\n  Running 5-fold stratified cross-validation...")

    # Use class weights to handle imbalance
    cw = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y), y=y
    )
    cw_dict = dict(enumerate(cw))

    base_clf = LinearSVC(
        C=1.0,
        max_iter=5000,
        class_weight=cw_dict,
    )
    # Wrap for probability calibration — use cv=5 to handle small classes
    clf = CalibratedClassifierCV(base_clf, cv=5, method='sigmoid')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        clf, X, y,
        cv=skf,
        scoring=['accuracy', 'f1_macro', 'f1_weighted'],
        return_train_score=True,
        error_score='raise',
    )

    print(f"\n  Cross-validation results (5-fold):")
    print(f"  {'Metric':<30} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8}")
    for metric in ['test_accuracy', 'test_f1_macro', 'test_f1_weighted']:
        vals = cv_results[metric]
        print(f"  {metric:<30} {vals.mean():>8.4f} {vals.std():>8.4f}")

    # ── Train final model on all data ─────────────────────────────────────
    print(f"\n  Training final model on all {len(texts)} samples...")
    clf.fit(X, y)

    # ── Per-class report ──────────────────────────────────────────────────
    y_pred = clf.predict(X)
    print(f"\n  Training set classification report:")
    print(classification_report(
        y, y_pred,
        target_names=le.classes_,
        zero_division=0,
    ))

    # ── Save model ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_bundle = {
        "tfidf":         tfidf,
        "scaler":        scaler,
        "clf":           clf,
        "label_encoder": le,
        "feature_names": feature_names,
        "cv_accuracy":   float(cv_results['test_accuracy'].mean()),
        "cv_f1_macro":   float(cv_results['test_f1_macro'].mean()),
        "cv_f1_weighted":float(cv_results['test_f1_weighted'].mean()),
        "n_classes":     len(le.classes_),
        "n_samples":     len(texts),
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_bundle, f)

    print(f"\n  Model saved to: {model_path}")
    print(f"  CV Accuracy:    {cv_results['test_accuracy'].mean():.4f}")
    print(f"  CV F1 (macro):  {cv_results['test_f1_macro'].mean():.4f}")
    print(f"  CV F1 (weighted): {cv_results['test_f1_weighted'].mean():.4f}")

    return model_bundle


# ============================================================
# Model loading
# ============================================================

def load_model(model_path=MODEL_PATH):
    """Load the trained model bundle from disk."""
    global _model
    if _model is None:
        if not os.path.exists(model_path):
            return None
        with open(model_path, 'rb') as f:
            _model = pickle.load(f)
    return _model


# ============================================================
# Prediction
# ============================================================

def predict(text, category=0, model_path=MODEL_PATH, min_confidence=MIN_CONFIDENCE):
    """
    Predict the annotation type for a single text string.

    Args:
        text:           The normalised text string to classify.
        category:       Drawing category (1, 2, 3) — used for category-gated types.
        model_path:     Path to the trained model pickle.
        min_confidence: Minimum probability to accept prediction (else "unknown").

    Returns:
        (predicted_type, confidence) tuple.
        Returns ("unknown", 0.0) if model not loaded or confidence too low.
    """
    import scipy.sparse as sp

    bundle = load_model(model_path)
    if bundle is None:
        return "unknown", 0.0

    tfidf    = bundle["tfidf"]
    scaler   = bundle["scaler"]
    clf      = bundle["clf"]
    le       = bundle["label_encoder"]
    feat_names = bundle["feature_names"]

    # Build feature vector
    X_tfidf   = tfidf.transform([text])
    X_numeric = np.array([_features_to_vector(_extract_features(text), feat_names)])
    X_numeric_scaled = scaler.transform(X_numeric)
    X = sp.hstack([X_tfidf, sp.csr_matrix(X_numeric_scaled)])

    # Get probabilities
    proba = clf.predict_proba(X)[0]
    best_idx  = int(np.argmax(proba))
    best_conf = float(proba[best_idx])
    pred_type = le.classes_[best_idx]

    # Category-gated types: only valid in certain categories
    if pred_type == "balloon_number" and category == 1:
        # In Cat1, single digits are dimension values, not balloons
        pred_type = "dimension_value"
    if pred_type == "section_marker" and category == 2:
        # Section markers don't appear in Cat2 assembly drawings
        # Find next best prediction
        sorted_idx = np.argsort(proba)[::-1]
        for idx in sorted_idx[1:]:
            if le.classes_[idx] != "section_marker":
                pred_type = le.classes_[idx]
                best_conf = float(proba[idx])
                break

    if best_conf < min_confidence:
        return "unknown", best_conf

    return pred_type, best_conf


def predict_batch(texts, categories=None, model_path=MODEL_PATH, min_confidence=MIN_CONFIDENCE):
    """
    Predict types for a list of texts efficiently (batch inference).

    Args:
        texts:      List of text strings.
        categories: List of category ints (same length as texts), or None.
        model_path: Path to trained model.

    Returns:
        List of (type, confidence) tuples.
    """
    import scipy.sparse as sp

    bundle = load_model(model_path)
    if bundle is None:
        return [("unknown", 0.0)] * len(texts)

    if categories is None:
        categories = [0] * len(texts)

    tfidf      = bundle["tfidf"]
    scaler     = bundle["scaler"]
    clf        = bundle["clf"]
    le         = bundle["label_encoder"]
    feat_names = bundle["feature_names"]

    X_tfidf   = tfidf.transform(texts)
    X_numeric = np.array([
        _features_to_vector(_extract_features(t), feat_names) for t in texts
    ])
    X_numeric_scaled = scaler.transform(X_numeric)
    X = sp.hstack([X_tfidf, sp.csr_matrix(X_numeric_scaled)])

    probas    = clf.predict_proba(X)
    best_idxs = np.argmax(probas, axis=1)

    results = []
    for i, (idx, cat) in enumerate(zip(best_idxs, categories)):
        conf      = float(probas[i][idx])
        pred_type = le.classes_[idx]

        # Category gates
        if pred_type == "balloon_number" and cat == 1:
            pred_type = "dimension_value"
        if pred_type == "section_marker" and cat == 2:
            sorted_idx = np.argsort(probas[i])[::-1]
            for j in sorted_idx[1:]:
                if le.classes_[j] != "section_marker":
                    pred_type = le.classes_[j]
                    conf = float(probas[i][j])
                    break

        if conf < min_confidence:
            pred_type = "unknown"

        results.append((pred_type, conf))

    return results


# ============================================================
# Evaluation
# ============================================================

def evaluate(results_dir="results/batch", model_path=MODEL_PATH):
    """
    Evaluate the trained model with proper precision/recall/F1.

    Uses the regex labels as ground truth (they're correct for unambiguous cases).
    Reports per-class and macro/weighted averages.
    """
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score, f1_score)

    print("\n=== MODEL EVALUATION ===")

    texts, labels, categories = load_training_data(results_dir)
    if not texts:
        print("  No data found")
        return

    preds = predict_batch(texts, categories, model_path)
    pred_types = [p[0] for p in preds]
    confidences = [p[1] for p in preds]

    # Overall accuracy
    acc = accuracy_score(labels, pred_types)
    f1_macro    = f1_score(labels, pred_types, average='macro',    zero_division=0)
    f1_weighted = f1_score(labels, pred_types, average='weighted', zero_division=0)

    print(f"\n  Overall metrics:")
    print(f"  {'Accuracy':<30} {acc:.4f}")
    print(f"  {'F1 (macro)':<30} {f1_macro:.4f}")
    print(f"  {'F1 (weighted)':<30} {f1_weighted:.4f}")

    # How many "unknown" predictions (model wasn't confident)
    n_unknown = sum(1 for p in pred_types if p == "unknown")
    print(f"  {'Unknown predictions':<30} {n_unknown}/{len(pred_types)} "
          f"({n_unknown/len(pred_types)*100:.1f}%)")

    # Per-class report (excluding unknowns from ground truth)
    print(f"\n  Per-class classification report:")
    print(classification_report(
        labels, pred_types,
        labels=TRAINABLE_TYPES,
        target_names=TRAINABLE_TYPES,
        zero_division=0,
    ))

    # Confidence distribution
    confs = [c for c in confidences if c > 0]
    if confs:
        print(f"  Confidence distribution:")
        print(f"    Mean:   {sum(confs)/len(confs):.3f}")
        print(f"    >0.9:   {sum(1 for c in confs if c > 0.9)}/{len(confs)} "
              f"({sum(1 for c in confs if c > 0.9)/len(confs)*100:.1f}%)")
        print(f"    >0.7:   {sum(1 for c in confs if c > 0.7)}/{len(confs)} "
              f"({sum(1 for c in confs if c > 0.7)/len(confs)*100:.1f}%)")
        print(f"    <0.45:  {sum(1 for c in confs if c < 0.45)}/{len(confs)} "
              f"({sum(1 for c in confs if c < 0.45)/len(confs)*100:.1f}%)")

    return {
        "accuracy":    acc,
        "f1_macro":    f1_macro,
        "f1_weighted": f1_weighted,
        "n_unknown":   n_unknown,
        "n_total":     len(pred_types),
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/evaluate text classifier")
    parser.add_argument("--train",    action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--predict",  type=str, default=None, help="Predict a single text")
    parser.add_argument("--results",  default="results/batch", help="Results directory")
    parser.add_argument("--model",    default=MODEL_PATH, help="Model path")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if args.train:
        train(args.results, args.model)

    if args.evaluate:
        evaluate(args.results, args.model)

    if args.predict:
        t, conf = predict(args.predict, model_path=args.model)
        print(f"Text: '{args.predict}'")
        print(f"Predicted type: {t}")
        print(f"Confidence: {conf:.3f}")

    if not any([args.train, args.evaluate, args.predict]):
        print("Training and evaluating...")
        train(args.results, args.model)
        evaluate(args.results, args.model)
