# CAD Dimension Extraction & Association Tool

Automated 2D CAD drawing analysis using Computer Vision.

## What it does
- **Extracts** all dimension text from engineering drawings using OCR
- **Segments** geometric elements: lines, circles, contours, hatching
- **Classifies** each annotation into 16 engineering types (regex + ML)
- **Associates** each dimension to the geometric feature it describes
- **Labels** what each dimension means (length, height, bore diameter, etc.)
- **Attributes** each dimension to the correct part in assembly drawings

## Pipeline
1. **Preprocessing** — CLAHE contrast, unsharp masking, adaptive thresholding
2. **Geometric Segmentation** — Hough lines, contour analysis, circle detection
3. **Dimension Extraction** — EasyOCR (2-pass: full image 2× + BOM region 3×)
4. **Classification** — Regex priority chain + ML fallback (TF-IDF + SVM)
5. **Geometric Association** — nearest-element linking with distance metrics
6. **Semantic Labelling** — assigns meaning (length, height, bore_diameter)
7. **Part Attribution** — links dimensions to parts via balloon → BOM chain
8. **Web Dashboard** — Flask app with visualizations and downloads

## Setup
```bash
python -m venv cad_env
cad_env\Scripts\activate
pip install -r requirements.txt

# Train ML classifier (one-time)
python src/text_classifier.py --train --results results/batch
```

## Run
```bash
# Single image
python src/pipeline.py data/category_1/cad1_001.png results

# Full dataset
python batch_process.py

# Evaluate (18 metrics)
python evaluate_full.py

# Web dashboard
python app/app.py
# Open http://localhost:5000
```

## Dataset
- **Category 1:** Part drawings with dimensions and tolerances (23 images)
- **Category 2:** Assembly drawings with balloon numbers and BOM (11 images)
- **Category 3:** Assembly views only (2 images)

Source: K.L. Narayana *Machine Drawing* (3rd ed.)

## Results (18/18 metrics passing)
- OCR confidence: 89.3% mean
- Classification: 93.2% meaningful rate (6.8% unknown)
- Association: 78.1% match rate, 86% within 50px
- Semantic labelling: 100% labelled, 88.3% consistency
- Part attribution: 94.7% named rate

## Tech Stack
- OpenCV 4.x (CLAHE, Hough, contours, morphology)
- EasyOCR (CRAFT + CRNN)
- scikit-learn (TF-IDF + LinearSVC text classifier)
- Flask (web dashboard)
- PyTorch (CUDA - RTX 4060)
