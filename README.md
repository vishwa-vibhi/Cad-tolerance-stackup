# CAD Tolerance Stack-Up Analysis

Automated 2D CAD drawing analysis using Computer Vision.

## What it does
- Segments multiple views from a single drawing sheet
- Detects dimension lines, extension lines, arrowheads, balloons
- Reads dimension values and tolerances using OCR
- Links each dimension to the correct part

## Pipeline
1. Preprocessing — OpenCV binarization, morphological cleanup
2. Element Detection — Hough lines, contour analysis, text region proposals
3. OCR — EasyOCR with upscaling and rotation-aware detection
4. Validation / Structuring — regex-based engineering text classification
5. Association — nearest-contour linking for dimension/label association

## Setup
```bash
python -m venv cad_env
cad_env\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python src/pipeline.py data/category_1/cad1_001.png results
python src/pipeline.py --dataset data/category_1 results/batch
```

## Dataset
- Category 1: Part drawings with dimensions and tolerances
- Category 2: Assembly drawings with balloon numbers and BOM
- Category 3: Assembly views only

## Tech Stack
- OpenCV 4.x
- EasyOCR
- PyTorch (CUDA - RTX 4060)