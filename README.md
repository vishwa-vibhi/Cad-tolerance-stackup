# CAD Tolerance Stack-Up Analysis

Automated 2D CAD drawing analysis using Computer Vision.

## What it does
- Segments multiple views from a single drawing sheet
- Detects dimension lines, extension lines, arrowheads, balloons
- Reads dimension values and tolerances using OCR
- Links each dimension to the correct part

## Pipeline
1. Preprocessing — OpenCV binarization, deskew, morphological cleanup
2. View Segmentation — projection profiles, connected components
3. Element Detection — Hough lines, template matching, contour analysis
4. OCR — PaddleOCR with rotation correction
5. Association — geometric line tracing, nearest contour

## Setup
```bash
python -m venv cad_env
cad_env\Scripts\activate
pip install -r requirements.txt
```

## Dataset
- Category 1: Part drawings with dimensions and tolerances
- Category 2: Assembly drawings with balloon numbers and BOM
- Category 3: Assembly views only

## Tech Stack
- OpenCV 4.x
- PaddleOCR
- YOLOv8
- PyTorch (CUDA - RTX 4060)