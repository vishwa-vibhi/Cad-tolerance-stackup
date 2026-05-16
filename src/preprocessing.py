"""
Stage 1: Preprocessing
Cleans up engineering drawing screenshots before sending to OCR.
Pure OpenCV — adaptive per-region processing for best results.
"""

import cv2
import numpy as np
import os
import sys


def load_image(image_path):
    """Load image from disk."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    print(f"  Loaded: {img.shape[1]}x{img.shape[0]} pixels")
    return img


def to_grayscale(img):
    """Convert to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray):
    """Gentle Gaussian blur to remove screenshot noise."""
    return cv2.GaussianBlur(gray, (3, 3), 0)


def binarize(gray):
    """
    Adaptive threshold — handles uneven brightness across the page.
    Better than Otsu for screenshots with mixed lighting.
    """
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=10
    )
    return binary


def enhance_contrast(gray):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Improves local contrast — critical for small BOM table text
    and faint dimension lines in scanned drawings.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def sharpen(gray):
    """
    Unsharp masking — sharpens edges of text characters.
    Reduces OCR misreads on blurry or low-res drawings.
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return sharpened


def remove_frame(binary, margin_percent=0.02):
    """
    Crop a small margin from the edges.
    Removes scanner edges, screenshot borders, etc.
    """
    h, w = binary.shape
    mh = int(h * margin_percent)
    mw = int(w * margin_percent)
    return binary[mh:h-mh, mw:w-mw]


def cleanup_morphology(binary):
    """Close tiny gaps in lines, remove isolated noise pixels."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cleaned


def detect_bom_region(gray):
    """
    Detect the BOM table region (typically bottom-right of drawing).
    Returns (x, y, w, h) bounding box or None.

    The BOM region has dense horizontal lines forming a table grid.
    We detect it by finding the area with the highest density of
    short horizontal line segments.
    """
    h, w = gray.shape
    # BOM is always in the bottom half
    bottom_half = gray[h//2:, :]

    # Detect horizontal edges
    edges = cv2.Sobel(bottom_half, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.abs(edges).astype(np.uint8)
    _, thresh = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)

    # Find region with most horizontal lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)

    # Sum horizontally to find columns with many lines
    col_sums = np.sum(h_lines, axis=0)
    row_sums = np.sum(h_lines, axis=1)

    if col_sums.max() == 0 or row_sums.max() == 0:
        return None

    # Find the dense region
    col_thresh = col_sums.max() * 0.3
    row_thresh = row_sums.max() * 0.3

    col_mask = col_sums > col_thresh
    row_mask = row_sums > row_thresh

    cols = np.where(col_mask)[0]
    rows = np.where(row_mask)[0]

    if len(cols) < 20 or len(rows) < 5:
        return None

    bom_x = int(cols.min())
    bom_y = int(rows.min()) + h // 2
    bom_w = int(cols.max() - cols.min())
    bom_h = int(rows.max() - rows.min())

    # Sanity check: BOM should be at least 50px wide and 30px tall
    if bom_w < 50 or bom_h < 30:
        return None

    return (bom_x, bom_y, bom_w, bom_h)


def preprocess(image_path, save_result=True, output_dir="results"):
    """
    Main preprocessing pipeline with adaptive per-region enhancement.

    Improvements over basic pipeline:
    - CLAHE contrast enhancement before binarization
    - Unsharp masking for text sharpening
    - Better noise removal
    """
    print(f"\nProcessing: {os.path.basename(image_path)}")
    print("-" * 50)

    img      = load_image(image_path)
    gray     = to_grayscale(img)

    # Enhance contrast before denoising (CLAHE works on grayscale)
    enhanced = enhance_contrast(gray)

    # Sharpen text edges
    sharpened = sharpen(enhanced)

    # Gentle denoise after sharpening
    denoised = denoise(sharpened)

    # Binarize
    binary   = binarize(denoised)
    cropped  = remove_frame(binary)
    cleaned  = cleanup_morphology(cropped)

    print(f"  Final size: {cleaned.shape[1]}x{cleaned.shape[0]}")

    if save_result:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(output_dir, f"{filename}_clean.png")
        cv2.imwrite(out_path, cleaned)
        print(f"  Saved: {out_path}")

    print("-" * 50)
    return cleaned


def preprocess_for_ocr(image_path):
    """
    Return both the full enhanced image AND a BOM-region crop
    for region-specific OCR upscaling.

    Returns:
        (enhanced_bgr, bom_region_bgr_or_None, bom_bbox_or_None)
    """
    img  = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Full image enhancement
    enhanced_gray = enhance_contrast(gray)
    sharpened     = sharpen(enhanced_gray)

    # Rebuild BGR from enhanced gray
    enhanced_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    # Detect BOM region for extra upscaling
    bom_bbox = detect_bom_region(gray)
    bom_crop = None
    if bom_bbox is not None:
        bx, by, bw, bh = bom_bbox
        # Add margin
        margin = 10
        x1 = max(0, bx - margin)
        y1 = max(0, by - margin)
        x2 = min(img.shape[1], bx + bw + margin)
        y2 = min(img.shape[0], by + bh + margin)
        bom_crop = enhanced_bgr[y1:y2, x1:x2]
        bom_bbox = (x1, y1, x2 - x1, y2 - y1)

    return enhanced_bgr, bom_crop, bom_bbox


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/preprocessing.py <image_path>")
    else:
        preprocess(sys.argv[1])
        print("Done.")