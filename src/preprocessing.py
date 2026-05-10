"""
Stage 1: Preprocessing
Cleans up engineering drawing screenshots before sending to VLM.
Pure OpenCV - no smart logic, just reliable cleanup.
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
    """Convert to grayscale - we don't need color."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray):
    """Gentle Gaussian blur to remove screenshot noise."""
    return cv2.GaussianBlur(gray, (3, 3), 0)


def binarize(gray):
    """
    Adaptive threshold - handles uneven brightness across the page.
    Better than Otsu for screenshots.
    """
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=10
    )
    return binary


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
    """Close tiny gaps in lines, remove tiny dots."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cleaned


def preprocess(image_path, save_result=True, output_dir="results"):
    """Main preprocessing pipeline."""
    print(f"\nProcessing: {os.path.basename(image_path)}")
    print("-" * 50)

    img      = load_image(image_path)
    gray     = to_grayscale(img)
    denoised = denoise(gray)
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/preprocessing.py <image_path>")
    else:
        preprocess(sys.argv[1])
        print("Done.")