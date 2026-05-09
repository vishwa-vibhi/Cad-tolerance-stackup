import cv2
import numpy as np
import os
import sys


def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    print(f"  Loaded: {img.shape[1]}x{img.shape[0]} pixels")
    return img


def to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"  Grayscale: done")
    return gray


def denoise(gray):
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    print(f"  Denoise: done")
    return denoised


def binarize(gray):
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"  Binarize: done")
    return binary


def deskew(binary):
    coords = np.column_stack(np.where(binary < 128))
    if len(coords) == 0:
        return binary
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        print(f"  Deskew: no tilt detected")
        return binary
    h, w = binary.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    print(f"  Deskew: corrected {angle:.2f} degrees")
    return rotated


def morphological_cleanup(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    print(f"  Morphological cleanup: done")
    return cleaned


def preprocess(image_path, save_result=True, output_dir="results"):
    print(f"\nProcessing: {os.path.basename(image_path)}")
    print("-" * 40)
    img      = load_image(image_path)
    gray     = to_grayscale(img)
    denoised = denoise(gray)
    binary   = binarize(denoised)
    deskewed = deskew(binary)
    cleaned  = morphological_cleanup(deskewed)
    if save_result:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(output_dir, f"{filename}_preprocessed.png")
        cv2.imwrite(out_path, cleaned)
        print(f"  Saved to: {out_path}")
    print("-" * 40)
    print(f"Preprocessing complete.")
    return cleaned


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/preprocessing.py <image_path>")
    else:
        result = preprocess(sys.argv[1])
        print(f"Output shape: {result.shape}")