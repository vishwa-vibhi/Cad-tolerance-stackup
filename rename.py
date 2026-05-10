"""
Rename dataset images to systematic names.
data/category_1/Screenshot xxx.png  →  data/category_1/cat1_001.png
data/category_2/Screenshot xxx.png  →  data/category_2/cat2_001.png
"""

import os
from pathlib import Path

DATASET_DIR = "data"

def rename_category(category_path, prefix):
    images = sorted([f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"\nFound {len(images)} images in {category_path}")
    print("-" * 60)

    for idx, old_name in enumerate(images, start=1):
        ext = os.path.splitext(old_name)[1].lower()
        new_name = f"{prefix}_{idx:03d}{ext}"

        old_path = os.path.join(category_path, old_name)
        new_path = os.path.join(category_path, new_name)

        if old_path == new_path:
            print(f"  Skip (already named): {new_name}")
            continue

        os.rename(old_path, new_path)
        print(f"  {old_name}  →  {new_name}")

    print(f"Done. Renamed {len(images)} files.")


if __name__ == "__main__":
    categories = [
        ("data/category_1", "cad1"),
        ("data/category_2", "cad2"),
        ("data/category_3", "cad3"),
    ]

    for path, prefix in categories:
        if os.path.exists(path):
            rename_category(path, prefix)
        else:
            print(f"Skipping {path} (does not exist)")

    print("\nAll categories renamed.")