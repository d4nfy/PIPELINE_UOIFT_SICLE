#!/usr/bin/env python3
"""prepare dataset: scan tif images, convert to png, create metadata"""

import json
from pathlib import Path
from PIL import Image
from datetime import datetime

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_DIR = PIPELINE_DIR.parent
PROJECT_ROOT = REPO_DIR.parent
NEW_IMAGES_DIR = PROJECT_ROOT / "new_images"
PROCESSED_DIR = PROJECT_ROOT / "new_images_processed"
RESULTS_DIR = PROJECT_ROOT / "new_images_results"


def scan_images():
    """scan new_images for tif files"""
    images = []
    categories = []

    for category_dir in sorted(NEW_IMAGES_DIR.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue

        category = category_dir.name
        categories.append(category)
        print(f"  {category}:")

        tif_files = list(category_dir.glob("*.tif")) + list(category_dir.glob("*.tiff"))
        tif_files.sort()

        for tif_path in tif_files:
            print(f"    {tif_path.name}")
            images.append({
                "filename": tif_path.name,
                "stem": tif_path.stem,
                "category": category,
                "path": str(tif_path),
                "size_bytes": tif_path.stat().st_size
            })

    print(f"  total: {len(images)} images, {len(categories)} categories")
    return images, categories


def convert_images(images):
    """convert tif to png format"""
    PROCESSED_DIR.mkdir(exist_ok=True)
    converted = []

    for i, img_info in enumerate(images, 1):
        category = img_info['category']
        stem = img_info['stem']
        tif_path = Path(img_info['path'])

        category_dir = PROCESSED_DIR / category
        category_dir.mkdir(exist_ok=True)
        png_path = category_dir / f"{stem}.png"

        try:
            img = Image.open(tif_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            width, height = img.size
            img.save(png_path, 'PNG')
            print(f"  [{i}/{len(images)}] {category}/{tif_path.name} ({width}x{height})")

            img_info['png_path'] = str(png_path)
            img_info['width'] = width
            img_info['height'] = height
            img_info['converted'] = True
            converted.append(img_info)
        except Exception as e:
            print(f"  [{i}/{len(images)}] {category}/{tif_path.name} ERROR: {e}")
            img_info['converted'] = False
            img_info['error'] = str(e)

    print(f"  converted: {len(converted)}/{len(images)}")
    return converted


def save_metadata(images, categories):
    """save metadata json"""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_images": len(images),
        "categories": categories,
        "images_by_category": {
            cat: sum(1 for img in images if img['category'] == cat)
            for cat in categories
        },
        "images": images
    }

    metadata_path = PROCESSED_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  metadata: {metadata_path}")
    for cat in categories:
        print(f"    {cat}: {metadata['images_by_category'][cat]} images")

    return metadata_path


def create_output_directories():
    """create results directory structure"""
    dirs = [
        RESULTS_DIR / "benchmark_stardist",
        RESULTS_DIR / "benchmark_cellpose",
        RESULTS_DIR / "pipeline_with_saliency",
        RESULTS_DIR / "pipeline_no_saliency",
        RESULTS_DIR / "visualizations",
        RESULTS_DIR / "comparisons",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def main():
    """prepare dataset for benchmark"""
    print("scanning images...")
    images, categories = scan_images()

    print("converting to png...")
    converted_images = convert_images(images)

    print("saving metadata...")
    save_metadata(converted_images, categories)

    print("creating output dirs...")
    create_output_directories()

    print(f"done: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
