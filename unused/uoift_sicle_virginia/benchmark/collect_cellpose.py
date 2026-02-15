#!/usr/bin/env python3
"""
collect existing cellpose results from img_used/ into benchmark_results/

Uses pre-computed cellpose_alone results to avoid re-running (slow without GPU)
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import ALL_IMAGES

import functools
print = functools.partial(print, flush=True)

import numpy as np
from skimage import io
import json
from datetime import datetime
from PIL import Image
import cv2
import colorsys

# base directories
BASE_IMG_DIR = Path(__file__).parent.parent / "img_used"
OUTPUT_BASE = Path(__file__).parent / "benchmark_results"


def create_visualization(masks, bg_gray, alpha=0.7):
    """create colored visualization of masks on grayscale background"""
    h, w = masks.shape
    result = np.stack([bg_gray, bg_gray, bg_gray], axis=-1).astype(np.float32)

    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels != 0]

    if len(unique_labels) == 0:
        return result.astype(np.uint8)

    # generate distinct colors
    np.random.seed(42)
    n = len(unique_labels)
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    np.random.shuffle(colors)

    # apply colors
    for i, lbl in enumerate(unique_labels):
        mask = masks == lbl
        color = colors[i % len(colors)]
        for c in range(3):
            result[:, :, c][mask] = (1 - alpha) * result[:, :, c][mask] + alpha * color[c]

    return result.astype(np.uint8)


def load_cellpose_mask_from_png(mask_path):
    """load cellpose mask from png file (indexed color)"""
    img = io.imread(str(mask_path))
    if img.ndim == 3:
        # convert to grayscale/labels - take first channel
        # cellpose masks are typically saved as indexed images
        if img.shape[2] == 4:  # RGBA
            img = img[:, :, 0]
        else:  # RGB
            # try to reconstruct labels from colors
            # this is a heuristic - may not work perfectly
            h, w = img.shape[:2]
            labels = np.zeros((h, w), dtype=np.int32)
            unique_colors = {}
            for y in range(h):
                for x in range(w):
                    pixel = tuple(img[y, x])
                    if pixel == (0, 0, 0):  # background
                        continue
                    if pixel not in unique_colors:
                        unique_colors[pixel] = len(unique_colors) + 1
                    labels[y, x] = unique_colors[pixel]
            return labels
    return img.astype(np.int32)


def collect_cellpose_result(img_name, output_dir):
    """collect existing cellpose result from img_used"""
    img_dir = BASE_IMG_DIR / img_name
    cellpose_dir = img_dir / "cellpose_alone"

    output_dir.mkdir(parents=True, exist_ok=True)

    # get nuclei count from centroids.csv
    centroids_csv = cellpose_dir / "centroids.csv"
    if not centroids_csv.exists():
        return {'error': 'centroids.csv not found'}

    with open(centroids_csv) as f:
        lines = f.readlines()
    n_nuclei = len(lines) - 1  # minus header

    # load mask image
    mask_png = cellpose_dir / "cellpose_masks.png"
    if mask_png.exists():
        masks = load_cellpose_mask_from_png(mask_png)
        np.save(output_dir / "cellpose_masks.npy", masks)

        # create new visualization
        img_original = img_dir / "pre_seg" / "img_original.png"
        if img_original.exists():
            img = io.imread(str(img_original))
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
            vis = create_visualization(masks, img_gray)
            Image.fromarray(vis).save(output_dir / "cellpose_vis.png")

    # time not available from existing results - estimate based on typical run
    # (85 min per image on CPU is typical, but original results may have used GPU)
    return {
        'nuclei': int(n_nuclei),
        'time': 0,  # unknown - not recorded in original results
        'source': 'pre-computed (cellpose_alone)'
    }


def main():
    print("collecting cellpose results from img_used/")
    print(f"started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"images: {len(ALL_IMAGES)}")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    results = {}

    for img_name in ALL_IMAGES:
        print(f"\n  collecting: {img_name}")

        output_dir = OUTPUT_BASE / img_name / "cellpose"

        try:
            result = collect_cellpose_result(img_name, output_dir)
            results[img_name] = result

            # save individual report
            with open(output_dir / "report.json", 'w') as f:
                json.dump(result, f, indent=2)

            if 'error' in result:
                print(f"    error: {result['error']}")
            else:
                print(f"    collected: {result['nuclei']} nuclei")

        except Exception as e:
            print(f"    error: {e}")
            results[img_name] = {'error': str(e)}

    # summary
    print("\ncellpose collection summary")
    print(f"\n{'Image':<12} {'Nuclei':>10}")

    total_nuclei = 0
    count = 0

    for img_name, result in results.items():
        if 'error' in result:
            print(f"{img_name:<12} {'ERROR':>10}")
        else:
            nuclei = result['nuclei']
            print(f"{img_name:<12} {nuclei:>10}")
            total_nuclei += nuclei
            count += 1

    if count > 0:
        print(f"{'AVERAGE':<12} {total_nuclei/count:>10.0f}")

    # save summary
    summary_path = OUTPUT_BASE / "cellpose_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'source': 'collected from img_used/cellpose_alone/',
            'note': 'timing not available - results were pre-computed',
            'results': results,
            'average': {
                'nuclei': total_nuclei / count if count > 0 else 0,
            }
        }, f, indent=2)

    print(f"\nsummary saved: {summary_path}")
    print(f"\ncellpose collection complete")
    print(f"ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
