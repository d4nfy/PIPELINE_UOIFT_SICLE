#!/usr/bin/env python3
"""
run cellpose 4.0+ on all images for benchmark comparison.

outputs results to benchmark_results/<image>/cellpose/
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import ALL_IMAGES

# force unbuffered output for real-time progress
import functools
print = functools.partial(print, flush=True)

import numpy as np
from skimage import io
import json
import time
from datetime import datetime
from PIL import Image
import cv2

# base directories
BASE_IMG_DIR = Path(__file__).parent.parent / "img_used"
OUTPUT_BASE = Path(__file__).parent / "benchmark_results"


def run_cellpose(img_path, output_dir):
    """run cellpose 4.0+ nuclei model"""
    from cellpose import models

    output_dir.mkdir(parents=True, exist_ok=True)

    img = io.imread(str(img_path))
    print(f"      image shape: {img.shape}")

    t0 = time.time()

    # cellpose 4.0.1+ new api: use Cellpose without model_type
    # default is cyto3 but we want nuclei, so use pretrained_model
    model = models.CellposeModel(pretrained_model='nuclei', gpu=False)

    # run inference - v4.0.1+ removed channels parameter
    masks, flows, styles = model.eval(
        img,
        diameter=None,  # auto-detect
        flow_threshold=0.4,
        cellprob_threshold=0.0
    )

    elapsed = time.time() - t0

    n_nuclei = len(np.unique(masks)) - 1
    print(f"      detected: {n_nuclei} nuclei in {elapsed:.1f}s")

    # save masks
    np.save(output_dir / "cellpose_masks.npy", masks)

    # create visualization
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    vis = create_visualization(masks, img_gray)
    Image.fromarray(vis).save(output_dir / "cellpose_vis.png")

    return {
        'nuclei': int(n_nuclei),
        'time': elapsed
    }


def create_visualization(masks, bg_gray, alpha=0.7):
    """create colored visualization of masks on grayscale background"""
    import colorsys

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


def main():
    print("cellpose 4.0+ benchmark")
    print(f"started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"images: {len(ALL_IMAGES)}")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    results = {}

    for img_name in ALL_IMAGES:
        print(f"\n  processing: {img_name}")

        img_dir = BASE_IMG_DIR / img_name
        img_original = img_dir / "pre_seg" / "img_original.png"

        if not img_original.exists():
            print(f"    img_original not found, skipping")
            results[img_name] = {'error': 'img_original not found'}
            continue

        output_dir = OUTPUT_BASE / img_name / "cellpose"

        try:
            result = run_cellpose(img_original, output_dir)
            results[img_name] = result

            # save individual report
            with open(output_dir / "report.json", 'w') as f:
                json.dump(result, f, indent=2)

        except Exception as e:
            print(f"    error: {e}")
            results[img_name] = {'error': str(e)}

    # summary
    print("\ncellpose benchmark summary")
    print(f"\n{'Image':<12} {'Nuclei':>10} {'Time (s)':>10}")

    total_nuclei = 0
    total_time = 0
    count = 0

    for img_name, result in results.items():
        if 'error' in result:
            print(f"{img_name:<12} {'ERROR':>10}")
        else:
            nuclei = result['nuclei']
            t = result['time']
            print(f"{img_name:<12} {nuclei:>10} {t:>10.1f}")
            total_nuclei += nuclei
            total_time += t
            count += 1

    if count > 0:
        print(f"{'AVERAGE':<12} {total_nuclei/count:>10.0f} {total_time/count:>10.1f}")

    # save summary
    summary_path = OUTPUT_BASE / "cellpose_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'average': {
                'nuclei': total_nuclei / count if count > 0 else 0,
                'time': total_time / count if count > 0 else 0
            }
        }, f, indent=2)

    print(f"\nsummary saved: {summary_path}")
    print(f"\ncellpose benchmark complete")
    print(f"ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
