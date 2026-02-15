#!/usr/bin/env python3
"""generate uoift saliency for all images (sec 3.2)

creates saliency maps in img_used/<IMAGE>/pre_seg/UOIFT/.
uoift params: alpha=-0.7 (dark nuclei, eq. 1).
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import functools
print = functools.partial(print, flush=True)

import numpy as np
from skimage import io
import time
from datetime import datetime

from core.config import ALL_IMAGES, UOIFT_ALPHA, UOIFT_SP_SIZE
from core.uoift_saliency_python import generate_uoift_saliency, save_saliency_map

# directories
IMG_USED = Path(__file__).parent.parent / "img_used"


def generate_for_image(img_name):
    """generate uoift saliency for one image"""
    img_dir = IMG_USED / img_name / "pre_seg"
    img_path = img_dir / "img_original.png"

    if not img_path.exists():
        print(f"    error: img_original.png not found")
        return None

    uoift_dir = img_dir / "UOIFT"
    uoift_dir.mkdir(exist_ok=True)

    img_rgb = io.imread(str(img_path))
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb]*3, axis=2)
    elif img_rgb.shape[2] == 4:
        img_rgb = img_rgb[:, :, :3]

    print(f"    image shape: {img_rgb.shape}")

    t_start = time.time()

    saliency, metadata = generate_uoift_saliency(
        img_rgb,
        sp_size=UOIFT_SP_SIZE,
        alpha=UOIFT_ALPHA,
        config_overrides={
            'extinction_attribute': 'volume',
            'use_lab_channel': 'L'
        }
    )

    elapsed = time.time() - t_start

    print(f"    superpixels: {metadata['n_superpixels']}")
    print(f"    mean saliency: {metadata['saliency_stats']['mean']:.4f}")
    print(f"    time: {elapsed:.1f}s")

    # save saliency
    png_path = uoift_dir / "saliency_uoift.png"
    pgm_path = uoift_dir / "saliency_uoift.pgm"

    save_saliency_map(saliency, str(png_path))
    save_saliency_map(saliency, str(pgm_path))

    print(f"    saved: {png_path.name}, {pgm_path.name}")

    return {
        'time': elapsed,
        'superpixels': metadata['n_superpixels'],
        'mean_saliency': metadata['saliency_stats']['mean'],
    }


def main():
    print("uoift saliency generation (sec 3.2)")
    print(f"started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"alpha: {UOIFT_ALPHA}, sp_size: {UOIFT_SP_SIZE}")
    print(f"images: {len(ALL_IMAGES)}")

    results = {}
    total_time = 0

    for img_name in ALL_IMAGES:
        print(f"\n  {img_name}:")

        try:
            result = generate_for_image(img_name)
            if result:
                results[img_name] = result
                total_time += result['time']
        except Exception as e:
            print(f"    error: {e}")
            results[img_name] = {'error': str(e)}

    # summary
    print("\nsummary")
    print(f"\n{'Image':<12} {'Time (s)':>10} {'Superpixels':>12} {'Mean Sal':>10}")

    for img_name, result in results.items():
        if 'error' in result:
            print(f"{img_name:<12} {'ERROR':>10}")
        else:
            print(f"{img_name:<12} {result['time']:>10.1f} {result['superpixels']:>12} {result['mean_saliency']:>10.4f}")

    if len(results) > 0:
        print(f"{'TOTAL':<12} {total_time:>10.1f}s")
        print(f"{'AVG':<12} {total_time/len(results):>10.1f}s")

    print(f"\ncompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
