#!/usr/bin/env python3
"""batch visualization generation

rules:
  monochrome: single module output (path a, path b, stardist)
  multicolor: mixed modules (pool, fused, final)

generates vis_01 through vis_10 + comparison grid per config.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import functools
print = functools.partial(print, flush=True)

import numpy as np
from PIL import Image
import cv2
import colorsys
from skimage import io
from datetime import datetime

from core.config import ALL_IMAGES

# directories
BATCH_RESULTS = Path(__file__).parent / "batch_results"
IMG_USED = Path(__file__).parent.parent / "img_used"

# hue values for monochrome mode
COLORS = {
    "path_a": 0.33,      # green
    "path_b": 0.50,      # cyan
    "stardist": 0.75,    # violet
    "pool_a": 0.33,      # green (from a)
    "pool_b": 0.50,      # cyan (from b)
}


def labels_to_monochrome(labels, bg_gray, hue, saturation=0.8, value=0.9, alpha=0.7):
    """colorize labels with single hue"""
    result = np.stack([bg_gray, bg_gray, bg_gray], axis=-1).astype(np.float32)

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    color = (int(r * 255), int(g * 255), int(b * 255))

    mask = labels > 0
    for c in range(3):
        result[:, :, c][mask] = (1 - alpha) * result[:, :, c][mask] + alpha * color[c]

    from skimage.segmentation import find_boundaries
    contours = find_boundaries(labels, mode='thick')
    result[contours] = [255, 255, 255]

    return result.astype(np.uint8)


def labels_to_multicolor(labels, bg_gray, alpha=0.7):
    """colorize labels with distinct colors"""
    result = np.stack([bg_gray, bg_gray, bg_gray], axis=-1).astype(np.float32)

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    if len(unique_labels) == 0:
        return result.astype(np.uint8)

    np.random.seed(42)
    n = len(unique_labels)
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    np.random.shuffle(colors)

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        color = colors[i % len(colors)]
        for c in range(3):
            result[:, :, c][mask] = (1 - alpha) * result[:, :, c][mask] + alpha * color[c]

    return result.astype(np.uint8)


def pool_bicolor(pool, path_a_filtered, path_b_pooled, bg_gray, alpha=0.7):
    """colorize pool: green=path a, cyan=path b"""
    result = np.stack([bg_gray, bg_gray, bg_gray], axis=-1).astype(np.float32)

    r_a, g_a, b_a = colorsys.hsv_to_rgb(COLORS["pool_a"], 0.8, 0.9)
    color_a = (int(r_a * 255), int(g_a * 255), int(b_a * 255))

    r_b, g_b, b_b = colorsys.hsv_to_rgb(COLORS["pool_b"], 0.8, 0.9)
    color_b = (int(r_b * 255), int(g_b * 255), int(b_b * 255))

    unique_labels = np.unique(pool)
    unique_labels = unique_labels[unique_labels != 0]

    for lbl in unique_labels:
        mask = pool == lbl
        overlap_a = np.sum((path_a_filtered > 0) & mask)
        overlap_b = np.sum((path_b_pooled > 0) & mask)

        color = color_a if overlap_a > overlap_b else color_b

        for c in range(3):
            result[:, :, c][mask] = (1 - alpha) * result[:, :, c][mask] + alpha * color[c]

    from skimage.segmentation import find_boundaries
    contours = find_boundaries(pool, mode='thick')
    result[contours] = [255, 255, 255]

    return result.astype(np.uint8)


def create_fusion_decision_vis(pool, stardist, final, bg_gray, output_path):
    """fusion decision visualization with legend"""
    _, w = pool.shape
    result = np.stack([bg_gray, bg_gray, bg_gray], axis=-1).astype(np.float32)

    COLOR_SICLE = (0, 200, 100)
    COLOR_STARDIST = (100, 150, 255)
    COLOR_REJECTED = (200, 80, 80)

    alpha = 0.7

    pool_labels = set(np.unique(pool)) - {0}
    for lbl in pool_labels:
        mask = pool == lbl
        overlap_final = np.sum((final > 0) & mask) / np.sum(mask) if np.sum(mask) > 0 else 0
        color = COLOR_SICLE if overlap_final > 0.5 else COLOR_REJECTED
        for c in range(3):
            result[:, :, c][mask] = (1 - alpha) * result[:, :, c][mask] + alpha * color[c]

    stardist_labels = set(np.unique(stardist)) - {0}
    for lbl in stardist_labels:
        mask = stardist == lbl
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlap_final = np.sum((final > 0) & mask) / np.sum(mask) if np.sum(mask) > 0 else 0
        color = COLOR_STARDIST if overlap_final > 0.5 else (150, 150, 150)
        thickness = 2 if overlap_final > 0.5 else 1
        result_uint8 = result.astype(np.uint8)
        cv2.drawContours(result_uint8, contours, -1, color, thickness)
        result = result_uint8.astype(np.float32)

    # legend
    legend_h = 100
    legend = np.ones((legend_h, w, 3), dtype=np.uint8) * 40

    cv2.putText(legend, "Fusion Decisions:", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(legend, (10, 40), (30, 60), COLOR_SICLE, -1)
    cv2.putText(legend, "SICLE kept", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(legend, (200, 40), (220, 60), COLOR_STARDIST, -1)
    cv2.putText(legend, "StarDist kept", (230, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(legend, (400, 40), (420, 60), COLOR_REJECTED, -1)
    cv2.putText(legend, "Rejected", (430, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    result = np.clip(result, 0, 255).astype(np.uint8)
    full_vis = np.vstack([result, legend])
    Image.fromarray(full_vis).save(output_path)


def create_comparison_grid(images, titles, output_path, cols=3):
    """create grid of images with titles"""
    if len(images) == 0:
        return

    target_h = 400
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        resized.append(cv2.resize(img, (new_w, target_h)))

    max_w = max(img.shape[1] for img in resized)
    padded = []
    for img in resized:
        h, w = img.shape[:2]
        if w < max_w:
            pad = np.zeros((h, max_w - w, 3), dtype=np.uint8)
            img = np.hstack([img, pad])
        padded.append(img)

    titled = []
    for img, title in zip(padded, titles):
        title_bar = np.zeros((30, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_bar, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        titled.append(np.vstack([title_bar, img]))

    rows = []
    for i in range(0, len(titled), cols):
        row_imgs = titled[i:i+cols]
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros_like(row_imgs[0]))
        rows.append(np.hstack(row_imgs))

    grid = np.vstack(rows)
    Image.fromarray(grid).save(output_path)


def visualize_config(config_dir, img_name):
    """generate all visualizations for one config"""
    print(f"    generating vis for {config_dir.name}...")

    img_original_path = IMG_USED / img_name / "pre_seg" / "img_original.png"
    if not img_original_path.exists():
        print(f"      error: img_original not found")
        return

    img_orig = np.array(Image.open(img_original_path).convert('RGB'))
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)

    # 1. input rgb
    Image.fromarray(img_orig).save(config_dir / "vis_01_input.png")

    # 2. saliency
    sal_path = config_dir / "temp_saliency.png"
    if sal_path.exists():
        sal = io.imread(str(sal_path))
        if sal.ndim == 2:
            sal_vis = cv2.applyColorMap(sal, cv2.COLORMAP_JET)
            Image.fromarray(sal_vis).save(config_dir / "vis_02_saliency.png")

    # 3. path a filtered (green)
    path_a_filtered_path = config_dir / "path_a_filtered.npy"
    path_a_filtered = None
    n_a = 0
    if path_a_filtered_path.exists():
        path_a_filtered = np.load(path_a_filtered_path)
        n_a = len(np.unique(path_a_filtered)) - 1
        vis = labels_to_monochrome(path_a_filtered, img_gray, COLORS["path_a"])
        Image.fromarray(vis).save(config_dir / "vis_03_path_a_filtered.png")

    # 4. path b veta (cyan)
    path_b_pooled_path = config_dir / "path_b_pooled.npy"
    path_b_pooled = None
    n_b = 0
    if path_b_pooled_path.exists():
        path_b_pooled = np.load(path_b_pooled_path)
        n_b = len(np.unique(path_b_pooled)) - 1
        vis = labels_to_monochrome(path_b_pooled, img_gray, COLORS["path_b"])
        Image.fromarray(vis).save(config_dir / "vis_04_path_b_veta.png")

    # 5. pool bicolor
    pool_path = config_dir / "pool.npy"
    pool = None
    n_pool = 0
    if pool_path.exists() and path_a_filtered is not None and path_b_pooled is not None:
        pool = np.load(pool_path)
        n_pool = len(np.unique(pool)) - 1
        vis = pool_bicolor(pool, path_a_filtered, path_b_pooled, img_gray)
        Image.fromarray(vis).save(config_dir / "vis_05_pool.png")

    # 6. stardist (violet)
    stardist_path = config_dir / "stardist.npy"
    stardist = None
    n_sd = 0
    if stardist_path.exists():
        stardist = np.load(stardist_path)
        n_sd = len(np.unique(stardist)) - 1
        vis = labels_to_monochrome(stardist, img_gray, COLORS["stardist"])
        Image.fromarray(vis).save(config_dir / "vis_06_stardist.png")

    # 7. fused (multicolor)
    fused_path = config_dir / "fused.npy"
    fused = None
    n_fused = 0
    if fused_path.exists():
        fused = np.load(fused_path)
        n_fused = len(np.unique(fused)) - 1
        vis = labels_to_multicolor(fused, img_gray)
        Image.fromarray(vis).save(config_dir / "vis_07_fused.png")

    # 8. final (multicolor)
    final_path = config_dir / "final.npy"
    final = None
    n_final = 0
    if final_path.exists():
        final = np.load(final_path)
        n_final = len(np.unique(final)) - 1
        vis = labels_to_multicolor(final, img_gray)
        Image.fromarray(vis).save(config_dir / "vis_08_final.png")

        # 9. overlay on rgb
        from skimage.color import label2rgb
        overlay = label2rgb(final, image=img_orig, bg_label=0, alpha=0.3)
        Image.fromarray((overlay * 255).astype(np.uint8)).save(config_dir / "vis_09_overlay.png")

    # 10. fusion decisions
    if pool is not None and stardist is not None and final is not None:
        create_fusion_decision_vis(pool, stardist, final, img_gray, config_dir / "vis_10_fusion_decisions.png")

    # 11. comparison grid
    images = []
    titles = []

    for vis_file, title in [
        ("vis_01_input.png", "Input"),
        ("vis_03_path_a_filtered.png", f"Path A ({n_a})"),
        ("vis_04_path_b_veta.png", f"Path B ({n_b})"),
        ("vis_05_pool.png", f"Pool ({n_pool})"),
        ("vis_06_stardist.png", f"StarDist ({n_sd})"),
        ("vis_08_final.png", f"Final ({n_final})"),
    ]:
        path = config_dir / vis_file
        if path.exists():
            images.append(np.array(Image.open(path)))
            titles.append(title)

    if len(images) >= 3:
        create_comparison_grid(images, titles, config_dir / "vis_comparison.png", cols=3)


def main():
    print("batch visualization generation")
    print(f"started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"images: {len(ALL_IMAGES)}")

    for img_name in ALL_IMAGES:
        print(f"\n  {img_name}:")

        img_dir = BATCH_RESULTS / img_name
        if not img_dir.exists():
            print(f"    skipped: no batch results")
            continue

        config1_dir = img_dir / "config1_orig_orig"
        if config1_dir.exists():
            visualize_config(config1_dir, img_name)

        config2_dir = img_dir / "config2_orig_simp"
        if config2_dir.exists():
            visualize_config(config2_dir, img_name)

    print("\nvisualization complete")
    print(f"ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
