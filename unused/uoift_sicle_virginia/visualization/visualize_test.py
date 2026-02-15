#!/usr/bin/env python3
"""visualization for single pipeline test results

generates step by step visualizations:
  grayscale bg + colored superpixels, comparison grids,
  fusion decision visualization with legend.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import cv2
import colorsys

from generate_batch_vis import (
    labels_to_monochrome,
    create_comparison_grid,
)

# hue values for each pipeline stage
STAGE_COLORS = {
    "path_a": 0.33,      # green
    "path_b": 0.50,      # cyan
    "stardist": 0.75,    # violet
    "pool": 0.08,        # orange
    "fused": 0.0,        # red
    "final": 0.0,        # red
}


def visualize_config_results(config_dir: Path, img_original_path: Path):
    """generate all visualizations for a single config"""
    print(f"\nvisualizing: {config_dir.name}")

    img_orig = np.array(Image.open(img_original_path).convert('RGB'))
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)
    h, w = img_gray.shape

    # input image
    if not (config_dir / "vis_01_input.png").exists():
        Image.fromarray(img_orig).save(config_dir / "vis_01_input.png")

    # path a (green)
    path_a_npy = config_dir / "path_a_mask.npy"
    n_a = 0
    if path_a_npy.exists():
        path_a_mask = np.load(path_a_npy)
        n_a = len(np.unique(path_a_mask)) - 1
        vis_a = labels_to_monochrome(path_a_mask, img_gray, STAGE_COLORS["path_a"], alpha=0.75)
        Image.fromarray(vis_a).save(config_dir / "vis_03_path_a.png")

    # path b (cyan)
    path_b_npy = config_dir / "path_b_pooled.npy"
    n_b = 0
    if path_b_npy.exists():
        path_b_mask = np.load(path_b_npy)
        n_b = len(np.unique(path_b_mask)) - 1
        vis_b = labels_to_monochrome(path_b_mask, img_gray, STAGE_COLORS["path_b"], alpha=0.75)
        Image.fromarray(vis_b).save(config_dir / "vis_04_path_b_pooled.png")

    # pool (orange)
    pool_npy = config_dir / "pool.npy"
    n_pool = 0
    pool_mask = None
    if pool_npy.exists():
        pool_mask = np.load(pool_npy)
        n_pool = len(np.unique(pool_mask)) - 1
        vis_pool = labels_to_monochrome(pool_mask, img_gray, STAGE_COLORS["pool"], alpha=0.75)
        Image.fromarray(vis_pool).save(config_dir / "vis_05_pool.png")

    # stardist (violet)
    stardist_npy = config_dir / "stardist.npy"
    n_sd = 0
    stardist_mask = None
    if stardist_npy.exists():
        stardist_mask = np.load(stardist_npy)
        n_sd = len(np.unique(stardist_mask)) - 1
        vis_sd = labels_to_monochrome(stardist_mask, img_gray, STAGE_COLORS["stardist"], alpha=0.75)
        Image.fromarray(vis_sd).save(config_dir / "vis_06_stardist.png")

    # fused (red)
    fused_npy = config_dir / "fused.npy"
    n_fused = 0
    if fused_npy.exists():
        fused_mask = np.load(fused_npy)
        n_fused = len(np.unique(fused_mask)) - 1
        vis_fused = labels_to_monochrome(fused_mask, img_gray, STAGE_COLORS["fused"], alpha=0.75)
        Image.fromarray(vis_fused).save(config_dir / "vis_07_fused.png")

    # final (red)
    final_npy = config_dir / "final.npy"
    n_final = 0
    final_mask = None
    if final_npy.exists():
        final_mask = np.load(final_npy)
        n_final = len(np.unique(final_mask)) - 1
        vis_final = labels_to_monochrome(final_mask, img_gray, STAGE_COLORS["final"], alpha=0.75)
        Image.fromarray(vis_final).save(config_dir / "vis_08_final.png")

        # overlay on rgb
        from skimage.color import label2rgb
        overlay = label2rgb(final_mask, image=img_orig, bg_label=0, alpha=0.3)
        Image.fromarray((overlay * 255).astype(np.uint8)).save(config_dir / "vis_09_overlay.png")

    # fusion decisions
    if pool_mask is not None and stardist_mask is not None and final_mask is not None:
        create_fusion_decision_vis_detailed(
            pool_mask, stardist_mask, final_mask, img_gray,
            config_dir / "vis_10_fusion_decisions.png"
        )

    # comparison grid
    images = []
    titles = []

    if (config_dir / "vis_01_input.png").exists():
        images.append(np.array(Image.open(config_dir / "vis_01_input.png")))
        titles.append("Input RGB")

    if (config_dir / "vis_05_pool.png").exists():
        images.append(np.array(Image.open(config_dir / "vis_05_pool.png")))
        titles.append(f"Pool ({n_pool})")

    if (config_dir / "vis_06_stardist.png").exists():
        images.append(np.array(Image.open(config_dir / "vis_06_stardist.png")))
        titles.append(f"StarDist ({n_sd})")

    if (config_dir / "vis_07_fused.png").exists():
        images.append(np.array(Image.open(config_dir / "vis_07_fused.png")))
        titles.append(f"Fused ({n_fused})")

    if (config_dir / "vis_08_final.png").exists():
        images.append(np.array(Image.open(config_dir / "vis_08_final.png")))
        titles.append(f"Final ({n_final})")

    if len(images) >= 3:
        create_comparison_grid(
            images, titles,
            config_dir / "vis_comparison_pipeline.png",
            cols=3
        )

    print(f"  visualizations complete")


def create_fusion_decision_vis_detailed(pool_mask, stardist_mask, final_mask, bg_gray, output_path):
    """fusion decision visualization with legend"""
    h, w = pool_mask.shape
    result = np.stack([bg_gray, bg_gray, bg_gray], axis=-1).astype(np.float32)

    COLOR_SICLE = (0, 200, 100)
    COLOR_STARDIST = (100, 150, 255)
    COLOR_REJECTED = (200, 80, 80)

    alpha = 0.7

    # process pool regions
    pool_labels = set(np.unique(pool_mask)) - {0}
    for lbl in pool_labels:
        mask = pool_mask == lbl
        overlap_final = np.sum((final_mask > 0) & mask) / np.sum(mask) if np.sum(mask) > 0 else 0
        color = COLOR_SICLE if overlap_final > 0.5 else COLOR_REJECTED
        for c in range(3):
            result[:, :, c][mask] = (1 - alpha) * result[:, :, c][mask] + alpha * color[c]

    # stardist contours
    stardist_labels = set(np.unique(stardist_mask)) - {0}
    for lbl in stardist_labels:
        mask = stardist_mask == lbl
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlap_final = np.sum((final_mask > 0) & mask) / np.sum(mask) if np.sum(mask) > 0 else 0
        color = COLOR_STARDIST if overlap_final > 0.5 else (150, 150, 150)
        thickness = 2 if overlap_final > 0.5 else 1
        result_uint8 = result.astype(np.uint8)
        cv2.drawContours(result_uint8, contours, -1, color, thickness)
        result = result_uint8.astype(np.float32)

    # legend
    legend_h = 120
    legend = np.ones((legend_h, w, 3), dtype=np.uint8) * 40

    cv2.putText(legend, "Fusion Decision Legend:", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.rectangle(legend, (10, 45), (30, 65), COLOR_SICLE, -1)
    cv2.putText(legend, "SICLE regions kept", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.rectangle(legend, (10, 75), (30, 95), COLOR_STARDIST, -1)
    cv2.putText(legend, "StarDist regions kept (contours)", (40, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.rectangle(legend, (w//2, 45), (w//2 + 20, 65), COLOR_REJECTED, -1)
    cv2.putText(legend, "Rejected regions", (w//2 + 30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    result = np.clip(result, 0, 255).astype(np.uint8)
    full_vis = np.vstack([result, legend])
    Image.fromarray(full_vis).save(output_path)
    print(f"    saved {output_path.name}")


def main():
    output_base = Path(__file__).parent / "output_complete_test"
    img_original_path = Path(__file__).parent.parent / "img_used" / "B3-T_1" / "pre_seg" / "img_original.png"

    if not output_base.exists():
        print(f"  error: output dir not found: {output_base}")
        return

    if not img_original_path.exists():
        print(f"  error: original image not found: {img_original_path}")
        return

    print("visualization generation")

    config1_dir = output_base / "config1_orig_orig"
    if config1_dir.exists():
        visualize_config_results(config1_dir, img_original_path)

    config2_dir = output_base / "config2_orig_simp"
    if config2_dir.exists():
        visualize_config_results(config2_dir, img_original_path)

    print("\nvisualization generation complete")


if __name__ == "__main__":
    main()
