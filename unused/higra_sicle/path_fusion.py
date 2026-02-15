"""fuse dual sicle paths and prepare superpixels for instance segmentation

corresponds to phase 3 fusion in doc-general.md architecture.

merges precise (path a) and coarse (path b) sicle outputs using iou matching,
filters by lab color space, generates roi mask and centroid markers.

outputs ready for cellpose, stardist, unet, plantseg or other instance segmentation tools.
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

from pipeline_utils import (
    CFG,
    ensure_dir,
    extract_markers,
    filter_lab,
    filter_path_regions,
    find_image_path,
    fuse_segmentations,
    get_lab_channels,
    label,
    save_uint8,
    save_uint16,
    summarize_mask,
)


def create_fusion_visualization(
    img: np.ndarray,
    labels_a_clean: np.ndarray,
    labels_b_clean: np.ndarray,
    unified_lab: np.ndarray,
    kept_a: int,
    kept_b: int,
    kept_lab: int,
    out_path: Path,
) -> None:
    """generate 4-panel comparison: original, path a, path b, final fusion"""
    img_bw = (rgb2gray(img) * 255).astype(np.uint8)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    ax[0].imshow(img_bw, cmap='gray')
    ax[0].set_title('original')
    ax[0].axis('off')

    overlay_a = np.stack([img_bw, img_bw, img_bw], axis=-1).copy()
    overlay_a[labels_a_clean > 0] = [180, 0, 255]
    ax[1].imshow(overlay_a)
    ax[1].set_title(f'path A ({kept_a})')
    ax[1].axis('off')

    overlay_b = np.stack([img_bw, img_bw, img_bw], axis=-1).copy()
    overlay_b[labels_b_clean > 0] = [180, 0, 255]
    ax[2].imshow(overlay_b)
    ax[2].set_title(f'path B ({kept_b})')
    ax[2].axis('off')

    overlay_final = np.stack([img_bw, img_bw, img_bw], axis=-1).copy()
    overlay_final[unified_lab > 0] = [180, 0, 255]
    ax[3].imshow(overlay_final)
    ax[3].set_title(f'final ({kept_lab})')
    ax[3].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved fusion visualization to {out_path}")


def run_fusion(dataset_dir: Path, path_a: Path | None, path_b: Path | None) -> None:
    dataset_dir = dataset_dir.resolve()
    pre_dir = dataset_dir / "pre_seg"
    sicle_dir = dataset_dir / "sicle_results"
    out_dir = ensure_dir(dataset_dir / "for_cellpose")

    img_path = find_image_path(dataset_dir)
    if not img_path.exists():
        raise FileNotFoundError(f"missing original image at {dataset_dir}")
    img = imageio.imread(img_path).astype(np.uint8)
    l_ch, a_ch, b_ch = get_lab_channels(img)

    labels_a_path = path_a if path_a is not None else sicle_dir / "pathA_precise.pgm"
    labels_b_path = path_b if path_b is not None else sicle_dir / "pathB_coarse.pgm"
    if not labels_a_path.exists() or not labels_b_path.exists():
        raise FileNotFoundError("SICLE outputs not found; run SICLE first")

    labels_a = imageio.imread(labels_a_path)
    labels_b = imageio.imread(labels_b_path)

    # shape filtering without lab
    labels_a_clean, kept_a, rej_a = filter_path_regions(labels_a, CFG)
    labels_b_clean, kept_b, rej_b = filter_path_regions(labels_b, CFG)

    unified = fuse_segmentations(labels_a_clean, labels_b_clean, CFG["iou_threshold"])
    
    # lab filtering after fusion
    mask_lab, kept_lab, rej_lab = filter_lab(unified, l_ch, a_ch, b_ch, CFG)
    unified_lab = unified.copy()
    unified_lab[~mask_lab] = 0
    
    roi_mask = (unified_lab > 0)
    markers = extract_markers(roi_mask)

    save_uint16(out_dir / "unified.pgm", unified)
    save_uint16(out_dir / "unified_for_cellpose.pgm", unified_lab)
    save_uint8(out_dir / "roi_mask.png", (roi_mask.astype(np.uint8) * 255))
    save_uint8(out_dir / "markers.png", (markers * 255).astype(np.uint8))

    # Visualization
    create_fusion_visualization(
        img, labels_a_clean, labels_b_clean, unified_lab,
        kept_a, kept_b, kept_lab,
        out_dir / "fusion_comparison.png",
    )

    print(f"path A kept={kept_a}, rejected={rej_a}")
    print(f"path B kept={kept_b}, rejected={rej_b}")
    print(f"fusion regions={int(unified_lab.max())}, lab kept={kept_lab}, lab rejected={rej_lab}")
    print(f"roi: {summarize_mask(roi_mask)}")
    print(f"saved fusion outputs to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse SICLE outputs and filter by LAB")
    parser.add_argument("dataset_dir", type=Path, help="dataset folder under img_used")
    parser.add_argument("--path-a", type=Path, default=None, help="optional path to pathA PGM")
    parser.add_argument("--path-b", type=Path, default=None, help="optional path to pathB PGM")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_fusion(args.dataset_dir, args.path_a, args.path_b)


if __name__ == "__main__":
    main()