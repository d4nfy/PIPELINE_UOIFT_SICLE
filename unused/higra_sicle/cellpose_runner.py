"""run cellpose instance segmentation on sicle roi

optional comparison with standalone cellpose (no sicle preprocessing).
"""
import argparse
import os
import sys
from pathlib import Path

# avoid shadowing the installed `cellpose` package by this script name
if "" in sys.path:
    sys.path.remove("")
cwd = os.getcwd()
sys.path = [p for p in sys.path if os.path.abspath(p) != cwd]

import imageio.v2 as imageio
import numpy as np
import torch
from cellpose import models
from skimage.measure import label, regionprops

from pipeline_utils import find_image_path


def run_cellpose(
    dataset_dir: Path,
    model_type: str,
    flow_threshold: float,
    cellprob_threshold: float,
    use_gpu: bool,
    also_alone: bool,
) -> None:
    dataset_dir = dataset_dir.resolve()
    fusion_dir = dataset_dir / "for_cellpose"
    alone_dir = dataset_dir / "cellpose_alone"

    img_path = find_image_path(dataset_dir)
    roi_path = fusion_dir / "unified_for_cellpose.pgm"
    mask_path = fusion_dir / "roi_mask.png"

    if not img_path.exists():
        raise FileNotFoundError(f"missing original image in {dataset_dir}")
    if not roi_path.exists() or not mask_path.exists():
        raise FileNotFoundError("missing fusion outputs; run fusion.py first")

    img = imageio.imread(img_path).astype(np.uint8)
    roi_labels = imageio.imread(roi_path)
    roi_mask = imageio.imread(mask_path) > 0

    # Cellpose 4.x uses CellposeModel; model_type defaults to Cyto2 when None
    model = models.CellposeModel(gpu=use_gpu, model_type=model_type)

    def _run_one(input_img: np.ndarray, save_dir: Path, apply_roi: bool) -> None:
        masks_cp, _, diams = model.eval(
            input_img,
            diameter=None,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        if apply_roi:
            masks_cp[~roi_mask] = 0
        masks_final = label(masks_cp > 0)
        centroids = np.array([[p.centroid[1], p.centroid[0]] for p in regionprops(masks_final)])
        save_dir.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(save_dir / "cellpose_masks.png", masks_final.astype(np.uint16))
        np.savetxt(save_dir / "centroids.csv", centroids, delimiter=",", header="x,y", comments="")
        print(
            f"cellpose -> {save_dir.name}: raw={int(masks_cp.max())}, after_roi={int(masks_final.max())}, diam~{diams[0]:.1f}"
        )

    img_masked = img.copy()
    img_masked[~roi_mask] = [255, 255, 255]

    _run_one(img_masked, fusion_dir, apply_roi=True)

    if also_alone:
        _run_one(img, alone_dir, apply_roi=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cellpose with optional ROI mask")
    parser.add_argument("dataset_dir", type=Path, help="dataset folder under img_used")
    parser.add_argument("--model-type", default=None, help="Cellpose model type (None = default Cyto2)")
    parser.add_argument("--flow-threshold", type=float, default=0.4, help="flow threshold")
    parser.add_argument("--cellprob-threshold", type=float, default=0.0, help="cellprob threshold")
    parser.add_argument("--cpu", dest="use_gpu", action="store_false", help="force CPU")
    parser.add_argument("--also-alone", action="store_true", help="also run Cellpose on raw image (no ROI)")
    parser.set_defaults(use_gpu=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_gpu and not torch.cuda.is_available():
        print("cuda not available, falling back to cpu")
        args.use_gpu = False
    run_cellpose(
        dataset_dir=args.dataset_dir,
        model_type=args.model_type,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        use_gpu=args.use_gpu,
        also_alone=args.also_alone,
    )


if __name__ == "__main__":
    main()
