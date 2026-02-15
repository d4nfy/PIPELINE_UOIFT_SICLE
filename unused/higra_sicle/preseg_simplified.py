"""generate simplified image and saliency map for dual path sicle segmentation

corresponds to phase 1 preprocessing in doc-general.md architecture.

outputs:
- img_simplified.png: higra-simplified rgb for coarse path
- saliency.png: prob map + gradient for sicle objsm
- binary_mask.png: optional qa mask
- run_sicle.sh: wsl commands with default sicle parameters
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from pipeline_utils import (
    CFG,
    create_binary_mask,
    create_prob_map,
    create_saliency,
    ensure_dir,
    find_image_path,
    get_lab_channels,
    higra_filter,
    higra_simplify_image,
    morph_reconstruction,
    save_uint8,
    to_wsl_path,
)


SICLE_CMD_TEMPLATE = """#!/bin/bash
# path a: original image for precise segmentation
./bin/RunSICLE \
  --img "{img_sharp}" \
  --out "{out_a}" \
  --objsm "{saliency}" \
  --n0 52000 \
  --nf 3000 \
  --alpha 0.7 \
  --max-iters 22 \
  --conn-opt fmax \
  --crit-opt minsc \
  --sampl-opt grid \
  --irreg 0.12 \
  --adhr 16

# path b: simplified image for coarse segmentation
./bin/RunSICLE \
  --img "{img_simplified}" \
  --out "{out_b}" \
  --objsm "{saliency}" \
  --n0 40000 \
  --nf 4000 \
  --alpha 0.85 \
  --max-iters 12 \
  --conn-opt fmax \
  --crit-opt minsc \
  --sampl-opt grid \
  --irreg 0.18 \
  --adhr 8
"""


def run_pre_seg(dataset_dir: Path, image_path: Path | None, write_sicle_cmds: bool) -> None:
    dataset_dir = dataset_dir.resolve()
    pre_dir = ensure_dir(dataset_dir / "pre_seg")
    img_path = image_path.resolve() if image_path else find_image_path(dataset_dir)
    img = imageio.imread(img_path).astype(np.uint8)

    l_ch, a_ch, b_ch = get_lab_channels(img)
    prob = create_prob_map(l_ch, a_ch, b_ch, CFG)
    prob_rec = morph_reconstruction(prob, CFG)
    prob_filtered = higra_filter(prob_rec, CFG["filter_min_area"], CFG["filter_min_height"])
    binary_mask = create_binary_mask(prob_filtered)
    saliency = create_saliency(prob_filtered, CFG)
    img_simplified = higra_simplify_image(img, CFG["simplify_min_area"], CFG["simplify_min_height"])

    save_uint8(pre_dir / "img_simplified.png", img_simplified)
    save_uint8(pre_dir / "saliency.png", saliency)
    save_uint8(pre_dir / "binary_mask.png", (binary_mask * 255).astype(np.uint8))

    if write_sicle_cmds:
        sicle_dir = ensure_dir(dataset_dir / "sicle_results")
        cmd_text = SICLE_CMD_TEMPLATE.format(
            img_sharp=to_wsl_path(img_path),
            img_simplified=to_wsl_path(pre_dir / "img_simplified.png"),
            saliency=to_wsl_path(pre_dir / "saliency.png"),
            out_a=to_wsl_path(sicle_dir / "pathA_precise.pgm"),
            out_b=to_wsl_path(sicle_dir / "pathB_coarse.pgm"),
        )
        script_path = sicle_dir / "run_sicle.sh"
        script_path.write_text(cmd_text)
        print(f"wrote SICLE commands to {script_path}")

    print(f"saved pre-seg outputs to {pre_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-segmentation for HIGRA-SICLE pipeline")
    parser.add_argument("dataset_dir", type=Path, help="dataset folder under img_used")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="optional explicit path to the original image (auto-detected by find_image_path)",
    )
    parser.add_argument(
        "--write-sicle-cmds",
        action="store_true",
        help="write ready-to-run WSL commands into sicle_results/run_sicle.sh",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pre_seg(args.dataset_dir, args.image, args.write_sicle_cmds)


if __name__ == "__main__":
    main()
