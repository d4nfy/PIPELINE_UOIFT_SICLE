# Technical Architecture: HIGRA-SICLE Pipeline (deprecated)

**Deprecated**: This documents the original HIGRA-based pipeline. The current pipeline is in `uoift_sicle/`. See `uoift_sicle/ARCHITECTURE_OPTIONS.md` for the active architecture.

## Overview

Cell segmentation pipeline for H&E images. Two parallel paths (sharp/simplified) feed SICLE with a common saliency, then fusion and validation.

```
RGB Image
    |
    +---> LAB Conversion
    |         |
    |         v
    |     Prob Map: (100-L)*0.4 + a*0.6 - b*0.2
    |         |
    |         v
    |     H-Dome Reconstruction
    |         |
    |         v
    |     HIGRA Filter (MaxTree + MinTree)
    |         |
    |         +---> Saliency Map (prob+grad, gamma=1.5)
    |         |
    |         +---> Binary Mask (Otsu, optional)
    |
    +---> HIGRA Simplification (Path B input)
    |
    +---> SICLE Path A (original image, sharp)
    |
    +---> SICLE Path B (simplified image, smooth)
    |
    +---> Shape Filter (area + solidity)
    |
    +---> IoU Fusion (selective union)
    |
    +---> Color Validation (LAB thresholds)
    |
    +---> Cellpose (instance segmentation)
    |
    v
    Output: masks + centroids
```

## Pre-SICLE (python)

| Step | Function | Description |
|------|----------|-------------|
| 1 | `get_lab_channels` | extract l, a, b channels |
| 2 | `create_prob_map` | color filter: (100-L)*w_l + a*w_a - b*w_b |
| 3 | `morph_reconstruction` | h-dome: flattens background, removes texture |
| 4 | `higra_filter` | max-tree + min-tree on prob map (area=500, height=8) |
| 5 | `create_saliency` | saliency = 0.7*filtered_prob + 0.3*morph_gradient, exponent 1.5 |
| 6 | `create_binary_mask` | closing + fill holes + opening (visual qa only) |
| 7 | `higra_simplify_image` | per-channel simplification for path b (area=1800, height=10) |

Saliency is computed from filtered prob map mixed with morphological gradient (disk=2), with fixed weights 0.7/0.3, exponent 1.5 and blur sigma=1.5.

Generated files:
- `img_simplified.png` - path b input
- `saliency.png` - objsm for both paths
- `binary_mask.png` - visual qa only

Script: `preseg_simplified.py`

## SICLE (external, c++)

| Param | Path A (sharp) | Path B (simplified) |
|-------|----------------|---------------------|
| alpha | 0.7 | 0.85 |
| n0 | 52000 | 40000 |
| nf | 3000 | 4000 |
| adhr | 16 | 8 |
| max_iters | 22 | 12 |

## Post-SICLE (python)

Filter order:
1. Shape filter (`filter_path_regions`): area + dynamic solidity on A and B individually
2. IoU fusion (`fuse_segmentations`): merges A and B, overlap > threshold keeps single region
3. LAB filter (`filter_lab`): strict color validation on fused regions
4. Cellpose: instance segmentation on validated ROIs

Shape filter keeps regions > min area, applies solidity 0.75 by default, relaxes to 0.50 for large regions (area > 2000) or dark/magenta regions (L < 80 and a > 15).

Final LAB classification:
- L < 105, a > 4, b < 25 = GOOD (dark, magenta, not yellow)
- Other = REJECT

Script: `path_fusion.py`

## Configuration (CFG)

```python
CFG = {
    "w_l": 0.4, "w_a": 0.6, "w_b": 0.2,
    "morph_h": 20,
    "filter_min_area": 500, "filter_min_height": 8,
    "simplify_min_area": 1800, "simplify_min_height": 10,
    "sal_sigma": 1.5,
    "iou_threshold": 0.5,
    "final_min_area": 80, "final_max_area": 9000,
    "final_min_solidity": 0.75,
    "final_l_max": 105, "final_a_min": 4, "final_b_max": 25,
}
```

Note: in the current `uoift_sicle/` pipeline, this config is preserved as `CFG_PREPROC` in `core/config.py` for backwards compatibility.

## I/O Interfaces

| Phase | Input | Output |
|-------|-------|--------|
| PRE (preseg_simplified.py) | image.png (RGB) | pre_seg/img_simplified.png, saliency.png, binary_mask.png |
| SICLE (wsl RunSICLE) | original, simplified, saliency | sicle_results/pathA_precise.pgm, pathB_coarse.pgm |
| POST (path_fusion.py) | pathA.pgm, pathB.pgm, LAB | for_cellpose/unified.pgm, roi_mask.png, markers.png |
| CELLS (cellpose_runner.py) | original, unified, roi_mask | cellpose_masks.png, centroids.csv |

## Validated design decisions

- Saliency from prob map + gradient: preserves nuances for small dark/magenta objects
- Aggressive path b simplification (area=1800): path b clearly distinct from path a
- Dynamic solidity in shape filter: relaxed for large or dark/magenta regions

## Rejected design decisions

- Vectorial RGB simplification: no significant improvement, added complexity

## References

- SICLE: Belem et al., "Efficient Multiscale Object-based Superpixel Framework", arXiv 2022
- DISF+TBM: Lopes et al., "Cell Classification based on Superpixel Segmentation and Transport-based Morphometry", IEEE 2021
- HIGRA: Lohou et al., "Higra: Hierarchical Graph Analysis", JOSS 2019
