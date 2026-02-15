# UOIFT+SICLE Pipeline Architecture

## Overview

The pipeline segments nuclei in H&E oral histology images using two complementary SICLE paths (A and B) combined with StarDist anchor detection for arbitration. Two strategies exist depending on saliency usage.

All parameters are centralized in `core/config.py`.

## Boundary Aware Strategy (best quality)

Both paths run on img_original with UOIFT boundary saliency.

```
img_original.png
    |
    +---> UOIFT saliency (alpha=-0.7, uoift_saliency_light.py)
    |         |
    |         v
    +---> PATH A (coarse, NO multiscale)
    |         SICLE: n0=52000, nf=3000, alpha=0.9
    |         Filter: solidity>=0.80, 100<=area<=10000, 15<=a<=40
    |         Output: ~200-400 filtered regions
    |
    +---> PATH B (multiscale, 6-11 scales)
              SICLE: n0=52000, nf=500, alpha=0.85
              Per-scale filter: solidity>=0.80, LAB filters
              Veta selection: OV>0.2, fitness=solidity
              Output: ~400-800 best nuclei

    v
CANDIDATE POOL (A + B)
    - Merge filtered A + Veta B
    - 50% overlap deduplication
    - Output: ~600-1200 regions

    v
STARDIST (2D_versatile_he)
    - prob_thresh=0.5, nms_thresh=0.4
    - Output: ~500-800 nuclei

    v
ARBITRATION
    - IoU > 0.2 = conflict
    - Conflict resolution: max(solidity) wins
    - Output: ~600-900 nuclei = FINAL
```

## Modular Dual-Path Strategy (faster)

Same pipeline but skips UOIFT saliency computation. 1.1x faster with minimal quality loss.

```bash
python run.py --method modular
```

## SICLE Parameters (core/config.py)

| Parameter | Path A | Path B |
|---|---|---|
| n0 | 52000 | 52000 |
| nf | 3000 | 500 |
| alpha | 0.9 | 0.85 |
| max_iters | 22 | 12 |
| irreg | 0.12 | 0.15 |
| adhr | 16 | 12 |
| multiscale | NO | YES (6-11 scales) |

## Veta Filter Parameters (FILTER_CFG)

```python
FILTER_CFG = {
    "min_area": 100,
    "max_area": 10000,
    "min_solidity": 0.80,
    "l_max": 100,
    "a_min": 15,
    "a_max": 40,
    "b_max": 35,
}
```

## Arbitration Thresholds

| Parameter | Value | Role |
|---|---|---|
| ION_THRESH | 0.2 | IoU conflict detection |
| POOL_OVERLAP_THRESH | 0.5 | pool deduplication |
| KDTREE_MAX_DIST | 100 | spatial indexing radius |

## Results (30 oral lesion histology images, 2048x1532 px)

| Method | Avg nuclei | Time/image | Notes |
|---|---|---|---|
| SAM-Cellpose (ref) | 453 | 832s | deep learning baseline |
| StarDist alone | 574 | 2.9s | fast convex detector |
| **Boundary aware** | **660** | **48.3s** | best quality |
| Modular dual-path | 653 | 44.6s | faster variant |

## UOIFT Saliency Variants

| Module | Speed | Usage |
|---|---|---|
| `uoift_saliency_light.py` | ~0.3s/image | recommended (vectorized) |
| `uoift_saliency_python.py` | ~30s/image | original |

## Visualization Layout

```
<output_dir>/<image_name>/
    report.json
    01_saliency.png           # raw UOIFT saliency (if boundary aware)
    vis_01_input.png
    vis_02_path_a_filtered.png
    vis_03_path_b_veta.png
    vis_04_pool.png
    vis_05_stardist.png
    vis_06_fused.png
    vis_07_final.png
    vis_08_overlay.png
    vis_09_fusion_decisions.png
    vis_comparison.png
```
