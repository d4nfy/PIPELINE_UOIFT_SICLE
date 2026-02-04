# UOIFT+SICLE Pipeline Architecture

## Overview

The pipeline segments nuclei in H&E oral histology images using two complementary SICLE paths (A and B) combined with StarDist anchor detection for arbitration. Two configurations exist depending on the Path B input image.

All parameters are centralized in `core/config.py`.

## Config 1: Original-Original (best quality)

Both paths run on img_original. Config1 produces the best precision.

```
img_original.png
    |
    +---> UOIFT saliency (alpha=-0.7, sec 3.2)
    |         |
    |         v
    +---> PATH A (coarse, NO multiscale, sec 3.3)
    |         SICLE: n0=52000, nf=3000, alpha=0.9
    |         Filter: solidity>=0.80, 100<=area<=10000, 15<=a<=40
    |         Output: ~200-400 filtered regions
    |
    +---> PATH B (multiscale, 6-11 scales, sec 3.3)
              Input: img_original (same as Path A)
              SICLE: n0=52000, nf=500, alpha=0.85
              Per-scale filter: solidity>=0.80, LAB filters
              Veta selection: OV>0.2, fitness=solidity (sec 3.4)
              Output: ~400-800 best nuclei

    v
CANDIDATE POOL (A + B)
    - Merge filtered A + Veta B
    - 50% overlap deduplication
    - Output: ~600-1200 regions

    v
STARDIST (2D_versatile_he, sec 3.5)
    - prob_thresh=0.5, nms_thresh=0.4
    - Output: ~800-1000 nuclei

    v
ARBITRATION (sec 3.5)
    - IoU > 0.2 = conflict
    - Conflict resolution: max(solidity) wins
    - Output: ~900-1000 nuclei = FINAL
```

## Config 2: Original-Simplified

Path A on img_original, Path B on img_simplified (HIGRA pre-processed). Config2 detects more nuclei (+88) but includes more false positives.

```
img_original.png                    img_simplified.png (HIGRA)
    |                                       |
    +---> UOIFT saliency (alpha=-0.7)      |
    |         |                             |
    |         v                             |
    +---> PATH A (coarse, NO multiscale)    |
    |         SICLE: n0=52000, nf=3000      |
    |         Output: ~200-400 regions      |
    |                                       |
    +---------------------------------------+
                        |
                        v
                  PATH B (multiscale, 6-11 scales)
                      Input: img_simplified
                      SICLE: n0=52000, nf=500, alpha=0.85
                      Per-scale filter: LAB from img_original
                      Veta selection: OV>0.2, fitness=solidity
                      Output: ~600-1000 best nuclei

    v
CANDIDATE POOL -> STARDIST -> ARBITRATION (same as Config 1)
```

Path B LAB filtering uses channels from img_original, NOT img_simplified, for consistent cytoplasm rejection.

## No-Saliency Variant (recommended)

Same as Config1 but skips UOIFT saliency computation. 2.4x faster with only -0.4% quality loss.

```bash
cd batch/
python run_no_saliency.py
```

## SICLE Parameters (core/config.py)

| Parameter | Path A (SICLE_PATH_A) | Path B (SICLE_PATH_B) |
|-----------|----------------------|----------------------|
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
|-----------|-------|------|
| ION_THRESH | 0.2 | IoU conflict detection |
| POOL_OVERLAP_THRESH | 0.5 | pool deduplication |
| KDTREE_MAX_DIST | 100 | spatial indexing radius |

## Results Comparison

| Config | Avg nuclei | Time/image | Notes |
|--------|-----------|------------|-------|
| Config1 (best) | 916 | 149s | both paths on original |
| Config2 | 1004 | 197s | more detections, higher noise |
| No-saliency | 818 | 93s | skip uoift, 2.4x faster |
| Cellpose (ref) | 691 | 600s | deep learning baseline |
| StarDist alone | 638 | 3.3s | fast convex detector |

## Image Data Layout

```
img_used/<IMAGE_NAME>/
    pre_seg/
        img_original.png          # required
        img_simplified.png        # required for config 2 (from higra)
    path_B_original_uoift/
        alpha_n0_7/
            01_saliency_uoift.png # uoift saliency for config 1
    path_B_simplified_uoift/
        alpha_n0_7/
            01_saliency_uoift.png # uoift saliency for config 2
```

## Known Limitations

1. **Config 2 LAB filtering**: When using img_simplified for Path B, region boundaries differ from img_original. Mean LAB per region may differ because different pixels are included.

2. **SICLE WSL dependency**: SICLE binary runs via WSL2, adding ~1-2s overhead per call.

3. **StarDist model loading**: First inference takes ~60-90s. Subsequent runs use cached model (~3-5s).
