# HIGRA+SICLE Pipeline (deprecated)

**Deprecated**: This pipeline is superseded by `uoift_sicle/`. Use `uoift_sicle/` with Config1 or no-saliency variant instead.

This was the original approach using HIGRA for saliency generation and image simplification, before UOIFT boundary saliency was available. The pipeline used a hardcoded LAB probability map `(100-L)*0.4 + a*0.6 - b*0.2`, morphological reconstruction, and HIGRA tree filtering.

## Why deprecated

1. Config1 (uoift_sicle) outperforms Config2 in precision despite fewer detections
2. No-saliency variant provides 2.4x speedup with -0.4% quality loss
3. UOIFT saliency (sec 3.2 of the paper) replaces the HIGRA-based saliency

## Structure

```
higra_sicle/
├── preseg_simplified.py   # img_simplified via higra + otsu + probmap
├── sicle_wrapper.py       # sicle wrapper (higra version)
├── path_fusion.py         # fusion path a + path b
├── cellpose_runner.py     # cellpose integration
├── __init__.py
├── GET_STARTED.md         # quick start (deprecated)
└── ARCHITECTURE.md        # technical architecture (deprecated)
```

## Workflow

```
phase 1: preprocessing
  [original image] -> preseg_simplified.py
    -> img_simplified.png, saliency.png, binary_mask.png

phase 2: sicle (via wsl)
  [SICLE binary] -> pathA_precise.pgm, pathB_coarse.pgm

phase 3: fusion
  [path_fusion.py] -> unified labels, roi_mask, markers

phase 4: instance segmentation
  [cellpose or stardist] -> final masks
```

## How it relates to uoift_sicle

Both pipelines share the same dual-path SICLE architecture. The difference is how saliency is computed:

| Component | higra_sicle (deprecated) | uoift_sicle (current) |
|-----------|--------------------------|----------------------|
| Saliency | HIGRA probmap + gradient | UOIFT boundary polarity (sec 3.2) |
| Simplification | HIGRA tree filtering | same (for Config2 only) |
| Path A | same SICLE params | same |
| Path B | same SICLE params | same |
| Post-filter | area + solidity + LAB | Veta filtering (sec 3.4) |
| Arbitration | IoU fusion | StarDist + IoU arbitration (sec 3.5) |

## Existing outputs

All 10 images have been processed:

```
higra_sicle/output/
├── B3-T_1/
├── B3-T_2/
├── B4-T_1/
├── B5-T_1/
├── B5-T_3/
├── C1-T_1/
├── C2-T_1/
├── C4-T_1/
├── C4-T_2/
└── C5-T_1/
```

## Benchmark

| Config | Nuclei | Notes |
|--------|--------|-------|
| Config1 (uoift_sicle, best) | 916 | both paths on original |
| Config2 (uses img_simplified) | 1004 | more detections, higher noise |
| No-saliency (uoift_sicle) | 818 | 2.4x faster, -0.4% quality |
