# Get Started - HIGRA+SICLE Pipeline (deprecated)

**Deprecated**: Use `uoift_sicle/` instead. This pipeline was the original approach before UOIFT saliency was available. It used HIGRA for saliency generation and image simplification via a hardcoded LAB probability map.

## Structure

```
higra_sicle/
├── preseg_simplified.py   # img_simplified via higra + otsu + probmap
├── sicle_wrapper.py       # sicle wrapper (higra version)
├── path_fusion.py         # fusion path a + path b
├── cellpose_runner.py     # cellpose integration
└── __init__.py
```

## What HIGRA does here

HIGRA generates `img_simplified.png` via:
1. LAB probability map: `(100-L)*0.4 + a*0.6 - b*0.2`
2. Otsu thresholding on the probability map
3. Morphological filtering via HIGRA component tree (area=1800, height=10)

```bash
python preseg_simplified.py --image B3-T_1
```

Output: `img_used/<IMAGE>/pre_seg/img_simplified.png`

## How it integrates with uoift_sicle

Both Config1 and Config2 in uoift_sicle use UOIFT saliency. The only difference is the Path B input:

| Config | Path A input | Path B input | Saliency |
|--------|--------------|--------------|----------|
| Config1 | img_original | img_original | UOIFT |
| Config2 | img_original | img_simplified (from HIGRA) | UOIFT |

The simplified image smooths textures, making SICLE superpixels more uniform. However, it also merges some nuclei that were distinguishable in the original.

## Benchmark results

| Config | Nuclei | Notes |
|--------|--------|-------|
| Config1 (best) | 916 | both paths on original |
| Config2 | 1004 | more detections, higher noise |
| No-saliency | 818 | skip uoift, 2.4x faster |

Config2 detects more nuclei (+88) but includes more false positives. The simplified image creates merged regions that cannot be properly sub-segmented without ground truth.

## Why deprecated

1. Config1 outperforms Config2 in precision despite fewer detections
2. No-saliency variant provides 2.4x speedup with -0.4% quality loss
3. UOIFT is the bottleneck, not the HIGRA simplification (133s cost for 0.4% gain)

**Recommendation**: Use `uoift_sicle/` with Config1 or no-saliency.

## Existing outputs

All 10 images have been processed and results are in `higra_sicle/output/`.
