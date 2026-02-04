# Nuclei Segmentation Pipeline

Hybrid superpixel-based nuclei segmentation for H&E oral histology images.
Combines UOIFT saliency, dual-path SICLE superpixels, Veta-inspired filtering, and StarDist anchor detection.

## Structure

```
final/
├── uoift_sicle/          # main pipeline
│   ├── core/             # shared modules (config, wrappers, filtering)
│   ├── batch/            # runners for 10 images
│   ├── visualization/    # figure generation
│   ├── benchmark/        # cellpose comparison
│   └── tests/            # import verification
│
├── higra_sicle/          # deprecated (higra saliency variant)
│
├── paper/                # latex article
│   └── overleaf_paper.tex
│
└── outputs/              # pre-computed results
```

## Quick start

```bash
# recommended: no-saliency variant (2.4x faster, -0.4% quality)
cd uoift_sicle/batch/
python run_no_saliency.py

# full pipeline with uoift saliency
cd uoift_sicle/batch/
python run_config1_config2.py

# single image with argparse
python run_single_image.py path/to/image.png --output logs/

# generate all visualizations
cd uoift_sicle/visualization/
python generate_all.py

# run import verification
cd uoift_sicle/
python tests/test_imports.py
```

## Paper to code mapping

| Paper section | Topic | Code module |
|---------------|-------|-------------|
| sec 3.1 | pipeline overview | `core/__init__.py` |
| sec 3.2 | uoift saliency (eq. 1, alpha=-0.7) | `core/uoift_wrapper.py`, `core/uoift_saliency_python.py` |
| sec 3.3 | dual-path sicle (path a coarse, path b multiscale) | `core/sicle_wrapper.py`, `core/config.py` |
| sec 3.4 | veta filtering (solidity, lab, area, kdtree) | `core/veta_filtering.py` |
| sec 3.5 | stardist anchor + iou arbitration | `core/stardist_wrapper.py`, `core/veta_filtering.py` |
| sec 4.1 | dataset (10 oral histology images, 1920x1200, 40x) | `core/config.py` (ALL_IMAGES) |

## Results summary

| Configuration | Avg nuclei | Time/image | Notes |
|---------------|-----------|------------|-------|
| Cellpose (ref) | 691 | 600s | deep learning baseline |
| StarDist alone | 638 | 3.3s | fast convex detector |
| SICLE alone | 1799 | 58s | over-segmentation |
| Config1 (best) | 916 | 149s | both paths on original |
| No-saliency | 818 | 93s | skip uoift, 2.4x faster |

## Dependencies

```
numpy, scikit-image, scipy, imageio, higra>=0.6
stardist>=0.8, tensorflow
cellpose>=3.0  (benchmark only)
```

WSL2 required for SICLE and UOIFT C binaries.
