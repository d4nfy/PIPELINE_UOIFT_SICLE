# Nuclei Segmentation Pipeline

Hybrid superpixel-based nuclei segmentation for H&E oral histology images.
Combines UOIFT boundary saliency, dual-path SICLE superpixels, Veta-inspired filtering, and StarDist anchor-based arbitration.

Dataset: 30 oral lesion histology images (2048x1532 px, 6 tissue categories), PUC Minas Odontology Lab, Belo Horizonte.

## Structure

```
PIPELINE_UOIFT_SICLE/
├── pipeline/                        # 30-image benchmark
│   ├── run.py                       # main entry (4 methods, --method flag)
│   ├── prepare_new_images.py        # dataset preparation (tif -> png + metadata)
│   ├── requirements.txt
│   ├── setup_env.sh
│   ├── ARCHITECTURE_OPTIONS.md      # pipeline flow diagrams and parameters
│   └── core/                        # shared pipeline modules
│       ├── uoift_saliency_light.py  # vectorized UOIFT saliency (sec 2.1.1)
│       ├── uoift_saliency_python.py # original UOIFT saliency (sec 2.1.1)
│       ├── sicle_wrapper.py         # SICLE C binary wrapper (sec 2.1.2)
│       ├── veta_filtering.py        # chromatic + morphological filtering (sec 2.2)
│       ├── stardist_wrapper.py      # StarDist anchor detection (sec 2.3)
│       └── config.py                # all pipeline parameters
│
└── unused/                          # legacy code
    ├── higra_sicle/                 # deprecated (higra saliency variant)
    └── uoift_sicle_virginia/        # Virginia dataset (10 images)
```

## Quick start

```bash
cd pipeline
./setup_env.sh
source venv/bin/activate

# 1. prepare dataset (converts tif images to png, creates metadata.json)
python prepare_new_images.py

# 2. run benchmark
python run.py                        # all 4 methods
python run.py --method stardist      # stardist only
python run.py --method cellpose      # sam-cellpose only
python run.py --method boundary_aware  # with uoift saliency (best quality)
python run.py --method modular       # without saliency (faster)
```

Input images go in `../new_images/<category>/` (tif format, organized by tissue category).
Results are written to `../new_images_results/`.

## Paper to code mapping

| Paper section | Topic | Code module |
|---|---|---|
| Sec 2.1.1 | Boundary-Aware Saliency (UOIFT, alpha=-0.7) | `pipeline/core/uoift_saliency_light.py` |
| Sec 2.1.2 | Modular Dual-Path (SICLE coarse + multiscale) | `pipeline/core/sicle_wrapper.py`, `pipeline/core/config.py` |
| Sec 2.2 | Chromatic & Morphological Filtering (Veta) | `pipeline/core/veta_filtering.py` |
| Sec 2.3 | Anchor-Based Arbitration (StarDist + IoU) | `pipeline/core/stardist_wrapper.py` |

## Results (30 oral lesion histology images)

6 tissue categories of increasing difficulty:
normal mucosa, mild dysplasia, moderate dysplasia, severe dysplasia, carcinoma in situ, CCE (OSCC).

### Overall

| Method | Avg nuclei | Time/image | Explainable |
|---|---|---|---|
| SAM-Cellpose (ref) | 453 | 832s | No (DL) |
| StarDist (baseline) | 574 | 2.9s | Partial |
| **Boundary aware** | **660** | **48.3s** | **Yes** |
| Modular dual-path | 653 | 44.6s | Yes |

### Per-category (5 images each, boundary aware)

| Category | SAM-Cellpose | StarDist | Boundary aware | Difficulty |
|---|---|---|---|---|
| Normal mucosa | 400 | 500 | 529 | Easiest |
| Mild dysplasia | 523 | 612 | 737 | Easy |
| Moderate dysplasia | 548 | 622 | 734 | Medium |
| Severe dysplasia | 470 | 657 | 701 | Hard |
| Carcinoma in situ | 600 | 787 | 880 | Hard |
| CCE (OSCC) | 177 | 267 | 380 | Hardest |

## Dependencies

```
numpy, scikit-image, scipy, Pillow, imageio, opencv-python
stardist, tensorflow, csbdeep
cellpose>=4.0 (benchmark only)
```

### SICLE C binary

Compile with `make` in the SICLE/ source directory.

- **macOS/Linux**: runs natively, images auto-converted to PPM/PGM (compiled without libpng)
- **Windows**: runs via WSL2, images passed as PNG with WSL path conversion
