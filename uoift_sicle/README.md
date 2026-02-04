# UOIFT+SICLE Pipeline

Main nuclei segmentation pipeline using UOIFT (boundary saliency) + SICLE (superpixels) + StarDist (anchor detection).

## Structure

```
uoift_sicle/
├── core/                        # shared modules
│   ├── __init__.py              # package init, public api
│   ├── config.py                # all parameters and image list
│   ├── pipeline_utils.py        # shared utilities (lab, paths, timer)
│   ├── sicle_wrapper.py         # sicle c binary wrapper via wsl2
│   ├── stardist_wrapper.py      # stardist 2d_versatile_he wrapper
│   ├── uoift_wrapper.py         # uoift wrapper via wsl2
│   ├── uoift_saliency_python.py # uoift pure python fallback
│   └── veta_filtering.py        # veta filtering, pool, arbitration
│
├── batch/                       # runners for all 10 images
│   ├── run_config1_config2.py   # config1 + config2 (with uoift)
│   ├── run_no_saliency.py       # no-saliency variant (2.4x faster)
│   ├── run_profile5_precise.py  # profile 5 (path a precise nf=1800)
│   ├── run_single_image.py      # single image via argparse
│   └── generate_saliency_all.py # uoift saliency for all images
│
├── visualization/               # figure generation
│   ├── generate_all.py          # wrapper for all visuals
│   ├── generate_batch_vis.py    # visuals for batch_results/
│   ├── generate_profile5_vis.py # visuals for profile 5
│   └── visualize_test.py        # test visualization
│
├── benchmark/                   # comparisons and reports
│   ├── config.py                # benchmark configuration
│   ├── run_cellpose.py          # cellpose benchmark
│   ├── collect_cellpose.py      # collect cellpose results
│   ├── create_summary.py        # unified_benchmark_summary.json
│   ├── generate_report.py       # full report generation
│   └── test_pipeline_B3T1.py    # full test on B3-T_1
│
├── tests/                       # verification suite (49 tests)
│   ├── test_imports.py          # 18 tests (imports, signatures, ast)
│   ├── test_core.py             # 31 tests (filtering, pool, fusion, io)
│   └── run_all.py               # unified runner
│
├── run.py                       # unified orchestrator (single entry point)
├── ARCHITECTURE_OPTIONS.md      # config1 vs config2 architecture
└── GET_STARTED.md               # quick start guide
```

## Quick Start

```bash
# run everything on one image (no-saliency, recommended)
python run.py all B3-T_1 --no-saliency

# run everything on all 10 images
python run.py all --all --no-saliency

# run individual steps
python run.py sicle-a B3-T_1
python run.py stardist B3-T_1
python run.py arbitrate B3-T_1

# see all steps
python run.py --help
```

## Generated Outputs

```
uoift_sicle/
├── batch_results/                 # config1/config2 (10 images)
├── batch_results_no_saliency/     # no-saliency variant (10 images)
├── batch_results_profile5_pathA/  # profile 5 (10 images)
├── output_complete_test/          # test B3-T_1
└── benchmark_results/             # cellpose + reports
```

## Key Results

| Config | Avg nuclei | Time/image | Notes |
|--------|-----------|------------|-------|
| **Config1 (best)** | **916** | **149s** | both paths on original |
| No-saliency | 818 | 93s | 2.4x faster, -0.4% quality |
| Cellpose (ref) | 691 | 600s | deep learning baseline |

## Paper to Code

| Paper section | Code module |
|---------------|-------------|
| sec 3.1 pipeline overview | `core/__init__.py` |
| sec 3.2 uoift saliency | `core/uoift_wrapper.py`, `core/uoift_saliency_python.py` |
| sec 3.3 dual-path sicle | `core/sicle_wrapper.py`, `core/config.py` |
| sec 3.4 veta filtering | `core/veta_filtering.py` |
| sec 3.5 stardist + arbitration | `core/stardist_wrapper.py`, `core/veta_filtering.py` |
| sec 4.1 dataset | `core/config.py` (ALL_IMAGES) |
