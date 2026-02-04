# Get Started - UOIFT+SICLE Pipeline

## 1. Prerequisites

### WSL2 (required on windows)

```bash
wsl --version
```

Both SICLE and UOIFT are C/C++ binaries that run inside WSL2.

### Conda Environments

Two environments are provided:

```bash
# main pipeline (sicle, uoift, stardist, filtering)
conda activate sicle_uoift

# visualization only
conda activate sicle-viz
```

Activate `sicle_uoift` before running any pipeline command.

### Python Dependencies

```
numpy, scikit-image, scipy, imageio, higra>=0.6
stardist>=0.8, tensorflow
cellpose>=3.0  (benchmark only)
```

## 2. Binary Setup

### SICLE binary (required)

SICLE is the core superpixel algorithm. Compile from source:

```bash
# inside wsl
cd /path/to/SICLE
make
# produces bin/RunSICLE
```

The wrapper searches these paths in order:
1. `<uoift_sicle>/SICLE/bin/RunSICLE`
2. `~/SICLE/bin/RunSICLE`
3. `/usr/local/bin/RunSICLE`

To use a custom location, set the `SICLE_BIN` env var:

```bash
# powershell
$env:SICLE_BIN = "C:\path\to\SICLE\bin\RunSICLE"

# bash / wsl
export SICLE_BIN=/path/to/SICLE/bin/RunSICLE
```

### UOIFT binary (optional, for saliency)

UOIFT generates boundary saliency maps (sec 3.2). Only needed if using Config1/Config2 with saliency. The no-saliency variant skips this entirely.

```bash
# inside wsl
cd references/unsupseg/unsupseg
make
# produces unsupseg_mergehistory
```

Default path: `<uoift_sicle>/references/unsupseg/unsupseg/unsupseg_mergehistory`

To use a custom location, set the `UOIFT_BIN` env var:

```bash
# powershell
$env:UOIFT_BIN = "C:\path\to\unsupseg_mergehistory"

# bash / wsl
export UOIFT_BIN=/path/to/unsupseg_mergehistory
```

## 3. Run the Pipeline

`run.py` is the unified entry point. Each pipeline step is a subcommand.

### Run everything (recommended)

```bash
conda activate sicle_uoift
export SICLE_BIN=/path/to/SICLE/bin/RunSICLE

# single image, no saliency (fastest)
python run.py all B3-T_1 --no-saliency

# single image, with uoift saliency
python run.py all B3-T_1

# all 10 images
python run.py all --all --no-saliency
```

Output: `pipeline_output/<image_name>/`

### Run individual steps

Each step caches its output as .npy. Re-running skips cached steps (use `--force` to override).

```bash
python run.py saliency  B3-T_1    # uoift saliency (sec 3.2)
python run.py sicle-a   B3-T_1    # path a coarse nf=3000 (sec 3.3)
python run.py filter-a  B3-T_1    # veta filter path a (sec 3.4)
python run.py sicle-b   B3-T_1    # path b multiscale nf=500 (sec 3.3)
python run.py filter-b  B3-T_1    # veta multiscale selection (sec 3.4)
python run.py pool      B3-T_1    # merge a + b candidates
python run.py stardist  B3-T_1    # stardist anchors (sec 3.5)
python run.py fuse      B3-T_1    # stardist-pool iou fusion (sec 3.5)
python run.py arbitrate B3-T_1    # final solidity filter (sec 3.5)
python run.py visualize B3-T_1    # overlay visualizations
```

Steps depend on previous outputs. If a dependency is missing, the error tells you which step to run first.

### Custom image path

```bash
python run.py all path/to/image.png --output results/
```

### Batch runners (legacy)

The old batch scripts in `batch/` still work for reproducing published results:

```bash
cd batch/
python run_no_saliency.py          # all 10 images, no saliency
python run_config1_config2.py      # all 10 images, config1 + config2
python generate_saliency_all.py    # generate uoift saliency maps
```

## 4. Generate Visualizations

```bash
cd visualization/
python generate_all.py
```

Files generated per image:
- `vis_01_input.png` - original image
- `vis_02_saliency.png` - uoift saliency
- `vis_03_path_a_filtered.png` - path a filtered
- `vis_04_path_b_veta.png` - path b after veta
- `vis_05_pool.png` - combined pool
- `vis_06_stardist.png` - stardist detection
- `vis_07_fused.png` - final fusion
- `vis_08_final.png` - final result
- `vis_09_overlay.png` - overlay on image
- `vis_comparison.png` - config1 vs config2

## 5. Run Tests

```bash
cd uoift_sicle/

# all 49 tests (imports + functional)
python tests/run_all.py

# or individually:
python tests/test_imports.py   # 18 import/signature tests
python tests/test_core.py      # 31 functional tests
```

Coverage: circular imports, AST import resolution, old name detection, function signatures, config validation, LAB filtering, shape filtering, candidate pool, stardist-pool fusion, instance recovery, solidity arbitration, multiscale selection, env var overrides, pgm round trip.

## 6. Cellpose Benchmark (reference)

```bash
cd benchmark/
python run_cellpose.py
python create_summary.py
```

Output: `benchmark_results/`, `unified_benchmark_summary.json`

## Key Parameters

All parameters live in `core/config.py`.

### SICLE (sec 3.3)

| Param | Path A (coarse) | Path B (multiscale) |
|-------|-----------------|---------------------|
| n0 | 52000 | 52000 |
| nf | 3000 | 500 |
| alpha | 0.9 | 0.85 |
| max_iters | 22 | 12 |
| scales | 1 | 6-11 |

### Veta Filtering (sec 3.4)

```python
FILTER_CFG = {
    "min_solidity": 0.80,
    "a_min": 15,
    "a_max": 40,
    "min_area": 100,
    "max_area": 10000,
    "l_max": 100,
    "b_max": 35,
}
```

### UOIFT (sec 3.2)

- alpha: -0.7 (favors dark-on-bright = nuclei boundaries)

### Arbitration (sec 3.5)

- IoU threshold: 0.2 (conflict detection)
- Pool overlap: 0.5 (deduplication)
- KDTree max distance: 100 (spatial indexing)
