#!/usr/bin/env python3
"""pipeline test for B3-T_1

tests two configurations:
  config 1 (orig+orig): path a + path b on img_original
  config 2 (orig+simp): path a on original + path b on simplified

architecture:
  path a: coarse, no multiscale (n0=52000, nf=3000)
  path b: always multiscale (n0=52000, nf=500, 6-11 scales)
  pool: merge a+b with 50% overlap dedup
  stardist: convex nuclei detection
  arbitration: iou > 0.2 -> max solidity
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from skimage import io
import json
import time
from datetime import datetime
from dataclasses import dataclass

from core.stardist_wrapper import load_stardist_model, stardist_predict
from core.sicle_wrapper import run_sicle_path_a, run_sicle_path_b
from core.pipeline_utils import get_lab_channels, Timer
from core.veta_filtering import (
    filter_sicle_regions,
    veta_multiscale_selection,
    create_candidate_pool,
    fuse_stardist_pool,
)
from core.config import FILTER_CFG


# configuration

BASE_DIR = Path(__file__).parent.parent / "img_used" / "B3-T_1"
OUTPUT_BASE = Path(__file__).parent / "output_complete_test"

@dataclass
class PipelineConfig:
    """single pipeline run config"""
    name: str
    path_a_img: Path
    path_b_img: Path
    saliency_path: Path
    output_dir: Path

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


CONFIG_1 = PipelineConfig(
    name="config1_orig_orig",
    path_a_img=BASE_DIR / "pre_seg" / "img_original.png",
    path_b_img=BASE_DIR / "pre_seg" / "img_original.png",
    saliency_path=BASE_DIR / "path_B_original_uoift" / "alpha_n0_7" / "01_saliency_uoift.png",
    output_dir=OUTPUT_BASE / "config1_orig_orig"
)

CONFIG_2 = PipelineConfig(
    name="config2_orig_simp",
    path_a_img=BASE_DIR / "pre_seg" / "img_original.png",
    path_b_img=BASE_DIR / "pre_seg" / "img_simplified.png",
    saliency_path=BASE_DIR / "path_B_simplified_uoift" / "alpha_n0_7" / "01_saliency_uoift.png",
    output_dir=OUTPUT_BASE / "config2_orig_simp"
)


# utility functions

def save_mask(path: Path, mask: np.ndarray):
    """save mask as npy"""
    np.save(path, mask)


def load_or_compute(cache_path: Path, compute_fn, *args, **kwargs):
    """load from cache or compute"""
    if cache_path.exists():
        try:
            data = np.load(cache_path)
            print(f"    loaded from cache: {cache_path.name}")
            return data, 0.0
        except Exception as e:
            print(f"    cache failed ({e}), recomputing...")

    t0 = time.time()
    data = compute_fn(*args, **kwargs)
    elapsed = time.time() - t0
    np.save(cache_path, data)
    return data, elapsed


# pipeline

def run_complete_pipeline(config: PipelineConfig, stardist_model) -> dict:
    """run full pipeline for one config"""
    print(f"\npipeline: {config.name}")

    report = {
        'config': config.name,
        'start_time': datetime.now().isoformat(),
        'timings': {},
        'stats': {}
    }

    # step 1: load inputs
    print("\nloading inputs...")
    with Timer("Load") as t:
        img_a = io.imread(str(config.path_a_img))
        print(f"    path a image: {config.path_a_img.name} {img_a.shape}")

        img_b = io.imread(str(config.path_b_img))
        print(f"    path b image: {config.path_b_img.name} {img_b.shape}")

        saliency = io.imread(str(config.saliency_path))
        if saliency.ndim == 3:
            saliency = saliency[:, :, 0]
        print(f"    saliency: {saliency.shape}, range [{saliency.min()}, {saliency.max()}]")

        l_ch, a_ch, b_ch = get_lab_channels(img_a)

    report['timings']['load'] = t.elapsed
    report['stats']['img_shape'] = list(img_a.shape)

    # step 2: sicle path a
    print("\nsicle path a (coarse, no multiscale)...")
    path_a_cache = config.output_dir / "path_a_mask.npy"

    def compute_path_a():
        temp_img = config.output_dir / "temp_img_a.png"
        temp_sal = config.output_dir / "temp_saliency.png"
        io.imsave(str(temp_img), img_a, check_contrast=False)
        io.imsave(str(temp_sal), saliency.astype(np.uint8), check_contrast=False)

        result = run_sicle_path_a(
            img_path=str(temp_img),
            saliency_path=str(temp_sal),
            output_path=str(config.output_dir / "path_a.pgm"),
            multiscale=False
        )

        if isinstance(result, dict):
            return list(result.values())[0]
        return result

    path_a_mask, t_path_a = load_or_compute(path_a_cache, compute_path_a)
    n_path_a = len(np.unique(path_a_mask)) - 1
    report['timings']['sicle_path_a'] = t_path_a
    report['stats']['path_a_regions'] = int(n_path_a)
    print(f"    path a: {n_path_a} regions in {t_path_a:.2f}s")

    # step 2.5: filter path a
    print("\nfilter path a (solidity >=0.80, lab)...")
    path_a_filtered_cache = config.output_dir / "path_a_filtered.npy"

    def compute_path_a_filtered():
        filtered, n_kept, n_rejected = filter_sicle_regions(
            path_a_mask, l_ch, a_ch, b_ch, FILTER_CFG
        )
        print(f"    kept: {n_kept}, rejected: {n_rejected}")
        return filtered

    path_a_filtered, t_filter_a = load_or_compute(path_a_filtered_cache, compute_path_a_filtered)
    n_path_a_filtered = len(np.unique(path_a_filtered)) - 1
    report['timings']['filter_path_a'] = t_filter_a
    report['stats']['path_a_filtered'] = int(n_path_a_filtered)
    print(f"    path a filtered: {n_path_a_filtered} regions (from {n_path_a})")

    # step 3: sicle path b multiscale
    print("\nsicle path b (multiscale, 11 scales)...")
    path_b_cache = config.output_dir / "path_b_pooled.npy"
    path_b_scales_cache = config.output_dir / "path_b_scales.pkl"

    cache_valid = path_b_cache.exists() and path_b_scales_cache.exists()

    if cache_valid:
        try:
            path_b_mask = np.load(path_b_cache)
            print(f"    loaded from cache: {path_b_cache.name}")
            t_path_b = 0.0
        except Exception as e:
            print(f"    cache failed ({e}), recomputing...")
            cache_valid = False

    if not cache_valid:
        temp_img_b = config.output_dir / "temp_img_b.png"
        temp_sal = config.output_dir / "temp_saliency.png"
        io.imsave(str(temp_img_b), img_b, check_contrast=False)
        io.imsave(str(temp_sal), saliency.astype(np.uint8), check_contrast=False)

        t0 = time.time()
        result = run_sicle_path_b(
            img_path=str(temp_img_b),
            saliency_path=str(temp_sal),
            output_path=str(config.output_dir / "path_b.pgm"),
            multiscale=True
        )
        t_path_b = time.time() - t0

        if isinstance(result, dict):
            print(f"    generated {len(result)} scales")

            if len(result) == 0:
                raise RuntimeError(
                    "sicle path b multiscale returned empty dict.\n"
                    "check wsl and sicle binary."
                )

            for i, (scale_name, scale_mask) in enumerate(result.items(), 1):
                scale_path = config.output_dir / f"path_b_scale_{i:02d}.npy"
                np.save(scale_path, scale_mask)

            import pickle
            with open(path_b_scales_cache, 'wb') as f:
                pickle.dump(result, f)

            # veta multiscale selection
            print(f"    veta multiscale selection (ov>0.2, fitness=solidity)...")
            t_veta_start = time.time()

            path_b_mask = veta_multiscale_selection(
                scales_dict=result,
                l_ch=l_ch,
                a_ch=a_ch,
                b_ch=b_ch,
                filter_cfg=FILTER_CFG,
                min_solidity=0.80,
                ov_thresh=0.2
            )
            t_veta = time.time() - t_veta_start
            print(f"    veta selection: {t_veta:.2f}s")

            np.save(path_b_cache, path_b_mask)
        else:
            path_b_mask = result
            import pickle
            single_scale_dict = {"SICLE_01": result}
            with open(path_b_scales_cache, 'wb') as f:
                pickle.dump(single_scale_dict, f)
            np.save(path_b_cache, path_b_mask)

    n_path_b = len(np.unique(path_b_mask)) - 1
    report['timings']['sicle_path_b'] = t_path_b
    report['stats']['path_b_regions'] = int(n_path_b)
    print(f"    path b pooled: {n_path_b} regions in {t_path_b:.2f}s")

    # step 4: candidate pool
    print("\ncandidate pool (filtered a + veta b, 50% overlap dedup)...")
    pool_cache = config.output_dir / "pool.npy"

    def compute_pool():
        path_b_single_scale = {"veta_selected": path_b_mask}
        pool = create_candidate_pool(
            path_a_filtered.astype(np.int32),
            path_b_single_scale
        )
        return pool

    pool, t_pool = load_or_compute(pool_cache, compute_pool)
    n_pool = len(np.unique(pool)) - 1
    report['timings']['pool'] = t_pool
    report['stats']['pool_regions'] = int(n_pool)
    print(f"    pool: {n_pool} regions in {t_pool:.2f}s")

    n_from_a = len(np.unique(path_a_filtered)) - 1
    n_from_b_added = n_pool - n_from_a
    print(f"    pool breakdown: {n_from_a} from a, ~{n_from_b_added} from b")
    report['stats']['pool_from_a'] = int(n_from_a)
    report['stats']['pool_from_b'] = int(n_from_b_added)

    # step 5: stardist
    print("\nstardist detection...")
    stardist_cache = config.output_dir / "stardist.npy"

    def compute_stardist():
        mask = stardist_predict(img_a, model=stardist_model, prob_thresh=0.5, nms_thresh=0.4)
        return mask

    stardist_mask, t_stardist = load_or_compute(stardist_cache, compute_stardist)
    n_stardist = len(np.unique(stardist_mask)) - 1
    report['timings']['stardist'] = t_stardist
    report['stats']['stardist_nuclei'] = int(n_stardist)
    print(f"    stardist: {n_stardist} nuclei in {t_stardist:.2f}s")

    # step 6: arbitration
    print("\narbitration (iou > 0.2, max solidity)...")
    fused_cache = config.output_dir / "fused.npy"

    def compute_fused():
        fused = fuse_stardist_pool(
            stardist_mask.astype(np.int32),
            pool.astype(np.int32),
            iou_thresh=0.2
        )
        return fused

    fused, t_fused = load_or_compute(fused_cache, compute_fused)
    n_fused = len(np.unique(fused)) - 1
    report['timings']['fusion'] = t_fused
    report['stats']['fused_regions'] = int(n_fused)
    print(f"    fused: {n_fused} regions in {t_fused:.2f}s")

    # no post arbitration filter
    final = fused
    n_final = n_fused

    final_cache = config.output_dir / "final.npy"
    np.save(final_cache, final)

    report['stats']['final_nuclei'] = int(n_final)
    print(f"    result: {n_final} nuclei")

    total_time = sum(report['timings'].values())
    report['timings']['total'] = total_time
    report['end_time'] = datetime.now().isoformat()

    report_path = config.output_dir / "report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  report saved: {report_path}")

    print(f"\n  summary for {config.name}:")
    print(f"    total time: {total_time:.2f}s")
    print(f"    nuclei: {n_final}")

    return report


# comparative analysis

def generate_comparison_report(results: list):
    """compare config results"""
    print("\ncomparative analysis")

    if len(results) != 2:
        print(f"  warning: expected 2 results, got {len(results)}")
        return

    r1, r2 = results

    print(f"\n{'Metric':<30} {'Config 1':>15} {'Config 2':>15} {'Diff':>10}")

    metrics = [
        ('Path A regions', 'path_a_regions'),
        ('Path B regions', 'path_b_regions'),
        ('Pool regions', 'pool_regions'),
        ('StarDist nuclei', 'stardist_nuclei'),
        ('Fused regions', 'fused_regions'),
        ('Nuclei count', 'final_nuclei'),
    ]

    for label, key in metrics:
        v1 = r1['stats'].get(key, 0)
        v2 = r2['stats'].get(key, 0)
        diff = v2 - v1
        print(f"{label:<30} {v1:>15} {v2:>15} {diff:>+10}")

    print()

    timing_metrics = [
        ('SICLE Path A (s)', 'sicle_path_a'),
        ('SICLE Path B (s)', 'sicle_path_b'),
        ('Pool (s)', 'pool'),
        ('StarDist (s)', 'stardist'),
        ('Fusion (s)', 'fusion'),
        ('Arbitration (s)', 'arbitration'),
        ('TOTAL (s)', 'total'),
    ]

    for label, key in timing_metrics:
        v1 = r1['timings'].get(key, 0)
        v2 = r2['timings'].get(key, 0)
        diff = v2 - v1
        print(f"{label:<30} {v1:>15.2f} {v2:>15.2f} {diff:>+10.2f}")

    comparison = {
        'timestamp': datetime.now().isoformat(),
        'config1': r1,
        'config2': r2,
        'comparison': {
            'nuclei_diff': r2['stats']['final_nuclei'] - r1['stats']['final_nuclei'],
            'time_diff': r2['timings']['total'] - r1['timings']['total']
        }
    }

    comp_path = OUTPUT_BASE / "comparison_report.json"
    with open(comp_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  comparison saved: {comp_path}")


# main

def main():
    print("pipeline test B3-T_1")
    print(f"started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # preload stardist
    print("\nloading stardist model...")
    t0_model = time.time()
    stardist_model = load_stardist_model()
    t_model_load = time.time() - t0_model
    print(f"  model loaded in {t_model_load:.2f}s")

    results = []

    # config 1
    print("\nconfig 1: original + original")
    print("  path a: img_original, path b: img_original multiscale")
    results.append(run_complete_pipeline(CONFIG_1, stardist_model))

    # config 2
    print("\nconfig 2: original + simplified")
    print("  path a: img_original, path b: img_simplified multiscale")
    results.append(run_complete_pipeline(CONFIG_2, stardist_model))

    generate_comparison_report(results)

    print(f"\npipeline complete")
    print(f"ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"results: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
