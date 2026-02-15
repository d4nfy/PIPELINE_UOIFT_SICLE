#!/usr/bin/env python3
"""batch run: config 1 (orig+orig) and config 2 (orig+simp)

runs dual path sicle pipeline on all 10 images (sec 4.1).
config 1: original image for both paths.
config 2: original for path a, simplified for path b.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from skimage import io
import json
import time
from datetime import datetime
from dataclasses import dataclass
import traceback

from core.config import ALL_IMAGES, FILTER_CFG
from core.stardist_wrapper import load_stardist_model, stardist_predict
from core.sicle_wrapper import run_sicle_path_a, run_sicle_path_b
from core.pipeline_utils import get_lab_channels
from core.veta_filtering import (
    filter_sicle_regions,
    veta_multiscale_selection,
    create_candidate_pool,
    fuse_stardist_pool,
)

# directories
BASE_IMG_DIR = Path(__file__).parent.parent / "img_used"
OUTPUT_BASE = Path(__file__).parent / "batch_results"


@dataclass
class ImageConfig:
    """config for one image run"""
    name: str
    img_original: Path
    img_simplified: Path
    saliency_orig: Path
    saliency_simp: Path
    output_dir: Path

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


def run_pipeline_config(img_config, config_name, img_path, saliency_path,
                        stardist_model, l_ch, a_ch, b_ch, img_a):
    """run pipeline for one config (1 or 2)"""
    output_dir = img_config.output_dir / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'image': img_config.name,
        'config': config_name,
        'start_time': datetime.now().isoformat(),
        'timings': {},
        'stats': {}
    }

    img_b = io.imread(str(img_path))

    if not saliency_path.exists():
        print(f"      saliency not found: {saliency_path}")
        report['error'] = f"saliency not found: {saliency_path}"
        return report

    saliency = io.imread(str(saliency_path))
    if saliency.ndim == 3:
        saliency = saliency[:, :, 0]

    # sicle path a (sec 3.3)
    print(f"      sicle path a...")
    temp_img = output_dir / "temp_img_a.png"
    temp_sal = output_dir / "temp_saliency.png"
    io.imsave(str(temp_img), img_a, check_contrast=False)
    io.imsave(str(temp_sal), saliency.astype(np.uint8), check_contrast=False)

    t0 = time.time()
    try:
        result_a = run_sicle_path_a(
            img_path=str(temp_img),
            saliency_path=str(temp_sal),
            output_path=str(output_dir / "path_a.pgm"),
            multiscale=False
        )
        if isinstance(result_a, dict):
            path_a_mask = list(result_a.values())[0]
        else:
            path_a_mask = result_a
        t_path_a = time.time() - t0
    except Exception as e:
        print(f"      sicle path a failed: {e}")
        report['error'] = f"sicle path a failed: {e}"
        return report

    n_path_a = len(np.unique(path_a_mask)) - 1
    report['timings']['sicle_path_a'] = t_path_a
    report['stats']['path_a_regions'] = int(n_path_a)
    print(f"      path a: {n_path_a} regions ({t_path_a:.1f}s)")

    # filter path a (sec 3.4)
    print(f"      filtering path a...")
    t0 = time.time()
    path_a_filtered, n_kept_a, n_rej_a = filter_sicle_regions(
        path_a_mask, l_ch, a_ch, b_ch, FILTER_CFG
    )
    t_filter_a = time.time() - t0
    n_path_a_filtered = len(np.unique(path_a_filtered)) - 1
    report['timings']['filter_path_a'] = t_filter_a
    report['stats']['path_a_filtered'] = int(n_path_a_filtered)
    print(f"      path a filtered: {n_path_a_filtered} ({t_filter_a:.1f}s)")

    # sicle path b multiscale (sec 3.3)
    print(f"      sicle path b (multiscale)...")
    temp_img_b = output_dir / "temp_img_b.png"
    io.imsave(str(temp_img_b), img_b, check_contrast=False)

    t0 = time.time()
    try:
        result_b = run_sicle_path_b(
            img_path=str(temp_img_b),
            saliency_path=str(temp_sal),
            output_path=str(output_dir / "path_b.pgm"),
            multiscale=True
        )
        t_path_b = time.time() - t0
    except Exception as e:
        print(f"      sicle path b failed: {e}")
        report['error'] = f"sicle path b failed: {e}"
        return report

    if isinstance(result_b, dict) and len(result_b) > 0:
        n_scales = len(result_b)
        # veta selection (sec 3.4)
        print(f"      veta selection ({n_scales} scales)...")
        t0_veta = time.time()
        path_b_mask = veta_multiscale_selection(
            result_b, l_ch, a_ch, b_ch, FILTER_CFG
        )
        t_veta = time.time() - t0_veta
        report['timings']['veta_selection'] = t_veta
    else:
        path_b_mask = result_b if not isinstance(result_b, dict) else np.zeros_like(path_a_mask)
        n_scales = 1
        t_veta = 0

    n_path_b = len(np.unique(path_b_mask)) - 1
    report['timings']['sicle_path_b'] = t_path_b
    report['stats']['path_b_regions'] = int(n_path_b)
    report['stats']['path_b_scales'] = int(n_scales)
    print(f"      path b: {n_path_b} regions, {n_scales} scales ({t_path_b:.1f}s + {t_veta:.1f}s veta)")

    # pool (sec 3.3)
    print(f"      creating pool...")
    t0 = time.time()
    path_b_single_scale = {"veta_selected": path_b_mask}
    pool = create_candidate_pool(
        path_a_filtered.astype(np.int32),
        path_b_single_scale
    )
    t_pool = time.time() - t0
    n_pool = len(np.unique(pool)) - 1
    report['timings']['pool'] = t_pool
    report['stats']['pool_regions'] = int(n_pool)
    print(f"      pool: {n_pool} regions ({t_pool:.1f}s)")

    # stardist (sec 3.5)
    print(f"      stardist...")
    t0 = time.time()
    stardist_mask = stardist_predict(img_a, model=stardist_model, prob_thresh=0.5, nms_thresh=0.4)
    t_stardist = time.time() - t0
    n_stardist = len(np.unique(stardist_mask)) - 1
    report['timings']['stardist'] = t_stardist
    report['stats']['stardist_nuclei'] = int(n_stardist)
    print(f"      stardist: {n_stardist} nuclei ({t_stardist:.1f}s)")

    # arbitration (sec 3.5)
    print(f"      arbitration...")
    t0 = time.time()
    fused = fuse_stardist_pool(
        stardist_mask.astype(np.int32),
        pool.astype(np.int32),
        iou_thresh=0.2
    )
    t_fused = time.time() - t0
    n_fused = len(np.unique(fused)) - 1
    report['timings']['arbitration'] = t_fused
    report['stats']['fused_regions'] = int(n_fused)

    final = fused
    n_final = n_fused
    report['stats']['final_nuclei'] = int(n_final)
    print(f"      final: {n_final} nuclei ({t_fused:.1f}s)")

    # save masks
    np.save(output_dir / "path_a_filtered.npy", path_a_filtered)
    np.save(output_dir / "path_b_pooled.npy", path_b_mask)
    np.save(output_dir / "pool.npy", pool)
    np.save(output_dir / "stardist.npy", stardist_mask)
    np.save(output_dir / "fused.npy", fused)
    np.save(output_dir / "final.npy", final)

    total_time = sum(report['timings'].values())
    report['timings']['total'] = total_time
    report['end_time'] = datetime.now().isoformat()

    with open(output_dir / "report.json", 'w') as f:
        json.dump(report, f, indent=2)

    return report


def run_image(img_name, stardist_model):
    """run both configs on one image"""
    print(f"\n  processing: {img_name}")

    img_dir = BASE_IMG_DIR / img_name
    img_original = img_dir / "pre_seg" / "img_original.png"
    img_simplified = img_dir / "pre_seg" / "img_simplified.png"

    if not img_original.exists():
        print(f"    img_original not found, skipping")
        return None

    # saliency paths
    saliency_uoift = img_dir / "pre_seg" / "UOIFT" / "saliency_uoift.png"
    saliency_higra = img_dir / "pre_seg" / "HIGRA" / "saliency.png"

    if saliency_uoift.exists():
        saliency_orig = saliency_uoift
        saliency_simp = saliency_uoift
        print(f"    using uoift saliency")
    elif saliency_higra.exists():
        saliency_orig = saliency_higra
        saliency_simp = saliency_higra
        print(f"    fallback to higra saliency")
    else:
        print(f"    no saliency found")
        saliency_orig = None
        saliency_simp = None

    output_dir = OUTPUT_BASE / img_name

    img_config = ImageConfig(
        name=img_name,
        img_original=img_original,
        img_simplified=img_simplified,
        saliency_orig=saliency_orig,
        saliency_simp=saliency_simp,
        output_dir=output_dir
    )

    img_a = io.imread(str(img_original))
    l_ch, a_ch, b_ch = get_lab_channels(img_a)

    results = {}

    # config 1: orig+orig
    print(f"    config 1 (orig+orig)...")
    if saliency_orig is not None and saliency_orig.exists():
        results['config1'] = run_pipeline_config(
            img_config, "config1_orig_orig",
            img_original, saliency_orig,
            stardist_model, l_ch, a_ch, b_ch, img_a
        )
    else:
        print(f"      saliency not found for config1")
        results['config1'] = {'error': 'saliency not found'}

    # config 2: orig+simp
    print(f"    config 2 (orig+simp)...")
    if saliency_simp is not None and saliency_simp.exists() and img_simplified.exists():
        results['config2'] = run_pipeline_config(
            img_config, "config2_orig_simp",
            img_simplified, saliency_simp,
            stardist_model, l_ch, a_ch, b_ch, img_a
        )
    else:
        print(f"      files not found for config2")
        results['config2'] = {'error': 'files not found'}

    return results


def generate_summary_report(all_results):
    """generate summary report for all images"""
    print("\nsummary report")

    print(f"\n{'Image':<12} {'Config':<15} {'PathA':>8} {'PathB':>8} {'Pool':>8} {'SD':>8} {'Final':>8} {'Time':>8}")

    summary_data = []

    for img_name, results in all_results.items():
        for cfg_name, report in results.items():
            if 'error' in report:
                print(f"{img_name:<12} {cfg_name:<15} {'ERROR':>8}")
                continue

            stats = report.get('stats', {})
            timings = report.get('timings', {})

            path_a = stats.get('path_a_filtered', 0)
            path_b = stats.get('path_b_regions', 0)
            pool = stats.get('pool_regions', 0)
            sd = stats.get('stardist_nuclei', 0)
            final = stats.get('final_nuclei', 0)
            total = timings.get('total', 0)

            print(f"{img_name:<12} {cfg_name:<15} {path_a:>8} {path_b:>8} {pool:>8} {sd:>8} {final:>8} {total:>7.1f}s")

            summary_data.append({
                'image': img_name,
                'config': cfg_name,
                'path_a_filtered': path_a,
                'path_b_regions': path_b,
                'pool_regions': pool,
                'stardist_nuclei': sd,
                'final_nuclei': final,
                'total_time': total
            })

    summary_path = OUTPUT_BASE / "summary_report.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': summary_data
        }, f, indent=2)
    print(f"\nsummary saved: {summary_path}")

    return summary_data


def main():
    print("batch run: config 1 + config 2")
    print(f"started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"images: {len(ALL_IMAGES)}")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    print("\nloading stardist model...")
    t0 = time.time()
    stardist_model = load_stardist_model()
    print(f"  model loaded in {time.time() - t0:.1f}s")

    all_results = {}

    for img_name in ALL_IMAGES:
        try:
            results = run_image(img_name, stardist_model)
            if results:
                all_results[img_name] = results
        except Exception as e:
            print(f"  error processing {img_name}: {e}")
            traceback.print_exc()
            all_results[img_name] = {'error': str(e)}

    generate_summary_report(all_results)

    print(f"\nbatch complete")
    print(f"ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"results: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
