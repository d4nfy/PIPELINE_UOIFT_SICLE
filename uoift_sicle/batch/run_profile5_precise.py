#!/usr/bin/env python3
"""batch run: profile 5 path a (n0=45000, nf=1800)

uses sicle profile 5 for path a instead of default path a params.
rest of pipeline unchanged: path b multiscale, stardist, arbitration.
"""

import sys
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from skimage import io
import json
import time
from datetime import datetime

from core.config import ALL_IMAGES, SICLE_PROFILE5, SICLE_PATH_B
from core.sicle_wrapper import run_sicle_with_params
from core.stardist_wrapper import load_stardist_model, stardist_predict
from core.pipeline_utils import get_lab_channels, fuse_segmentations
from core.veta_filtering import filter_sicle_regions, veta_multiscale_selection

# directories
BASE_DIR = Path(__file__).parent
IMG_DIR = BASE_DIR.parent / "img_used"
OUTPUT_DIR = BASE_DIR / "batch_results_profile5_pathA"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# filter config override for profile 5
# a_min=10 differs from default (15)
FILTER_CFG_PROFILE5 = {
    "min_area": 100,
    "max_area": 10000,
    "min_solidity": 0.80,
    "l_max": 100,
    "a_min": 10,
    "a_max": 40,
    "b_max": 35,
}


def run_pipeline_config_profile5(img_name, config_name, img_a_path, img_b_path,
                                  saliency_path, stardist_model):
    """run pipeline with profile 5 path a"""
    print(f"\n  {config_name}...")

    output_dir = OUTPUT_DIR / img_name / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'image': img_name,
        'config': config_name,
        'note': f"profile 5 path a (n0={SICLE_PROFILE5['n0']}, nf={SICLE_PROFILE5['nf']})",
        'start_time': datetime.now().isoformat(),
        'timings': {},
        'stats': {}
    }

    img_a = io.imread(str(img_a_path))
    img_b = io.imread(str(img_b_path))
    l_ch, a_ch, b_ch = get_lab_channels(img_a)

    # load saliency
    if saliency_path and saliency_path.exists():
        saliency = io.imread(str(saliency_path))
        if saliency.ndim == 3:
            saliency = saliency[:, :, 0]
    else:
        print(f"      no saliency (running without)")
        saliency = None

    # save temp files
    temp_img_a = output_dir / "temp_img_a.png"
    temp_img_b = output_dir / "temp_img_b.png"
    io.imsave(str(temp_img_a), img_a, check_contrast=False)
    io.imsave(str(temp_img_b), img_b, check_contrast=False)

    if saliency is not None:
        temp_sal = output_dir / "temp_saliency.png"
        io.imsave(str(temp_sal), saliency.astype(np.uint8), check_contrast=False)
        sal_path_str = str(temp_sal)
    else:
        sal_path_str = None

    # sicle path a with profile 5 (sec 3.3)
    print(f"      sicle path a (profile 5: n0={SICLE_PROFILE5['n0']}, nf={SICLE_PROFILE5['nf']})...")
    t0 = time.time()
    try:
        result_a = run_sicle_with_params(
            img_path=str(temp_img_a),
            saliency_path=sal_path_str,
            output_path=str(output_dir / "path_a.pgm"),
            params=SICLE_PROFILE5,
            multiscale=False
        )
        if isinstance(result_a, dict):
            path_a_mask = list(result_a.values())[0]
        else:
            path_a_mask = result_a
        t_path_a = time.time() - t0
    except Exception as e:
        print(f"      path a failed: {e}")
        report['error'] = f"path a failed: {e}"
        return report

    n_path_a = len(np.unique(path_a_mask)) - 1
    report['timings']['sicle_path_a'] = t_path_a
    report['stats']['path_a_regions'] = int(n_path_a)
    print(f"      path a: {n_path_a} regions ({t_path_a:.1f}s)")

    # filter path a (sec 3.4)
    print(f"      filter path a...")
    t0 = time.time()
    path_a_filtered, n_kept, n_rejected = filter_sicle_regions(
        path_a_mask, l_ch, a_ch, b_ch, FILTER_CFG_PROFILE5
    )
    t_filter_a = time.time() - t0
    report['timings']['filter_path_a'] = t_filter_a
    report['stats']['path_a_filtered'] = int(n_kept)
    print(f"      path a filtered: {n_kept} regions ({t_filter_a:.1f}s)")
    np.save(str(output_dir / "path_a_filtered.npy"), path_a_filtered)

    # sicle path b multiscale (sec 3.3)
    print(f"      sicle path b (multiscale)...")
    t0 = time.time()
    try:
        result_b = run_sicle_with_params(
            img_path=str(temp_img_b),
            saliency_path=sal_path_str,
            output_path=str(output_dir / "path_b.pgm"),
            params=SICLE_PATH_B,
            multiscale=True
        )
        t_path_b = time.time() - t0
    except Exception as e:
        print(f"      path b failed: {e}")
        report['error'] = f"path b failed: {e}"
        return report

    report['timings']['sicle_path_b'] = t_path_b
    report['stats']['path_b_scales'] = len(result_b)
    print(f"      path b: {len(result_b)} scales ({t_path_b:.1f}s)")

    # veta selection (sec 3.4)
    print(f"      veta selection...")
    t0 = time.time()
    path_b_veta = veta_multiscale_selection(
        result_b, l_ch, a_ch, b_ch, FILTER_CFG_PROFILE5,
        min_solidity=0.80, ov_thresh=0.2
    )
    t_veta = time.time() - t0
    n_path_b = len(np.unique(path_b_veta)) - 1
    report['timings']['veta_selection'] = t_veta
    report['stats']['path_b_regions'] = int(n_path_b)
    print(f"      path b veta: {n_path_b} regions ({t_veta:.1f}s)")
    np.save(str(output_dir / "path_b_veta.npy"), path_b_veta)

    # pool (sec 3.3)
    print(f"      create pool...")
    t0 = time.time()
    pool = path_a_filtered.copy()
    next_id = pool.max() + 1
    for region_id in np.unique(path_b_veta):
        if region_id == 0:
            continue
        mask = path_b_veta == region_id
        overlap = (pool > 0) & mask
        if overlap.sum() / mask.sum() < 0.5:
            pool[mask] = next_id
            next_id += 1
    t_pool = time.time() - t0
    n_pool = len(np.unique(pool)) - 1
    report['timings']['pool'] = t_pool
    report['stats']['pool_regions'] = int(n_pool)
    print(f"      pool: {n_pool} regions ({t_pool:.1f}s)")
    np.save(str(output_dir / "pool.npy"), pool)

    # stardist (sec 3.5)
    print(f"      stardist...")
    t0 = time.time()
    stardist_mask = stardist_predict(img_a, stardist_model)
    t_stardist = time.time() - t0
    n_stardist = len(np.unique(stardist_mask)) - 1
    report['timings']['stardist'] = t_stardist
    report['stats']['stardist_nuclei'] = int(n_stardist)
    print(f"      stardist: {n_stardist} nuclei ({t_stardist:.1f}s)")
    np.save(str(output_dir / "stardist.npy"), stardist_mask)

    # arbitration (sec 3.5)
    print(f"      arbitration...")
    t0 = time.time()
    fused = fuse_segmentations(stardist_mask, pool, iou_thresh=0.2)
    t_arb = time.time() - t0
    n_fused = len(np.unique(fused)) - 1
    report['timings']['arbitration'] = t_arb
    report['stats']['fused_regions'] = int(n_fused)
    report['stats']['final_nuclei'] = int(n_fused)
    print(f"      fused: {n_fused} regions ({t_arb:.1f}s)")
    np.save(str(output_dir / "fused.npy"), fused)
    np.save(str(output_dir / "final.npy"), fused)

    report['timings']['total'] = sum(report['timings'].values())
    report['end_time'] = datetime.now().isoformat()

    with open(output_dir / "report.json", 'w') as f:
        json.dump(report, f, indent=2)

    return report


def main():
    print("batch run: profile 5 path a")
    print(f"started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"path a: profile 5 (n0={SICLE_PROFILE5['n0']}, nf={SICLE_PROFILE5['nf']})")
    print(f"path b: multiscale (n0={SICLE_PATH_B['n0']}, nf={SICLE_PATH_B['nf']})")

    stardist_model = load_stardist_model()
    print("  stardist model loaded")

    all_results = []

    for img_name in ALL_IMAGES:
        print(f"\n[{ALL_IMAGES.index(img_name) + 1}/{len(ALL_IMAGES)}] {img_name}")

        img_dir = IMG_DIR / img_name
        img_original = img_dir / "pre_seg" / "img_original.png"
        img_simplified = img_dir / "pre_seg" / "img_simplified.png"
        saliency_uoift = img_dir / "pre_seg" / "UOIFT" / "saliency_uoift.png"

        if not img_original.exists():
            print(f"  skip: img_original not found")
            continue

        # config 1: both on original
        result1 = run_pipeline_config_profile5(
            img_name, "config1_profile5_orig_orig",
            img_original, img_original,
            saliency_uoift if saliency_uoift.exists() else None,
            stardist_model
        )
        all_results.append(result1)

        # config 2: path b on simplified
        if img_simplified.exists():
            result2 = run_pipeline_config_profile5(
                img_name, "config2_profile5_orig_simp",
                img_original, img_simplified,
                saliency_uoift if saliency_uoift.exists() else None,
                stardist_model
            )
            all_results.append(result2)

    # summary
    print("\nsummary")
    print(f"\n{'Image':<12} {'Config':<25} {'Final':>8} {'Time(s)':>10}")

    for result in all_results:
        if 'error' not in result:
            print(f"{result['image']:<12} {result['config']:<25} {result['stats']['final_nuclei']:>8} {result['timings']['total']:>10.1f}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "note": f"profile 5 path a (n0={SICLE_PROFILE5['n0']}, nf={SICLE_PROFILE5['nf']}) + multiscale path b",
        "results": all_results
    }
    with open(OUTPUT_DIR / "summary_profile5_pathA.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nresults saved to: {OUTPUT_DIR}")
    print(f"completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
