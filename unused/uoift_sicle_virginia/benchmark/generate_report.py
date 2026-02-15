#!/usr/bin/env python3
"""
Generate comprehensive benchmark report comparing all methods.

Methods compared:
1. Cellpose 4.0+ alone (pre-computed)
2. StarDist alone
3. SICLE Profile 5 (Path A only, no multiscale)
4. UOIFT+SICLE Pipeline Config 1 (orig-orig)
5. UOIFT+SICLE Pipeline Config 2 (orig-simp)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import ALL_IMAGES

import functools
print = functools.partial(print, flush=True)

import json
from datetime import datetime
import numpy as np

OUTPUT_BASE = Path(__file__).parent / "benchmark_results"
BATCH_RESULTS = Path(__file__).parent / "batch_results"


def load_cellpose_results():
    """Load cellpose summary"""
    path = OUTPUT_BASE / "cellpose_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_batch_results():
    """Load batch pipeline results"""
    path = BATCH_RESULTS / "summary_report.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_stardist_results():
    """Load stardist results from batch"""
    results = {}
    for img_name in ALL_IMAGES:
        report_path = BATCH_RESULTS / img_name / "config1_orig_orig" / "report.json"
        if report_path.exists():
            with open(report_path) as f:
                data = json.load(f)
                results[img_name] = {
                    'nuclei': data.get('stats', {}).get('stardist_nuclei', 0),
                    'time': data.get('timings', {}).get('stardist', 0)
                }
    return results


def main():
    print("comprehensive benchmark report")
    print(f"generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # load all results
    cellpose_data = load_cellpose_results()
    batch_data = load_batch_results()
    stardist_data = load_stardist_results()

    # build comparison table
    print("nuclei count comparison")
    print()
    header = f"{'Image':<12} {'Cellpose':>10} {'StarDist':>10} {'Config1':>10} {'Config2':>10} {'Diff C1-C2':>12}"
    print(header)

    totals = {'cellpose': [], 'stardist': [], 'config1': [], 'config2': []}

    for img_name in ALL_IMAGES:
        # Cellpose
        cp_nuclei = 0
        if cellpose_data and img_name in cellpose_data.get('results', {}):
            cp_result = cellpose_data['results'][img_name]
            if 'nuclei' in cp_result:
                cp_nuclei = cp_result['nuclei']
                totals['cellpose'].append(cp_nuclei)

        # StarDist
        sd_nuclei = 0
        if img_name in stardist_data:
            sd_nuclei = stardist_data[img_name].get('nuclei', 0)
            totals['stardist'].append(sd_nuclei)

        # Config 1 and Config 2
        c1_nuclei = 0
        c2_nuclei = 0
        if batch_data:
            for result in batch_data.get('results', []):
                if result['image'] == img_name:
                    if result['config'] == 'config1':
                        c1_nuclei = result.get('final_nuclei', 0)
                        totals['config1'].append(c1_nuclei)
                    elif result['config'] == 'config2':
                        c2_nuclei = result.get('final_nuclei', 0)
                        totals['config2'].append(c2_nuclei)

        diff = c2_nuclei - c1_nuclei
        diff_pct = f"{(c2_nuclei/c1_nuclei - 1)*100:+.1f}%" if c1_nuclei > 0 else "N/A"

        print(f"{img_name:<12} {cp_nuclei:>10} {sd_nuclei:>10} {c1_nuclei:>10} {c2_nuclei:>10} {diff:>6} ({diff_pct})")

    # Averages
    avg_cp = np.mean(totals['cellpose']) if totals['cellpose'] else 0
    avg_sd = np.mean(totals['stardist']) if totals['stardist'] else 0
    avg_c1 = np.mean(totals['config1']) if totals['config1'] else 0
    avg_c2 = np.mean(totals['config2']) if totals['config2'] else 0

    print(f"{'AVERAGE':<12} {avg_cp:>10.0f} {avg_sd:>10.0f} {avg_c1:>10.0f} {avg_c2:>10.0f}")
    print()

    # Timing comparison (only for pipeline - cellpose timing unknown)
    print("TIMING COMPARISON (seconds)")
    print()
    print(f"{'Image':<12} {'StarDist':>10} {'Config1':>10} {'Config2':>10}")

    time_totals = {'stardist': [], 'config1': [], 'config2': []}

    for img_name in ALL_IMAGES:
        sd_time = 0
        if img_name in stardist_data:
            sd_time = stardist_data[img_name].get('time', 0)
            time_totals['stardist'].append(sd_time)

        c1_time = 0
        c2_time = 0
        if batch_data:
            for result in batch_data.get('results', []):
                if result['image'] == img_name:
                    if result['config'] == 'config1':
                        c1_time = result.get('total_time', 0)
                        time_totals['config1'].append(c1_time)
                    elif result['config'] == 'config2':
                        c2_time = result.get('total_time', 0)
                        time_totals['config2'].append(c2_time)

        print(f"{img_name:<12} {sd_time:>10.1f} {c1_time:>10.1f} {c2_time:>10.1f}")

    avg_sd_t = np.mean(time_totals['stardist']) if time_totals['stardist'] else 0
    avg_c1_t = np.mean(time_totals['config1']) if time_totals['config1'] else 0
    avg_c2_t = np.mean(time_totals['config2']) if time_totals['config2'] else 0
    print(f"{'AVERAGE':<12} {avg_sd_t:>10.1f} {avg_c1_t:>10.1f} {avg_c2_t:>10.1f}")
    print()

    # summary statistics
    print("summary statistics")
    print()

    print("nuclei detection:")
    print(f"  Cellpose 4.0+:    avg {avg_cp:.0f} nuclei (timing unknown - pre-computed results)")
    print(f"  StarDist alone:   avg {avg_sd:.0f} nuclei in {avg_sd_t:.1f}s")
    print(f"  Pipeline Config1: avg {avg_c1:.0f} nuclei in {avg_c1_t:.1f}s")
    print(f"  Pipeline Config2: avg {avg_c2:.0f} nuclei in {avg_c2_t:.1f}s")
    print()

    print("key findings:")
    print(f"  - Config2 detects {(avg_c2/avg_c1 - 1)*100:.1f}% more nuclei than Config1")
    print(f"  - Pipeline detects {(avg_c1/avg_sd - 1)*100:.1f}% more nuclei than StarDist alone (Config1)")
    print(f"  - Pipeline detects {(avg_c2/avg_sd - 1)*100:.1f}% more nuclei than StarDist alone (Config2)")
    if avg_cp > 0:
        print(f"  - Pipeline Config1 vs Cellpose: {(avg_c1/avg_cp - 1)*100:+.1f}%")
        print(f"  - Pipeline Config2 vs Cellpose: {(avg_c2/avg_cp - 1)*100:+.1f}%")
    print()

    print("architecture notes:")
    print("  - config1: path a (img_original) + path b (img_original) - same input, different sicle params")
    print("  - config2: path a (img_original) + path b (img_simplified) - different inputs")
    print("  - both configs use veta multiscale selection for path b")
    print("  - both configs use arbitration (iou > 0.2, max solidity)")
    print()

    # Save comprehensive JSON report
    report = {
        'timestamp': datetime.now().isoformat(),
        'methods': {
            'cellpose': {
                'description': 'Cellpose 4.0+ nuclei model (pre-computed)',
                'avg_nuclei': avg_cp,
                'avg_time': None,  # unknown
            },
            'stardist': {
                'description': 'StarDist 2D_versatile_he model',
                'avg_nuclei': avg_sd,
                'avg_time': avg_sd_t,
            },
            'pipeline_config1': {
                'description': 'UOIFT+SICLE Pipeline (orig-orig)',
                'avg_nuclei': avg_c1,
                'avg_time': avg_c1_t,
            },
            'pipeline_config2': {
                'description': 'UOIFT+SICLE Pipeline (orig-simp)',
                'avg_nuclei': avg_c2,
                'avg_time': avg_c2_t,
            },
        },
        'per_image': {},
    }

    for img_name in ALL_IMAGES:
        report['per_image'][img_name] = {
            'cellpose': cellpose_data['results'].get(img_name, {}) if cellpose_data else {},
            'stardist': stardist_data.get(img_name, {}),
        }

        if batch_data:
            for result in batch_data.get('results', []):
                if result['image'] == img_name:
                    config = result['config']
                    report['per_image'][img_name][f'pipeline_{config}'] = {
                        'nuclei': result.get('final_nuclei', 0),
                        'time': result.get('total_time', 0),
                        'path_a_filtered': result.get('path_a_filtered', 0),
                        'path_b_regions': result.get('path_b_regions', 0),
                        'pool_regions': result.get('pool_regions', 0),
                    }

    report_path = OUTPUT_BASE / "comprehensive_benchmark_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"comprehensive report saved: {report_path}")
    print()
    print("benchmark complete")


if __name__ == "__main__":
    main()
