#!/usr/bin/env python3
"""Create unified benchmark summary consolidating all results."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import ALL_IMAGES
import json
from datetime import datetime

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).parent

print("unified benchmark summary")
print()

# load cellpose
print("Loading Cellpose results...")
cellpose_file = BASE_DIR / "benchmark_results" / "cellpose_summary.json"
with open(cellpose_file) as f:
    cellpose_data = json.load(f)

cellpose_results = {}
for img_name in ALL_IMAGES:
    if img_name in cellpose_data['results']:
        cellpose_results[img_name] = cellpose_data['results'][img_name]

print(f"  - Cellpose: {len(cellpose_results)} images")

# load sicle profile 5
print("Loading SICLE Profile 5 results...")
profile5_file = BASE_DIR / "sicle_profile5_results" / "profile5_summary.json"
with open(profile5_file) as f:
    profile5_data = json.load(f)

profile5_results = {}
for img_name in ALL_IMAGES:
    if img_name in profile5_data['results']:
        profile5_results[img_name] = profile5_data['results'][img_name]

print(f"  - SICLE Profile 5: {len(profile5_results)} images")

# load uoift pipeline
print("Loading UOIFT pipeline results...")
uoift_file = BASE_DIR / "batch_results" / "summary_report.json"
with open(uoift_file) as f:
    uoift_data = json.load(f)

# Organize by image and config
uoift_by_image = {}
for result in uoift_data['results']:
    img = result['image']
    cfg = result['config']
    if img not in uoift_by_image:
        uoift_by_image[img] = {}
    uoift_by_image[img][cfg] = result

print(f"  - UOIFT pipeline: {len(uoift_by_image)} images, 2 configs each")

# load no-saliency pipeline
print("Loading no-saliency pipeline results...")
no_sal_dir = BASE_DIR / "batch_results_no_saliency"

no_sal_by_image = {}
for img_name in ALL_IMAGES:
    img_dir = no_sal_dir / img_name

    if not img_dir.exists():
        continue

    no_sal_by_image[img_name] = {}

    # Config 3: no sal orig
    config3_report = img_dir / "config3_no_sal_orig" / "report.json"
    if config3_report.exists():
        with open(config3_report) as f:
            no_sal_by_image[img_name]['config3'] = json.load(f)

    # Config 4: no sal simp
    config4_report = img_dir / "config4_no_sal_simp" / "report.json"
    if config4_report.exists():
        with open(config4_report) as f:
            no_sal_by_image[img_name]['config4'] = json.load(f)

print(f"  - No-saliency pipeline: {len(no_sal_by_image)} images")

# create unified table
print()
print("results table")
print()

# header
print(f"{'Image':<12} {'Cellpose':>10} {'Profile5':>10} {'UOIFT-C1':>10} {'UOIFT-C2':>10} {'NoSal-C3':>10} {'NoSal-C4':>10}")
print(f"{'':12} {'(600s)':>10} {'(~55s)':>10} {'(~220s)':>10} {'(~195s)':>10} {'(~215s)':>10} {'(~215s)':>10}")

# Data rows
for img in ALL_IMAGES:
    row = [img]

    # Cellpose
    if img in cellpose_results:
        row.append(f"{cellpose_results[img]['nuclei']}")
    else:
        row.append("-")

    # SICLE Profile 5 (average of with/without saliency)
    if img in profile5_results:
        methods = profile5_results[img]['methods']
        p5_higra = methods.get('profile5_higra', {}).get('nuclei', 0)
        p5_nosal = methods.get('profile5_no_sal', {}).get('nuclei', 0)
        avg = (p5_higra + p5_nosal) / 2 if p5_higra and p5_nosal else 0
        row.append(f"{int(avg)}")
    else:
        row.append("-")

    # UOIFT config1
    if img in uoift_by_image and 'config1' in uoift_by_image[img]:
        row.append(f"{uoift_by_image[img]['config1']['final_nuclei']}")
    else:
        row.append("-")

    # UOIFT config2
    if img in uoift_by_image and 'config2' in uoift_by_image[img]:
        row.append(f"{uoift_by_image[img]['config2']['final_nuclei']}")
    else:
        row.append("-")

    # No-sal config3
    if img in no_sal_by_image and 'config3' in no_sal_by_image[img]:
        row.append(f"{no_sal_by_image[img]['config3']['stats']['final_nuclei']}")
    else:
        row.append("-")

    # No-sal config4
    if img in no_sal_by_image and 'config4' in no_sal_by_image[img]:
        row.append(f"{no_sal_by_image[img]['config4']['stats']['final_nuclei']}")
    else:
        row.append("-")

    # Print row
    print(f"{row[0]:<12} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10} {row[5]:>10} {row[6]:>10}")

# averages

# Compute averages
avg_cellpose = sum(r['nuclei'] for r in cellpose_results.values()) / len(cellpose_results) if cellpose_results else 0
avg_profile5 = sum(
    (r['methods'].get('profile5_higra', {}).get('nuclei', 0) +
     r['methods'].get('profile5_no_sal', {}).get('nuclei', 0)) / 2
    for r in profile5_results.values()
) / len(profile5_results) if profile5_results else 0

avg_uoift_c1 = sum(d['config1']['final_nuclei'] for d in uoift_by_image.values() if 'config1' in d) / len([d for d in uoift_by_image.values() if 'config1' in d])
avg_uoift_c2 = sum(d['config2']['final_nuclei'] for d in uoift_by_image.values() if 'config2' in d) / len([d for d in uoift_by_image.values() if 'config2' in d])

avg_nosal_c3 = sum(d['config3']['stats']['final_nuclei'] for d in no_sal_by_image.values() if 'config3' in d) / len([d for d in no_sal_by_image.values() if 'config3' in d]) if no_sal_by_image else 0
avg_nosal_c4 = sum(d['config4']['stats']['final_nuclei'] for d in no_sal_by_image.values() if 'config4' in d) / len([d for d in no_sal_by_image.values() if 'config4' in d]) if no_sal_by_image else 0

print(f"{'AVERAGE':<12} {int(avg_cellpose):>10} {int(avg_profile5):>10} {int(avg_uoift_c1):>10} {int(avg_uoift_c2):>10} {int(avg_nosal_c3):>10} {int(avg_nosal_c4):>10}")

# timing comparison
print()
print("TIMING COMPARISON (average per image)")
print()

avg_time_cellpose = 600  # seconds (10 min, from cellpose_summary)
avg_time_profile5 = sum(
    (r['methods'].get('profile5_higra', {}).get('time', 0) +
     r['methods'].get('profile5_no_sal', {}).get('time', 0)) / 2
    for r in profile5_results.values()
) / len(profile5_results) if profile5_results else 0

avg_time_uoift_c1 = sum(d['config1']['total_time'] for d in uoift_by_image.values() if 'config1' in d) / len([d for d in uoift_by_image.values() if 'config1' in d])
avg_time_uoift_c2 = sum(d['config2']['total_time'] for d in uoift_by_image.values() if 'config2' in d) / len([d for d in uoift_by_image.values() if 'config2' in d])

avg_time_nosal_c3 = sum(d['config3']['timings']['total'] for d in no_sal_by_image.values() if 'config3' in d) / len([d for d in no_sal_by_image.values() if 'config3' in d]) if no_sal_by_image else 0
avg_time_nosal_c4 = sum(d['config4']['timings']['total'] for d in no_sal_by_image.values() if 'config4' in d) / len([d for d in no_sal_by_image.values() if 'config4' in d]) if no_sal_by_image else 0

print(f"{'Method':<20} {'Avg Time (s)':>15} {'Speedup vs Cellpose':>25}")
print(f"{'Cellpose':<20} {avg_time_cellpose:>15.1f} {1.0:>25.2f}x")
print(f"{'SICLE Profile 5':<20} {avg_time_profile5:>15.1f} {avg_time_cellpose/avg_time_profile5:>25.2f}x")
print(f"{'UOIFT Config1':<20} {avg_time_uoift_c1:>15.1f} {avg_time_cellpose/avg_time_uoift_c1:>25.2f}x")
print(f"{'UOIFT Config2':<20} {avg_time_uoift_c2:>15.1f} {avg_time_cellpose/avg_time_uoift_c2:>25.2f}x")
print(f"{'NoSal Config3':<20} {avg_time_nosal_c3:>15.1f} {avg_time_cellpose/avg_time_nosal_c3:>25.2f}x")
print(f"{'NoSal Config4':<20} {avg_time_nosal_c4:>15.1f} {avg_time_cellpose/avg_time_nosal_c4:>25.2f}x")

# save json
print()
print("Saving unified summary to JSON...")

unified_summary = {
    "timestamp": datetime.now().isoformat(),
    "images": ALL_IMAGES,
    "methods": {
        "cellpose": {
            "description": "Cellpose baseline (CPU, 10 min/image)",
            "avg_nuclei": int(avg_cellpose),
            "avg_time": avg_time_cellpose,
            "results": cellpose_results
        },
        "profile5": {
            "description": "SICLE Profile 5 alone (nf=1800, ~55s/image)",
            "avg_nuclei": int(avg_profile5),
            "avg_time": avg_time_profile5,
            "results": profile5_results
        },
        "uoift_config1": {
            "description": "UOIFT+SICLE+StarDist, orig-orig (~220s/image)",
            "avg_nuclei": int(avg_uoift_c1),
            "avg_time": avg_time_uoift_c1,
            "speedup_vs_cellpose": avg_time_cellpose / avg_time_uoift_c1
        },
        "uoift_config2": {
            "description": "UOIFT+SICLE+StarDist, orig-simp (~195s/image)",
            "avg_nuclei": int(avg_uoift_c2),
            "avg_time": avg_time_uoift_c2,
            "speedup_vs_cellpose": avg_time_cellpose / avg_time_uoift_c2
        },
        "nosal_config3": {
            "description": "SICLE+StarDist without saliency, orig (~215s/image)",
            "avg_nuclei": int(avg_nosal_c3),
            "avg_time": avg_time_nosal_c3,
            "speedup_vs_cellpose": avg_time_cellpose / avg_time_nosal_c3 if avg_time_nosal_c3 > 0 else 0
        },
        "nosal_config4": {
            "description": "SICLE+StarDist without saliency, simp (~215s/image)",
            "avg_nuclei": int(avg_nosal_c4),
            "avg_time": avg_time_nosal_c4,
            "speedup_vs_cellpose": avg_time_cellpose / avg_time_nosal_c4 if avg_time_nosal_c4 > 0 else 0
        }
    },
    "by_image": {}
}

# Per-image breakdown
for img in ALL_IMAGES:
    unified_summary['by_image'][img] = {
        "cellpose": cellpose_results.get(img, {}).get('nuclei', None),
        "profile5_avg": int((
            profile5_results.get(img, {}).get('methods', {}).get('profile5_higra', {}).get('nuclei', 0) +
            profile5_results.get(img, {}).get('methods', {}).get('profile5_no_sal', {}).get('nuclei', 0)
        ) / 2) if img in profile5_results else None,
        "uoift_config1": uoift_by_image.get(img, {}).get('config1', {}).get('final_nuclei', None),
        "uoift_config2": uoift_by_image.get(img, {}).get('config2', {}).get('final_nuclei', None),
        "nosal_config3": no_sal_by_image.get(img, {}).get('config3', {}).get('stats', {}).get('final_nuclei', None),
        "nosal_config4": no_sal_by_image.get(img, {}).get('config4', {}).get('stats', {}).get('final_nuclei', None),
    }

output_file = BASE_DIR / "unified_benchmark_summary.json"
with open(output_file, 'w') as f:
    json.dump(unified_summary, f, indent=2)

print(f"  - Saved to: {output_file}")
