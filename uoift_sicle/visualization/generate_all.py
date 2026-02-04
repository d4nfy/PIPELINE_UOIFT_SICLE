#!/usr/bin/env python3
"""generate visualizations for all completed batch results"""

import sys
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_batch_vis import visualize_config
from core.config import ALL_IMAGES

BATCH_RESULTS = Path(__file__).parent / "batch_results"

SKIP_IMAGES = ["B3-T_1", "B3-T_2"]

print("generating visualizations for remaining images...")
print(f"total: {len(ALL_IMAGES)}, skip: {len(SKIP_IMAGES)}, generate: {len(ALL_IMAGES) - len(SKIP_IMAGES)}\n")

for img_name in ALL_IMAGES:
    if img_name in SKIP_IMAGES:
        print(f"  {img_name}: skipped (already has vis)")
        continue

    print(f"\n  {img_name}:")
    img_dir = BATCH_RESULTS / img_name

    if not img_dir.exists():
        print(f"    error: no results directory")
        continue

    # config 1: orig+orig
    config1_dir = img_dir / "config1_orig_orig"
    if config1_dir.exists():
        print(f"    config1...")
        try:
            visualize_config(config1_dir, img_name)
            print(f"      config1 vis generated")
        except Exception as e:
            print(f"      config1 error: {e}")
    else:
        print(f"    config1: not found")

    # config 2: orig+simp
    config2_dir = img_dir / "config2_orig_simp"
    if config2_dir.exists():
        print(f"    config2...")
        try:
            visualize_config(config2_dir, img_name)
            print(f"      config2 vis generated")
        except Exception as e:
            print(f"      config2 error: {e}")
    else:
        print(f"    config2: not found")

print("\ndone")
