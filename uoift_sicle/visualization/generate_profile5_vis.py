#!/usr/bin/env python3
"""generate visualizations for profile 5 path a results"""

import sys
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_batch_vis import visualize_config
from core.config import ALL_IMAGES

BATCH_RESULTS = Path(__file__).parent / "batch_results_profile5_pathA"

CONFIGS = [
    "config1_profile5_orig_orig",
    "config2_profile5_orig_simp"
]


def main():
    print("generate visualizations: profile 5 path a")

    total = len(ALL_IMAGES) * len(CONFIGS)
    count = 0

    for img_name in ALL_IMAGES:
        print(f"\n[{img_name}]")
        img_dir = BATCH_RESULTS / img_name

        if not img_dir.exists():
            print(f"  directory not found: {img_dir}")
            continue

        for config_name in CONFIGS:
            count += 1
            config_dir = img_dir / config_name

            if not config_dir.exists():
                print(f"  {count}/{total} {config_name}: not found")
                continue

            print(f"  {count}/{total} {config_name}...", end=" ", flush=True)

            try:
                visualize_config(config_dir, img_name)
                print("ok")
            except Exception as e:
                print(f"error: {e}")

    print("\ndone")


if __name__ == "__main__":
    main()
