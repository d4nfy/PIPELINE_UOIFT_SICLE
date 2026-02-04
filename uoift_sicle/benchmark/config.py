"""benchmark configuration for nuclei segmentation methods"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import ALL_IMAGES

# sicle parameter profiles

# Profile 5: PRECISE segmentation (from profile5_profile7.sh)
# Lower n0/nf = fewer, more precise regions
PROFILE_5_PARAMS = {
    "n0": 45000,
    "nf": 1800,
    "alpha": 0.9,
    "max_iters": 22,
    "irreg": 0.12,
    "adhr": 16,
    "sampl_opt": "grid",
    "conn_opt": "fmax",
    "crit_opt": "minsc",
}

# Coarse Path A: from batch_run_all (current working params)
# Higher n0/nf = more regions, coarser
COARSE_PATH_A_PARAMS = {
    "n0": 52000,
    "nf": 3000,
    "alpha": 0.9,
    "max_iters": 22,
    "irreg": 0.12,
    "adhr": 16,
    "sampl_opt": "grid",
    "conn_opt": "fmax",
    "crit_opt": "minsc",
}

# Multiscale Path B: from batch_run_all
# High n0/nf ratio (52000/500=104) for MORE scales
MULTISCALE_PATH_B_PARAMS = {
    "n0": 52000,
    "nf": 500,
    "alpha": 0.85,
    "max_iters": 12,
    "irreg": 0.15,
    "adhr": 12,
    "sampl_opt": "grid",
    "conn_opt": "fmax",
    "crit_opt": "minsc",
    "multiscale": True,
}

# benchmark methods configuration

BENCHMARK_METHODS = {
    # standalone methods

    "cellpose": {
        "description": "Cellpose nuclei model (CPU)",
        "type": "standalone",
        "model": "nuclei",
        "expected_time": 600,  # ~10 min on CPU for 1920x1200
        "notes": "Pre-computed results in img_used/<IMG>/cellpose_alone/",
    },

    "stardist": {
        "description": "StarDist 2D_versatile_he",
        "type": "standalone",
        "model": "2D_versatile_he",
        "expected_time": 3,  # ~3s
        "notes": "Fast, convex nuclei detector",
    },

    # sicle standalone

    "sicle_profile5_higra": {
        "description": "SICLE Profile 5 with HIGRA saliency",
        "type": "sicle_standalone",
        "sicle_params": PROFILE_5_PARAMS,
        "saliency_source": "HIGRA",  # pre_seg/HIGRA/saliency.png
        "expected_time": 40,
        "notes": "TO FIX: verify saliency path is correct",
    },

    "sicle_profile5_no_saliency": {
        "description": "SICLE Profile 5 without saliency",
        "type": "sicle_standalone",
        "sicle_params": PROFILE_5_PARAMS,
        "saliency_source": None,
        "expected_time": 40,
        "notes": "TO FIX: run without --objsm flag",
    },

    # pipeline options (uoift saliency)

    "pipeline_option_a_uoift": {
        "description": "Pipeline Option A with UOIFT saliency",
        "type": "pipeline",
        "saliency_source": "UOIFT",  # pre_seg/UOIFT/saliency_uoift.png
        "path_a": {
            "params": COARSE_PATH_A_PARAMS,
            "input": "img_original",
            "multiscale": False,
        },
        "path_b": {
            "params": MULTISCALE_PATH_B_PARAMS,
            "input": "img_original",  # Option A = both on original
            "multiscale": True,
        },
        "fusion": "stardist_arbitration",
        "expected_time": 150,
        "notes": "Path A coarse (52k/3k) + Path B multiscale (52k/500)",
    },

    "pipeline_option_b_uoift": {
        "description": "Pipeline Option B with UOIFT saliency",
        "type": "pipeline",
        "saliency_source": "UOIFT",
        "path_a": {
            "params": COARSE_PATH_A_PARAMS,
            "input": "img_original",
            "multiscale": False,
        },
        "path_b": {
            "params": MULTISCALE_PATH_B_PARAMS,
            "input": "img_simplified",  # Option B = Path B on simplified
            "multiscale": True,
        },
        "fusion": "stardist_arbitration",
        "expected_time": 150,
        "notes": "Path A on original + Path B on HIGRA simplified",
    },

    # pipeline options (higra saliency) - existing

    "pipeline_option_a_higra": {
        "description": "Pipeline Option A with HIGRA saliency",
        "type": "pipeline",
        "saliency_source": "HIGRA",  # pre_seg/HIGRA/saliency.png
        "path_a": {
            "params": COARSE_PATH_A_PARAMS,
            "input": "img_original",
            "multiscale": False,
        },
        "path_b": {
            "params": MULTISCALE_PATH_B_PARAMS,
            "input": "img_original",
            "multiscale": True,
        },
        "fusion": "stardist_arbitration",
        "expected_time": 150,
        "notes": "EXISTS for all except B3-T_1 (was pipeline_config1)",
    },

    "pipeline_option_b_higra": {
        "description": "Pipeline Option B with HIGRA saliency",
        "type": "pipeline",
        "saliency_source": "HIGRA",
        "path_a": {
            "params": COARSE_PATH_A_PARAMS,
            "input": "img_original",
            "multiscale": False,
        },
        "path_b": {
            "params": MULTISCALE_PATH_B_PARAMS,
            "input": "img_simplified",
            "multiscale": True,
        },
        "fusion": "stardist_arbitration",
        "expected_time": 150,
        "notes": "EXISTS for all except B3-T_1 (was pipeline_config2)",
    },

    # pipeline options with profile 5 path a (uoift)

    "pipeline_option_a_uoift_profile5": {
        "description": "Pipeline Option A (UOIFT) with Profile 5 precise Path A",
        "type": "pipeline",
        "saliency_source": "UOIFT",
        "path_a": {
            "params": PROFILE_5_PARAMS,  # PRECISE: n0=45000, nf=1800
            "input": "img_original",
            "multiscale": False,
        },
        "path_b": {
            "params": MULTISCALE_PATH_B_PARAMS,
            "input": "img_original",
            "multiscale": True,
        },
        "fusion": "stardist_arbitration",
        "expected_time": 150,
        "notes": "Profile 5 precise Path A (45k/1.8k) + multiscale Path B",
    },

    "pipeline_option_b_uoift_profile5": {
        "description": "Pipeline Option B (UOIFT) with Profile 5 precise Path A",
        "type": "pipeline",
        "saliency_source": "UOIFT",
        "path_a": {
            "params": PROFILE_5_PARAMS,  # PRECISE: n0=45000, nf=1800
            "input": "img_original",
            "multiscale": False,
        },
        "path_b": {
            "params": MULTISCALE_PATH_B_PARAMS,
            "input": "img_simplified",
            "multiscale": True,
        },
        "fusion": "stardist_arbitration",
        "expected_time": 150,
        "notes": "Profile 5 precise Path A + multiscale Path B on simplified",
    },
}

# saliency paths

def get_saliency_path(img_dir, saliency_source):
    """Get saliency file path based on source type."""
    from pathlib import Path
    img_dir = Path(img_dir)

    if saliency_source == "UOIFT":
        return img_dir / "pre_seg" / "UOIFT" / "saliency_uoift.pgm"
    elif saliency_source == "HIGRA":
        return img_dir / "pre_seg" / "HIGRA" / "saliency.png"
    elif saliency_source is None:
        return None
    else:
        raise ValueError(f"Unknown saliency source: {saliency_source}")


def get_input_image_path(img_dir, input_type):
    """Get input image path based on type."""
    from pathlib import Path
    img_dir = Path(img_dir)

    if input_type == "img_original":
        return img_dir / "pre_seg" / "img_original.png"
    elif input_type == "img_simplified":
        return img_dir / "pre_seg" / "img_simplified.png"
    else:
        raise ValueError(f"Unknown input type: {input_type}")


# timing summary

TIMING_ESTIMATES = {
    "cellpose": 600,           # ~10 min CPU
    "stardist": 3,             # ~3s
    "sicle_profile5": 40,      # ~40s
    "uoift_saliency": 32,      # ~32s (from generation logs)
    "higra_simplification": 5, # ~5s
    "sicle_path_a": 35,        # ~35s (no multiscale)
    "sicle_path_b_multiscale": 45,  # ~45s (with multiscale)
    "pool_creation": 2,        # ~2s
    "stardist_inference": 3,   # ~3s
    "arbitration": 5,          # ~5s
    "filtering": 2,            # ~2s
}

# Pipeline total estimate:
# UOIFT(32) + PathA(35) + PathB(45) + Pool(2) + StarDist(3) + Arb(5) + Filter(2) = ~125s
# With WSL overhead: ~150s


if __name__ == "__main__":
    print("benchmark configuration summary")

    print(f"\ntotal methods: {len(BENCHMARK_METHODS)}")
    print(f"total images: {len(ALL_IMAGES)}")

    print("\nmethods:")
    for name, config in BENCHMARK_METHODS.items():
        print(f"  - {name}")
        print(f"    {config['description']}")
        print(f"    expected time: ~{config['expected_time']}s")
        if 'notes' in config:
            print(f"    notes: {config['notes']}")
        print()

    print("\nprofile 5 params (precise):")
    for k, v in PROFILE_5_PARAMS.items():
        print(f"  {k}: {v}")

    print("\ncoarse path a params:")
    for k, v in COARSE_PATH_A_PARAMS.items():
        print(f"  {k}: {v}")

    print("\nmultiscale path b params:")
    for k, v in MULTISCALE_PATH_B_PARAMS.items():
        print(f"  {k}: {v}")
