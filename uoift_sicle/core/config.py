"""centralized pipeline configuration

maps to paper sections:
  sec 3.2  uoift saliency with boundary polarity
  sec 3.3  dual path sicle (coarse path a + multiscale path b)
  sec 3.4  veta inspired filtering (solidity, lab, area)
  sec 3.5  stardist and iou based arbitration
  sec 4.1  dataset (10 oral histology images, 1920x1200, 40x)
"""

# sec 4.1: oral histology dataset
ALL_IMAGES = [
    "B3-T_1", "B3-T_2", "B4-T_1", "B5-T_1", "B5-T_3",
    "C1-T_1", "C2-T_1", "C4-T_1", "C4-T_2", "C5-T_1",
]

# sec 3.3: sicle path a (coarse, diverse retrieval, nf=3000)
SICLE_PATH_A = {
    "n0": 52000,
    "nf": 3000,
    "alpha": 0.9,
    "max_iters": 22,
    "irreg": 0.12,
    "adhr": 16,
    "sampl_opt": "grid",
}

# sec 3.3: sicle path b (multiscale, precision, nf=500)
SICLE_PATH_B = {
    "n0": 52000,
    "nf": 500,
    "alpha": 0.85,
    "max_iters": 12,
    "irreg": 0.15,
    "adhr": 12,
    "sampl_opt": "grid",
}

# sec 3.3: sicle profile 5 (precise variant, nf=1800)
SICLE_PROFILE5 = {
    "n0": 45000,
    "nf": 1800,
    "alpha": 0.9,
    "max_iters": 22,
    "irreg": 0.12,
    "adhr": 16,
    "sampl_opt": "grid",
}

# sec 3.4: veta filter thresholds
FILTER_CFG = {
    "min_area": 100,
    "max_area": 10000,
    "min_solidity": 0.80,
    "l_max": 100,
    "a_min": 15,
    "a_max": 40,
    "b_max": 35,
}

# sec 3.2: uoift saliency params (eq. 1, alpha=-0.7)
UOIFT_ALPHA = -0.7
UOIFT_SP_SIZE = 100

# sec 3.5: arbitration thresholds
ION_THRESH = 0.2
POOL_OVERLAP_THRESH = 0.5
KDTREE_MAX_DIST = 100

# preprocessing (used by higra_sicle compat)
CFG_PREPROC = {
    "w_l": 0.4, "w_a": 0.6, "w_b": 0.2,
    "morph_h": 20,
    "filter_min_area": 500, "filter_min_height": 8,
    "simplify_min_area": 1800, "simplify_min_height": 10,
    "sal_sigma": 1.5, "sal_gamma": 0.5,
    "sal_prob_gamma": 1.2, "sal_mix_soft": 0.7, "sal_mix_dist": 0.3,
    "iou_threshold": 0.5,
    "final_min_area": 80, "final_max_area": 9000, "final_min_solidity": 0.75,
    "final_l_max": 105, "final_a_min": 4, "final_b_max": 25,
}
