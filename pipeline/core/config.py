"""centralized pipeline configuration

maps to paper sections:
  sec 2.1.1  uoift saliency with boundary polarity
  sec 2.1.2  dual path sicle (coarse path a + multiscale path b)
  sec 2.2    veta inspired filtering (solidity, lab, area)
  sec 2.3    stardist and iou based arbitration
"""

# sec 2.1.2: sicle path a (coarse, diverse retrieval, nf=3000)
SICLE_PATH_A = {
    "n0": 52000,
    "nf": 3000,
    "alpha": 0.9,
    "max_iters": 22,
    "irreg": 0.12,
    "adhr": 16,
    "sampl_opt": "grid",
}

# sec 2.1.2: sicle path b (multiscale, precision, nf=500)
SICLE_PATH_B = {
    "n0": 52000,
    "nf": 500,
    "alpha": 0.85,
    "max_iters": 12,
    "irreg": 0.15,
    "adhr": 12,
    "sampl_opt": "grid",
}

# sec 2.2: veta filter thresholds
FILTER_CFG = {
    "min_area": 100,
    "max_area": 10000,
    "min_solidity": 0.80,
    "l_max": 100,
    "a_min": 15,
    "a_max": 40,
    "b_max": 35,
}

# sec 2.1.1: uoift saliency params (eq. 1, alpha=-0.7)
UOIFT_ALPHA = -0.7
UOIFT_SP_SIZE = 100

# sec 2.3: arbitration thresholds
ION_THRESH = 0.2
POOL_OVERLAP_THRESH = 0.5
KDTREE_MAX_DIST = 100
