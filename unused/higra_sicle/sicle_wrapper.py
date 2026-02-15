"""sicle integration via wsl subprocess - HIGRA_SICLE version (baseline)"""
import sys
from pathlib import Path
import subprocess
import numpy as np
import imageio.v2 as imageio

sys.path.append('..')
from pipeline_utils import ensure_dir, to_wsl_path


# HIGRA_SICLE BASELINE parameters
# Path A: original image (sharp)
# Path B: HIGRA simplified image (coarse)
SICLE_PARAMS = {
    "path_a": {
        # precise: original image, high seed count
        "n0": 52000,
        "nf": 3000,
        "alpha": 0.7,
        "max_iters": 22,
        "irreg": 0.12,
        "adhr": 16,
        "sampl_opt": "grid",
    },
    "path_b": {
        # coarse: HIGRA simplified image, fewer seeds (image already simplified)
        "n0": 40000,
        "nf": 4000,
        "alpha": 0.85,
        "max_iters": 12,
        "irreg": 0.18,
        "adhr": 8,
        "sampl_opt": "grid",
    },
}


def find_sicle_binary():
    """locate sicle binary in workspace"""
    workspace_sicle = Path(__file__).parent.parent / "SICLE" / "bin" / "RunSICLE"
    
    if workspace_sicle.exists():
        return workspace_sicle
    
    possible_paths = [
        Path.home() / "SICLE" / "bin" / "RunSICLE",
        Path("/usr/local/bin/RunSICLE"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError(
        f"sicle binary not found at {workspace_sicle}\n"
        "check: workspace should have SICLE/bin/RunSICLE"
    )


def run_sicle_path_a(img_path, saliency_path, output_path):
    """
    run sicle path a (precise) on ORIGINAL image
    
    args:
        img_path: input ORIGINAL rgb image
        saliency_path: objsm saliency map (from ProbMap + higra)
        output_path: output .pgm file path
    
    returns:
        labels: segmentation mask
    """
    params = SICLE_PARAMS["path_a"]
    return _run_sicle_wsl(
        img_path, saliency_path, output_path,
        n0=params["n0"], nf=params["nf"], alpha=params["alpha"],
        max_iters=params["max_iters"], irreg=params["irreg"], adhr=params["adhr"],
        sampl_opt=params["sampl_opt"]
    )


def run_sicle_path_b(img_simplified_path, saliency_path, output_path):
    """
    run sicle path b (coarse) on HIGRA SIMPLIFIED image
    
    args:
        img_simplified_path: input HIGRA SIMPLIFIED rgb image (from higra_simplify_image)
        saliency_path: objsm saliency map
        output_path: output .pgm file path
    
    returns:
        labels: segmentation mask
    """
    params = SICLE_PARAMS["path_b"]
    return _run_sicle_wsl(
        img_simplified_path, saliency_path, output_path,
        n0=params["n0"], nf=params["nf"], alpha=params["alpha"],
        max_iters=params["max_iters"], irreg=params["irreg"], adhr=params["adhr"],
        sampl_opt=params["sampl_opt"]
    )


def _run_sicle_wsl(img_path, saliency_path, output_path, 
                   n0, nf, alpha, max_iters, irreg, adhr, sampl_opt="grid"):
    """
    internal: run sicle via wsl subprocess
    
    args:
        img_path: input image
        saliency_path: objsm saliency map
        output_path: output .pgm file
        n0, nf, alpha, max_iters, irreg, adhr: sicle params
        sampl_opt: sampling option ("grid" or "rnd")
    
    returns:
        labels array
    """
    import logging
    log = logging.getLogger(__name__)
    
    sicle_bin = find_sicle_binary()
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    
    # convert paths to wsl format
    wsl_sicle = to_wsl_path(sicle_bin)
    wsl_img = to_wsl_path(Path(img_path).resolve())
    wsl_saliency = to_wsl_path(Path(saliency_path).resolve())
    wsl_output = to_wsl_path(output_path)
    
    # build wsl command
    wsl_cmd = [
        "wsl", wsl_sicle,
        "--img", wsl_img,
        "--objsm", wsl_saliency,
        "--out", wsl_output,
        "--n0", str(n0),
        "--nf", str(nf),
        "--alpha", str(alpha),
        "--max-iters", str(max_iters),
        "--conn-opt", "fmax",
        "--crit-opt", "minsc",
        "--sampl-opt", sampl_opt,
        "--irreg", str(irreg),
        "--adhr", str(adhr),
    ]
    
    log.info(f"running sicle via wsl (n0={n0}, nf={nf})...")
    log.debug(f"wsl command: {' '.join(wsl_cmd)}")
    
    result = subprocess.run(wsl_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"sicle failed: {result.stderr}")
    
    log.info("sicle completed")
    
    labels = imageio.imread(output_path)
    log.info(f"sicle output: {labels.max()} regions")
    return labels
