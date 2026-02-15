"""uoift c++ binary integration via wsl (sec 3.2)

runs unsupseg_mergehistory to generate saliency maps
with boundary polarity (eq. 1, alpha=-0.7 for dark nuclei).
"""
import time
import shutil
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from skimage.color import rgb2gray

from .pipeline_utils import to_wsl_path


def find_uoift_binary():
    """locate unsupseg_mergehistory via UOIFT_BIN env var or default path"""
    import os

    env_path = os.environ.get("UOIFT_BIN")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"UOIFT_BIN={env_path} does not exist")

    binary = Path(__file__).parent.parent / "references" / "unsupseg" / "unsupseg" / "unsupseg_mergehistory"
    if not binary.exists():
        raise FileNotFoundError(
            "uoift binary not found. options:\n"
            "  1. set UOIFT_BIN=/path/to/unsupseg_mergehistory\n"
            "  2. compile: cd references/unsupseg/unsupseg && make"
        )
    return binary.resolve()


def run_uoift_wsl(img_rgb, polarity=0.0, method=0, sp_size=100, output_dir=None):
    """run uoift via wsl, returns normalized saliency map"""
    binary = find_uoift_binary()

    if output_dir is None:
        temp_dir = Path(tempfile.mkdtemp())
        cleanup = True
    else:
        temp_dir = Path(output_dir)
        temp_dir.mkdir(exist_ok=True, parents=True)
        cleanup = False

    try:
        img_gray = rgb2gray(img_rgb)
        img_gray_uint8 = (img_gray * 255).astype(np.uint8)
        input_pgm = temp_dir / "input.pgm"
        imageio.imsave(input_pgm, img_gray_uint8)

        wsl_binary = to_wsl_path(binary)
        wsl_input = to_wsl_path(input_pgm)
        wsl_temp_dir = to_wsl_path(temp_dir)

        cmd_str = f"cd {wsl_temp_dir} && {wsl_binary} 0 {wsl_input} {method} {polarity} {sp_size}"
        wsl_cmd = ["wsl", "bash", "-c", cmd_str]

        result = subprocess.run(wsl_cmd, capture_output=True, text=True)
        time.sleep(0.5)

        if result.returncode != 0:
            print(f"uoift stderr: {result.stderr}")
            raise RuntimeError(f"uoift failed with code {result.returncode}")

        saliency_path = temp_dir / "saliencymap.pgm"
        if not saliency_path.exists():
            saliency_path = temp_dir / "spixels.pgm"
        if not saliency_path.exists():
            raise FileNotFoundError(f"no uoift output in {temp_dir}")

        saliency = imageio.imread(saliency_path)

        if saliency.max() > 0:
            saliency = ((saliency - saliency.min()) / (saliency.max() - saliency.min()) * 255).astype(np.uint8)

        return saliency

    finally:
        if cleanup:
            time.sleep(1)
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
