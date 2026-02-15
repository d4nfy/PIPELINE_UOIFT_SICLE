"""sicle binary integration via wsl subprocess (sec 2.1.2)

runs the sicle c binary for superpixel segmentation.
path a: coarse diverse retrieval (nf=3000, no multiscale)
path b: multiscale precision (nf=500, 6 to 11 scales)
"""
import re
import logging
from pathlib import Path
import subprocess
import numpy as np
import imageio.v2 as imageio

from .config import SICLE_PATH_A, SICLE_PATH_B


def ensure_dir(path):
    """create directory if missing"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def to_wsl_path(path):
    """convert windows path to /mnt/c/... format"""
    import platform
    resolved = Path(path).resolve()
    if platform.system() != "Windows":
        return str(resolved)
    drive = resolved.drive.rstrip(":\\/").lower()
    relative = resolved.as_posix().split(":", 1)[-1]
    if relative.startswith("/"):
        relative = relative[1:]
    return f"/mnt/{drive}/{relative}"

log = logging.getLogger(__name__)


def find_sicle_binary():
    """locate RunSICLE binary via SICLE_BIN env var or default paths"""
    import os

    env_path = os.environ.get("SICLE_BIN")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"SICLE_BIN={env_path} does not exist")

    candidates = [
        Path(__file__).parent.parent / "SICLE" / "bin" / "RunSICLE",
        Path.home() / "SICLE" / "bin" / "RunSICLE",
        Path("/usr/local/bin/RunSICLE"),
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "sicle binary not found. options:\n"
        "  1. set SICLE_BIN=/path/to/RunSICLE\n"
        "  2. place binary at SICLE/bin/RunSICLE next to uoift_sicle/\n"
        "  3. place binary at ~/SICLE/bin/RunSICLE"
    )


def run_sicle_with_params(img_path, saliency_path, output_path, params, multiscale=False):
    """run sicle with explicit parameter dict"""
    return _run_sicle_wsl(
        img_path, saliency_path, output_path,
        n0=params["n0"], nf=params["nf"], alpha=params["alpha"],
        max_iters=params["max_iters"], irreg=params["irreg"], adhr=params["adhr"],
        sampl_opt=params.get("sampl_opt", "grid"),
        multiscale=multiscale
    )


def run_sicle_path_a(img_path, saliency_path, output_path, multiscale=False):
    """sec 2.1.2: coarse diverse retrieval (nf=3000, single scale)"""
    return run_sicle_with_params(img_path, saliency_path, output_path, SICLE_PATH_A, multiscale)


def run_sicle_path_b(img_path, saliency_path, output_path, multiscale=True):
    """sec 2.1.2: multiscale precision (nf=500, 6 to 11 scales)"""
    return run_sicle_with_params(img_path, saliency_path, output_path, SICLE_PATH_B, multiscale)


def _run_sicle_wsl(img_path, saliency_path, output_path,
                   n0, nf, alpha, max_iters, irreg, adhr, sampl_opt="grid", multiscale=False):
    """run sicle via wsl, returns labels or dict of scale labels"""
    sicle_bin = find_sicle_binary()
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    wsl_sicle = to_wsl_path(sicle_bin)
    wsl_img = to_wsl_path(Path(img_path).resolve())
    wsl_saliency = to_wsl_path(Path(saliency_path).resolve()) if saliency_path else None
    wsl_output = to_wsl_path(output_path)

    wsl_cmd = ["wsl", wsl_sicle, "--img", wsl_img]

    if wsl_saliency:
        wsl_cmd.extend(["--objsm", wsl_saliency])

    wsl_cmd.extend([
        "--out", wsl_output,
        "--n0", str(n0), "--nf", str(nf),
        "--alpha", str(alpha), "--max-iters", str(max_iters),
        "--conn-opt", "fmax", "--crit-opt", "minsc",
        "--sampl-opt", sampl_opt,
        "--irreg", str(irreg), "--adhr", str(adhr),
    ])

    if multiscale:
        wsl_cmd.append("--multiscale")

    log.info(f"running sicle (n0={n0}, nf={nf}, multiscale={multiscale})")

    max_attempts = 2
    timeout_sec = 600 if multiscale else 300

    for attempt in range(max_attempts):
        result = subprocess.run(wsl_cmd, capture_output=True, text=True, timeout=timeout_sec)
        if result.returncode == 0:
            break
        if attempt < max_attempts - 1:
            log.warning(f"sicle failed (attempt {attempt+1}), restarting wsl")
            print(f"    sicle failed, restarting wsl...")
            try:
                subprocess.run(["wsl", "--shutdown"], capture_output=True, timeout=120)
            except subprocess.TimeoutExpired:
                pass
            import time
            time.sleep(5)
        else:
            raise RuntimeError(f"sicle failed after {max_attempts} attempts: {result.stderr}")

    log.info("sicle completed")

    if multiscale:
        return _collect_multiscale(output_path)
    else:
        labels = imageio.imread(output_path)
        log.info(f"sicle: {labels.max()} regions")
        return labels


def _collect_multiscale(output_path):
    """collect numbered pgm files from multiscale run"""
    base_name = output_path.stem
    pattern = f"{base_name}_*.pgm"
    pgm_files = sorted(output_path.parent.glob(pattern))
    pgm_files = [f for f in pgm_files if re.match(f"{base_name}_\\d+\\.pgm", f.name)]

    print(f"    sicle multiscale: {len(pgm_files)} scales")

    candidates = {}
    for i, pgm_file in enumerate(pgm_files, 1):
        scale_name = f"SICLE_{i:02d}"
        labels = imageio.imread(pgm_file)
        candidates[scale_name] = labels
        log.info(f"{scale_name}: {labels.max()} regions ({pgm_file.name})")

    if not candidates:
        log.warning(f"no multiscale pgm files found: {pattern}")
        if output_path.exists():
            labels = imageio.imread(output_path)
            candidates["SICLE_01"] = labels
        else:
            raise RuntimeError(f"sicle multiscale produced no output in {output_path.parent}")

    return candidates
