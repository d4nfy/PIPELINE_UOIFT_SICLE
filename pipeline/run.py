#!/usr/bin/env python3
"""benchmark 4 nuclei segmentation methods on oral histology dataset

methods: stardist, cellpose, boundary_aware (with saliency), modular (no saliency)
generates vis_01-vis_09 + comparisons + timing json per image
"""

import os
import sys
import json
import time
import argparse
import colorsys
import subprocess
import warnings
import functools
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage import io as skio
from skimage.color import rgb2lab, label2rgb
from skimage.measure import regionprops
from skimage.segmentation import slic, find_boundaries
from scipy.spatial import cKDTree
import scipy.ndimage as ndi
import cv2

print = functools.partial(print, flush=True)

# PATHS
PIPELINE_DIR = Path(__file__).resolve().parent
REPO_DIR = PIPELINE_DIR.parent
PROJECT_ROOT = REPO_DIR.parent
SICLE_BIN = REPO_DIR / "SICLE" / "bin" / "RunSICLE"
PROCESSED_DIR = PROJECT_ROOT / "new_images_processed"
RESULTS_DIR = PROJECT_ROOT / "new_images_results"
METADATA_PATH = PROCESSED_DIR / "metadata.json"

# CONFIG (from core/config.py)
SICLE_PATH_A = {
    "n0": 52000, "nf": 3000, "alpha": 0.9,
    "max_iters": 22, "irreg": 0.12, "adhr": 16, "sampl_opt": "grid",
}
SICLE_PATH_B = {
    "n0": 52000, "nf": 500, "alpha": 0.85,
    "max_iters": 12, "irreg": 0.15, "adhr": 12, "sampl_opt": "grid",
}
FILTER_CFG = {
    "min_area": 100, "max_area": 10000, "min_solidity": 0.80,
    "l_max": 100, "a_min": 15, "a_max": 40, "b_max": 35,
}
UOIFT_ALPHA = -0.7
UOIFT_SP_SIZE = 100
ION_THRESH = 0.2
POOL_OVERLAP_THRESH = 0.5
KDTREE_MAX_DIST = 100

VIS_HUES = {
    "path_a": 0.33, "path_b": 0.50, "stardist": 0.75,
    "pool_a": 0.33, "pool_b": 0.50,
}


# TIMER
class Timer:
    def __init__(self, name):
        self.name = name
        self.elapsed = 0
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"    {self.name}: {self.elapsed:.2f}s")


# LAB UTILS
def get_lab_channels(img):
    """extract l, a, b channels from rgb"""
    lab = rgb2lab(img)
    return lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]


# SICLE WRAPPER (macOS native - no WSL)
def _to_ppm(img_path, tmp_dir):
    """Convert PNG to PPM for SICLE (compiled without libpng)"""
    img = Image.open(img_path).convert('RGB')
    ppm_path = Path(tmp_dir) / "input.ppm"
    img.save(str(ppm_path))
    return str(ppm_path)


def _sal_to_pgm(saliency_path, tmp_dir):
    """Convert saliency PNG to PGM for SICLE"""
    img = Image.open(saliency_path).convert('L')
    pgm_path = Path(tmp_dir) / "saliency.pgm"
    arr = np.array(img)
    h, w = arr.shape
    with open(pgm_path, 'wb') as f:
        f.write(f"P5\n{w} {h}\n255\n".encode('ascii'))
        f.write(arr.astype(np.uint8).tobytes())
    return str(pgm_path)


def run_sicle(img_path, saliency_path, output_path, params, multiscale=False):
    """Run SICLE binary directly on macOS (converts to PPM first)"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to PPM (SICLE compiled without libpng)
    tmp_dir = output_path.parent
    ppm_path = _to_ppm(img_path, tmp_dir)
    sal_pgm = None
    if saliency_path and Path(saliency_path).exists():
        sal_pgm = _sal_to_pgm(saliency_path, tmp_dir)

    cmd = [
        str(SICLE_BIN),
        "--img", ppm_path,
        "--out", str(output_path),
        "--n0", str(params["n0"]),
        "--nf", str(params["nf"]),
        "--alpha", str(params["alpha"]),
        "--max-iters", str(params["max_iters"]),
        "--conn-opt", "fmax",
        "--crit-opt", "minsc",
        "--sampl-opt", params.get("sampl_opt", "grid"),
        "--irreg", str(params["irreg"]),
        "--adhr", str(params["adhr"]),
    ]

    if sal_pgm:
        cmd.extend(["--objsm", sal_pgm])

    if multiscale:
        cmd.append("--multiscale")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"SICLE failed: {result.stderr}")

    if multiscale:
        return _collect_multiscale(output_path)
    else:
        return _read_pgm(output_path)


def _read_pgm(path):
    """Read PGM label file (P2=ASCII, P5=binary)"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PGM not found: {path}")
    with open(path, 'rb') as f:
        magic = f.readline().decode().strip()
        if magic not in ('P5', 'P2'):
            raise ValueError(f"Invalid PGM: {magic}")
        line = f.readline().decode()
        while line.startswith('#'):
            line = f.readline().decode()
        w, h = map(int, line.strip().split())
        maxval = int(f.readline().decode().strip())

        if magic == 'P2':
            # ASCII format: values as text
            raw_text = f.read().decode()
            values = np.array(raw_text.split(), dtype=np.int32)
            data = values.reshape((h, w))
        else:
            # Binary format
            raw = f.read()
            n_pixels = h * w
            if maxval < 256:
                data = np.frombuffer(raw[:n_pixels], dtype=np.uint8).reshape((h, w))
            else:
                data = np.frombuffer(raw[:n_pixels * 2], dtype='>u2').reshape((h, w))
    return data.astype(np.int32)


def _collect_multiscale(output_path):
    """Collect numbered pgm files from multiscale run"""
    import re
    base_name = output_path.stem
    pattern = f"{base_name}_*.pgm"
    pgm_files = sorted(output_path.parent.glob(pattern))
    pgm_files = [f for f in pgm_files if re.match(rf"{re.escape(base_name)}_\d+\.pgm", f.name)]

    candidates = {}
    for i, pgm_file in enumerate(pgm_files, 1):
        scale_name = f"SICLE_{i:02d}"
        labels = _read_pgm(pgm_file)
        candidates[scale_name] = labels

    if not candidates and output_path.exists():
        candidates["SICLE_01"] = _read_pgm(output_path)

    if not candidates:
        raise RuntimeError(f"SICLE multiscale produced no output in {output_path.parent}")

    return candidates


# saliency from light module
from core.uoift_saliency_light import generate as generate_uoift_saliency


# VETA FILTERING (from core/veta_filtering.py)
def filter_sicle_regions(labels, l_ch, a_ch, b_ch, cfg=None):
    """filter regions by lab color and morphology (sec 2.2)"""
    if cfg is None:
        cfg = FILTER_CFG
    labels = labels.astype(np.int32)
    if labels.max() == 0:
        return labels, 0, 0
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    n_total = len(unique_labels)
    if n_total == 0:
        return np.zeros_like(labels), 0, 0

    mean_l = ndi.mean(l_ch, labels, unique_labels)
    mean_a = ndi.mean(a_ch, labels, unique_labels)
    mean_b = ndi.mean(b_ch, labels, unique_labels)
    keep_mask = (mean_l <= cfg["l_max"]) & (mean_a >= cfg["a_min"]) & (mean_b <= cfg["b_max"])
    if "a_max" in cfg:
        keep_mask &= (mean_a <= cfg["a_max"])
    labels_after_lab = unique_labels[keep_mask]
    if len(labels_after_lab) == 0:
        return np.zeros_like(labels), 0, n_total

    temp_mask = np.isin(labels, labels_after_lab)
    temp_labels = labels * temp_mask
    areas = ndi.sum(np.ones_like(labels), temp_labels, labels_after_lab)
    area_mask = (areas >= cfg["min_area"]) & (areas <= cfg["max_area"])
    labels_after_area = labels_after_lab[area_mask]
    if len(labels_after_area) == 0:
        return np.zeros_like(labels), 0, n_total

    temp_mask2 = np.isin(temp_labels, labels_after_area)
    temp_labels2 = temp_labels * temp_mask2
    labels_to_keep = set()
    for region in regionprops(temp_labels2):
        if region.solidity >= cfg["min_solidity"]:
            labels_to_keep.add(region.label)
    n_kept = len(labels_to_keep)
    if n_kept == 0:
        return np.zeros_like(labels), 0, n_total

    lut = np.zeros(labels.max() + 1, dtype=np.int32)
    new_id = 1
    for lbl in sorted(labels_to_keep):
        lut[lbl] = new_id
        new_id += 1
    return lut[labels], n_kept, n_total - n_kept


def veta_multiscale_selection(scales_dict, l_ch, a_ch, b_ch, min_solidity=0.80, ov_thresh=0.2):
    """select best nuclei across scales via solidity (sec 2.2)"""
    filtered_scales = {}
    for scale_name, scale_mask in scales_dict.items():
        filtered, n_kept, _ = filter_sicle_regions(scale_mask, l_ch, a_ch, b_ch)
        if n_kept > 0:
            filtered_scales[scale_name] = filtered
    if len(filtered_scales) == 0:
        h, w = list(scales_dict.values())[0].shape
        return np.zeros((h, w), dtype=np.int32)

    candidates = []
    for scale_name, scale_mask in filtered_scales.items():
        for region in regionprops(scale_mask):
            if region.solidity >= min_solidity:
                candidates.append({
                    'id': len(candidates), 'label': region.label,
                    'scale': scale_name, 'mask': scale_mask == region.label,
                    'bbox': region.bbox, 'centroid': region.centroid,
                    'area': region.area, 'solidity': region.solidity,
                })
    if len(candidates) == 0:
        h, w = list(scales_dict.values())[0].shape
        return np.zeros((h, w), dtype=np.int32)

    centroids = np.array([c['centroid'] for c in candidates])
    tree = cKDTree(centroids)
    conflicts = {}
    for i, cand in enumerate(candidates):
        nearby = tree.query_ball_point(cand['centroid'], r=KDTREE_MAX_DIST)
        for j in nearby:
            if i >= j:
                continue
            other = candidates[j]
            r1, c1, r2, c2 = cand['bbox']
            r1o, c1o, r2o, c2o = other['bbox']
            if r2 < r1o or r2o < r1 or c2 < c1o or c2o < c1:
                continue
            intersection = (cand['mask'] & other['mask']).sum()
            if intersection == 0:
                continue
            min_area = min(cand['area'], other['area'])
            ov = intersection / min_area
            if ov > ov_thresh:
                conflicts.setdefault(i, set()).add(j)
                conflicts.setdefault(j, set()).add(i)

    candidates.sort(key=lambda x: x['solidity'], reverse=True)
    accepted = []
    accepted_ids = set()
    for cand in candidates:
        cand_id = cand['id']
        if not (conflicts.get(cand_id, set()) & accepted_ids):
            accepted.append(cand)
            accepted_ids.add(cand_id)

    h, w = list(scales_dict.values())[0].shape
    merged = np.zeros((h, w), dtype=np.int32)
    for idx, cand in enumerate(accepted, 1):
        merged[cand['mask']] = idx
    return merged


def create_candidate_pool(path_a, path_b_scales):
    """merge path a and path b into candidate pool (sec 2.1.2)"""
    path_a = path_a.astype(np.int32)
    pool = np.zeros_like(path_a)
    unique_a = np.unique(path_a)
    unique_a = unique_a[unique_a > 0]
    next_id = 1
    for lbl in unique_a:
        pool[path_a == lbl] = next_id
        next_id += 1
    n_from_a = next_id - 1
    existing_mask = pool > 0
    n_from_b = 0
    for scale_name, scale_mask in path_b_scales.items():
        scale_mask = scale_mask.astype(np.int32)
        if scale_mask.max() == 0:
            continue
        for lbl in np.unique(scale_mask):
            if lbl == 0:
                continue
            region_mask = scale_mask == lbl
            region_size = region_mask.sum()
            if region_size == 0:
                continue
            overlap = (existing_mask & region_mask).sum()
            if overlap / region_size < POOL_OVERLAP_THRESH:
                new_pixels = region_mask & (~existing_mask)
                if new_pixels.sum() > 0:
                    pool[new_pixels] = next_id
                    existing_mask |= region_mask
                    next_id += 1
                    n_from_b += 1
    return pool, n_from_a, n_from_b


def fuse_stardist_pool(stardist, pool, iou_thresh=0.2):
    """fuse stardist detections with sicle pool (sec 2.3)"""
    stardist = stardist.astype(np.int32)
    pool = pool.astype(np.int32)
    fused = np.zeros_like(stardist)
    next_id = 1
    props_sd = {r.label: r for r in regionprops(stardist)}
    props_pool = {r.label: r for r in regionprops(pool)}
    matched_pool = set()
    for lid_sd, prop_sd in props_sd.items():
        mask_sd = stardist == lid_sd
        fused[mask_sd] = next_id
        for lid_pool, prop_pool in props_pool.items():
            if lid_pool in matched_pool:
                continue
            r1, c1, r2, c2 = prop_sd.bbox
            r3, c3, r4, c4 = prop_pool.bbox
            if r2 < r3 or r4 < r1 or c2 < c3 or c4 < c1:
                continue
            mask_pool = pool == lid_pool
            inter = (mask_sd & mask_pool).sum()
            union = (mask_sd | mask_pool).sum()
            iou = inter / union if union > 0 else 0
            if iou > iou_thresh:
                matched_pool.add(lid_pool)
        next_id += 1
    for lid_pool, prop_pool in props_pool.items():
        if lid_pool not in matched_pool:
            mask_pool = pool == lid_pool
            empty_pixels = (fused == 0) & mask_pool
            if empty_pixels.sum() > 0.5 * prop_pool.area:
                fused[mask_pool & (fused == 0)] = next_id
                next_id += 1
    return fused


def arbitrate_by_solidity(fused, min_area=100, min_solidity=0.80):
    """final arbitration by solidity and area (sec 2.3)"""
    fused = fused.astype(np.int32)
    arbitrated = np.zeros_like(fused)
    next_id = 1
    kept = 0
    rejected = 0
    for region in regionprops(fused):
        if region.area < min_area:
            rejected += 1
            continue
        if region.solidity >= min_solidity:
            arbitrated[fused == region.label] = next_id
            next_id += 1
            kept += 1
        else:
            rejected += 1
    return arbitrated, kept, rejected


# STARDIST
_stardist_model = None

def load_stardist():
    """load pretrained stardist 2d_versatile_he model"""
    global _stardist_model
    if _stardist_model is None:
        from stardist.models import StarDist2D
        _stardist_model = StarDist2D.from_pretrained('2D_versatile_he')
    return _stardist_model

def run_stardist(img_rgb, prob_thresh=0.5, nms_thresh=0.4):
    """run stardist prediction on one image"""
    model = load_stardist()
    from csbdeep.utils import normalize
    img_norm = normalize(img_rgb, 1, 99.8, axis=(0, 1))
    n_tiles = (2, 2, 1) if img_rgb.shape[0] > 512 else None
    labels, _ = model.predict_instances(img_norm, prob_thresh=prob_thresh, nms_thresh=nms_thresh, n_tiles=n_tiles)
    return labels


# CELLPOSE
def run_cellpose(img_rgb):
    """run cellpose sam model on one image"""
    from cellpose.models import CellposeModel
    # Cellpose 4.0+ universal SAM-based model (cpsam)
    model = CellposeModel(gpu=False, pretrained_model='cpsam')
    # eval() returns 3 values for single image: (masks, flows, styles)
    masks, _, _ = model.eval(img_rgb, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0)
    return masks


# VISUALIZATION
def to_gray(img_rgb):
    """convert rgb to grayscale uint8"""
    return np.mean(img_rgb, axis=2).astype(np.uint8)


def labels_to_monochrome(labels, bg_gray, hue, sat=0.8, val=0.9, alpha=0.7):
    """overlay label regions with single hue"""
    viz = np.stack([bg_gray, bg_gray, bg_gray], axis=-1).astype(np.float32)
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    color = np.array([r * 255, g * 255, b * 255])
    mask = labels > 0
    for c in range(3):
        viz[:, :, c][mask] = (1 - alpha) * viz[:, :, c][mask] + alpha * color[c]
    bounds = find_boundaries(labels, mode='thick')
    viz[bounds] = [255, 255, 255]
    return np.clip(viz, 0, 255).astype(np.uint8)


def labels_to_multicolor(labels, bg_gray, alpha=0.7):
    """overlay label regions with distinct colors"""
    viz = np.stack([bg_gray, bg_gray, bg_gray], axis=-1).astype(np.float32)
    unique = np.unique(labels)
    unique = unique[unique != 0]
    if len(unique) == 0:
        return viz.astype(np.uint8)
    rng = np.random.RandomState(42)
    n = len(unique)
    colors = [np.array([r * 255, g * 255, b * 255])
              for r, g, b in [colorsys.hsv_to_rgb(i / n, 0.8, 0.9) for i in range(n)]]
    rng.shuffle(colors)
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        color = colors[i % n]
        for c in range(3):
            viz[:, :, c][mask] = (1 - alpha) * viz[:, :, c][mask] + alpha * color[c]
    bounds = find_boundaries(labels, mode='thick')
    viz[bounds] = [255, 255, 255]
    return np.clip(viz, 0, 255).astype(np.uint8)


def viz_pool_bicolor(pool, path_a, path_b, bg_gray, alpha=0.7):
    """show pool with path a/b color coding"""
    viz = np.stack([bg_gray, bg_gray, bg_gray], axis=-1).astype(np.float32)
    r_a, g_a, b_a = colorsys.hsv_to_rgb(VIS_HUES["pool_a"], 0.8, 0.9)
    color_a = np.array([r_a * 255, g_a * 255, b_a * 255])
    r_b, g_b, b_b = colorsys.hsv_to_rgb(VIS_HUES["pool_b"], 0.8, 0.9)
    color_b = np.array([r_b * 255, g_b * 255, b_b * 255])
    for lbl in set(np.unique(pool)) - {0}:
        mask = pool == lbl
        ov_a = np.sum((path_a > 0) & mask)
        ov_b = np.sum((path_b > 0) & mask)
        color = color_a if ov_a > ov_b else color_b
        for c in range(3):
            viz[:, :, c][mask] = (1 - alpha) * viz[:, :, c][mask] + alpha * color[c]
    bounds = find_boundaries(pool, mode='thick')
    viz[bounds] = [255, 255, 255]
    return np.clip(viz, 0, 255).astype(np.uint8)


def viz_fusion_decisions(pool, stardist, final, bg_gray):
    """visualize fusion kept/rejected decisions"""
    h, w = pool.shape
    viz = np.stack([bg_gray, bg_gray, bg_gray], axis=-1).astype(np.float32)
    C_KEPT = np.array([0, 200, 100])
    C_STAR = (100, 150, 255)
    C_REJ = np.array([200, 80, 80])
    alpha = 0.7
    for lbl in set(np.unique(pool)) - {0}:
        mask = pool == lbl
        overlap = np.sum((final > 0) & mask) / max(np.sum(mask), 1)
        color = C_KEPT if overlap > 0.5 else C_REJ
        for c in range(3):
            viz[:, :, c][mask] = (1 - alpha) * viz[:, :, c][mask] + alpha * color[c]
    for lbl in set(np.unique(stardist)) - {0}:
        mask = stardist == lbl
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlap = np.sum((final > 0) & mask) / max(np.sum(mask), 1)
        color = C_STAR if overlap > 0.5 else (150, 150, 150)
        thickness = 2 if overlap > 0.5 else 1
        viz_u8 = np.clip(viz, 0, 255).astype(np.uint8)
        cv2.drawContours(viz_u8, contours, -1, color, thickness)
        viz = viz_u8.astype(np.float32)
    legend = np.ones((80, w, 3), dtype=np.uint8) * 40
    cv2.putText(legend, "Fusion Decisions:", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(legend, (10, 40), (30, 60), tuple(int(x) for x in C_KEPT), -1)
    cv2.putText(legend, "SICLE kept", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(legend, (200, 40), (220, 60), C_STAR, -1)
    cv2.putText(legend, "StarDist kept", (230, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(legend, (400, 40), (420, 60), tuple(int(x) for x in C_REJ), -1)
    cv2.putText(legend, "Rejected", (430, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.vstack([np.clip(viz, 0, 255).astype(np.uint8), legend])


def create_comparison_grid(images, titles, cols=3):
    """create tiled comparison grid of visualizations"""
    if len(images) == 0:
        return None
    target_h = 400
    resized = []
    for img in images:
        h, w_ = img.shape[:2]
        scale = target_h / h
        resized.append(cv2.resize(img, (int(w_ * scale), target_h)))
    max_w = max(img.shape[1] for img in resized)
    padded = []
    for img in resized:
        h, w_ = img.shape[:2]
        if w_ < max_w:
            img = np.hstack([img, np.zeros((h, max_w - w_, 3), dtype=np.uint8)])
        padded.append(img)
    titled = []
    for img, title in zip(padded, titles):
        bar = np.zeros((30, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        titled.append(np.vstack([bar, img]))
    rows = []
    for i in range(0, len(titled), cols):
        row_imgs = titled[i:i + cols]
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros_like(row_imgs[0]))
        rows.append(np.hstack(row_imgs))
    return np.vstack(rows)


def save_all_visualizations(out_dir, img_rgb, intermediates, has_saliency=True):
    """Generate vis_01 through vis_09 + comparison"""
    bg = to_gray(img_rgb)
    counts = {}

    # vis_01: Input
    Image.fromarray(img_rgb).save(out_dir / "vis_01_input.png")

    # vis_02: Path A filtered (green)
    if 'path_a_filtered' in intermediates:
        n = len(np.unique(intermediates['path_a_filtered'])) - 1
        counts['path_a'] = n
        viz = labels_to_monochrome(intermediates['path_a_filtered'], bg, VIS_HUES["path_a"])
        Image.fromarray(viz).save(out_dir / "vis_02_path_a_filtered.png")

    # vis_03: Path B veta (cyan)
    if 'path_b_veta' in intermediates:
        n = len(np.unique(intermediates['path_b_veta'])) - 1
        counts['path_b'] = n
        viz = labels_to_monochrome(intermediates['path_b_veta'], bg, VIS_HUES["path_b"])
        Image.fromarray(viz).save(out_dir / "vis_03_path_b_veta.png")

    # vis_04: Pool bicolor
    if all(k in intermediates for k in ['pool', 'path_a_filtered', 'path_b_veta']):
        n = len(np.unique(intermediates['pool'])) - 1
        counts['pool'] = n
        viz = viz_pool_bicolor(intermediates['pool'], intermediates['path_a_filtered'],
                               intermediates['path_b_veta'], bg)
        Image.fromarray(viz).save(out_dir / "vis_04_pool.png")

    # vis_05: StarDist (violet)
    if 'stardist' in intermediates:
        n = len(np.unique(intermediates['stardist'])) - 1
        counts['stardist'] = n
        viz = labels_to_monochrome(intermediates['stardist'], bg, VIS_HUES["stardist"])
        Image.fromarray(viz).save(out_dir / "vis_05_stardist.png")

    # vis_06: Fused (multicolor)
    if 'fused' in intermediates:
        n = len(np.unique(intermediates['fused'])) - 1
        counts['fused'] = n
        viz = labels_to_multicolor(intermediates['fused'], bg)
        Image.fromarray(viz).save(out_dir / "vis_06_fused.png")

    # vis_07: Final (multicolor)
    if 'final' in intermediates:
        n = len(np.unique(intermediates['final'])) - 1
        counts['final'] = n
        viz = labels_to_multicolor(intermediates['final'], bg)
        Image.fromarray(viz).save(out_dir / "vis_07_final.png")

    # vis_08: Overlay on RGB
    if 'final' in intermediates:
        overlay = label2rgb(intermediates['final'], image=img_rgb, bg_label=0, alpha=0.3)
        Image.fromarray((overlay * 255).astype(np.uint8)).save(out_dir / "vis_08_overlay.png")

    # vis_09: Fusion decisions
    if all(k in intermediates for k in ['pool', 'stardist', 'final']):
        viz = viz_fusion_decisions(intermediates['pool'], intermediates['stardist'],
                                   intermediates['final'], bg)
        Image.fromarray(viz).save(out_dir / "vis_09_fusion_decisions.png")

    # Comparison grid
    grid_imgs = []
    grid_titles = []
    for vis_file, title in [
        ("vis_01_input.png", "Input"),
        ("vis_02_path_a_filtered.png", f"Path A ({counts.get('path_a', '?')})"),
        ("vis_03_path_b_veta.png", f"Path B ({counts.get('path_b', '?')})"),
        ("vis_04_pool.png", f"Pool ({counts.get('pool', '?')})"),
        ("vis_05_stardist.png", f"StarDist ({counts.get('stardist', '?')})"),
        ("vis_07_final.png", f"Final ({counts.get('final', '?')})"),
    ]:
        p = out_dir / vis_file
        if p.exists():
            grid_imgs.append(np.array(Image.open(p)))
            grid_titles.append(title)

    if len(grid_imgs) >= 3:
        grid = create_comparison_grid(grid_imgs, grid_titles)
        if grid is not None:
            Image.fromarray(grid).save(out_dir / "vis_comparison.png")

    return counts


# PIPELINE RUNNER
def run_full_pipeline(img_rgb, img_path, out_dir, with_saliency=True):
    """Run the complete pipeline on one image, return report dict"""
    out_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    stats = {}
    intermediates = {}

    l_ch, a_ch, b_ch = get_lab_channels(img_rgb)

    # Step 1: Saliency
    saliency_path = None
    if with_saliency:
        with Timer("uoift saliency") as t:
            sal = generate_uoift_saliency(img_rgb)
        timings['saliency'] = t.elapsed
        np.save(out_dir / "01_saliency.npy", sal)
        saliency_png = out_dir / "01_saliency.png"
        Image.fromarray(sal).save(saliency_png)
        saliency_path = str(saliency_png)
        intermediates['saliency'] = sal
    else:
        print("    saliency: skipped")

    # Step 2: SICLE Path A
    with Timer("sicle path a") as t:
        path_a_raw = run_sicle(str(img_path), saliency_path,
                               str(out_dir / "sicle_a.pgm"), SICLE_PATH_A, multiscale=False)
    timings['sicle_path_a'] = t.elapsed
    n_raw_a = len(np.unique(path_a_raw)) - 1
    stats['path_a_raw'] = n_raw_a
    print(f"      path a: {n_raw_a} raw regions")

    # Step 3: Filter A
    with Timer("filter path a") as t:
        path_a_filtered, n_kept_a, n_rej_a = filter_sicle_regions(path_a_raw, l_ch, a_ch, b_ch)
    timings['filter_a'] = t.elapsed
    stats['path_a_filtered'] = n_kept_a
    intermediates['path_a_filtered'] = path_a_filtered
    print(f"      path a filtered: {n_kept_a} kept, {n_rej_a} rejected")

    # Step 4: SICLE Path B (multiscale)
    with Timer("sicle path b (multiscale)") as t:
        scales = run_sicle(str(img_path), saliency_path,
                           str(out_dir / "sicle_b.pgm"), SICLE_PATH_B, multiscale=True)
    timings['sicle_path_b'] = t.elapsed
    stats['path_b_scales'] = len(scales)
    print(f"      path b: {len(scales)} scales")

    # Step 5: Filter B (veta multiscale)
    with Timer("veta multiscale selection") as t:
        path_b_veta = veta_multiscale_selection(scales, l_ch, a_ch, b_ch)
    timings['filter_b'] = t.elapsed
    n_b = len(np.unique(path_b_veta)) - 1
    stats['path_b_veta'] = n_b
    intermediates['path_b_veta'] = path_b_veta
    print(f"      path b veta: {n_b} selected")

    # Step 6: Pool
    with Timer("candidate pool") as t:
        pool, n_from_a, n_from_b = create_candidate_pool(path_a_filtered, {"veta": path_b_veta})
    timings['pool'] = t.elapsed
    n_pool = len(np.unique(pool)) - 1
    stats['pool'] = n_pool
    stats['pool_from_a'] = n_from_a
    stats['pool_from_b'] = n_from_b
    intermediates['pool'] = pool

    # Step 7: StarDist
    with Timer("stardist") as t:
        sd = run_stardist(img_rgb)
    timings['stardist'] = t.elapsed
    n_sd = len(np.unique(sd)) - 1
    stats['stardist'] = n_sd
    intermediates['stardist'] = sd
    print(f"      stardist: {n_sd} nuclei")

    # Step 8: Fusion
    with Timer("stardist-pool fusion") as t:
        fused = fuse_stardist_pool(sd, pool, iou_thresh=ION_THRESH)
    timings['fusion'] = t.elapsed
    n_fused = len(np.unique(fused)) - 1
    stats['fused'] = n_fused
    intermediates['fused'] = fused

    # Step 9: Arbitration
    with Timer("solidity arbitration") as t:
        final, n_final, n_rej_final = arbitrate_by_solidity(fused)
    timings['arbitration'] = t.elapsed
    stats['final'] = n_final
    intermediates['final'] = final
    print(f"      final: {n_final} nuclei")

    # Save intermediates
    for key in ['path_a_filtered', 'path_b_veta', 'pool', 'stardist', 'fused', 'final']:
        if key in intermediates:
            np.save(out_dir / f"{key}.npy", intermediates[key])

    # Visualizations
    with Timer("visualizations") as t:
        save_all_visualizations(out_dir, img_rgb, intermediates, has_saliency=with_saliency)
    timings['visualization'] = t.elapsed

    timings['total'] = sum(v for k, v in timings.items() if k != 'visualization')

    report = {
        'timings': timings,
        'stats': stats,
        'with_saliency': with_saliency,
    }
    return report


# BENCHMARK: STARDIST ALONE
def run_stardist_alone(img_rgb, out_dir):
    """Run StarDist 2D_versatile_he alone (no SICLE, no fusion)"""
    out_dir.mkdir(parents=True, exist_ok=True)

    with Timer("stardist") as t:
        masks = run_stardist(img_rgb)
    n_nuclei = len(np.unique(masks)) - 1

    bg = to_gray(img_rgb)

    # Standard visualizations for single-method benchmarks
    # vis_seg.png: blue contours on gray background
    viz_contours = np.stack([bg, bg, bg], axis=-1).astype(np.uint8)
    bounds = find_boundaries(masks, mode='thick')
    BLUE_COLOR = (100, 150, 255)  # RGB blue (same as C_STAR in vis_10)
    viz_contours[bounds] = BLUE_COLOR
    Image.fromarray(viz_contours).save(out_dir / "vis_seg.png")

    # vis_seg_filled.png: blue-filled nuclei on gray background
    viz_filled = np.stack([bg, bg, bg], axis=-1).astype(np.float32)
    BLUE_FILL = np.array([100, 150, 255])
    alpha = 0.7
    mask_all = masks > 0
    for c in range(3):
        viz_filled[:, :, c][mask_all] = (1 - alpha) * viz_filled[:, :, c][mask_all] + alpha * BLUE_FILL[c]
    Image.fromarray(np.clip(viz_filled, 0, 255).astype(np.uint8)).save(out_dir / "vis_seg_filled.png")

    # Save masks
    np.save(out_dir / "stardist_masks.npy", masks)

    return {
        'timings': {'stardist': t.elapsed, 'total': t.elapsed},
        'stats': {'nuclei': n_nuclei},
    }


# BENCHMARK: CELLPOSE ALONE
def run_cellpose_alone(img_rgb, out_dir):
    """Run Cellpose nuclei model alone"""
    out_dir.mkdir(parents=True, exist_ok=True)

    with Timer("cellpose") as t:
        masks = run_cellpose(img_rgb)
    n_nuclei = len(np.unique(masks)) - 1

    bg = to_gray(img_rgb)

    # Standard visualizations for single-method benchmarks
    # vis_seg.png: blue contours on gray background
    viz_contours = np.stack([bg, bg, bg], axis=-1).astype(np.uint8)
    bounds = find_boundaries(masks, mode='thick')
    BLUE_COLOR = (100, 150, 255)  # RGB blue (same as C_STAR in vis_10)
    viz_contours[bounds] = BLUE_COLOR
    Image.fromarray(viz_contours).save(out_dir / "vis_seg.png")

    # vis_seg_filled.png: blue-filled nuclei on gray background
    viz_filled = np.stack([bg, bg, bg], axis=-1).astype(np.float32)
    BLUE_FILL = np.array([100, 150, 255])
    alpha = 0.7
    mask_all = masks > 0
    for c in range(3):
        viz_filled[:, :, c][mask_all] = (1 - alpha) * viz_filled[:, :, c][mask_all] + alpha * BLUE_FILL[c]
    Image.fromarray(np.clip(viz_filled, 0, 255).astype(np.uint8)).save(out_dir / "vis_seg_filled.png")

    # Save masks
    np.save(out_dir / "cellpose_masks.npy", masks)

    return {
        'timings': {'cellpose': t.elapsed, 'total': t.elapsed},
        'stats': {'nuclei': n_nuclei},
    }


# MAIN
def load_image_list():
    """load image list from metadata json"""
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return metadata['images'], metadata['categories']


def sanitize_name(category, stem):
    """create filesystem-safe image name"""
    return f"{category.replace(' ', '_')}_{stem}"


METHODS = {
    'stardist': ('benchmark_stardist', 'stardist_alone'),
    'cellpose': ('benchmark_cellpose', 'cellpose_alone'),
    'boundary_aware': ('pipeline_with_saliency', 'pipeline_with_saliency'),
    'modular': ('pipeline_no_saliency', 'pipeline_no_saliency'),
}


def run_method(method, img_rgb, png_path, name, category):
    """run a single method on one image, return report"""
    subdir, key = METHODS[method]
    out = RESULTS_DIR / subdir / name
    if method == 'stardist':
        report = run_stardist_alone(img_rgb, out)
    elif method == 'cellpose':
        report = run_cellpose_alone(img_rgb, out)
    elif method == 'boundary_aware':
        report = run_full_pipeline(img_rgb, png_path, out, with_saliency=True)
    elif method == 'modular':
        report = run_full_pipeline(img_rgb, png_path, out, with_saliency=False)
    report['category'] = category
    report['image'] = name
    with open(out / "report.json", 'w') as f:
        json.dump(report, f, indent=2)
    return report


def main():
    """parse args and run benchmark on dataset"""
    parser = argparse.ArgumentParser(description="nuclei segmentation benchmark")
    parser.add_argument('--method', choices=['stardist', 'cellpose', 'boundary_aware',
                                             'modular', 'all'], default='all',
                        help='method to run (default: all)')
    args = parser.parse_args()

    methods = list(METHODS.keys()) if args.method == 'all' else [args.method]

    images, categories = load_image_list()
    print(f"dataset: {len(images)} images, {len(categories)} categories")
    print(f"methods: {', '.join(methods)}\n")

    for subdir in set(METHODS[m][0] for m in methods):
        (RESULTS_DIR / subdir).mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "comparisons").mkdir(parents=True, exist_ok=True)

    all_results = {METHODS[m][1]: {} for m in methods}

    for i, img_info in enumerate(images, 1):
        category = img_info['category']
        stem = img_info['stem']
        png_path = img_info['png_path']
        name = sanitize_name(category, stem)

        print(f"[{i}/{len(images)}] {name} ({category})")
        img_rgb = np.array(Image.open(png_path).convert('RGB'))

        for method in methods:
            _, key = METHODS[method]
            try:
                report = run_method(method, img_rgb, png_path, name, category)
                all_results[key][name] = report
                if method in ('stardist', 'cellpose'):
                    n = report['stats']['nuclei']
                else:
                    n = report['stats']['final']
                t = report['timings']['total']
                print(f"  {method}: {n} nuclei, {t:.1f}s")
            except Exception as e:
                print(f"  {method}: error - {e}")
                all_results[key][name] = {'error': str(e)}

    # unified comparison json

    comparison = {
        'timestamp': datetime.now().isoformat(),
        'total_images': len(images),
        'categories': categories,
        'methods': {},
        'by_image': {},
        'by_category': {},
    }

    for method_name, results in all_results.items():
        nuclei_counts = []
        times = []
        for name, report in results.items():
            if 'error' in report:
                continue
            if method_name in ('stardist_alone', 'cellpose_alone'):
                nuclei_counts.append(report['stats']['nuclei'])
            else:
                nuclei_counts.append(report['stats']['final'])
            times.append(report['timings']['total'])

        comparison['methods'][method_name] = {
            'avg_nuclei': float(np.mean(nuclei_counts)) if nuclei_counts else 0,
            'avg_time': float(np.mean(times)) if times else 0,
            'total_images': len(nuclei_counts),
        }

    for img_info in images:
        name = sanitize_name(img_info['category'], img_info['stem'])
        comparison['by_image'][name] = {
            'category': img_info['category'],
        }
        for method_name, results in all_results.items():
            if name in results and 'error' not in results[name]:
                r = results[name]
                if method_name in ('stardist_alone', 'cellpose_alone'):
                    comparison['by_image'][name][method_name] = {
                        'nuclei': r['stats']['nuclei'],
                        'time': r['timings']['total'],
                    }
                else:
                    comparison['by_image'][name][method_name] = {
                        'nuclei': r['stats']['final'],
                        'time': r['timings']['total'],
                        'timings': r['timings'],
                    }

    # by category
    for cat in categories:
        comparison['by_category'][cat] = {}
        for method_name, results in all_results.items():
            cat_nuclei = []
            cat_times = []
            for name, report in results.items():
                if 'error' in report:
                    continue
                if report.get('category') != cat:
                    continue
                if method_name in ('stardist_alone', 'cellpose_alone'):
                    cat_nuclei.append(report['stats']['nuclei'])
                else:
                    cat_nuclei.append(report['stats']['final'])
                cat_times.append(report['timings']['total'])
            comparison['by_category'][cat][method_name] = {
                'avg_nuclei': float(np.mean(cat_nuclei)) if cat_nuclei else 0,
                'avg_time': float(np.mean(cat_times)) if cat_times else 0,
                'n_images': len(cat_nuclei),
            }

    comp_path = RESULTS_DIR / "comparisons" / "unified_comparison.json"
    with open(comp_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\ndone: {comp_path}")


if __name__ == "__main__":
    main()
