"""shared utilities for image preprocessing and io

maps to paper:
  sec 3.4  lab color space operations, binary mask, saliency
  sec 3.3  higra simplification for path b input
"""
import time
from pathlib import Path

import higra as hg
import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2lab
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    dilation,
    erosion,
    reconstruction,
)

from .config import CFG_PREPROC as CFG


# lab color space

def get_lab_channels(img):
    """extract l, a, b channels from rgb"""
    lab = rgb2lab(img)
    return lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]


# probability map

def create_prob_map(l, a, b, cfg):
    """weighted lab combination for nuclei probability"""
    l_inv = 100 - l
    prob = (l_inv * cfg["w_l"]) + (a * cfg["w_a"]) - (b * cfg["w_b"])
    return np.clip(prob, 0, 100)


def morph_reconstruction(prob, cfg):
    """geodesic reconstruction (h-dome, fills mitoses)"""
    seed = prob - cfg["morph_h"]
    return reconstruction(seed, prob, method="dilation")


# higra filtering

def higra_filter(data, min_area, min_height):
    """max tree + min tree attribute filtering"""
    h, w = data.shape
    graph = hg.get_4_adjacency_graph((h, w))
    data_f = data.astype(np.float64)

    tree_max, alt_max = hg.component_tree_max_tree(graph, data_f)
    area_max = hg.attribute_area(tree_max)
    height_max = hg.attribute_height(tree_max, alt_max)
    deleted_max = (area_max < min_area) | (height_max < min_height)
    filtered = hg.reconstruct_leaf_data(tree_max, alt_max, deleted_max)

    tree_min, alt_min = hg.component_tree_min_tree(graph, filtered)
    area_min = hg.attribute_area(tree_min)
    height_min = hg.attribute_height(tree_min, alt_min)
    deleted_min = (area_min < min_area) | (height_min < min_height)
    filtered = hg.reconstruct_leaf_data(tree_min, alt_min, deleted_min)

    return filtered


# binary mask

def create_binary_mask(prob_filtered):
    """otsu threshold + morphological cleanup"""
    thresh = threshold_otsu(prob_filtered)
    mask = (prob_filtered > thresh).astype(bool)
    mask = binary_closing(mask, disk(4)).astype(bool)
    mask = binary_fill_holes(mask).astype(bool)
    mask = binary_opening(mask, disk(2)).astype(bool)
    return mask


# saliency

SAL_PROB_WEIGHT = 0.7
SAL_GRAD_WEIGHT = 0.3
SAL_GAMMA = 1.5


def create_saliency(prob_filtered, cfg):
    """prob + gradient saliency map"""
    prob = prob_filtered.astype(np.float32)
    prob_norm = prob - prob.min()
    if prob_norm.max() > 0:
        prob_norm /= prob_norm.max()
    grad = dilation(prob_norm, disk(2)) - erosion(prob_norm, disk(2))
    sal = SAL_PROB_WEIGHT * prob_norm + SAL_GRAD_WEIGHT * grad
    sal = np.power(sal, SAL_GAMMA)
    if sal.max() > 0:
        sal = sal / sal.max()
    sal = gaussian(sal, sigma=cfg["sal_sigma"], preserve_range=True)
    return (sal * 255).astype(np.uint8)


# higra image simplification (sec 3.3)

def higra_simplify_channel(channel, min_area, min_height):
    """simplify single channel via attribute filtering"""
    h, w = channel.shape
    graph = hg.get_4_adjacency_graph((h, w))
    channel_f = channel.astype(np.float64)

    tree_max, alt_max = hg.component_tree_max_tree(graph, channel_f)
    area_max = hg.attribute_area(tree_max)
    height_max = hg.attribute_height(tree_max, alt_max)
    deleted_max = (area_max < min_area) | (height_max < min_height)
    filtered = hg.reconstruct_leaf_data(tree_max, alt_max, deleted_max)

    tree_min, alt_min = hg.component_tree_min_tree(graph, filtered)
    area_min = hg.attribute_area(tree_min)
    height_min = hg.attribute_height(tree_min, alt_min)
    deleted_min = (area_min < min_area) | (height_min < min_height)
    filtered = hg.reconstruct_leaf_data(tree_min, alt_min, deleted_min)

    return filtered


def higra_simplify_image(img, min_area, min_height):
    """per channel higra simplification"""
    simplified = np.zeros_like(img, dtype=np.float64)
    for ch in range(3):
        simplified[:, :, ch] = higra_simplify_channel(
            img[:, :, ch].astype(np.float64), min_area, min_height
        )
    return np.clip(simplified, 0, 255).astype(np.uint8)


# segmentation fusion (iou based)

def fuse_segmentations(labels_a, labels_b, iou_thresh):
    """merge path a and b labels via iou matching"""
    unified = np.zeros_like(labels_a)
    next_id = 1
    matched_b = set()

    props_a = {r.label: r for r in regionprops(labels_a.astype(np.int32))}
    props_b = {r.label: r for r in regionprops(labels_b.astype(np.int32))}

    def bbox_overlap(bbox1, bbox2):
        r1_min, c1_min, r1_max, c1_max = bbox1
        r2_min, c2_min, r2_max, c2_max = bbox2
        return not (r1_max < r2_min or r2_max < r1_min or c1_max < c2_min or c2_max < c1_min)

    def fast_iou(prop_a, prop_b):
        r1_min, c1_min, r1_max, c1_max = prop_a.bbox
        r2_min, c2_min, r2_max, c2_max = prop_b.bbox
        rmin = min(r1_min, r2_min)
        rmax = max(r1_max, r2_max)
        cmin = min(c1_min, c2_min)
        cmax = max(c1_max, c2_max)
        roi_a = labels_a[rmin:rmax, cmin:cmax] == prop_a.label
        roi_b = labels_b[rmin:rmax, cmin:cmax] == prop_b.label
        inter = (roi_a & roi_b).sum()
        union = (roi_a | roi_b).sum()
        return inter / union if union > 0 else 0.0

    for lid_a, prop_a in props_a.items():
        best_iou = 0.0
        best_lid_b = None
        for lid_b, prop_b in props_b.items():
            if lid_b in matched_b:
                continue
            if not bbox_overlap(prop_a.bbox, prop_b.bbox):
                continue
            iou = fast_iou(prop_a, prop_b)
            if iou > best_iou:
                best_iou = iou
                best_lid_b = lid_b
        mask_a = labels_a == lid_a
        unified[mask_a] = next_id
        next_id += 1
        if best_iou > iou_thresh and best_lid_b is not None:
            matched_b.add(best_lid_b)

    for lid_b, prop_b in props_b.items():
        if lid_b in matched_b:
            continue
        mask_b = labels_b == lid_b
        unified[mask_b] = next_id
        next_id += 1

    return unified


# region filtering (higra_sicle compat)

def filter_path_regions(labels, cfg, l_ch=None, a_ch=None, b_ch=None):
    """shape + optional color filter for regions"""
    labels = labels.astype(np.int32)
    cleaned = np.zeros_like(labels)
    next_id = 1
    kept = 0
    rejected = 0
    unique = np.unique(labels)
    if unique.size > 0 and unique[0] == 0:
        unique = unique[1:]
    for region_id in unique:
        mask = (labels == region_id).astype(bool)
        area = mask.sum()
        if not (cfg["final_min_area"] < area < cfg["final_max_area"]):
            rejected += 1
            continue
        props = regionprops(mask.astype(np.uint8))[0]
        color_ok = False
        if l_ch is not None and a_ch is not None and b_ch is not None:
            rmin, cmin, rmax, cmax = props.bbox
            roi = mask[rmin:rmax, cmin:cmax]
            if roi.sum() > 0:
                mean_l = np.mean(l_ch[rmin:rmax, cmin:cmax][roi])
                mean_a = np.mean(a_ch[rmin:rmax, cmin:cmax][roi])
                if (mean_l < 80) and (mean_a > 15):
                    color_ok = True
        if area > 2000:
            if not (color_ok or props.solidity >= 0.50):
                rejected += 1
                continue
        else:
            if not (color_ok or props.solidity >= cfg["final_min_solidity"]):
                rejected += 1
                continue
        cleaned[mask] = next_id
        next_id += 1
        kept += 1
    return cleaned, kept, rejected


def filter_lab(labels, l_ch, a_ch, b_ch, cfg):
    """validate regions by mean lab color"""
    labels = labels.astype(np.int32)
    valid = []
    kept = 0
    rejected = 0
    unique = np.unique(labels)
    if unique.size > 0 and unique[0] == 0:
        unique = unique[1:]
    for region_id in unique:
        mask = (labels == region_id).astype(bool)
        props = regionprops(mask.astype(np.uint8))[0]
        rmin, cmin, rmax, cmax = props.bbox
        roi = mask[rmin:rmax, cmin:cmax]
        if roi.sum() == 0:
            rejected += 1
            continue
        mean_l = np.mean(l_ch[rmin:rmax, cmin:cmax][roi])
        mean_a = np.mean(a_ch[rmin:rmax, cmin:cmax][roi])
        mean_b = np.mean(b_ch[rmin:rmax, cmin:cmax][roi])
        if (mean_l > cfg["final_l_max"]) or (mean_a < cfg["final_a_min"]) or (mean_b > cfg["final_b_max"]):
            rejected += 1
            continue
        valid.append(region_id)
        kept += 1
    return np.isin(labels, valid).astype(bool), kept, rejected


# marker extraction

def extract_markers(final_mask):
    """centroid markers from binary mask"""
    labeled = label(final_mask)
    markers = np.zeros_like(final_mask, dtype=np.uint8)
    for region in regionprops(labeled):
        cy, cx = int(region.centroid[0]), int(region.centroid[1])
        markers[cy, cx] = 1
    return markers


def label(mask):
    """connected component labeling"""
    from skimage.measure import label as sk_label
    return sk_label(mask)


# io

def ensure_dir(path):
    """create directory if missing"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_image_path(dataset_dir, candidates=None):
    """locate original image in dataset folder"""
    dataset_dir = Path(dataset_dir)
    if candidates is None:
        candidates = (
            f"{dataset_dir.name}.png",
            "image.png",
            "img.png",
        )
    for rel in candidates:
        if (dataset_dir / rel).exists():
            return (dataset_dir / rel).resolve()
    raise FileNotFoundError(f"original image not found in {dataset_dir}")


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


def save_uint8(path, array):
    """save as uint8 png"""
    imageio.imwrite(path, array.astype(np.uint8))


def save_uint16(path, array):
    """save as uint16 pgm"""
    array = array.astype(np.uint16)
    h, w = array.shape
    max_val = array.max()
    with open(path, 'wb') as f:
        f.write(f"P5\n{w} {h}\n{max_val}\n".encode('ascii'))
        if max_val < 256:
            f.write(array.astype(np.uint8).tobytes())
        else:
            f.write(array.astype('>u2').tobytes())


def summarize_mask(mask):
    """quick stats string for mask"""
    return f"shape={mask.shape}, pixels={int(mask.sum())}, ratio={100*mask.sum()/mask.size:.2f}%"


# timing

class Timer:
    """context manager for timing pipeline steps"""
    def __init__(self, name):
        self.name = name
        self.elapsed = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"  {self.name}: {self.elapsed:.2f}s")


def format_time(seconds):
    """format seconds as mm:ss.ms"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"
