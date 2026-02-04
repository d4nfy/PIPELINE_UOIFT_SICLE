"""veta inspired filtering, pool creation and arbitration (sec 3.4, 3.5)

sec 3.4: lab color filter (10 <= a* <= 40), solidity >= 0.80, area 100..10000
sec 3.4: multiscale selection via fitness conflict resolution (max solidity)
sec 3.5: stardist + pool fusion via iou > 0.2, instance recovery
"""
import numpy as np
import scipy.ndimage as ndi
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from scipy.spatial import cKDTree

from .config import FILTER_CFG, KDTREE_MAX_DIST, POOL_OVERLAP_THRESH


# sec 3.4: combined lab + shape filter

def filter_sicle_regions(labels, l_ch, a_ch, b_ch, cfg=None):
    """combined lab color + area + solidity filter (sec 3.4)

    applies vectorized lab filter, then area, then solidity.
    returns (filtered_labels, n_kept, n_rejected).
    """
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

    # vectorized lab filter
    mean_l = ndi.mean(l_ch, labels, unique_labels)
    mean_a = ndi.mean(a_ch, labels, unique_labels)
    mean_b = ndi.mean(b_ch, labels, unique_labels)

    keep_mask = (
        (mean_l <= cfg["l_max"]) &
        (mean_a >= cfg["a_min"]) &
        (mean_b <= cfg["b_max"])
    )
    if "a_max" in cfg:
        keep_mask &= (mean_a <= cfg["a_max"])

    labels_after_lab = unique_labels[keep_mask]
    if len(labels_after_lab) == 0:
        return np.zeros_like(labels), 0, n_total

    # area filter
    temp_mask = np.isin(labels, labels_after_lab)
    temp_labels = labels * temp_mask
    areas = ndi.sum(np.ones_like(labels), temp_labels, labels_after_lab)
    area_mask = (areas >= cfg["min_area"]) & (areas <= cfg["max_area"])
    labels_after_area = labels_after_lab[area_mask]

    if len(labels_after_area) == 0:
        return np.zeros_like(labels), 0, n_total

    # solidity filter
    temp_mask2 = np.isin(temp_labels, labels_after_area)
    temp_labels2 = temp_labels * temp_mask2

    labels_to_keep = set()
    for region in regionprops(temp_labels2):
        if region.solidity >= cfg["min_solidity"]:
            labels_to_keep.add(region.label)

    n_kept = len(labels_to_keep)
    if n_kept == 0:
        return np.zeros_like(labels), 0, n_total

    # relabel contiguously
    lut = np.zeros(labels.max() + 1, dtype=np.int32)
    new_id = 1
    for lbl in sorted(labels_to_keep):
        lut[lbl] = new_id
        new_id += 1

    return lut[labels], n_kept, n_total - n_kept


# sec 3.4: separate lab filter (used by standalone runner)

def filter_by_lab(labels, l_ch, a_ch, b_ch, cfg=None):
    """vectorized lab color filter only"""
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

    keep_mask = (
        (mean_l <= cfg["l_max"]) &
        (mean_a >= cfg["a_min"]) &
        (mean_b <= cfg["b_max"])
    )

    labels_to_keep = set(unique_labels[keep_mask])
    n_kept = len(labels_to_keep)

    if n_kept == 0:
        return np.zeros_like(labels), 0, n_total - n_kept

    lut = np.zeros(labels.max() + 1, dtype=np.int32)
    new_id = 1
    for lbl in sorted(labels_to_keep):
        lut[lbl] = new_id
        new_id += 1

    return lut[labels], n_kept, n_total - n_kept


# sec 3.4: separate shape filter

def filter_by_shape(labels, cfg=None):
    """area + solidity shape filter only"""
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

    areas = ndi.sum(np.ones_like(labels), labels, unique_labels)
    area_mask = (areas >= cfg["min_area"]) & (areas <= cfg["max_area"])
    labels_after_area = unique_labels[area_mask]

    if len(labels_after_area) == 0:
        return np.zeros_like(labels), 0, n_total

    temp_mask = np.isin(labels, labels_after_area)
    temp_labels = labels * temp_mask

    labels_to_keep = set()
    for region in regionprops(temp_labels):
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


# sec 3.4: veta multiscale selection with kdtree

def veta_multiscale_selection(scales_dict, l_ch, a_ch, b_ch,
                              filter_cfg=None, min_solidity=0.80, ov_thresh=0.2):
    """filter per scale then resolve conflicts by max solidity (sec 3.4)

    uses kdtree spatial indexing for O(n log n + n*k) complexity.
    returns merged label mask with best nuclei across all scales.
    """
    if filter_cfg is None:
        filter_cfg = FILTER_CFG

    # filter each scale
    filtered_scales = {}
    for scale_name, scale_mask in scales_dict.items():
        filtered, n_kept, _ = filter_sicle_regions(scale_mask, l_ch, a_ch, b_ch, filter_cfg)
        if n_kept > 0:
            filtered_scales[scale_name] = filtered

    if len(filtered_scales) == 0:
        h, w = list(scales_dict.values())[0].shape
        return np.zeros((h, w), dtype=np.int32)

    # extract all candidates with solidity
    candidates = []
    for scale_name, scale_mask in filtered_scales.items():
        for region in regionprops(scale_mask):
            if region.solidity >= min_solidity:
                candidates.append({
                    'id': len(candidates),
                    'label': region.label,
                    'scale': scale_name,
                    'mask': scale_mask == region.label,
                    'bbox': region.bbox,
                    'centroid': region.centroid,
                    'area': region.area,
                    'solidity': region.solidity,
                })

    if len(candidates) == 0:
        h, w = list(scales_dict.values())[0].shape
        return np.zeros((h, w), dtype=np.int32)

    # build spatial index
    centroids = np.array([c['centroid'] for c in candidates])
    tree = cKDTree(centroids)

    # build conflict graph
    conflicts = {}
    for i, cand in enumerate(candidates):
        nearby = tree.query_ball_point(cand['centroid'], r=KDTREE_MAX_DIST)
        for j in nearby:
            if i >= j:
                continue
            other = candidates[j]

            # bbox check
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

    # greedy selection sorted by solidity
    candidates.sort(key=lambda x: x['solidity'], reverse=True)

    accepted = []
    accepted_ids = set()

    for cand in candidates:
        cand_id = cand['id']
        has_conflict = bool(conflicts.get(cand_id, set()) & accepted_ids)
        if not has_conflict:
            accepted.append(cand)
            accepted_ids.add(cand_id)

    # build merged mask
    h, w = list(scales_dict.values())[0].shape
    merged = np.zeros((h, w), dtype=np.int32)
    for idx, cand in enumerate(accepted, 1):
        merged[cand['mask']] = idx

    return merged


# sec 3.3: candidate pool from dual paths

def create_candidate_pool(path_a, path_b_scales):
    """merge path a + path b into unified pool (sec 3.3)

    starts with path a, adds path b regions with <50% overlap.
    preserves instance boundaries (no relabeling via connected components).
    """
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

        unique_labels = np.unique(scale_mask)
        unique_labels = unique_labels[unique_labels > 0]

        for lbl in unique_labels:
            region_mask = scale_mask == lbl
            region_size = region_mask.sum()
            if region_size == 0:
                continue

            overlap = (existing_mask & region_mask).sum()
            overlap_ratio = overlap / region_size

            if overlap_ratio < POOL_OVERLAP_THRESH:
                new_pixels = region_mask & (~existing_mask)
                if new_pixels.sum() > 0:
                    pool[new_pixels] = next_id
                    existing_mask |= region_mask
                    next_id += 1
                    n_from_b += 1

    print(f"      pool: {n_from_a} from path a, {n_from_b} from path b")
    return pool


# sec 3.5: stardist + pool fusion

def fuse_stardist_pool(stardist, pool, iou_thresh=0.2):
    """fuse stardist anchors with sicle pool via iou matching (sec 3.5)

    overlapping regions (iou > thresh): keep stardist.
    non overlapping pool regions: recover as additional detections.
    """
    stardist = stardist.astype(np.int32)
    pool = pool.astype(np.int32)
    fused = np.zeros_like(stardist)
    next_id = 1

    props_sd = {r.label: r for r in regionprops(stardist)}
    props_pool = {r.label: r for r in regionprops(pool)}
    matched_pool = set()

    # keep all stardist, mark matched pool regions
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

    # recover unmatched pool regions
    for lid_pool, prop_pool in props_pool.items():
        if lid_pool not in matched_pool:
            mask_pool = pool == lid_pool
            empty_pixels = (fused == 0) & mask_pool
            if empty_pixels.sum() > 0.5 * prop_pool.area:
                fused[mask_pool & (fused == 0)] = next_id
                next_id += 1

    return fused


# sec 3.5: solidity arbitration

def arbitrate_by_solidity(fused, cfg):
    """final quality filter by area and solidity (sec 3.5)"""
    fused = fused.astype(np.int32)
    arbitrated = np.zeros_like(fused)
    next_id = 1
    kept = 0
    rejected = 0

    for region in regionprops(fused):
        if region.area < cfg["arb_min_area"]:
            rejected += 1
            continue
        if region.solidity >= cfg["arb_min_solidity"]:
            arbitrated[fused == region.label] = next_id
            next_id += 1
            kept += 1
        else:
            rejected += 1

    return arbitrated, kept, rejected


# visualization helpers

# hue values for monochrome overlays (0-1 range)
VIS_HUES = {
    "path_a":   0.33,   # green
    "path_b":   0.50,   # cyan
    "stardist": 0.75,   # violet
    "pool_a":   0.33,   # green (from a)
    "pool_b":   0.50,   # cyan (from b)
}


def _to_grayscale(img_rgb):
    """convert rgb to grayscale uint8"""
    return np.mean(img_rgb, axis=2).astype(np.uint8)


def labels_to_monochrome(labels, bg_gray, hue, sat=0.8, val=0.9, alpha=0.7):
    """colorize labels with single hue on grayscale background"""
    import colorsys
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
    """colorize labels with distinct per-region colors on grayscale"""
    import colorsys
    viz = np.stack([bg_gray, bg_gray, bg_gray], axis=-1).astype(np.float32)
    unique = np.unique(labels)
    unique = unique[unique != 0]
    if len(unique) == 0:
        return viz.astype(np.uint8)
    rng = np.random.RandomState(42)
    n = len(unique)
    colors = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.8, 0.9)
        colors.append(np.array([r * 255, g * 255, b * 255]))
    rng.shuffle(colors)
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        color = colors[i % n]
        for c in range(3):
            viz[:, :, c][mask] = (1 - alpha) * viz[:, :, c][mask] + alpha * color[c]
    bounds = find_boundaries(labels, mode='thick')
    viz[bounds] = [255, 255, 255]
    return np.clip(viz, 0, 255).astype(np.uint8)


def pool_bicolor(pool, path_a, path_b, bg_gray, alpha=0.7):
    """colorize pool: green = from path a, cyan = from path b"""
    import colorsys
    viz = np.stack([bg_gray, bg_gray, bg_gray], axis=-1).astype(np.float32)
    r_a, g_a, b_a = colorsys.hsv_to_rgb(VIS_HUES["pool_a"], 0.8, 0.9)
    color_a = np.array([r_a * 255, g_a * 255, b_a * 255])
    r_b, g_b, b_b = colorsys.hsv_to_rgb(VIS_HUES["pool_b"], 0.8, 0.9)
    color_b = np.array([r_b * 255, g_b * 255, b_b * 255])
    unique = np.unique(pool)
    unique = unique[unique != 0]
    for lbl in unique:
        mask = pool == lbl
        overlap_a = np.sum((path_a > 0) & mask)
        overlap_b = np.sum((path_b > 0) & mask)
        color = color_a if overlap_a > overlap_b else color_b
        for c in range(3):
            viz[:, :, c][mask] = (1 - alpha) * viz[:, :, c][mask] + alpha * color[c]
    bounds = find_boundaries(pool, mode='thick')
    viz[bounds] = [255, 255, 255]
    return np.clip(viz, 0, 255).astype(np.uint8)


def create_overlay_on_rgb(img_rgb, labels, alpha=0.3):
    """final labels overlaid on original rgb with distinct colors"""
    from skimage.color import label2rgb
    overlay = label2rgb(labels, image=img_rgb, bg_label=0, alpha=alpha)
    return (overlay * 255).astype(np.uint8)


def create_fusion_decisions(pool, stardist, final, bg_gray, alpha=0.7):
    """fusion decision visualization with legend

    green fill  = sicle pool regions kept in final
    blue contour = stardist regions kept in final
    red fill    = rejected regions
    """
    import cv2
    h, w = pool.shape
    viz = np.stack([bg_gray, bg_gray, bg_gray], axis=-1).astype(np.float32)

    c_kept = np.array([0, 200, 100])
    c_star = (100, 150, 255)
    c_rej = np.array([200, 80, 80])

    # pool regions: green if kept, red if rejected
    for lbl in set(np.unique(pool)) - {0}:
        mask = pool == lbl
        overlap = np.sum((final > 0) & mask) / max(np.sum(mask), 1)
        color = c_kept if overlap > 0.5 else c_rej
        for c in range(3):
            viz[:, :, c][mask] = (1 - alpha) * viz[:, :, c][mask] + alpha * color[c]

    # stardist contours: blue if kept, gray if rejected
    for lbl in set(np.unique(stardist)) - {0}:
        mask = stardist == lbl
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        overlap = np.sum((final > 0) & mask) / max(np.sum(mask), 1)
        color = c_star if overlap > 0.5 else (150, 150, 150)
        thickness = 2 if overlap > 0.5 else 1
        viz_u8 = np.clip(viz, 0, 255).astype(np.uint8)
        cv2.drawContours(viz_u8, contours, -1, color, thickness)
        viz = viz_u8.astype(np.float32)

    # legend bar
    legend = np.ones((100, w, 3), dtype=np.uint8) * 40
    cv2.putText(legend, "Fusion Decisions:", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(legend, (10, 45), (30, 65), tuple(int(x) for x in c_kept), -1)
    cv2.putText(legend, "SICLE kept", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(legend, (200, 45), (220, 65), c_star, -1)
    cv2.putText(legend, "StarDist kept", (230, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(legend, (420, 45), (440, 65), tuple(int(x) for x in c_rej), -1)
    cv2.putText(legend, "Rejected", (450, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    result = np.clip(viz, 0, 255).astype(np.uint8)
    return np.vstack([result, legend])


def create_violet_overlay(img_rgb, mask, alpha=0.6):
    """grayscale bg + violet overlay for segmented regions"""
    gray = _to_grayscale(img_rgb)
    return labels_to_monochrome(mask, gray, VIS_HUES["stardist"], alpha=alpha)


def create_contour_overlay(img_rgb, mask, color=(255, 0, 255)):
    """contours of mask regions on original image"""
    contours = find_boundaries(mask, mode='outer')
    viz = img_rgb.copy()
    viz[contours] = color
    return viz
