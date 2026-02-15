"""vectorized uoift saliency without higra dependency (sec 2.1.1)"""

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from scipy.ndimage import maximum_filter, minimum_filter
import scipy.ndimage as ndi
from skimage.segmentation import slic
from skimage.color import rgb2lab


UOIFT_ALPHA = -0.7
UOIFT_SP_SIZE = 300


def generate(image, alpha=UOIFT_ALPHA, sp_size=UOIFT_SP_SIZE):
    """compute boundary saliency via oriented graph weights"""
    # normalize input
    if image.dtype == np.uint8:
        img_f = image.astype(np.float32) / 255.0
    else:
        img_f = image.astype(np.float32)

    # extract lightness channel
    lab = rgb2lab(img_f)
    L = lab[:, :, 0] / 100.0
    H, W = L.shape
    n_seg = max(20, (H * W) // sp_size)

    # slic superpixels
    superpixels = slic(
        img_f, n_segments=n_seg, compactness=30.0,
        sigma=1.0, min_size_factor=0.5, start_label=0,
        channel_axis=2, convert2lab=True
    )

    # relabel contiguously via lut
    unique_labels = np.unique(superpixels)
    n_sp = len(unique_labels)
    if unique_labels.max() != n_sp - 1:
        remap = np.zeros(unique_labels.max() + 1, dtype=np.int32)
        remap[unique_labels] = np.arange(n_sp, dtype=np.int32)
        superpixels = remap[superpixels]

    # mean intensity per superpixel
    mean_intensity = np.array(ndi.mean(L, superpixels, range(n_sp)))

    # build rag edges (vectorized)
    h_mask = superpixels[:, 1:] != superpixels[:, :-1]
    rows_h, cols_h = np.where(h_mask)
    left = superpixels[rows_h, cols_h]
    right = superpixels[rows_h, cols_h + 1]

    v_mask = superpixels[1:, :] != superpixels[:-1, :]
    rows_v, cols_v = np.where(v_mask)
    top = superpixels[rows_v, cols_v]
    bottom = superpixels[rows_v + 1, cols_v]

    all_s = np.concatenate([np.minimum(left, right), np.minimum(top, bottom)])
    all_t = np.concatenate([np.maximum(left, right), np.maximum(top, bottom)])
    edges = np.unique(np.stack([all_s, all_t], axis=1), axis=0)

    if len(edges) == 0:
        return (np.ones((H, W), dtype=np.float32) * 128).astype(np.uint8)

    # oriented weights (vectorized)
    sources, targets = edges[:, 0], edges[:, 1]
    Is = mean_intensity[sources]
    It = mean_intensity[targets]
    diff = np.abs(It - Is)
    weights = np.where(Is > It, diff * (1.0 + alpha), diff * (1.0 - alpha))
    weights = np.maximum(weights, 1e-8)

    # mst via sparse graph
    sparse_graph = csr_matrix((weights, (sources, targets)), shape=(n_sp, n_sp))
    sparse_graph = sparse_graph + sparse_graph.T
    minimum_spanning_tree(sparse_graph)

    # saliency: mean weight per superpixel
    sp_weight_sum = np.zeros(n_sp, dtype=np.float64)
    sp_weight_count = np.zeros(n_sp, dtype=np.int32)
    np.add.at(sp_weight_sum, sources, weights)
    np.add.at(sp_weight_sum, targets, weights)
    np.add.at(sp_weight_count, sources, 1)
    np.add.at(sp_weight_count, targets, 1)
    sp_mean_weight = np.divide(
        sp_weight_sum, sp_weight_count,
        where=sp_weight_count > 0, out=np.zeros(n_sp)
    )
    saliency_map = sp_mean_weight[superpixels].astype(np.float32)

    # assign boundary edge weights
    edge_keys = sources.astype(np.int64) * n_sp + targets.astype(np.int64)
    sort_idx = np.argsort(edge_keys)
    edge_keys_sorted = edge_keys[sort_idx]
    weights_sorted = weights[sort_idx]

    def _assign_boundary(sp_a, sp_b, bnd_rows, bnd_cols, bnd_cols2=None):
        """assign edge weights to boundary pixels"""
        s = np.minimum(sp_a, sp_b).astype(np.int64)
        t = np.maximum(sp_a, sp_b).astype(np.int64)
        keys = s * n_sp + t
        idx = np.clip(np.searchsorted(edge_keys_sorted, keys), 0, len(edge_keys_sorted) - 1)
        valid = edge_keys_sorted[idx] == keys
        w = np.where(valid, weights_sorted[idx], 0.0).astype(np.float32)
        np.maximum.at(saliency_map, (bnd_rows, bnd_cols), w)
        if bnd_cols2 is not None:
            np.maximum.at(saliency_map, (bnd_rows, bnd_cols2), w)

    _assign_boundary(left, right, rows_h, cols_h, cols_h + 1)
    _assign_boundary(top, bottom, rows_v, cols_v)
    np.maximum.at(saliency_map, (rows_v + 1, cols_v), saliency_map[rows_v, cols_v])

    # enhance boundaries with morphological filter
    dilated_sp = maximum_filter(superpixels, size=3)
    eroded_sp = minimum_filter(superpixels, size=3)
    boundary_mask = dilated_sp != eroded_sp
    saliency_map[boundary_mask] = maximum_filter(saliency_map, size=3)[boundary_mask]

    # normalize to uint8
    smin, smax = saliency_map.min(), saliency_map.max()
    if smax > smin:
        saliency_map = (saliency_map - smin) / (smax - smin)

    return (saliency_map * 255).astype(np.uint8)
