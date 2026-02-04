"""pure python uoift saliency generation (sec 3.2)

oriented image foresting transform for topological saliency maps.
eq. 1: alpha=-0.7 favors dark nuclei on bright background.
fallback when c++ binary is unavailable.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List, Union
from dataclasses import dataclass, field
import warnings

from skimage.segmentation import slic, find_boundaries
from skimage.color import rgb2lab
from skimage.measure import regionprops_table
from scipy import ndimage
import higra as hg


@dataclass
class UOIFTConfig:
    """uoift saliency generator configuration"""
    sp_size: int = 100
    alpha: float = -0.7
    slic_compactness: float = 10.0
    slic_sigma: float = 1.0
    slic_min_size_factor: float = 0.5
    extinction_attribute: str = 'volume'
    min_extinction_ratio: float = 0.001
    use_lab_channel: str = 'L'
    boundary_mode: str = 'gradient'
    enabled: bool = True

    def __post_init__(self):
        """validate parameters after init"""
        if self.alpha >= 0:
            warnings.warn(
                f"alpha={self.alpha} >= 0. For basophilic nuclei "
                "(light->dark transitions), use alpha in [-1.0, -0.5].",
                UserWarning
            )
        if self.sp_size < 10:
            raise ValueError("sp_size must be >= 10 pixels")
        if self.use_lab_channel not in ('L', 'a', 'b', 'gray'):
            raise ValueError(f"invalid channel: {self.use_lab_channel}")
        if self.extinction_attribute not in ('volume', 'area', 'height'):
            raise ValueError(f"invalid extinction attribute: {self.extinction_attribute}")


class UOIFTSaliencyGenerator:
    """topological saliency generator via uoift algorithm"""

    def __init__(self, config: Optional[UOIFTConfig] = None):
        """initialize uoift generator"""
        self.config = config or UOIFTConfig()
        self._superpixels: Optional[np.ndarray] = None
        self._rag = None
        self._working_channel: Optional[np.ndarray] = None

    def generate_saliency(
        self,
        image: np.ndarray,
        return_intermediate: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """generate complete uoift topological saliency map"""
        if not self.config.enabled:
            uniform_map = np.ones(image.shape[:2], dtype=np.float32) * 0.5
            return uniform_map, {'enabled': False}

        image_norm = self._normalize_input(image)
        self._working_channel = self._extract_working_channel(image_norm)

        self._superpixels, n_segments = self._compute_superpixels(image_norm)

        self._rag, node_features = self._build_rag(
            self._superpixels,
            self._working_channel
        )

        edge_weights = self._compute_oriented_weights(node_features)

        tree, altitudes, merge_history = self._build_hierarchy(edge_weights)

        extinction_values = self._compute_extinction(
            tree,
            altitudes,
            node_features
        )

        saliency_map = self._generate_ucm(
            tree,
            altitudes,
            extinction_values
        )

        metadata = {
            'n_superpixels': n_segments,
            'n_edges': len(edge_weights),
            'merge_history': merge_history,
            'alpha': self.config.alpha,
            'sp_size': self.config.sp_size,
            'extinction_attribute': self.config.extinction_attribute,
            'saliency_stats': {
                'min': float(saliency_map.min()),
                'max': float(saliency_map.max()),
                'mean': float(saliency_map.mean()),
                'std': float(saliency_map.std())
            }
        }
        
        if return_intermediate:
            metadata['superpixels'] = self._superpixels.copy()
            metadata['working_channel'] = self._working_channel.copy()
            metadata['extinction_values'] = extinction_values
        
        return saliency_map, metadata
    
    def _normalize_input(self, image: np.ndarray) -> np.ndarray:
        """normalize input image to float32 [0, 1]"""
        image = np.asarray(image)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected RGB (H,W,3), got shape={image.shape}")
        
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype in (np.float32, np.float64):
            if image.max() > 1.0:
                return (image / image.max()).astype(np.float32)
            return image.astype(np.float32)
        else:
            return image.astype(np.float32) / np.iinfo(image.dtype).max
    
    def _extract_working_channel(self, image: np.ndarray) -> np.ndarray:
        """extract working channel for weight computation"""
        if self.config.use_lab_channel == 'gray':
            gray = (0.299 * image[:,:,0] +
                   0.587 * image[:,:,1] +
                   0.114 * image[:,:,2])
            return gray.astype(np.float32)

        lab = rgb2lab(image)

        channel_idx = {'L': 0, 'a': 1, 'b': 2}[self.config.use_lab_channel]
        channel = lab[:, :, channel_idx]

        if self.config.use_lab_channel == 'L':
            channel = channel / 100.0
        else:
            channel = (channel + 128.0) / 255.0

        return np.clip(channel, 0, 1).astype(np.float32)
    
    def _compute_superpixels(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """compute slic superpixels for rag construction"""
        n_pixels = image.shape[0] * image.shape[1]
        n_segments_target = max(20, n_pixels // self.config.sp_size)

        superpixels = slic(
            image,
            n_segments=n_segments_target,
            compactness=30.0,
            sigma=self.config.slic_sigma,
            min_size_factor=self.config.slic_min_size_factor,
            start_label=0,
            channel_axis=2,
            convert2lab=True
        )

        unique_labels = np.unique(superpixels)
        n_segments = len(unique_labels)
        
        if unique_labels.max() != n_segments - 1:
            label_map = {old: new for new, old in enumerate(unique_labels)}
            superpixels = np.vectorize(label_map.get)(superpixels)
        
        return superpixels.astype(np.int32), n_segments
    
    def _build_rag(
        self,
        superpixels: np.ndarray,
        intensity: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], Dict[int, Dict]]:
        """build region adjacency graph from superpixels"""
        n_superpixels = superpixels.max() + 1

        props = regionprops_table(
            superpixels + 1,  # regionprops needs labels > 0
            intensity_image=intensity,
            properties=['label', 'area', 'centroid', 'mean_intensity']
        )
        
        node_features = {}
        for i in range(len(props['label'])):
            node_id = props['label'][i] - 1  # back to 0 indexed
            node_features[node_id] = {
                'mean_intensity': props['mean_intensity'][i],
                'area': props['area'][i],
                'centroid': (props['centroid-0'][i], props['centroid-1'][i])
            }
        
        edge_set = set()

        h_diff = superpixels[:, 1:] != superpixels[:, :-1]
        rows, cols = np.where(h_diff)
        for r, c in zip(rows, cols):
            s, t = superpixels[r, c], superpixels[r, c+1]
            edge_set.add((min(s,t), max(s,t)))

        v_diff = superpixels[1:, :] != superpixels[:-1, :]
        rows, cols = np.where(v_diff)
        for r, c in zip(rows, cols):
            s, t = superpixels[r, c], superpixels[r+1, c]
            edge_set.add((min(s,t), max(s,t)))

        edge_list = list(edge_set)

        return edge_list, node_features
    
    def _compute_oriented_weights(
        self,
        node_features: Dict[int, Dict]
    ) -> np.ndarray:
        """compute oriented weights for uoift"""
        alpha = self.config.alpha
        edge_list, _ = self._build_rag(self._superpixels, self._working_channel)

        weights = np.zeros(len(edge_list), dtype=np.float64)

        for idx, (s, t) in enumerate(edge_list):
            Is = node_features.get(s, {}).get('mean_intensity', 0.5)
            It = node_features.get(t, {}).get('mean_intensity', 0.5)

            diff = abs(It - Is)

            if Is > It:
                weight = diff * (1.0 + alpha)
            else:
                weight = diff * (1.0 - alpha)

            weights[idx] = max(weight, 1e-8)

        return weights
    
    def _build_hierarchy(
        self,
        edge_weights: np.ndarray
    ) -> Tuple['hg.Tree', np.ndarray, List[Tuple]]:
        """build binary partition tree via single linkage"""
        edge_list, _ = self._build_rag(self._superpixels, self._working_channel)
        n_vertices = self._superpixels.max() + 1

        if len(edge_list) == 0:
            raise ValueError("graph has no edges - check superpixel segmentation")

        sources = np.array([e[0] for e in edge_list], dtype=np.int64)
        targets = np.array([e[1] for e in edge_list], dtype=np.int64)

        hg_graph = hg.UndirectedGraph(n_vertices)
        hg_graph.add_edges(sources, targets)

        tree, altitudes = hg.bpt_canonical(hg_graph, edge_weights)

        merge_history = self._extract_merge_history(tree, altitudes)

        return tree, altitudes, merge_history

    def _extract_merge_history(
        self,
        tree: 'hg.Tree',
        altitudes: np.ndarray
    ) -> List[Tuple[int, int, int, float]]:
        """extract merge history from bpt"""
        merge_history = []
        n_leaves = tree.num_leaves()

        for node in range(n_leaves, tree.num_vertices()):
            children = tree.children(node)
            if len(children) >= 2:
                merge_history.append((
                    int(node),
                    int(children[0]),
                    int(children[1]),
                    float(altitudes[node])
                ))

        merge_history.sort(key=lambda x: x[3])

        return merge_history
    
    def _compute_extinction(
        self,
        tree: 'hg.Tree',
        altitudes: np.ndarray,
        node_features: Dict[int, Dict]
    ) -> np.ndarray:
        """compute extinction values via component tree"""
        n_leaves = tree.num_leaves()

        leaf_areas = np.zeros(n_leaves, dtype=np.float64)
        for node_id, features in node_features.items():
            if node_id < n_leaves:
                leaf_areas[node_id] = features.get('area', 1)

        tree_areas = hg.accumulate_sequential(
            tree,
            leaf_areas,
            hg.Accumulators.sum
        )

        if self.config.extinction_attribute == 'volume':
            attribute = tree_areas * altitudes
        elif self.config.extinction_attribute == 'area':
            attribute = tree_areas
        else:
            attribute = altitudes

        extinction_values = hg.attribute_extinction_value(tree, altitudes, attribute)

        return extinction_values
    
    def _generate_ucm(
        self,
        tree: 'hg.Tree',
        altitudes: np.ndarray,
        extinction_values: np.ndarray
    ) -> np.ndarray:
        """generate ultrametric contour map"""
        H, W = self._superpixels.shape
        n_leaves = tree.num_leaves()

        max_extinction = extinction_values.max()
        if max_extinction > 0:
            threshold = max_extinction * self.config.min_extinction_ratio

            filtered_altitudes = altitudes.copy()
            for node in range(tree.num_vertices()):
                if extinction_values[node] < threshold:
                    filtered_altitudes[node] = 0.0
        else:
            filtered_altitudes = altitudes

        leaf_saliency = np.zeros(n_leaves, dtype=np.float64)

        for leaf in range(n_leaves):
            parent = tree.parent(leaf)
            leaf_saliency[leaf] = altitudes[parent]

        saliency_map = np.zeros((H, W), dtype=np.float32)

        for sp_id in range(n_leaves):
            mask = self._superpixels == sp_id
            saliency_map[mask] = leaf_saliency[sp_id]

        saliency_map = self._enhance_boundaries(saliency_map, leaf_saliency)

        smap_min, smap_max = saliency_map.min(), saliency_map.max()
        if smap_max > smap_min:
            saliency_map = (saliency_map - smap_min) / (smap_max - smap_min)

        return saliency_map.astype(np.float32)

    def _enhance_boundaries(
        self,
        saliency_map: np.ndarray,
        leaf_saliency: np.ndarray
    ) -> np.ndarray:
        """enhance boundaries between superpixels"""
        H, W = self._superpixels.shape
        enhanced = saliency_map.copy()

        if self.config.boundary_mode == 'gradient':
            from scipy.ndimage import maximum_filter, minimum_filter
            dilated = maximum_filter(self._superpixels, size=3)
            eroded = minimum_filter(self._superpixels, size=3)
            boundary_mask = dilated != eroded

            saliency_dilated = maximum_filter(saliency_map, size=3)
            enhanced[boundary_mask] = saliency_dilated[boundary_mask]

        else:
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                shifted_sp = np.roll(np.roll(self._superpixels, dy, axis=0), dx, axis=1)
                boundary = self._superpixels != shifted_sp

                if dy == -1:
                    boundary[0, :] = False
                elif dy == 1:
                    boundary[-1, :] = False
                if dx == -1:
                    boundary[:, 0] = False
                elif dx == 1:
                    boundary[:, -1] = False
                
                shifted_sal = np.roll(np.roll(saliency_map, dy, axis=0), dx, axis=1)
                enhanced[boundary] = np.maximum(
                    enhanced[boundary], 
                    shifted_sal[boundary]
                )
        
        return enhanced


def generate_uoift_saliency(
    image: np.ndarray,
    sp_size: int = 100,
    alpha: float = -0.7,
    config_overrides: Optional[Dict] = None
) -> Tuple[np.ndarray, Dict]:
    """generate topological saliency map via uoift"""
    config = UOIFTConfig(sp_size=sp_size, alpha=alpha)

    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                warnings.warn(f"unknown parameter ignored: {key}")

    generator = UOIFTSaliencyGenerator(config)
    return generator.generate_saliency(image)


def save_merge_history(
    merge_history: List[Tuple],
    filepath: str,
    include_header: bool = True
) -> None:
    """save merge history to text file"""
    with open(filepath, 'w') as f:
        if include_header:
            f.write("# UOIFT Merge History\n")
            f.write("# Format: parent_node child1 child2 fusion_altitude\n")
            f.write(f"# Total fusions: {len(merge_history)}\n")


        for parent, c1, c2, alt in merge_history:
            f.write(f"{parent} {c1} {c2} {alt:.8f}\n")


def save_saliency_map(
    saliency_map: np.ndarray,
    filepath: str,
    enhance_contrast: bool = True,
    power: float = 1.5
) -> None:
    """save saliency map with optional contrast enhancement"""
    saliency = saliency_map.copy()

    if enhance_contrast:
        saliency = np.power(saliency, power)

        smin, smax = saliency.min(), saliency.max()
        if smax > smin:
            saliency = (saliency - smin) / (smax - smin)

    saliency_uint8 = np.clip(saliency * 255, 0, 255).astype(np.uint8)

    if filepath.lower().endswith('.pgm'):
        H, W = saliency_uint8.shape
        with open(filepath, 'wb') as f:
            header = f"P5\n{W} {H}\n255\n".encode('ascii')
            f.write(header)
            f.write(saliency_uint8.tobytes())
    else:
        try:
            import imageio
            imageio.imwrite(filepath, saliency_uint8)
        except ImportError:
            import cv2
            cv2.imwrite(filepath, saliency_uint8)


def load_saliency_map(filepath: str) -> np.ndarray:
    """load saliency map from file"""
    if filepath.lower().endswith('.pgm'):
        with open(filepath, 'rb') as f:
            magic = f.readline().decode().strip()
            if magic != 'P5':
                raise ValueError(f"invalid PGM format: {magic}")

            line = f.readline().decode()
            while line.startswith('#'):
                line = f.readline().decode()

            W, H = map(int, line.strip().split())
            maxval = int(f.readline().decode().strip())

            data = np.frombuffer(f.read(), dtype=np.uint8).reshape((H, W))
    else:
        try:
            import imageio
            data = imageio.imread(filepath)
        except ImportError:
            import cv2
            data = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    return data.astype(np.float32) / 255.0

