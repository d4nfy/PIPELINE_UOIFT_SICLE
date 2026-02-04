"""stardist anchor detection wrapper (sec 3.5)

fast convex nuclei detection using star-convex polygons.
pretrained 2D_versatile_he model for h&e staining.
"""
import numpy as np
import logging

log = logging.getLogger(__name__)


def load_stardist_model(model_type='2D_versatile_he'):
    """load pretrained stardist model"""
    try:
        from stardist.models import StarDist2D
        model = StarDist2D.from_pretrained(model_type)
        log.info(f"stardist model loaded: {model_type}")
        return model
    except ImportError:
        log.error("stardist not installed. run: pip install stardist")
        raise


def stardist_predict(img_rgb, model=None, prob_thresh=0.5, nms_thresh=0.4, n_tiles=None):
    """run stardist instance segmentation on rgb image"""
    if model is None:
        model = load_stardist_model()

    from csbdeep.utils import normalize

    img_norm = normalize(img_rgb, 1, 99.8, axis=(0, 1))

    if n_tiles is None and img_rgb.shape[0] > 512:
        n_tiles = (2, 2, 1)

    labels, details = model.predict_instances(
        img_norm,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        n_tiles=n_tiles
    )

    log.info(f"stardist: {labels.max()} objects detected")
    return labels


def validate_stardist_with_markers(stardist_mask, sicle_seeds):
    """post filter: keep stardist instances overlapping sicle markers"""
    from skimage.measure import regionprops

    valid_labels = []
    for prop in regionprops(stardist_mask):
        cy, cx = int(prop.centroid[0]), int(prop.centroid[1])
        if sicle_seeds[cy, cx] > 0:
            valid_labels.append(prop.label)

    validated = np.where(np.isin(stardist_mask, valid_labels), stardist_mask, 0)
    log.info(f"marker validation: {len(valid_labels)}/{stardist_mask.max()} kept")
    return validated
