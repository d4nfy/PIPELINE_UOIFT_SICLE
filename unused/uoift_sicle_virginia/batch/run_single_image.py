# -*- coding: utf-8 -*-
"""single image pipeline runner

full pipeline on one image:
1. uoift saliency (sec 3.2, via wsl binary)
2. sicle multiscale segmentation (sec 3.3)
3. stardist anchor detection (sec 3.5)
4. fusion via iou (sec 3.5)
5. lab + shape filter (sec 3.4)
"""
import sys
from pathlib import Path
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline_utils import (
    CFG,
    ensure_dir,
    save_uint8,
    save_uint16,
    get_lab_channels,
    fuse_segmentations,
    filter_path_regions,
    filter_lab,
)
from core.uoift_wrapper import run_uoift_wsl
from core.sicle_wrapper import run_sicle_with_params
from core.stardist_wrapper import stardist_predict, load_stardist_model

log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'pipeline.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# sicle params for single image mode
SICLE_PARAMS = {
    'n0': 52000,
    'nf': 3000,
    'alpha': 0.7,
    'max_iters': 22,
    'irreg': 0.12,
    'adhr': 16,
    'sampl_opt': 'grid',
}


def run_full_pipeline(
    img_path: str,
    output_dir: str = "logs",
    polarity: float = -1.0,
    skip_stardist: bool = False,
    skip_fusion: bool = False,
    skip_filters: bool = False,
    verbose: bool = True
) -> Dict:
    """complete pipeline: uoift + sicle + stardist + fusion + filter"""
    from skimage import io
    from skimage.color import label2rgb

    log.info(f"pipeline start: {img_path}")
    log.info(f"  polarity={polarity}")

    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    stats = {
        'pipeline_start': datetime.now().isoformat(),
        'input_path': str(img_path)
    }

    # load image
    img_rgb = io.imread(img_path)
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb] * 3, axis=-1)
    elif img_rgb.shape[2] == 4:
        img_rgb = img_rgb[:, :, :3]

    save_uint8(output_dir / "00_input.png", img_rgb)
    log.info(f"image loaded: {img_rgb.shape}")
    stats['image_shape'] = img_rgb.shape

    l_ch, a_ch, b_ch = get_lab_channels(img_rgb)

    # step 1: uoift saliency (sec 3.2)
    log.info("step 1: uoift saliency")
    saliency_map = run_uoift_wsl(img_rgb, polarity=polarity, output_dir=output_dir)
    save_uint8(output_dir / "01_saliency.png", saliency_map)

    # step 2: sicle multiscale (sec 3.3)
    log.info("step 2: sicle multiscale")

    temp_saliency = output_dir / "saliency_temp.png"
    temp_img = output_dir / "img_temp.png"
    save_uint8(temp_saliency, saliency_map)
    save_uint8(temp_img, img_rgb)

    sicle_scales = run_sicle_with_params(
        img_path=str(temp_img),
        saliency_path=str(temp_saliency),
        output_path=str(output_dir / "sicle.pgm"),
        params=SICLE_PARAMS,
        multiscale=True
    )
    log.info(f"generated {len(sicle_scales)} sicle scales")
    stats['n_sicle_scales'] = len(sicle_scales)

    for name, mask in sicle_scales.items():
        viz = label2rgb(mask, image=img_rgb, bg_label=0)
        save_uint8(output_dir / f"02_{name.lower()}.png", (viz * 255).astype(np.uint8))

    # use middle scale as primary
    scale_names = list(sicle_scales.keys())
    middle_scale = scale_names[len(scale_names) // 2]
    sicle_mask = sicle_scales[middle_scale]
    log.info(f"using {middle_scale} as primary sicle output")

    # step 3: stardist anchor (sec 3.5)
    stardist_mask = None
    if not skip_stardist:
        log.info("step 3: stardist anchor")
        try:
            model = load_stardist_model()
            stardist_mask = stardist_predict(img_rgb, model=model)
            save_uint16(output_dir / "03_stardist.pgm", stardist_mask.astype(np.uint16))

            viz = label2rgb(stardist_mask, image=img_rgb, bg_label=0)
            save_uint8(output_dir / "03_stardist.png", (viz * 255).astype(np.uint8))

            n_stardist = stardist_mask.max()
            log.info(f"stardist: {n_stardist} instances")
            stats['n_stardist'] = int(n_stardist)
        except Exception as e:
            log.warning(f"stardist failed: {e}")
            stardist_mask = None
            stats['stardist_error'] = str(e)
    else:
        log.info("stardist skipped")

    # step 4: fusion (sec 3.5)
    log.info("step 4: iou fusion")

    if stardist_mask is not None and not skip_fusion:
        fused_mask = fuse_segmentations(
            stardist_mask.astype(np.int32),
            sicle_mask.astype(np.int32),
            iou_thresh=CFG['iou_threshold']
        )
        log.info(f"fused: {fused_mask.max()} instances")
        stats['n_fused'] = int(fused_mask.max())

        save_uint16(output_dir / "04_fused.pgm", fused_mask.astype(np.uint16))
        viz = label2rgb(fused_mask, image=img_rgb, bg_label=0)
        save_uint8(output_dir / "04_fused.png", (viz * 255).astype(np.uint8))
    else:
        if stardist_mask is not None:
            fused_mask = stardist_mask
            log.info("fusion skipped: using stardist directly")
        else:
            fused_mask = sicle_mask
            log.info(f"no stardist: using {middle_scale} directly")
        stats['fusion_skipped'] = True

    # step 5: lab + shape filter (sec 3.4)
    log.info("step 5: lab + shape filter")

    current_mask = fused_mask

    if not skip_filters:
        # shape filter
        filtered_shape, kept_shape, rej_shape = filter_path_regions(
            current_mask.astype(np.int32), CFG, l_ch, a_ch, b_ch
        )
        log.info(f"shape filter: kept={kept_shape}, rejected={rej_shape}")

        # lab filter
        lab_mask, kept_lab, rej_lab = filter_lab(
            filtered_shape, l_ch, a_ch, b_ch, CFG
        )
        log.info(f"lab filter: kept={kept_lab}, rejected={rej_lab}")

        from skimage.measure import label as sk_label
        current_mask = sk_label(lab_mask).astype(np.int32)

        stats['filter_shape'] = {'kept': kept_shape, 'rejected': rej_shape}
        stats['filter_lab'] = {'kept': kept_lab, 'rejected': rej_lab}

        save_uint16(output_dir / "05_filtered.pgm", current_mask.astype(np.uint16))
        viz = label2rgb(current_mask, image=img_rgb, bg_label=0)
        save_uint8(output_dir / "05_filtered.png", (viz * 255).astype(np.uint8))
    else:
        log.info("filtering skipped")

    # output
    final_mask = current_mask
    save_uint16(output_dir / "06_final.pgm", final_mask.astype(np.uint16))

    final_viz = label2rgb(final_mask, image=img_rgb, bg_label=0)
    save_uint8(output_dir / "06_final.png", (final_viz * 255).astype(np.uint8))

    n_final = final_mask.max()
    stats['n_final'] = int(n_final)
    stats['pipeline_end'] = datetime.now().isoformat()

    log.info(f"pipeline complete: {n_final} nuclei")

    results = {
        'img_rgb': img_rgb,
        'saliency_map': saliency_map,
        'stardist_mask': stardist_mask,
        'sicle_scales': sicle_scales,
        'fused_mask': fused_mask,
        'final_mask': final_mask,
        'stats': stats
    }

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="single image pipeline runner")
    parser.add_argument("image", help="path to input rgb image")
    parser.add_argument("--output", "-o", default="logs", help="output folder")
    parser.add_argument("--polarity", type=float, default=-1.0,
                        help="boundary polarity: -1.0=dark nuclei (default)")
    parser.add_argument("--skip-stardist", action="store_true",
                        help="skip stardist, sicle only")
    parser.add_argument("--skip-fusion", action="store_true",
                        help="skip fusion, stardist only")
    parser.add_argument("--skip-filters", action="store_true",
                        help="skip lab + shape filtering")

    args = parser.parse_args()

    result = run_full_pipeline(
        img_path=args.image,
        output_dir=args.output,
        polarity=args.polarity,
        skip_stardist=args.skip_stardist,
        skip_fusion=args.skip_fusion,
        skip_filters=args.skip_filters
    )

    print(f"\npipeline completed: {result['stats']['n_final']} nuclei segmented")
    print(f"output saved to: {args.output}/")
