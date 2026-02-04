#!/usr/bin/env python3
"""unified pipeline orchestrator

one command per pipeline step, plus 'all' to run everything.
each step caches its output as .npy/.npz. use --force to re-run.

usage:
  python run.py all B3-T_1
  python run.py all B3-T_1 --no-saliency
  python run.py all --all --no-saliency
  python run.py stardist B3-T_1
  python run.py sicle-a B3-T_1 --force

steps (in pipeline order):
  saliency    uoift boundary saliency (sec 3.2)
  sicle-a     sicle path a, coarse nf=3000 (sec 3.3)
  filter-a    veta filter on path a (sec 3.4)
  sicle-b     sicle path b, multiscale nf=500 (sec 3.3)
  filter-b    veta multiscale selection (sec 3.4)
  pool        candidate pool: merge a + b (sec 3.3)
  stardist    stardist anchor detection (sec 3.5)
  fuse        stardist-pool iou fusion (sec 3.5)
  arbitrate   final solidity filter (sec 3.5)
  visualize   generate overlay visualizations
  all         run complete pipeline
"""
import sys
import json
import functools
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
print = functools.partial(print, flush=True)

from core.config import (
    ALL_IMAGES, SICLE_PATH_A, SICLE_PATH_B,
    FILTER_CFG, UOIFT_ALPHA, UOIFT_SP_SIZE, ION_THRESH,
)
from core.pipeline_utils import (
    get_lab_channels, save_uint8, save_uint16, ensure_dir, Timer,
)

BASE_IMG_DIR = Path(__file__).parent / "img_used"
DEFAULT_OUTPUT = Path(__file__).parent / "pipeline_output"

# step output filenames
STEP_FILES = {
    'saliency':  '01_saliency',
    'sicle_a':   '02_sicle_path_a',
    'filter_a':  '03_path_a_filtered',
    'sicle_b':   '04_sicle_path_b_scales',
    'filter_b':  '05_path_b_veta',
    'pool':      '06_pool',
    'stardist':  '07_stardist',
    'fuse':      '08_fused',
    'arbitrate': '09_final',
}


# -- helpers --

def resolve_image(name_or_path):
    """resolve dataset name or file path to (img_rgb, name, path)"""
    from skimage import io as skio

    p = Path(name_or_path)
    if p.exists() and p.is_file():
        img = skio.imread(str(p))
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img, p.stem, p

    candidates = [
        BASE_IMG_DIR / name_or_path / "pre_seg" / "img_original.png",
        BASE_IMG_DIR / name_or_path / f"{name_or_path}.png",
    ]
    for c in candidates:
        if c.exists():
            img = skio.imread(str(c))
            if img.shape[2] == 4:
                img = img[:, :, :3]
            return img, name_or_path, c

    raise FileNotFoundError(
        f"image not found: {name_or_path}\n"
        f"tried: {[str(c) for c in candidates]}"
    )


def get_output_dir(args, img_name):
    """resolve output directory"""
    if hasattr(args, 'output') and args.output:
        return Path(args.output) / img_name
    return DEFAULT_OUTPUT / img_name


def save_step(output_dir, key, data):
    """save step output as npy (array) or npz (dict)"""
    if isinstance(data, dict):
        np.savez(output_dir / f"{STEP_FILES[key]}.npz", **data)
    else:
        np.save(output_dir / f"{STEP_FILES[key]}.npy", data)


def load_step(output_dir, key, required=True):
    """load cached step output"""
    npy = output_dir / f"{STEP_FILES[key]}.npy"
    npz = output_dir / f"{STEP_FILES[key]}.npz"

    if npy.exists():
        return np.load(str(npy), allow_pickle=False)
    if npz.exists():
        data = np.load(str(npz), allow_pickle=False)
        return {k: data[k] for k in data.files}

    if required:
        step_cmd = key.replace('_', '-')
        raise FileNotFoundError(
            f"missing output for step '{key}': {npy}\n"
            f"run first: python run.py {step_cmd} <image>"
        )
    return None


def is_cached(output_dir, key):
    """check if step output already exists"""
    npy = output_dir / f"{STEP_FILES[key]}.npy"
    npz = output_dir / f"{STEP_FILES[key]}.npz"
    return npy.exists() or npz.exists()


# -- step implementations --

def step_saliency(args):
    """generate uoift boundary saliency (sec 3.2)"""
    img_rgb, img_name, img_path = resolve_image(args.image)
    out = get_output_dir(args, img_name)
    ensure_dir(out)

    if is_cached(out, 'saliency') and not args.force:
        print(f"  cached: saliency")
        return

    from core.uoift_wrapper import run_uoift_wsl

    with Timer("uoift saliency"):
        sal = run_uoift_wsl(img_rgb, polarity=UOIFT_ALPHA, sp_size=UOIFT_SP_SIZE)

    save_step(out, 'saliency', sal)
    save_uint8(out / "01_saliency.png", sal)


def step_sicle_a(args):
    """run sicle path a, coarse nf=3000 (sec 3.3)"""
    _, img_name, img_path = resolve_image(args.image)
    out = get_output_dir(args, img_name)
    ensure_dir(out)

    if is_cached(out, 'sicle_a') and not args.force:
        print(f"  cached: sicle path a")
        return

    from core.sicle_wrapper import run_sicle_path_a

    sal_png = out / "01_saliency.png"
    sal_arg = str(sal_png) if sal_png.exists() else None

    with Timer("sicle path a"):
        labels = run_sicle_path_a(str(img_path), sal_arg, str(out / "sicle_a.pgm"))

    save_step(out, 'sicle_a', labels)
    n = len(np.unique(labels)) - 1
    print(f"  path a: {n} raw regions")


def step_filter_a(args):
    """veta filter on path a (sec 3.4)"""
    img_rgb, img_name, _ = resolve_image(args.image)
    out = get_output_dir(args, img_name)

    if is_cached(out, 'filter_a') and not args.force:
        print(f"  cached: filter path a")
        return

    path_a = load_step(out, 'sicle_a')
    from core.veta_filtering import filter_sicle_regions

    l, a, b = get_lab_channels(img_rgb)

    with Timer("filter path a"):
        filtered, kept, rej = filter_sicle_regions(path_a, l, a, b)

    save_step(out, 'filter_a', filtered)
    print(f"  path a: {kept} kept, {rej} rejected")


def step_sicle_b(args):
    """run sicle path b, multiscale nf=500 (sec 3.3)"""
    _, img_name, img_path = resolve_image(args.image)
    out = get_output_dir(args, img_name)
    ensure_dir(out)

    if is_cached(out, 'sicle_b') and not args.force:
        print(f"  cached: sicle path b")
        return

    from core.sicle_wrapper import run_sicle_path_b

    sal_png = out / "01_saliency.png"
    sal_arg = str(sal_png) if sal_png.exists() else None

    with Timer("sicle path b (multiscale)"):
        scales = run_sicle_path_b(
            str(img_path), sal_arg, str(out / "sicle_b.pgm"), multiscale=True
        )

    save_step(out, 'sicle_b', scales)
    print(f"  path b: {len(scales)} scales")


def step_filter_b(args):
    """veta multiscale selection on path b (sec 3.4)"""
    img_rgb, img_name, _ = resolve_image(args.image)
    out = get_output_dir(args, img_name)

    if is_cached(out, 'filter_b') and not args.force:
        print(f"  cached: filter path b")
        return

    scales = load_step(out, 'sicle_b')
    from core.veta_filtering import veta_multiscale_selection

    l, a, b = get_lab_channels(img_rgb)

    with Timer("veta multiscale selection"):
        merged = veta_multiscale_selection(scales, l, a, b)

    save_step(out, 'filter_b', merged)
    n = len(np.unique(merged)) - 1
    print(f"  path b veta: {n} selected")


def step_pool(args):
    """merge path a + path b into candidate pool (sec 3.3)"""
    _, img_name, _ = resolve_image(args.image)
    out = get_output_dir(args, img_name)

    if is_cached(out, 'pool') and not args.force:
        print(f"  cached: pool")
        return

    path_a = load_step(out, 'filter_a')
    path_b = load_step(out, 'filter_b')

    from core.veta_filtering import create_candidate_pool

    with Timer("candidate pool"):
        pool = create_candidate_pool(path_a, {"veta": path_b})

    save_step(out, 'pool', pool)
    n = len(np.unique(pool)) - 1
    print(f"  pool: {n} candidates")


def step_stardist(args):
    """stardist 2d_versatile_he anchor detection (sec 3.5)"""
    img_rgb, img_name, _ = resolve_image(args.image)
    out = get_output_dir(args, img_name)
    ensure_dir(out)

    if is_cached(out, 'stardist') and not args.force:
        print(f"  cached: stardist")
        return

    from core.stardist_wrapper import load_stardist_model, stardist_predict

    with Timer("stardist"):
        model = load_stardist_model()
        sd = stardist_predict(img_rgb, model=model)

    save_step(out, 'stardist', sd)
    n = len(np.unique(sd)) - 1
    print(f"  stardist: {n} nuclei")


def step_fuse(args):
    """fuse stardist anchors with sicle pool via iou (sec 3.5)"""
    _, img_name, _ = resolve_image(args.image)
    out = get_output_dir(args, img_name)

    if is_cached(out, 'fuse') and not args.force:
        print(f"  cached: fuse")
        return

    pool = load_step(out, 'pool')
    stardist = load_step(out, 'stardist')

    from core.veta_filtering import fuse_stardist_pool

    with Timer("stardist-pool fusion"):
        fused = fuse_stardist_pool(stardist, pool, iou_thresh=ION_THRESH)

    save_step(out, 'fuse', fused)
    n = len(np.unique(fused)) - 1
    print(f"  fused: {n} regions")


def step_arbitrate(args):
    """final solidity + area arbitration (sec 3.5)"""
    _, img_name, _ = resolve_image(args.image)
    out = get_output_dir(args, img_name)

    if is_cached(out, 'arbitrate') and not args.force:
        print(f"  cached: arbitrate")
        return

    fused = load_step(out, 'fuse')
    from core.veta_filtering import arbitrate_by_solidity

    arb_cfg = {
        "arb_min_area": FILTER_CFG["min_area"],
        "arb_min_solidity": FILTER_CFG["min_solidity"],
    }

    with Timer("solidity arbitration"):
        final, kept, rej = arbitrate_by_solidity(fused, arb_cfg)

    save_step(out, 'arbitrate', final)
    print(f"  final: {kept} nuclei, {rej} rejected")


def step_visualize(args):
    """generate colored visualizations for all cached steps"""
    img_rgb, img_name, _ = resolve_image(args.image)
    out = get_output_dir(args, img_name)
    ensure_dir(out)

    from core.veta_filtering import (
        _to_grayscale, labels_to_monochrome, labels_to_multicolor,
        pool_bicolor, create_overlay_on_rgb, create_fusion_decisions,
        VIS_HUES,
    )

    bg = _to_grayscale(img_rgb)

    # input for reference
    save_uint8(out / "vis_00_input.png", img_rgb)

    # path a filtered: green monochrome
    path_a = load_step(out, 'filter_a', required=False)
    if path_a is not None:
        viz = labels_to_monochrome(path_a, bg, VIS_HUES["path_a"])
        save_uint8(out / "vis_03_path_a.png", viz)
        print(f"  saved: vis_03_path_a.png ({len(np.unique(path_a)) - 1} regions)")

    # path b veta: cyan monochrome
    path_b = load_step(out, 'filter_b', required=False)
    if path_b is not None:
        viz = labels_to_monochrome(path_b, bg, VIS_HUES["path_b"])
        save_uint8(out / "vis_05_path_b.png", viz)
        print(f"  saved: vis_05_path_b.png ({len(np.unique(path_b)) - 1} regions)")

    # pool: bicolor green=a cyan=b
    pool = load_step(out, 'pool', required=False)
    if pool is not None and path_a is not None and path_b is not None:
        viz = pool_bicolor(pool, path_a, path_b, bg)
        save_uint8(out / "vis_06_pool.png", viz)
        print(f"  saved: vis_06_pool.png ({len(np.unique(pool)) - 1} regions)")

    # stardist: violet monochrome
    sd = load_step(out, 'stardist', required=False)
    if sd is not None:
        viz = labels_to_monochrome(sd, bg, VIS_HUES["stardist"])
        save_uint8(out / "vis_07_stardist.png", viz)
        print(f"  saved: vis_07_stardist.png ({len(np.unique(sd)) - 1} nuclei)")

    # fused: multicolor
    fused = load_step(out, 'fuse', required=False)
    if fused is not None:
        viz = labels_to_multicolor(fused, bg)
        save_uint8(out / "vis_08_fused.png", viz)
        print(f"  saved: vis_08_fused.png ({len(np.unique(fused)) - 1} regions)")

    # final: multicolor + overlay on rgb
    final = load_step(out, 'arbitrate', required=False)
    if final is not None:
        viz = labels_to_multicolor(final, bg)
        save_uint8(out / "vis_09_final.png", viz)
        print(f"  saved: vis_09_final.png ({len(np.unique(final)) - 1} nuclei)")

        overlay = create_overlay_on_rgb(img_rgb, final)
        save_uint8(out / "vis_09_final_overlay.png", overlay)
        print(f"  saved: vis_09_final_overlay.png")

    # fusion decisions: green=kept, blue=stardist, red=rejected
    if pool is not None and sd is not None and final is not None:
        viz = create_fusion_decisions(pool, sd, final, bg)
        save_uint8(out / "vis_10_fusion_decisions.png", viz)
        print(f"  saved: vis_10_fusion_decisions.png")


def step_all(args):
    """run complete pipeline end to end"""
    _, img_name, _ = resolve_image(args.image)
    out = get_output_dir(args, img_name)
    ensure_dir(out)

    no_sal = getattr(args, 'no_saliency', False)
    print(f"\npipeline: {img_name}")
    print(f"output:   {out}")
    print(f"saliency: {'skip' if no_sal else 'uoift'}\n")

    if not no_sal:
        step_saliency(args)
    else:
        print("  saliency: skipped (no-saliency mode)")

    step_sicle_a(args)
    step_filter_a(args)
    step_sicle_b(args)
    step_filter_b(args)
    step_pool(args)
    step_stardist(args)
    step_fuse(args)
    step_arbitrate(args)
    step_visualize(args)

    final = load_step(out, 'arbitrate')
    n_final = len(np.unique(final)) - 1

    report = {
        'image': img_name,
        'n_final': n_final,
        'no_saliency': no_sal,
    }
    with open(out / "report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\ndone: {n_final} nuclei -> {out / 'report.json'}")


# -- cli --

STEP_MAP = {
    'saliency':  step_saliency,
    'sicle-a':   step_sicle_a,
    'filter-a':  step_filter_a,
    'sicle-b':   step_sicle_b,
    'filter-b':  step_filter_b,
    'pool':      step_pool,
    'stardist':  step_stardist,
    'fuse':      step_fuse,
    'arbitrate': step_arbitrate,
    'visualize': step_visualize,
    'all':       step_all,
}


def main():
    parser = argparse.ArgumentParser(
        description="unified pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
steps (in pipeline order):
  saliency    uoift boundary saliency (sec 3.2)      [optional]
  sicle-a     sicle path a, coarse nf=3000 (sec 3.3)
  filter-a    veta filter on path a (sec 3.4)
  sicle-b     sicle path b, multiscale nf=500 (sec 3.3)
  filter-b    veta multiscale selection (sec 3.4)
  pool        merge path a + path b candidates
  stardist    stardist anchor detection (sec 3.5)
  fuse        stardist-pool iou fusion (sec 3.5)
  arbitrate   final solidity filter (sec 3.5)
  visualize   generate overlay images
  all         run complete pipeline

examples:
  python run.py all B3-T_1
  python run.py all B3-T_1 --no-saliency
  python run.py all --all --no-saliency
  python run.py stardist B3-T_1
  python run.py sicle-a B3-T_1 --force
  python run.py all path/to/image.png -o results/
""",
    )

    sub = parser.add_subparsers(dest='step', help='pipeline step')

    def add_common(sp):
        sp.add_argument('image', nargs='?', help='dataset name (B3-T_1) or image path')
        sp.add_argument('--output', '-o', help='output directory')
        sp.add_argument('--force', '-f', action='store_true', help='ignore cache')
        sp.add_argument('--all', dest='run_all', action='store_true',
                        help='process all 10 images')

    for name in STEP_MAP:
        sp = sub.add_parser(name, help=STEP_MAP[name].__doc__)
        add_common(sp)
        if name == 'all':
            sp.add_argument('--no-saliency', action='store_true',
                            help='skip uoift saliency (2.4x faster)')

    args = parser.parse_args()

    if args.step is None:
        parser.print_help()
        sys.exit(0)

    handler = STEP_MAP[args.step]

    if getattr(args, 'run_all', False):
        for img_name in ALL_IMAGES:
            args.image = img_name
            print(f"\n{'='*50}")
            print(f"  {img_name}")
            print(f"{'='*50}")
            handler(args)
    elif args.image:
        handler(args)
    else:
        print("error: provide an image name or --all")
        sys.exit(1)


if __name__ == "__main__":
    main()
