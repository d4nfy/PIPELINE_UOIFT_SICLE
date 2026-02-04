"""nuclei segmentation pipeline: uoift + sicle + stardist

architecture (sec 3.1):
    uoift saliency -> dual path sicle -> veta filtering -> stardist anchor -> iou arbitration
"""
__version__ = "2.0.0"
