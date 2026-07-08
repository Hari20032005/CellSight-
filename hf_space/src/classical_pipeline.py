"""Custom OpenCV instance-segmentation pipeline (marker-controlled watershed).

This is the hand-built "custom algorithm" that the JD explicitly values over
off-the-shelf tools. It turns an enhanced grayscale tile into an integer label
image where each nucleus gets a unique id.

Steps:
    1. Otsu threshold -> binary foreground
    2. Morphological opening -> remove speckle
    3. Distance transform + adaptive peak threshold -> "sure foreground"
       (this is what separates *touching* nuclei)
    4. Sure background via dilation; unknown = bg - fg
    5. connectedComponents -> markers; watershed -> instance boundaries
    6. Size filtering -> drop debris
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .preprocessing import PreprocessParams, enhance


@dataclass
class SegParams:
    open_kernel: int = 3
    open_iters: int = 2
    dist_peak_frac: float = 0.5   # fraction of max distance = sure-fg threshold
    bg_dilate_iters: int = 3
    min_area: int = 15            # drop objects smaller than this (pixels)


def _foreground_mask(gray: np.ndarray, p: SegParams) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.open_kernel, p.open_kernel))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=p.open_iters)


def segment_watershed(gray: np.ndarray, p: SegParams | None = None) -> np.ndarray:
    """Return an int32 label image (0 = background) from an enhanced tile."""
    p = p or SegParams()
    binary = _foreground_mask(gray, p)

    # Sure background = dilate the foreground.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(binary, k, iterations=p.bg_dilate_iters)

    # Sure foreground = peaks of the distance transform (separates touching cells).
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, p.dist_peak_frac * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)

    # Markers: label sure-fg components, shift so background is 1, mark unknown 0.
    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)

    # watershed labels: -1 = boundaries, 1 = background, >=2 = instances.
    label = np.zeros(gray.shape, dtype=np.int32)
    next_id = 1
    for m in range(2, markers.max() + 1):
        blob = markers == m
        if blob.sum() < p.min_area:
            continue
        label[blob] = next_id
        next_id += 1
    return label


def run_classical(gray: np.ndarray,
                  pre: PreprocessParams | None = None,
                  seg: SegParams | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Full classical path: enhance -> watershed. Returns (enhanced, label)."""
    enhanced = enhance(gray, pre)
    label = segment_watershed(enhanced, seg)
    return enhanced, label
