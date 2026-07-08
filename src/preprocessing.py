"""Image enhancement / denoising with OpenCV.

This is the "image enhancement / denoising" half of the pipeline. Every step is
classical OpenCV so it runs on a CPU-only 8 GB laptop. The output feeds the
custom watershed segmentation in `classical_pipeline.py`.

Pipeline:
    1. Illumination / background correction  (rolling-ball-style via large
       morphological opening, corrects uneven microscope lighting)
    2. CLAHE                                 (local contrast enhancement)
    3. Non-Local-Means denoising             (removes sensor noise, keeps edges)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessParams:
    background_kernel: int = 51   # size of morphological structuring element
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    nlm_h: float = 10.0           # denoise strength; higher = smoother
    invert_if_dark_bg: bool = True


def _maybe_invert(gray: np.ndarray) -> np.ndarray:
    """Ensure nuclei are bright on a dark background.

    Brightfield tiles often have dark nuclei on a light background; fluorescence
    is the opposite. We standardize to bright-foreground by checking the median.
    """
    if gray.mean() > 127:  # mostly-bright image => foreground likely dark
        return cv2.bitwise_not(gray)
    return gray


def correct_illumination(gray: np.ndarray, kernel: int) -> np.ndarray:
    """Subtract a smooth background estimate (uneven-illumination fix)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k)
    corrected = cv2.subtract(gray, background)
    return cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def enhance(gray: np.ndarray, params: PreprocessParams | None = None) -> np.ndarray:
    """Run the full enhancement chain. Input/return: uint8 grayscale."""
    params = params or PreprocessParams()
    g = gray.copy()
    if params.invert_if_dark_bg:
        g = _maybe_invert(g)
    g = correct_illumination(g, params.background_kernel)
    clahe = cv2.createCLAHE(
        clipLimit=params.clahe_clip,
        tileGridSize=(params.clahe_grid, params.clahe_grid),
    )
    g = clahe.apply(g)
    g = cv2.fastNlMeansDenoising(g, None, h=params.nlm_h,
                                 templateWindowSize=7, searchWindowSize=21)
    return g
