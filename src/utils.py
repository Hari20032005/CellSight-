"""Visualization helpers: colored instance overlays for figures and the demo."""
from __future__ import annotations

import cv2
import numpy as np


def colorize_labels(label: np.ndarray, seed: int = 0) -> np.ndarray:
    """Map an instance-label image to a random BGR color image (0 = black)."""
    rng = np.random.default_rng(seed)
    max_id = int(label.max())
    colors = rng.integers(60, 255, size=(max_id + 1, 3), dtype=np.uint8)
    colors[0] = (0, 0, 0)
    return colors[label.clip(min=0)]


def overlay_labels(gray: np.ndarray, label: np.ndarray, alpha: float = 0.45,
                   draw_contours: bool = True) -> np.ndarray:
    """Blend colored instances over the grayscale tile, outline each nucleus."""
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    colored = colorize_labels(label)
    blended = cv2.addWeighted(base, 1 - alpha, colored, alpha, 0)
    if draw_contours:
        mask = (label > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (255, 255, 255), 1)
    return blended


def label_count_banner(image_bgr: np.ndarray, text: str) -> np.ndarray:
    """Draw a small caption in the top-left corner."""
    out = image_bgr.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 220), 24), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out
