"""Dataset loading for CellSight.

Primary format: the 2018 Data Science Bowl (DSB / BBBC038) layout, where each
sample is a folder:

    <sample_id>/
        images/<sample_id>.png      # the microscopy tile (RGB or grayscale)
        masks/<mask_id>.png         # one binary PNG per nucleus (instance masks)

We collapse the per-nucleus mask PNGs into a single integer *label image* where
each nucleus has a unique id (0 = background). That label image is the ground
truth used everywhere else in the project.

Backup format: BBBC039 ships images and a single indexed-mask PNG per image;
`load_label_image` handles a single indexed mask too.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2


def read_gray(path: str | Path) -> np.ndarray:
    """Read an image as single-channel uint8 grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 3:
        # Drop alpha, convert BGR(A) -> gray
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        # Normalize 16-bit fluorescence to 8-bit for display/processing
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def masks_to_label(mask_dir: str | Path) -> np.ndarray:
    """Combine a directory of single-nucleus binary PNGs into one label image."""
    mask_dir = Path(mask_dir)
    mask_paths = sorted(mask_dir.glob("*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No mask PNGs found in {mask_dir}")
    first = cv2.imread(str(mask_paths[0]), cv2.IMREAD_GRAYSCALE)
    label = np.zeros(first.shape, dtype=np.int32)
    for idx, mp in enumerate(mask_paths, start=1):
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        label[m > 0] = idx
    return label


def load_dsb_sample(sample_dir: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load one DSB sample folder. Returns (gray_image, label_or_None)."""
    sample_dir = Path(sample_dir)
    img_dir = sample_dir / "images"
    img_paths = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.tif"))
    if not img_paths:
        raise FileNotFoundError(f"No image found under {img_dir}")
    image = read_gray(img_paths[0])

    label = None
    mask_dir = sample_dir / "masks"
    if mask_dir.is_dir() and any(mask_dir.glob("*.png")):
        label = masks_to_label(mask_dir)
    return image, label


def iter_dsb_samples(root: str | Path):
    """Yield (sample_id, image, label) for every DSB sample folder under `root`."""
    root = Path(root)
    for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (sample_dir / "images").is_dir():
            continue
        image, label = load_dsb_sample(sample_dir)
        yield sample_dir.name, image, label


def load_label_image(path: str | Path) -> np.ndarray:
    """Load a single indexed-label PNG (BBBC039-style) as an int32 label image."""
    lbl = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if lbl is None:
        raise FileNotFoundError(f"Could not read label image: {path}")
    if lbl.ndim == 3:
        lbl = lbl[..., 0]
    return lbl.astype(np.int32)
