"""Per-cell featurization (quantitative image analysis).

Given an instance-label image and the original intensity image, extract a
morphology + intensity feature table (one row per nucleus) plus summary stats.
This is the "feature extraction / featurization" and "quantitative image
analysis" deliverable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table


FEATURES = [
    "label",
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "equivalent_diameter",
    "mean_intensity",
    "max_intensity",
    "centroid",
]


def featurize(label: np.ndarray, intensity: np.ndarray) -> pd.DataFrame:
    """Return a per-nucleus feature DataFrame."""
    if label.max() == 0:
        return pd.DataFrame(columns=[
            "label", "area", "perimeter", "eccentricity", "solidity",
            "equivalent_diameter", "mean_intensity", "max_intensity",
            "centroid_y", "centroid_x",
        ])
    props = regionprops_table(
        label.astype(np.int32),
        intensity_image=intensity,
        properties=FEATURES,
    )
    df = pd.DataFrame(props)
    df = df.rename(columns={"centroid-0": "centroid_y", "centroid-1": "centroid_x"})
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Image-level summary: cell count and mean morphology."""
    if len(df) == 0:
        return {"cell_count": 0}
    return {
        "cell_count": int(len(df)),
        "mean_area": float(df["area"].mean()),
        "median_area": float(df["area"].median()),
        "mean_eccentricity": float(df["eccentricity"].mean()),
        "mean_solidity": float(df["solidity"].mean()),
        "mean_intensity": float(df["mean_intensity"].mean()),
    }
