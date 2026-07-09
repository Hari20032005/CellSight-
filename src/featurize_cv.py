"""Pure-OpenCV/NumPy per-cell featurization (no scikit-image).

Used by the lightweight web frontend so the Render/free-tier deployment doesn't
need scikit-image. Produces the same core morphology fields as
`featurize.py`: area, perimeter, eccentricity, solidity, equivalent diameter,
and mean/max intensity per nucleus.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def featurize_cv(label: np.ndarray, intensity: np.ndarray) -> list[dict]:
    """Return one dict of features per instance id in `label` (0 = background)."""
    rows = []
    ids = [i for i in np.unique(label) if i != 0]
    for lab in ids:
        mask = (label == lab).astype(np.uint8)
        area = int(mask.sum())
        if area == 0:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(cnt, True))

        # Solidity = area / convex-hull area
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull)) or float(area)
        solidity = float(area) / hull_area if hull_area else 1.0

        # Eccentricity from a fitted ellipse (needs >= 5 points)
        eccentricity = 0.0
        if len(cnt) >= 5:
            (_, (MA, ma), _) = cv2.fitEllipse(cnt)
            major, minor = max(MA, ma), min(MA, ma)
            if major > 0:
                ratio = (minor / major) ** 2
                eccentricity = math.sqrt(max(0.0, 1.0 - ratio))

        vals = intensity[mask.astype(bool)]
        m = cv2.moments(mask, binaryImage=True)
        cy = m["m01"] / m["m00"] if m["m00"] else 0.0
        cx = m["m10"] / m["m00"] if m["m00"] else 0.0

        rows.append({
            "label": int(lab),
            "area": area,
            "perimeter": round(perimeter, 2),
            "eccentricity": round(eccentricity, 3),
            "solidity": round(solidity, 3),
            "equivalent_diameter": round(math.sqrt(4 * area / math.pi), 2),
            "mean_intensity": round(float(vals.mean()), 1),
            "max_intensity": int(vals.max()),
            "centroid_y": round(cy, 1),
            "centroid_x": round(cx, 1),
        })
    return rows


def summarize_cv(rows: list[dict]) -> dict:
    if not rows:
        return {"cell_count": 0}
    areas = [r["area"] for r in rows]
    return {
        "cell_count": len(rows),
        "mean_area": round(sum(areas) / len(areas), 1),
        "mean_eccentricity": round(sum(r["eccentricity"] for r in rows) / len(rows), 3),
        "mean_solidity": round(sum(r["solidity"] for r in rows) / len(rows), 3),
        "mean_intensity": round(sum(r["mean_intensity"] for r in rows) / len(rows), 1),
    }
