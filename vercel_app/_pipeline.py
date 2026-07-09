"""Self-contained CellSight pipeline for the Vercel serverless function.

Pure OpenCV + NumPy (no torch / scikit-image) so the function bundle stays small.
This mirrors the logic in the repo's `src/` (preprocessing -> watershed ->
featurization -> overlay); it is vendored here because Vercel only bundles files
inside the deployment root.
"""
import base64
import math

import cv2
import numpy as np


# ----------------------------- enhancement -------------------------------- #
def _maybe_invert(gray):
    if gray.mean() > 127:
        return cv2.bitwise_not(gray)
    return gray


def enhance(gray):
    g = _maybe_invert(gray.copy())
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    bg = cv2.morphologyEx(g, cv2.MORPH_OPEN, k)
    g = cv2.normalize(cv2.subtract(g, bg), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    g = cv2.fastNlMeansDenoising(g, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return g


# --------------------------- segmentation --------------------------------- #
def segment_watershed(gray, min_area=15):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=2)

    sure_bg = cv2.dilate(binary, k, iterations=3)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)

    label = np.zeros(gray.shape, dtype=np.int32)
    next_id = 1
    for m in range(2, markers.max() + 1):
        blob = markers == m
        if blob.sum() < min_area:
            continue
        label[blob] = next_id
        next_id += 1
    return label


def run_classical(gray):
    enhanced = enhance(gray)
    return enhanced, segment_watershed(enhanced)


# --------------------------- featurization -------------------------------- #
def featurize(label, intensity):
    rows = []
    for lab in [i for i in np.unique(label) if i != 0]:
        mask = (label == lab).astype(np.uint8)
        area = int(mask.sum())
        if area == 0:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        perim = float(cv2.arcLength(cnt, True))
        hull_area = float(cv2.contourArea(cv2.convexHull(cnt))) or float(area)
        solidity = float(area) / hull_area if hull_area else 1.0
        ecc = 0.0
        if len(cnt) >= 5:
            (_, (MA, ma), _) = cv2.fitEllipse(cnt)
            major, minor = max(MA, ma), min(MA, ma)
            if major > 0:
                ecc = math.sqrt(max(0.0, 1.0 - (minor / major) ** 2))
        vals = intensity[mask.astype(bool)]
        rows.append({
            "label": int(lab),
            "area": area,
            "perimeter": round(perim, 1),
            "eccentricity": round(ecc, 3),
            "solidity": round(solidity, 3),
            "equivalent_diameter": round(math.sqrt(4 * area / math.pi), 1),
            "mean_intensity": round(float(vals.mean()), 1),
        })
    return rows


def summarize(rows):
    if not rows:
        return {"cell_count": 0, "mean_area": 0, "mean_eccentricity": 0,
                "mean_solidity": 0, "mean_intensity": 0}
    n = len(rows)
    return {
        "cell_count": n,
        "mean_area": round(sum(r["area"] for r in rows) / n, 1),
        "mean_eccentricity": round(sum(r["eccentricity"] for r in rows) / n, 3),
        "mean_solidity": round(sum(r["solidity"] for r in rows) / n, 3),
        "mean_intensity": round(sum(r["mean_intensity"] for r in rows) / n, 1),
    }


# ----------------------------- overlay ------------------------------------ #
def overlay(gray, label, seed=0):
    rng = np.random.default_rng(seed)
    max_id = int(label.max())
    colors = rng.integers(60, 255, size=(max_id + 1, 3), dtype=np.uint8)
    colors[0] = (0, 0, 0)
    colored = colors[label.clip(min=0)]
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(base, 0.55, colored, 0.45, 0)
    contours, _ = cv2.findContours((label > 0).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 255, 255), 1)
    return blended


# --------------------------- io helpers ----------------------------------- #
def decode_image(data_uri, max_side=1024):
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    buf = np.frombuffer(base64.b64decode(data_uri), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not read image.")
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    return img


def to_data_uri(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    return "data:image/png;base64," + base64.b64encode(buf).decode()
