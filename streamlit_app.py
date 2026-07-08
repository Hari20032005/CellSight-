"""CellSight — Streamlit live demo (free hosting on Streamlit Community Cloud).

Upload a microscopy tile -> enhancement -> custom OpenCV watershed segmentation
-> per-cell morphology. Lightweight (no torch), so it fits free hosting tiers.

The Cellpose-SAM foundation-model comparison is benchmarked offline (see README);
it is too memory-heavy for a free host, so this live tool runs the custom
pipeline. If cellpose happens to be installed, a second method appears.
"""
import io

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.classical_pipeline import run_classical
from src.featurize import featurize, summarize
from src.utils import overlay_labels, label_count_banner

try:
    from src.cellpose_infer import segment_cellpose
    _HAS_CELLPOSE = True
except Exception:
    _HAS_CELLPOSE = False

st.set_page_config(page_title="CellSight — Microscopy Cell Analysis",
                   page_icon="🔬", layout="wide")

st.title("🔬 CellSight")
st.caption("Microscopy cell/nuclei segmentation & quantification — "
           "custom OpenCV pipeline. "
           "[Code + Cellpose-SAM benchmark](https://github.com/Hari20032005/CellSight-)")


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


methods = ["OpenCV watershed (custom)"]
if _HAS_CELLPOSE:
    methods.append("Cellpose-SAM")

with st.sidebar:
    st.header("Input")
    upload = st.file_uploader("Microscopy image (PNG/TIF/JPG)",
                              type=["png", "tif", "tiff", "jpg", "jpeg"])
    method = st.radio("Segmentation method", methods)
    st.markdown("---")
    st.markdown(
        "**What it does**\n\n"
        "1. Illumination correction + CLAHE + NLM denoise\n"
        "2. Marker-controlled watershed (separates touching nuclei)\n"
        "3. Per-cell morphology via `regionprops`"
    )

if upload is None:
    st.info("⬅️ Upload a microscopy tile (e.g. a DSB-2018 / BBBC nuclei image) "
            "to segment and quantify individual nuclei.")
    st.stop()

file_bytes = np.frombuffer(upload.read(), np.uint8)
image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
gray = to_gray(image)

with st.spinner("Segmenting…"):
    enhanced, label_cv = run_classical(gray)
    if method == "Cellpose-SAM" and _HAS_CELLPOSE:
        label = segment_cellpose(gray, gpu=False)
    else:
        label = label_cv
    feats = featurize(label, gray)
    stats = summarize(feats)
    overlay = overlay_labels(enhanced, label)
    overlay = label_count_banner(overlay, f"{stats['cell_count']} cells")
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Input")
    st.image(gray, clamp=True, use_container_width=True)
with c2:
    st.subheader("Segmentation overlay")
    st.image(overlay_rgb, use_container_width=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Cells detected", stats["cell_count"])
m2.metric("Mean area (px)", f"{stats.get('mean_area', 0):.0f}")
m3.metric("Mean eccentricity", f"{stats.get('mean_eccentricity', 0):.3f}")
m4.metric("Mean intensity", f"{stats.get('mean_intensity', 0):.0f}")

st.subheader("Per-cell features")
st.dataframe(feats.round(2), use_container_width=True, height=300)

csv = feats.round(3).to_csv(index=False).encode()
st.download_button("⬇️ Download per-cell CSV", csv,
                   file_name="cellsight_features.csv", mime="text/csv")
