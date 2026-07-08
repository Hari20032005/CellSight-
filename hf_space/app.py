"""CellSight — Hugging Face Space (Gradio) entry point.

Upload a microscopy tile -> enhance -> segment nuclei -> per-cell morphology.
Two methods: a custom OpenCV watershed pipeline (instant) and the Cellpose-SAM
foundation model (slower on the free CPU Space, loaded lazily on first use).
"""
import os
import sys

import cv2
import numpy as np
import gradio as gr

sys.path.insert(0, os.path.dirname(__file__))

from src.classical_pipeline import run_classical
from src.featurize import featurize, summarize
from src.utils import overlay_labels, label_count_banner

_CELLPOSE_OK = True
try:
    from src.cellpose_infer import segment_cellpose
except Exception:
    _CELLPOSE_OK = False

CLASSICAL = "OpenCV watershed (custom, instant)"
CELLPOSE = "Cellpose-SAM (foundation model, slower on CPU)"
METHODS = [CLASSICAL] + ([CELLPOSE] if _CELLPOSE_OK else [])


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def analyze(image, method):
    if image is None:
        return None, None, "Upload a microscopy image first."
    gray = _to_gray(image)
    enhanced, label_cv = run_classical(gray)

    if method == CELLPOSE and _CELLPOSE_OK:
        label = segment_cellpose(gray, gpu=False)
    else:
        label = label_cv

    feats = featurize(label, gray)
    stats = summarize(feats)
    overlay = overlay_labels(enhanced, label)
    overlay = label_count_banner(overlay, f"{stats['cell_count']} cells")
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    msg = (f"**Cells detected:** {stats['cell_count']}  \n"
           f"**Mean area:** {stats.get('mean_area', 0):.1f} px  \n"
           f"**Mean eccentricity:** {stats.get('mean_eccentricity', 0):.3f}  \n"
           f"**Mean intensity:** {stats.get('mean_intensity', 0):.1f}")
    return overlay_rgb, feats.round(2), msg


with gr.Blocks(title="CellSight — Microscopy Cell Analysis") as demo:
    gr.Markdown(
        "# 🔬 CellSight — Microscopy Cell/Nuclei Segmentation & Quantification\n"
        "Upload a microscopy tile (brightfield / fluorescence nuclei). CellSight "
        "enhances it, segments individual nuclei, and reports per-cell morphology.\n\n"
        "- **OpenCV watershed** — a custom classical pipeline, returns instantly.\n"
        "- **Cellpose-SAM** — a vision foundation model; more accurate on dense "
        "clusters but takes ~30–90 s on this free CPU Space (first run also "
        "downloads the model)."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Microscopy image")
            method = gr.Radio(METHODS, value=METHODS[0], label="Segmentation method")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Segmentation overlay")
            out_msg = gr.Markdown()
    out_tbl = gr.Dataframe(label="Per-cell features")
    btn.click(analyze, [inp, method], [out_img, out_tbl, out_msg])

if __name__ == "__main__":
    demo.launch()
