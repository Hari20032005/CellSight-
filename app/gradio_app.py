"""CellSight demo — upload a microscopy tile, get segmentation + quantification.

Run:
    python app/gradio_app.py          # local, OpenCV pipeline (no GPU)
    python app/gradio_app.py --share  # public link for the video

The Cellpose-SAM option appears if `cellpose` is installed; otherwise the app
gracefully falls back to the custom OpenCV pipeline only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from src.classical_pipeline import run_classical  # noqa: E402
from src.featurize import featurize, summarize     # noqa: E402
from src.utils import overlay_labels, label_count_banner  # noqa: E402

try:
    from src.cellpose_infer import segment_cellpose
    _HAS_CELLPOSE = True
except Exception:
    _HAS_CELLPOSE = False


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def analyze(image: np.ndarray, method: str):
    if image is None:
        return None, None, "Upload a microscopy image first."
    gray = _to_gray(image)
    enhanced, label_cv = run_classical(gray)

    if method == "Cellpose-SAM" and _HAS_CELLPOSE:
        label = segment_cellpose(gray, gpu=False)
        base = enhanced
    else:
        label = label_cv
        base = enhanced

    feats = featurize(label, gray)
    stats = summarize(feats)

    overlay = overlay_labels(base, label)
    overlay = label_count_banner(overlay, f"{stats['cell_count']} cells")
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    msg = (f"**Cells detected:** {stats['cell_count']}  \n"
           f"**Mean area:** {stats.get('mean_area', 0):.1f} px  \n"
           f"**Mean eccentricity:** {stats.get('mean_eccentricity', 0):.3f}  \n"
           f"**Mean intensity:** {stats.get('mean_intensity', 0):.1f}")
    return overlay_rgb, feats.round(2), msg


def build_ui():
    methods = ["OpenCV watershed (custom)"]
    if _HAS_CELLPOSE:
        methods.append("Cellpose-SAM")

    with gr.Blocks(title="CellSight — Microscopy Cell Analysis") as demo:
        gr.Markdown(
            "# CellSight\n"
            "Upload a microscopy tile (brightfield / fluorescence nuclei). "
            "The pipeline enhances it, segments individual nuclei, and reports "
            "per-cell morphology."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="numpy", label="Microscopy image")
                method = gr.Radio(methods, value=methods[0], label="Segmentation method")
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="Segmentation overlay")
                out_msg = gr.Markdown()
        out_tbl = gr.Dataframe(label="Per-cell features")
        btn.click(analyze, [inp, method], [out_img, out_tbl, out_msg])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    build_ui().launch(share=args.share)
