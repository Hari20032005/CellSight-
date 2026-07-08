"""Cellpose-SAM foundation-model inference.

Cellpose v4 is Cellpose-SAM: a SAM/ViT transformer backbone adapted to the
Cellpose framework. It gives strong *zero-shot* instance masks for cells/nuclei
with no training, which is exactly what makes this project achievable in a day
on free compute.

Run the heavy inference on Google Colab / Kaggle (free GPU); it also runs on a
CPU laptop for a handful of small tiles, just slower.
"""
from __future__ import annotations

import numpy as np


_MODEL = None


def get_model(gpu: bool = True):
    """Lazily construct and cache the Cellpose-SAM model."""
    global _MODEL
    if _MODEL is None:
        from cellpose import models  # imported lazily so the rest works without it
        # Cellpose v4: CellposeModel with the built-in Cellpose-SAM weights.
        _MODEL = models.CellposeModel(gpu=gpu)
    return _MODEL


def segment_cellpose(gray: np.ndarray, gpu: bool = True,
                     diameter: float | None = None,
                     flow_threshold: float = 0.4,
                     cellprob_threshold: float = 0.0) -> np.ndarray:
    """Return an int32 instance-label image from Cellpose-SAM.

    `diameter=None` lets Cellpose estimate object size automatically.
    """
    model = get_model(gpu=gpu)
    result = model.eval(
        gray,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    # Cellpose returns (masks, flows, styles[, diams]); masks is the label image.
    masks = result[0]
    return np.asarray(masks, dtype=np.int32)
