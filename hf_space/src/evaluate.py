"""Segmentation metrics: semantic Dice/IoU and instance Average Precision.

`average_precision` follows the 2018 Data Science Bowl convention: at each IoU
threshold, count matched predicted nuclei as TP, unmatched predictions as FP,
unmatched GT nuclei as FN, then AP = TP / (TP + FP + FN). The reported score is
the mean over thresholds 0.5..0.95, which is the standard nucleus-segmentation
benchmark number.
"""
from __future__ import annotations

import numpy as np


def dice_iou(pred_label: np.ndarray, gt_label: np.ndarray) -> tuple[float, float]:
    """Foreground (semantic) Dice and IoU, treating any label > 0 as foreground."""
    pred = pred_label > 0
    gt = gt_label > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    dice = (2 * inter / denom) if denom else 1.0
    iou = (inter / union) if union else 1.0
    return float(dice), float(iou)


def _iou_matrix(pred_label: np.ndarray, gt_label: np.ndarray) -> np.ndarray:
    """IoU between every predicted instance and every GT instance."""
    pred_ids = [i for i in np.unique(pred_label) if i != 0]
    gt_ids = [i for i in np.unique(gt_label) if i != 0]
    mat = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
    for pi, p in enumerate(pred_ids):
        pmask = pred_label == p
        parea = pmask.sum()
        for gi, g in enumerate(gt_ids):
            gmask = gt_label == g
            inter = np.logical_and(pmask, gmask).sum()
            if inter == 0:
                continue
            union = parea + gmask.sum() - inter
            mat[pi, gi] = inter / union
    return mat


def average_precision(pred_label: np.ndarray, gt_label: np.ndarray,
                      thresholds=np.arange(0.5, 1.0, 0.05)) -> dict:
    """Instance AP averaged over IoU thresholds (DSB 2018 metric)."""
    iou = _iou_matrix(pred_label, gt_label)
    n_pred = iou.shape[0]
    n_gt = iou.shape[1]

    per_threshold = {}
    scores = []
    for t in thresholds:
        if n_pred == 0 or n_gt == 0:
            ap = 0.0 if (n_pred or n_gt) else 1.0
            per_threshold[round(float(t), 2)] = ap
            scores.append(ap)
            continue
        # Greedy one-to-one matching above threshold.
        matched_gt = set()
        tp = 0
        # Consider predictions in descending best-IoU order for stable matching.
        order = np.argsort(-iou.max(axis=1))
        for pi in order:
            gi = int(np.argmax(iou[pi]))
            if iou[pi, gi] >= t and gi not in matched_gt:
                matched_gt.add(gi)
                tp += 1
        fp = n_pred - tp
        fn = n_gt - len(matched_gt)
        ap = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
        per_threshold[round(float(t), 2)] = ap
        scores.append(ap)

    return {"mAP": float(np.mean(scores)), "per_threshold": per_threshold}


def evaluate(pred_label: np.ndarray, gt_label: np.ndarray) -> dict:
    """Bundle semantic + instance metrics for one image."""
    dice, iou = dice_iou(pred_label, gt_label)
    ap = average_precision(pred_label, gt_label)
    n_pred = len([i for i in np.unique(pred_label) if i != 0])
    n_gt = len([i for i in np.unique(gt_label) if i != 0])
    return {
        "dice": dice,
        "iou": iou,
        "mAP": ap["mAP"],
        "pred_count": n_pred,
        "gt_count": n_gt,
        "count_error": abs(n_pred - n_gt),
    }
