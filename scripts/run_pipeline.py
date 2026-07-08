"""End-to-end CellSight runner.

For each DSB sample under --data:
    * run the custom OpenCV watershed pipeline
    * (optionally) run Cellpose-SAM
    * featurize each result (per-cell morphology CSV)
    * if ground truth exists, evaluate both methods
    * save overlay figures

Outputs land in --out (default: outputs/).

Examples
--------
# Classical only, no GPU needed (great for the laptop):
    python scripts/run_pipeline.py --data data/stage1_train --limit 5

# Add Cellpose-SAM (run on Colab/Kaggle GPU, or CPU for a few tiles):
    python scripts/run_pipeline.py --data data/stage1_train --limit 5 --cellpose
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import iter_dsb_samples          # noqa: E402
from src.classical_pipeline import run_classical  # noqa: E402
from src.featurize import featurize, summarize  # noqa: E402
from src.evaluate import evaluate               # noqa: E402
from src.utils import overlay_labels, label_count_banner  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="CellSight end-to-end runner")
    ap.add_argument("--data", required=True, help="DSB-format root folder")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--limit", type=int, default=5, help="max samples to process")
    ap.add_argument("--cellpose", action="store_true",
                    help="also run Cellpose-SAM (needs cellpose + torch)")
    ap.add_argument("--gpu", action="store_true", help="use GPU for Cellpose")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    (out / "features").mkdir(parents=True, exist_ok=True)

    segment_cellpose = None
    if args.cellpose:
        from src.cellpose_infer import segment_cellpose  # lazy import

    rows = []
    for i, (sample_id, image, gt) in enumerate(iter_dsb_samples(args.data)):
        if i >= args.limit:
            break
        print(f"[{i+1}] {sample_id}  ({image.shape[1]}x{image.shape[0]})")

        # --- Custom OpenCV pipeline -------------------------------------
        enhanced, label_cv = run_classical(image)
        feats_cv = featurize(label_cv, image)
        feats_cv.to_csv(out / "features" / f"{sample_id}_classical.csv", index=False)
        sum_cv = summarize(feats_cv)

        ov_cv = overlay_labels(enhanced, label_cv)
        ov_cv = label_count_banner(ov_cv, f"OpenCV: {sum_cv['cell_count']} cells")
        cv2.imwrite(str(out / "overlays" / f"{sample_id}_classical.png"), ov_cv)

        row = {"sample": sample_id, "method": "opencv_watershed",
               "cells": sum_cv["cell_count"]}
        if gt is not None:
            row.update(evaluate(label_cv, gt))
        rows.append(row)

        # --- Cellpose-SAM ------------------------------------------------
        if segment_cellpose is not None:
            label_cp = segment_cellpose(image, gpu=args.gpu)
            feats_cp = featurize(label_cp, image)
            feats_cp.to_csv(out / "features" / f"{sample_id}_cellpose.csv",
                            index=False)
            sum_cp = summarize(feats_cp)

            ov_cp = overlay_labels(enhanced, label_cp)
            ov_cp = label_count_banner(ov_cp, f"Cellpose-SAM: {sum_cp['cell_count']} cells")
            cv2.imwrite(str(out / "overlays" / f"{sample_id}_cellpose.png"), ov_cp)

            row_cp = {"sample": sample_id, "method": "cellpose_sam",
                      "cells": sum_cp["cell_count"]}
            if gt is not None:
                row_cp.update(evaluate(label_cp, gt))
            rows.append(row_cp)

    results = pd.DataFrame(rows)
    results.to_csv(out / "results.csv", index=False)
    print("\n=== Results ===")
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(results.to_string(index=False))

    # Method-level summary (mean over samples) when GT metrics are present.
    if "mAP" in results.columns:
        summary = (results.groupby("method")[["dice", "iou", "mAP", "count_error"]]
                   .mean().round(3))
        print("\n=== Mean metrics by method ===")
        print(summary.to_string())
        summary.to_csv(out / "summary_metrics.csv")

    print(f"\nSaved overlays, per-cell CSVs, and results to {out}")


if __name__ == "__main__":
    main()
