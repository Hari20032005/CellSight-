"""Synthetic smoke test: exercises enhance -> watershed -> featurize -> evaluate
without needing the real dataset or any GPU. Run: python scripts/smoke_test.py
"""
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classical_pipeline import run_classical
from src.featurize import featurize, summarize
from src.evaluate import evaluate
from src.utils import overlay_labels

rng = np.random.default_rng(0)

# Build a synthetic fluorescence-like tile: bright blobs on dark bg + noise.
H = W = 256
img = np.zeros((H, W), np.uint8)
gt = np.zeros((H, W), np.int32)
centers = [(60, 60), (60, 90), (150, 70), (180, 180), (90, 190)]  # some touch
for i, (cy, cx) in enumerate(centers, start=1):
    blob = np.zeros((H, W), np.uint8)
    cv2.circle(blob, (cx, cy), 18, 255, -1)
    img[blob > 0] = 200
    gt[blob > 0] = i
img = cv2.GaussianBlur(img, (5, 5), 0)
img = np.clip(img.astype(int) + rng.normal(0, 15, img.shape), 0, 255).astype(np.uint8)

enhanced, label = run_classical(img)
feats = featurize(label, img)
stats = summarize(feats)
metrics = evaluate(label, gt)

print("detected cells:", stats["cell_count"], "| GT cells:", int(gt.max()))
print("feature columns:", list(feats.columns))
print("metrics:", {k: round(v, 3) if isinstance(v, float) else v
                    for k, v in metrics.items()})

overlay = overlay_labels(enhanced, label)
out = ROOT / "outputs" / "smoke_overlay.png"
out.parent.mkdir(exist_ok=True)
cv2.imwrite(str(out), overlay)
print("wrote", out)

assert stats["cell_count"] >= 4, "expected to detect most synthetic nuclei"
assert metrics["dice"] > 0.6, f"unexpectedly low Dice: {metrics['dice']}"
assert set(["area", "eccentricity", "mean_intensity"]).issubset(feats.columns)
print("SMOKE TEST PASSED")
