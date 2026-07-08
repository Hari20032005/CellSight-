# 3–5 min Video Script (Aganitha form requirement)

The form asks the video to cover four things. Hit them in order. Keep it ~4 min.
Screen-record the Gradio demo + one overlay figure + the metrics table.

---

### 0:00–0:30 — Intro (who + what)
> "Hi, I'm [Name], M.Tech at VIT. I built **CellSight**, an end-to-end computer
> vision pipeline for microscopy cell segmentation and quantification — the same
> kind of problem Aganitha's Microscopy Image Analysis solution tackles."

### 0:30–1:15 — Datasets (form prompt #1)
> "I worked with **fluorescence and brightfield microscopy** nuclei images from
> the 2018 Data Science Bowl dataset, which ships pixel-level instance masks as
> ground truth. Tiles are 256–512 px, so everything fits on my 8 GB laptop; the
> GPU step runs free on Colab."

Show: a raw tile next to its ground-truth overlay.

### 1:15–2:30 — Techniques (form prompt #2)
> "I segment nuclei **two ways**. First, a **custom OpenCV pipeline** I wrote:
> illumination correction, CLAHE contrast enhancement, and Non-Local-Means
> denoising, then a **marker-controlled watershed** using the distance transform
> to split touching nuclei. Second, I apply **Cellpose-SAM**, a vision foundation
> model with a SAM/ViT transformer backbone, for zero-shot instance masks."

Show: the enhancement stages, then both overlays side by side.

### 2:30–3:15 — Pipeline (form prompt #3)
> "The full pipeline is enhance → segment → **featurize** → evaluate. For every
> nucleus I extract morphology — area, perimeter, eccentricity, solidity, mean
> intensity — with scikit-image regionprops, and export a per-cell CSV plus a
> cell count. It's modular Python; a Gradio app runs the whole thing on an
> uploaded image."

Show: Gradio demo — upload a tile → overlay + cell count + feature table.

### 3:15–4:15 — Result / insight (form prompt #4)
> "I benchmarked both methods against ground truth with **Dice, IoU, and
> instance-level mean Average Precision** (the DSB metric). Key insight: the
> custom watershed pipeline is fast and competitive on well-separated nuclei but
> **merges touching cells**; Cellpose-SAM separates dense clusters far better,
> raising instance mAP by [X]. So the practical answer is a **hybrid** — cheap
> classical preprocessing plus a foundation model where cell density is high."

Show: the `summary_metrics.csv` table; point at the mAP / count-error columns.

### 4:15–4:30 — Close
> "Everything's on GitHub with a reproducible smoke test and a Colab notebook.
> I'd love to bring this to Aganitha's biomedical imaging work. Thank you!"

---

**Recording tips**
- Fill in `[X]` with your real numbers after running `scripts/run_pipeline.py`.
- Record at 1080p, share screen, talk over it — no need to show your face the whole time.
- Upload to Google Drive / YouTube (unlisted) with **open access**, paste the link in the form.
