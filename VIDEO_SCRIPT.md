# CellSight — Video Submission Script (Aganitha form)

Target length **~4 min** (form asks 3–5). Record 1080p, screen + mic (QuickTime/OBS).

## Before recording — open 3 tabs
1. Live demo: https://vercelapp-pied.vercel.app
2. GitHub repo: https://github.com/Hari20032005/CellSight- (pre-scroll to **Results**)
3. VS Code with `src/classical_pipeline.py` open

---

## SCRIPT (with on-screen cues)

**[0:00–0:25] Intro** — *your face or a "CellSight" title card*
> "Hi, I'm [Name], an M.Tech student at VIT. I'll walk you through CellSight — an
> end-to-end computer-vision pipeline I built for microscopy cell segmentation
> and quantification, exactly the kind of problem Aganitha's Microscopy Image
> Analysis work tackles."

**[0:25–1:05] ① Datasets** — *live site; click "Try a sample image"*
> "I worked with fluorescence and brightfield microscopy nuclei images from the
> 2018 Data Science Bowl, also known as BBBC038. Each image comes with
> pixel-level instance masks as ground truth. Here's a real tile from that
> dataset loaded into my live demo."

**[1:05–2:05] ② Techniques** — *input-vs-overlay result, then classical_pipeline.py*
> "I segment the nuclei two ways. The first is a custom OpenCV pipeline I wrote
> myself: correct uneven illumination, enhance contrast with CLAHE, denoise with
> Non-Local-Means, then a marker-controlled watershed that uses the distance
> transform to separate touching nuclei — you can see each cell outlined here.
> This is my own algorithm, not an off-the-shelf call. The second method is
> Cellpose-SAM — a vision foundation model with a SAM Vision-Transformer
> backbone — which I applied and evaluated on a free Kaggle GPU."

**[2:05–2:50] ③ Pipeline** — *live demo metrics + per-cell table + CSV*
> "The full pipeline is enhance, segment, featurize, then evaluate. For every
> nucleus I extract morphology — area, perimeter, eccentricity, solidity, and
> intensity — plus a cell count. On this sample it found 33 nuclei, and the
> per-cell features export as CSV. It's deployed as a live web app on Vercel with
> a Python serverless backend."

**[2:50–3:40] ④ Result / insight** — *README benchmark table + dense-tile figure*
> "I benchmarked both methods against ground truth using Dice, IoU, and
> instance-level mean Average Precision. Cellpose-SAM reached a Dice of 0.889
> versus my classical pipeline's 0.78, and more than doubled the instance mAP.
> The telling case is a dense tile with 70 true nuclei: my watershed finds 35,
> Cellpose-SAM finds 71. The insight: the classical pipeline is fast and
> competitive on sparse images but under-segments crowded ones — so the practical
> answer is a hybrid: cheap classical preprocessing plus a foundation model where
> cell density is high."

**[3:40–4:00] Close** — *GitHub repo home*
> "Everything is open-source on GitHub, with a reproducible Kaggle benchmark and
> the live demo you just saw. I'd be excited to bring this to Aganitha's
> biomedical imaging work. Thank you."

---

## 4 required points — checklist
- Imaging datasets → 0:25 (DSB-2018 / BBBC038 microscopy nuclei)
- CV / DL techniques → 1:05 (OpenCV watershed + Cellpose-SAM foundation model)
- How you built the pipeline → 2:05 (enhance→segment→featurize→evaluate, deployed)
- One result / insight → 2:50 (Dice 0.889 vs 0.78; 35-vs-71 dense-tile hybrid insight)

## Honesty guardrails
- Say Cellpose-SAM is a model you **applied and evaluated** — never "trained".
- Lead with the OpenCV pipeline as **your own algorithm** (the JD's "custom algorithm").

## After recording
Upload to Google Drive / YouTube (unlisted) → sharing = "Anyone with the link" →
paste in the form's video field.
