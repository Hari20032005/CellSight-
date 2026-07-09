# CellSight — Vercel frontend

A designed single-page site (hero + "How it works" + live demo + benchmark)
plus a Python serverless function that runs the custom OpenCV segmentation
pipeline. Lightweight (`opencv-python-headless` + `numpy`, ~170 MB) so it fits
Vercel's free serverless limits.

```
vercel_app/
├── index.html        # frontend (served on Vercel's CDN)
├── style.css
├── app.js            # uploads image -> POST /api/analyze -> renders result
├── api/
│   ├── analyze.py     # serverless function (POST /api/analyze)
│   └── _pipeline.py   # vendored OpenCV pipeline (enhance→watershed→featurize)
├── requirements.txt   # opencv-python-headless, numpy
└── vercel.json        # function memory/duration
```

## Deploy (free, no card)

**Dashboard (easiest):**
1. https://vercel.com → **Add New… → Project → Import Git Repository**.
2. Pick the `CellSight-` repo.
3. **Set "Root Directory" to `vercel_app`** ← important (so Vercel uses this
   folder's light `requirements.txt` and finds `api/` + `index.html`).
4. **Deploy.** You get a permanent URL like `https://cellsight.vercel.app`.

**CLI:** `npm i -g vercel && cd vercel_app && vercel --prod`

## Notes
- Live inference = custom OpenCV pipeline. Cellpose-SAM (~2–3 GB) is too heavy
  for any free serverless tier, so it is shown as the documented benchmark.
- First request after idle has a small cold-start warm-up.
