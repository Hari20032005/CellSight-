# Run Cellpose-SAM on Kaggle's free GPU — from the terminal

No browser, no local GPU, almost no local disk. All compute happens on Kaggle.

## One-time setup (2 min, only you can do this)
1. Sign in at https://www.kaggle.com and **verify your phone** (Settings) — this
   unlocks GPU + internet for kernels.
2. Get an API token: **Settings → API → Create New Token** → downloads
   `kaggle.json`.
3. Install it:
   ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```
4. Put your Kaggle username into `kernel-metadata.json` (replace
   `YOUR_KAGGLE_USERNAME`).

## Run it (from this folder)
```bash
pip install --user kaggle                       # tiny CLI
kaggle kernels push -p .                         # runs on Kaggle's GPU
kaggle kernels status <user>/cellsight-cellpose-sam   # poll until "complete"
kaggle kernels output <user>/cellsight-cellpose-sam -p ./kaggle_out
cat kaggle_out/outputs/summary_metrics.csv       # your Cellpose-SAM numbers
```
Then paste the Cellpose-SAM row into the main `README.md`.
