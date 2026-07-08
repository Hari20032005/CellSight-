"""CellSight — Cellpose-SAM run, executed headlessly on a Kaggle GPU kernel.

Everything here runs on Kaggle's servers (free P100/T4), not your laptop:
    1. install cellpose (== Cellpose-SAM)
    2. clone the CellSight repo (for src/ + scripts/)
    3. pull the DSB-2018 tiles from the Broad mirror (no login)
    4. run the end-to-end pipeline with the foundation model on 5 tiles
    5. print + save Dice / IoU / instance-mAP for both methods

Results land in /kaggle/working/outputs and are retrievable via:
    kaggle kernels output <user>/cellsight-cellpose-sam -p ./kaggle_out
"""
import os
import subprocess
import sys

WORK = "/kaggle/working"
REPO = f"{WORK}/CellSight"
DATA = f"{WORK}/data/stage1_train"
OUT = f"{WORK}/outputs"


def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


# 1. Cellpose-SAM
sh(f"{sys.executable} -m pip install -q cellpose scikit-image opencv-python-headless")

# 2. Code
sh(f"rm -rf {REPO} && git clone --depth 1 "
   f"https://github.com/Hari20032005/CellSight-.git {REPO}")

# 3. Data (Broad BBBC038 mirror of the DSB-2018 training set)
sh(f"curl -sL -o {WORK}/s.zip "
   f"https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip")
sh(f"mkdir -p {DATA} && unzip -q {WORK}/s.zip -d {DATA}")

# 4. Run classical + Cellpose-SAM on the same 5 tiles, on GPU
sh(f"cd {REPO} && {sys.executable} scripts/run_pipeline.py "
   f"--data {DATA} --limit 5 --cellpose --gpu --out {OUT}")

# 5. Echo the headline table into the kernel log
print("\n===== summary_metrics.csv =====", flush=True)
with open(f"{OUT}/summary_metrics.csv") as f:
    print(f.read(), flush=True)
print("===== results.csv =====", flush=True)
with open(f"{OUT}/results.csv") as f:
    print(f.read(), flush=True)
