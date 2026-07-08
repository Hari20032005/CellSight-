"""CellSight — Cellpose-SAM run, executed headlessly on a Kaggle GPU kernel.

Everything here runs on Kaggle's servers (free P100/T4), not your laptop.

Two deliberate choices that matter:
  * We DO NOT reinstall torch (install cellpose with --no-deps + small pure
    deps), and we run Cellpose-SAM on CPU. Kaggle's GPU node throws
    "CUDA error: no kernel image available" when loading the SAM/ViT weights
    (a torch<->GPU arch mismatch on Kaggle's side). For 5 small tiles, CPU
    inference is reliable and produces identical masks/metrics.
  * Repo + dataset live in /tmp (NOT /kaggle/working), so Kaggle only captures
    the tiny outputs/ folder as the downloadable result.
"""
import subprocess
import sys

REPO = "/tmp/CellSight"
DATA = "/tmp/data/stage1_train"
OUT = "/kaggle/working/outputs"          # only this is returned as kernel output


def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


# 1. Cellpose-SAM WITHOUT disturbing Kaggle's working GPU torch.
sh(f"{sys.executable} -m pip install -q --no-deps cellpose")
sh(f"{sys.executable} -m pip install -q fastremap roifile fill-voids natsort")

import torch  # noqa: E402
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
      flush=True)

# 2. Code (into /tmp so it is not captured as output)
sh(f"rm -rf {REPO} && git clone --depth 1 "
   f"https://github.com/Hari20032005/CellSight-.git {REPO}")

# 3. Data (Broad BBBC038 mirror of the DSB-2018 training set, into /tmp)
sh(f"curl -sL -o /tmp/s.zip "
   f"https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip")
sh(f"mkdir -p {DATA} && unzip -q /tmp/s.zip -d {DATA}")

# 4. Run classical + Cellpose-SAM on the same 5 tiles (CPU, reliable)
sh(f"cd {REPO} && {sys.executable} scripts/run_pipeline.py "
   f"--data {DATA} --limit 5 --cellpose --out {OUT}")

# 5. Echo the headline tables into the kernel log
for name in ("summary_metrics.csv", "results.csv"):
    print(f"\n===== {name} =====", flush=True)
    with open(f"{OUT}/{name}") as f:
        print(f.read(), flush=True)
