---
title: CellSight
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Microscopy cell/nuclei segmentation & quantification
---

# CellSight

Interactive demo for **microscopy cell/nuclei segmentation and quantification**.
Upload a brightfield/fluorescence tile → enhancement → instance segmentation
(custom OpenCV watershed **or** the Cellpose-SAM foundation model) → per-cell
morphology table.

Code + full pipeline + benchmarks: https://github.com/Hari20032005/CellSight-
