"""CellSight — single Flask WSGI app for Vercel's Python runtime.

Serves the static frontend (index.html / style.css / app.js) and the
POST /api/analyze endpoint that runs the custom OpenCV segmentation pipeline.
Declared as the Vercel entrypoint in pyproject.toml (`index:app`).
"""
import os

import cv2
from flask import Flask, Response, jsonify, request, send_file

from _pipeline import (
    run_classical, featurize, summarize, overlay, decode_image, to_data_uri,
)

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)


@app.route("/")
def index():
    return send_file(os.path.join(HERE, "index.html"))


@app.route("/style.css")
def style():
    return send_file(os.path.join(HERE, "style.css"), mimetype="text/css")


@app.route("/app.js")
def appjs():
    return send_file(os.path.join(HERE, "app.js"), mimetype="application/javascript")


@app.route("/example.png")
def example():
    return send_file(os.path.join(HERE, "example.png"), mimetype="image/png")


@app.route("/healthz")
def healthz():
    return "ok", 200


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    if not data.get("image"):
        return jsonify({"error": "No image provided."}), 400
    try:
        gray = decode_image(data["image"])
        enhanced, label = run_classical(gray)
        rows = featurize(label, gray)
        stats = summarize(rows)
        return jsonify({
            "input": to_data_uri(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
            "overlay": to_data_uri(overlay(enhanced, label)),
            "stats": stats,
            "features": rows[:200],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
