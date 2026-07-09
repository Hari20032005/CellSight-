"""Vercel Python serverless function: POST /api/analyze.

Body: JSON { "image": "<data-uri or base64>" }
Returns: JSON { input, overlay, stats, features }

Uses only opencv-python-headless + numpy (see requirements.txt) so the function
stays within Vercel's serverless bundle size.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _pipeline import (
    run_classical, featurize, summarize, overlay, decode_image, to_data_uri,
)


class handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
            if not data.get("image"):
                return self._send(400, {"error": "No image provided."})

            gray = decode_image(data["image"])
            enhanced, label = run_classical(gray)
            rows = featurize(label, gray)
            stats = summarize(rows)

            import cv2
            self._send(200, {
                "input": to_data_uri(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
                "overlay": to_data_uri(overlay(enhanced, label)),
                "stats": stats,
                "features": rows[:200],
            })
        except Exception as e:  # return the error to the UI instead of a 500 page
            self._send(400, {"error": str(e)})
