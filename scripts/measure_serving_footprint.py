"""Measure the ONNX serving process the way the free host runs it: one thread.

Reports median single-image inference time and the resident memory of the
process after the model is loaded and has served requests. Written to
``artifacts/results/serving_footprint.json`` so documents quote a measurement
rather than a remembered figure.

    python scripts/measure_serving_footprint.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["PDD_SERVING_BACKEND"] = "onnx"
os.environ["PDD_ONNX_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import numpy as np  # noqa: E402
import psutil  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.services.predictor import predictor  # noqa: E402


def main() -> int:
    process = psutil.Process()
    baseline_mb = process.memory_info().rss / 1048576
    rng = np.random.default_rng(0)
    # A smooth, leaf-coloured synthetic photograph: only timing matters here.
    yy, xx = np.mgrid[0:256, 0:256]
    base = np.stack([60 + 40 * np.sin(xx / 20), 130 + 40 * np.sin(yy / 17), 45 + 20 * np.cos(xx / 31)], -1)
    image = Image.fromarray(np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype("uint8"))

    for _ in range(5):                                     # warm-up, not timed
        predictor.predict(image, explain=False)
    inference, total, explain = [], [], []
    for _ in range(40):
        started = time.perf_counter()
        result = predictor.predict(image, explain=False)
        total.append((time.perf_counter() - started) * 1000)
        inference.append(result.timings["inference_ms"])
    for _ in range(10):
        result = predictor.predict(image, explain=True, save_dir=settings.upload_dir)
        explain.append(result.timings["explain_ms"])
    rss_mb = process.memory_info().rss / 1048576

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": result.model["label"],
        "threads": 1,
        "inference_ms_median": round(statistics.median(inference), 2),
        "pipeline_ms_median_no_explain": round(statistics.median(total), 2),
        "explain_ms_median": round(statistics.median(explain), 2),
        "resident_mb_after_requests": round(rss_mb, 1),
        "resident_mb_before_model": round(baseline_mb, 1),
        "host": {"cpu_count": os.cpu_count(), "note": "Developer machine, not the free host; "
                                                      "the free host's shared CPU is slower."},
    }
    out = settings.results_dir / "serving_footprint.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
