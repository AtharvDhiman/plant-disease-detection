"""Measure the serving path end to end and record it for the research dashboard.

Reports the four timings the brief asks for - model loading, preprocessing,
inference and total response - measured on the *serving* code path rather than
on a synthetic benchmark, so the numbers reflect what a user actually waits for.

    python scripts/benchmark_serving.py
    python scripts/benchmark_serving.py --iterations 60 --no-explain
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import app.core.runtime  # noqa: F401,E402  # isort:skip

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.ml.datasets import read_manifest  # noqa: E402
from app.ml.registry import registry  # noqa: E402
from app.services.predictor import Predictor  # noqa: E402

log = get_logger(__name__)


def summarise(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "mean_ms": round(statistics.fmean(samples), 3),
        "median_ms": round(statistics.median(samples), 3),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
        "stdev_ms": round(statistics.pstdev(samples), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-explain", action="store_true",
                        help="Measure without Grad-CAM, which needs a backward pass.")
    args = parser.parse_args()

    configure_logging(settings.log_level, settings.logs_dir / "benchmark_serving.log")
    settings.ensure_dirs()

    # 1 -------------------------------------------------------- model loading
    registry.clear()
    load_start = time.perf_counter()
    loaded = registry.load()
    cold_load_ms = (time.perf_counter() - load_start) * 1000

    warm_start = time.perf_counter()
    dummy = torch.zeros(1, 3, loaded.image_size, loaded.image_size, device=loaded.device)
    with torch.no_grad():
        loaded.model(dummy)
    if loaded.device.type == "cuda":
        torch.cuda.synchronize()
    warmup_ms = (time.perf_counter() - warm_start) * 1000

    log.info("Model loaded", fields={"model": loaded.name, "device": str(loaded.device),
                                     "cold_load_ms": round(cold_load_ms, 1)})

    # 2 ------------------------------------------------------------ requests
    manifest = settings.data_dir / "manifests" / "test_bench.csv"
    if not manifest.exists():
        raise SystemExit("data/manifests/test_bench.csv missing - run prepare_splits.py first.")
    entries = read_manifest(manifest)
    step = max(1, len(entries) // (args.iterations + args.warmup))
    sample_paths = [e.path for e in entries[::step]][: args.iterations + args.warmup]

    predictor = Predictor()
    scratch = settings.artifacts_dir / "_benchmark_tmp"
    scratch.mkdir(parents=True, exist_ok=True)

    stages: dict[str, list[float]] = {
        "quality_ms": [], "preprocess_ms": [], "inference_ms": [],
        "explain_ms": [], "total_ms": [],
    }

    for index, path in enumerate(sample_paths):
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        result = predictor.predict(
            image, top_k=5, explain=not args.no_explain,
            save_dir=scratch, file_stem=f"bench_{index}",
        )
        if index < args.warmup:
            continue
        for key in stages:
            stages[key].append(result.timings[key])

    for leftover in scratch.glob("bench_*"):
        leftover.unlink(missing_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": loaded.name,
            "label": loaded.label,
            "device": str(loaded.device),
            "image_size": loaded.image_size,
            "parameters": loaded.parameters,
            "size_mb": round(loaded.size_mb, 3),
        },
        "explain_enabled": not args.no_explain,
        "iterations": len(stages["total_ms"]),
        "model_loading": {
            "cold_load_ms": round(cold_load_ms, 2),
            "first_forward_ms": round(warmup_ms, 2),
            "note": (
                "Loading happens once, in the FastAPI startup hook. No request "
                "pays this cost. The first forward pass is measured separately "
                "because CUDA kernel initialisation makes it much slower than "
                "steady state - which is why the server warms the model at boot."
            ),
        },
        "per_request": {name: summarise(values) for name, values in stages.items() if values},
        "throughput_requests_per_second": round(
            1000 / statistics.fmean(stages["total_ms"]), 2
        ) if stages["total_ms"] else None,
    }

    destination = settings.results_dir / "serving_benchmark.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("SERVING PERFORMANCE")
    print("=" * 72)
    print(f"Model            : {loaded.label} on {loaded.device}")
    print(f"Explainability   : {'on' if not args.no_explain else 'off'}")
    print(f"Iterations       : {payload['iterations']} (after {args.warmup} warm-up)")
    print("-" * 72)
    print(f"Cold model load  : {cold_load_ms:8.1f} ms   (once, at startup)")
    print(f"First forward    : {warmup_ms:8.1f} ms   (once, at startup)")
    print("-" * 72)
    print(f"{'STAGE':<18}{'MEAN':>10}{'MEDIAN':>10}{'P95':>10}{'MAX':>10}")
    for name, stats in payload["per_request"].items():
        print(f"{name:<18}{stats['mean_ms']:>10.2f}{stats['median_ms']:>10.2f}"
              f"{stats['p95_ms']:>10.2f}{stats['max_ms']:>10.2f}")
    print("-" * 72)
    print(f"Throughput       : {payload['throughput_requests_per_second']} requests/second")
    print("=" * 72)
    print(f"Wrote {destination}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
