"""Measure what this machine can actually do, and write the evidence down.

Several training decisions in this project depend on measurements rather than
received practice - most notably that **mixed precision is disabled**, because
fp16 turned out to be roughly four times *slower* than fp32 on the development
GPU. This script reproduces every one of those measurements so the claims in
``docs/methodology.md`` can be checked rather than taken on trust.

    python scripts/probe_hardware.py
    python scripts/probe_hardware.py --models cnn_cbam resnet50 --image-size 224
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

import app.core.runtime  # noqa: F401,E402  # isort:skip

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from config import probe_hardware, recommended_workers  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.ml.architectures import build_model  # noqa: E402

DEFAULT_MODELS = ("cnn_baseline", "cnn_cbam", "mobilenet_v3_large",
                  "efficientnet_b0", "resnet50", "densenet121")


def timed(fn, iterations: int, warmup: int, synchronise: bool) -> float:
    """Mean seconds per call, after warm-up."""
    for _ in range(warmup):
        fn()
    if synchronise:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    if synchronise:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iterations


def matmul_throughput(device: torch.device, size: int = 4096) -> dict:
    """Raw TFLOPS at each precision - the clearest signal of tensor-core presence."""
    results = {}
    for dtype, name in ((torch.float32, "fp32"), (torch.float16, "fp16")):
        try:
            a = torch.randn(size, size, device=device, dtype=dtype)
            b = torch.randn(size, size, device=device, dtype=dtype)
            # Bind the operands as defaults: the lambda is created inside a loop
            # and its tensors are freed right after, so late binding would be a bug.
            seconds = timed(lambda x=a, y=b: x @ y, 20, 6, device.type == "cuda")
            results[name] = {
                "ms": round(seconds * 1000, 3),
                "tflops": round(2 * size**3 / seconds / 1e12, 3),
            }
            del a, b
        except RuntimeError as exc:
            results[name] = {"error": str(exc)[:160]}
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def training_step_throughput(model_name: str, device: torch.device, image_size: int,
                             batch_size: int) -> dict:
    """Images per second for one full training step, fp32 vs autocast fp16."""
    model = build_model(model_name, 38, pretrained=False).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    targets = torch.randint(0, 38, (batch_size,), device=device)
    result: dict = {"batch_size": batch_size, "image_size": image_size}

    def step_fp32(model=model, optimizer=optimizer, images=images, targets=targets):
        optimizer.zero_grad(set_to_none=True)
        criterion(model(images), targets).backward()
        optimizer.step()

    try:
        seconds = timed(step_fp32, 12, 5, device.type == "cuda")
        result["fp32"] = {"ms_per_step": round(seconds * 1000, 2),
                          "images_per_second": round(batch_size / seconds, 1)}
    except torch.cuda.OutOfMemoryError:
        result["fp32"] = {"error": "CUDA out of memory"}
        torch.cuda.empty_cache()

    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

        def step_amp(model=model, optimizer=optimizer, images=images, targets=targets):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        try:
            seconds = timed(step_amp, 12, 5, True)
            result["amp_fp16"] = {"ms_per_step": round(seconds * 1000, 2),
                                  "images_per_second": round(batch_size / seconds, 1)}
        except torch.cuda.OutOfMemoryError:
            result["amp_fp16"] = {"error": "CUDA out of memory"}
            torch.cuda.empty_cache()

    if "fp32" in result and "amp_fp16" in result and \
            "ms_per_step" in result["fp32"] and "ms_per_step" in result["amp_fp16"]:
        result["amp_speedup"] = round(
            result["fp32"]["ms_per_step"] / result["amp_fp16"]["ms_per_step"], 3
        )
    result["peak_vram_mb"] = (round(torch.cuda.max_memory_allocated() / 1024**2, 1)
                              if device.type == "cuda" else None)

    del model, optimizer, images, targets
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    settings.ensure_dirs()
    hardware = probe_hardware(settings.metadata_dir / "amp_probe.json")
    device = torch.device(hardware.device)

    print("=" * 78)
    print("HARDWARE PROBE")
    print("=" * 78)
    print(f"Platform    : {platform.platform()}")
    print(f"Python      : {platform.python_version()}")
    print(f"PyTorch     : {torch.__version__}  (CUDA {torch.version.cuda})")
    print(f"Device      : {hardware.gpu_name or hardware.cpu_name}")
    if hardware.device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM        : {hardware.vram_gb} GB   compute capability {props.major}.{props.minor}")
    print(f"System RAM  : {hardware.system_ram_gb} GB   CPU threads {hardware.cpu_count}")
    print(f"Workers     : {recommended_workers(hardware)} (auto-estimated)")
    print()

    matmul = {}
    if hardware.device == "cuda":
        print("-" * 78)
        print("RAW MATRIX-MULTIPLY THROUGHPUT")
        print("-" * 78)
        matmul = matmul_throughput(device)
        for name, stats in matmul.items():
            if "error" in stats:
                print(f"  {name:<6} error: {stats['error']}")
            else:
                print(f"  {name:<6} {stats['tflops']:>8.2f} TFLOPS   ({stats['ms']:.1f} ms)")
        fp32, fp16 = matmul.get("fp32", {}), matmul.get("fp16", {})
        if "tflops" in fp32 and "tflops" in fp16:
            ratio = fp16["tflops"] / fp32["tflops"]
            print(f"\n  fp16 / fp32 = {ratio:.2f}x")
            print("  " + ("Tensor cores present: half precision is genuinely faster."
                          if ratio > 1.5 else
                          "No tensor-core speed-up: half precision is NOT faster on this GPU."))
        print()

    print("-" * 78)
    print(f"TRAINING STEP THROUGHPUT  (batch {args.batch_size} @ {args.image_size}px)")
    print("-" * 78)
    print(f"  {'MODEL':<22}{'fp32 img/s':>12}{'amp img/s':>12}{'speedup':>10}{'VRAM MB':>10}")
    steps = {}
    for name in args.models:
        try:
            steps[name] = training_step_throughput(name, device, args.image_size, args.batch_size)
        except Exception as exc:  # noqa: BLE001 - one model must not abort the probe
            print(f"  {name:<22}  failed: {str(exc)[:60]}")
            continue
        entry = steps[name]
        fp32 = entry.get("fp32", {}).get("images_per_second")
        amp = entry.get("amp_fp16", {}).get("images_per_second")
        speedup = entry.get("amp_speedup")
        print(f"  {name:<22}{fp32 if fp32 else '—':>12}{amp if amp else '—':>12}"
              f"{f'{speedup:.2f}x' if speedup else '—':>10}{entry.get('peak_vram_mb') or '—':>10}")

    speedups = [e["amp_speedup"] for e in steps.values() if e.get("amp_speedup")]
    verdict = None
    if speedups:
        median = sorted(speedups)[len(speedups) // 2]
        verdict = {
            "median_amp_speedup": round(median, 3),
            "amp_recommended": median > 1.15,
            "statement": (
                f"Mixed precision is {median:.2f}x the speed of full precision on this GPU, "
                + ("so it is enabled." if median > 1.15 else
                   "so it is DISABLED - enabling it would make training slower."),
            )[0],
        }
        print("\n" + "=" * 78)
        print("VERDICT")
        print("=" * 78)
        print(f"  {verdict['statement']}")
        print(f"  The training pipeline reads this from models/metadata/amp_probe.json "
              f"and sets amp={verdict['amp_recommended']}.")
        print("  Note: absolute throughput drops if another job is using the GPU,")
        print("  but the fp16/fp32 ratio - which the decision turns on - is unaffected.")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hardware": hardware.as_dict(),
        "recommended_workers": recommended_workers(hardware),
        "matmul_throughput": matmul,
        "training_step_throughput": steps,
        "verdict": verdict,
    }
    destination = settings.results_dir / "hardware_probe.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {destination}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
