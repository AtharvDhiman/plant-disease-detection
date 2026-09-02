"""Evaluate a trained checkpoint: metrics, calibration, latency and plots.

Used by ``run_experiments.py`` after every training run and available standalone:

    python training/evaluate.py --checkpoint models/checkpoints/<id>.pt --split test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import app.core.runtime  # noqa: F401  # isort:skip

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from config import TrainConfig  # noqa: E402
from metrics import (  # noqa: E402
    brier_score,
    compute_metrics,
    expected_calibration_error,
    plot_confidence_distribution,
    plot_confusion_matrix,
    plot_per_class_performance,
    plot_reliability_diagram,
    predictive_entropy,
)
from torch.utils.data import DataLoader  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.ml.architectures import (  # noqa: E402
    MODEL_ZOO,
    build_model,
    count_parameters,
    model_size_mb,
)
from app.ml.datasets import load_split, load_split_config  # noqa: E402
from app.ml.transforms import build_eval_transform  # noqa: E402

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(logits, probabilities, labels)`` for a whole split."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                            dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        # Softmax in fp32: fp16 softmax loses resolution in the tail, which
        # matters for top-5 accuracy and for the entropy-based OOD score.
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append(targets.numpy())

    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    return logits, probabilities, labels


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
    batch_sizes: tuple[int, ...] = (1, 8, 32),
    warmup: int = 10,
    iterations: int = 40,
) -> dict:
    """Measure forward-pass latency, excluding data loading and preprocessing."""
    model.eval()
    results: dict[str, dict] = {}

    for batch_size in batch_sizes:
        try:
            sample = torch.randn(batch_size, 3, image_size, image_size, device=device)
            for _ in range(warmup):
                model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()

            timings = []
            for _ in range(iterations):
                start = time.perf_counter()
                model(sample)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000.0)

            timings_arr = np.array(timings)
            results[f"batch_{batch_size}"] = {
                "batch_size": batch_size,
                "mean_ms": float(timings_arr.mean()),
                "median_ms": float(np.median(timings_arr)),
                "p95_ms": float(np.percentile(timings_arr, 95)),
                "std_ms": float(timings_arr.std()),
                "ms_per_image": float(timings_arr.mean() / batch_size),
                "images_per_second": float(batch_size * 1000.0 / timings_arr.mean()),
            }
        except torch.cuda.OutOfMemoryError:  # pragma: no cover - hardware dependent
            log.warning("Latency probe OOM", fields={"batch_size": batch_size})
            torch.cuda.empty_cache()

    single = results.get("batch_1", {})
    return {
        "device": str(device),
        "image_size": image_size,
        "by_batch_size": results,
        "single_image_ms": single.get("mean_ms"),
        "single_image_p95_ms": single.get("p95_ms"),
    }


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 120) -> float:
    """Fit a single temperature on held-out logits (Guo et al., 2017).

    Temperature scaling divides the logits by a scalar T > 1 to soften
    over-confident predictions. It cannot change which class wins, so accuracy is
    provably unaffected - only the confidence values move.
    """
    logit_tensor = torch.from_numpy(logits).float()
    label_tensor = torch.from_numpy(labels).long()
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logit_tensor / log_temperature.exp(), label_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().item())


def calibration_report(
    logits: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    temperature: float | None = None,
) -> dict:
    """ECE/MCE/Brier before and (optionally) after temperature scaling."""
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correct = (predictions == labels).astype(np.float64)

    ece, mce, bins = expected_calibration_error(confidences, correct)
    report = {
        "ece": ece,
        "mce": mce,
        "brier": brier_score(probabilities, labels),
        "mean_confidence": float(confidences.mean()),
        "accuracy": float(correct.mean()),
        "over_confidence_gap": float(confidences.mean() - correct.mean()),
        "bins": bins,
        "temperature": None,
        "after_temperature": None,
    }

    if temperature is not None:
        scaled = torch.softmax(torch.from_numpy(logits / temperature), dim=1).numpy()
        scaled_conf = scaled.max(axis=1)
        scaled_correct = (scaled.argmax(axis=1) == labels).astype(np.float64)
        ece_t, mce_t, bins_t = expected_calibration_error(scaled_conf, scaled_correct)
        report["temperature"] = temperature
        report["after_temperature"] = {
            "ece": ece_t,
            "mce": mce_t,
            "brier": brier_score(scaled, labels),
            "mean_confidence": float(scaled_conf.mean()),
            "accuracy": float(scaled_correct.mean()),
            "bins": bins_t,
        }
    return report


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, TrainConfig, list[str]]:
    """Rebuild a model from a training checkpoint."""
    payload = torch.load(path, map_location=device, weights_only=False)
    config = TrainConfig.from_dict(payload["config"])
    class_names = payload["class_names"]
    model = build_model(config.model_name, len(class_names), pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval(), config, class_names


# --------------------------------------------------------------------------- #
# Full evaluation
# --------------------------------------------------------------------------- #


def save_raw_predictions(
    destination: Path, probabilities: np.ndarray, labels: np.ndarray
) -> None:
    """Persist raw probabilities so downstream analyses need no re-inference.

    About 700 KB per model for the benchmark test split. Cheap insurance: any
    later question about calibration, error patterns or OOD scoring can be
    answered from these arrays instead of another pass over the dataset.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, probabilities=probabilities.astype(np.float32), labels=labels)


def evaluate_checkpoint(
    checkpoint_path: Path,
    split_config: dict,
    split_name: str,
    device: torch.device,
    calibration_split: str | None = None,
    make_plots: bool = True,
    plot_prefix: str | None = None,
) -> dict:
    """Evaluate one checkpoint and return a JSON-serialisable result record."""
    model, config, class_names = load_checkpoint(checkpoint_path, device)
    spec = MODEL_ZOO.get(config.model_name)
    prefix = plot_prefix or config.experiment_id or config.model_name

    eval_tf = build_eval_transform(config.image_size)
    dataset = load_split(split_config, split_name, eval_tf)
    loader = DataLoader(
        dataset,
        batch_size=max(config.batch_size, 32),
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    started = time.perf_counter()
    logits, probabilities, labels = collect_predictions(model, loader, device, amp=config.amp)
    wall_time = time.perf_counter() - started
    predictions = probabilities.argmax(axis=1)

    metrics = compute_metrics(labels, predictions, probabilities, class_names)

    temperature = None
    if calibration_split:
        cal_ds = load_split(split_config, calibration_split, eval_tf)
        cal_loader = DataLoader(cal_ds, batch_size=max(config.batch_size, 32), shuffle=False,
                                num_workers=config.num_workers, pin_memory=device.type == "cuda")
        cal_logits, _, cal_labels = collect_predictions(model, cal_loader, device, amp=config.amp)
        temperature = fit_temperature(cal_logits, cal_labels)
        log.info("Fitted temperature", fields={"model": config.model_name,
                                               "T": round(temperature, 4),
                                               "on_split": calibration_split})

    calibration = calibration_report(logits, probabilities, labels, temperature)
    latency = measure_latency(model, device, config.image_size)
    total_params, _ = count_parameters(model)
    entropy = predictive_entropy(probabilities)
    confidences = probabilities.max(axis=1)
    correct = predictions == labels

    # Persist the raw probabilities so calibration, OOD and error analyses can be
    # redone later without re-running inference over the whole split.
    predictions_path = settings.results_dir / "predictions" / f"{prefix}_{split_name}.npz"
    save_raw_predictions(predictions_path, probabilities, labels)

    plots: dict[str, str] = {}
    if make_plots:
        cm_path = settings.confusion_dir / f"{prefix}_{split_name}.png"
        plot_confusion_matrix(labels, predictions, class_names, cm_path,
                              normalize=True,
                              title=f"{spec.label if spec else config.model_name} - {split_name}")
        plots["confusion_matrix"] = str(cm_path)

        pc_path = settings.plots_dir / f"{prefix}_per_class.png"
        plot_per_class_performance(metrics["per_class"], pc_path,
                                   f"Per-class performance - {spec.label if spec else config.model_name}")
        plots["per_class"] = str(pc_path)

        rel_path = settings.plots_dir / f"{prefix}_reliability.png"
        plot_reliability_diagram(calibration["bins"], calibration["ece"], rel_path,
                                 f"Reliability - {spec.label if spec else config.model_name}")
        plots["reliability"] = str(rel_path)

        conf_path = settings.plots_dir / f"{prefix}_confidence.png"
        plot_confidence_distribution(confidences, correct, conf_path,
                                     f"Confidence distribution - {spec.label if spec else config.model_name}")
        plots["confidence_distribution"] = str(conf_path)

    return {
        "model_name": config.model_name,
        "label": spec.label if spec else config.model_name,
        "proposed": bool(spec.proposed) if spec else False,
        "group": spec.group if spec else "benchmark",
        "checkpoint": str(checkpoint_path),
        "split": split_name,
        "image_size": config.image_size,
        "metrics": metrics,
        "calibration": calibration,
        "latency": latency,
        "total_parameters": total_params,
        "model_size_mb": model_size_mb(model),
        "evaluation_wall_time_s": wall_time,
        "throughput_images_per_s": len(dataset) / wall_time,
        "raw_predictions": str(predictions_path.relative_to(settings.project_root)),
        "confidence_stats": {
            "mean": float(confidences.mean()),
            "median": float(np.median(confidences)),
            "p05": float(np.percentile(confidences, 5)),
            "mean_when_correct": float(confidences[correct].mean()) if correct.any() else None,
            "mean_when_wrong": float(confidences[~correct].mean()) if (~correct).any() else None,
        },
        "entropy_stats": {
            "mean": float(entropy.mean()),
            "p95": float(np.percentile(entropy, 95)),
            "mean_when_correct": float(entropy[correct].mean()) if correct.any() else None,
            "mean_when_wrong": float(entropy[~correct].mean()) if (~correct).any() else None,
        },
        "plots": plots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--calibration-split", default="val")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    configure_logging(settings.log_level, settings.logs_dir / "evaluate.log")
    settings.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_config = load_split_config(settings.splits_path)

    result = evaluate_checkpoint(
        Path(args.checkpoint), split_config, args.split, device,
        calibration_split=args.calibration_split, make_plots=not args.no_plots,
    )
    out = settings.results_dir / f"eval_{Path(args.checkpoint).stem}_{args.split}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    m = result["metrics"]
    print(f"\n{result['label']} on '{args.split}'")
    print(f"  accuracy   : {m['accuracy']:.4f}")
    print(f"  macro F1   : {m['f1_macro']:.4f}")
    print(f"  weighted F1: {m['f1_weighted']:.4f}")
    print(f"  top-3      : {m['top3_accuracy']:.4f}")
    print(f"  top-5      : {m['top5_accuracy']:.4f}")
    print(f"  ECE        : {result['calibration']['ece']:.4f}")
    print(f"  latency    : {result['latency']['single_image_ms']:.2f} ms/image")
    print(f"  saved      : {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
