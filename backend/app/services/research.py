"""Read-only access to the training artefacts for the research dashboard.

The API never recomputes a metric. Everything served from here was produced by
the training pipeline and written to ``artifacts/`` - so the dashboard, the
README and the docs are guaranteed to agree, and none of them can drift into
reporting a number no experiment produced.

Files are cached in memory with an mtime check so a re-run of the training
pipeline is picked up without restarting the API.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


class JsonCache:
    """Cache a JSON file, reloading when its mtime changes."""

    def __init__(self) -> None:
        self._store: dict[Path, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def load(self, path: Path, default: Any = None) -> Any:
        path = Path(path)
        if not path.exists():
            return default
        mtime = path.stat().st_mtime
        with self._lock:
            cached = self._store.get(path)
            if cached and cached[0] == mtime:
                return cached[1]
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                log.warning("Malformed artefact", fields={"path": str(path), "error": str(exc)})
                return default
            self._store[path] = (mtime, data)
            return data

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


cache = JsonCache()


# --------------------------------------------------------------------------- #
# Artefact accessors
# --------------------------------------------------------------------------- #


def experiments() -> list[dict]:
    """Every logged experiment record."""
    return cache.load(settings.results_dir / "experiments.json", default=[]) or []


def dataset_report() -> dict | None:
    return cache.load(settings.dataset_report_path)


def split_config() -> dict | None:
    return cache.load(settings.splits_path)


def classical_results() -> dict | None:
    return cache.load(settings.results_dir / "classical_results.json")


def ablation_summary() -> dict | None:
    return cache.load(settings.results_dir / "ablation_summary.json")


def ood_calibration() -> dict | None:
    return cache.load(settings.metadata_dir / "ood_thresholds.json")


def serving_benchmark() -> dict | None:
    """End-to-end timings measured on the serving path by scripts/benchmark_serving.py."""
    return cache.load(settings.results_dir / "serving_benchmark.json")


def production_manifest() -> dict | None:
    return cache.load(settings.exported_dir / "production.json")


def history_for(experiment_id: str) -> dict | None:
    return cache.load(settings.results_dir / f"history_{experiment_id}.json")


# --------------------------------------------------------------------------- #
# Derived views
# --------------------------------------------------------------------------- #


def _metric(record: dict, path: str, default=None):
    node: Any = record
    for key in path.split("."):
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            return default
    return node


def canonical_seed() -> int | None:
    """The seed the study was run at; every other seed marks a replicate."""
    seeds = [(r.get("config") or {}).get("seed") for r in experiments()]
    seeds = [s for s in seeds if s is not None]
    return min(seeds) if seeds else None


def benchmark_rows(groups: tuple[str, ...] = ("benchmark", "classical")) -> list[dict]:
    """Flat per-model rows for the benchmark table and its charts.

    Seed replicates are excluded. They exist to measure run-to-run variation and
    share a model name with the canonical run, so including them listed the same
    architecture three times in the dashboard - identical name, identical size,
    three different scores - which reads as three models rather than one model
    measured three times.
    """
    baseline_seed = canonical_seed()
    rows = []
    for record in experiments():
        if record.get("group") not in groups:
            continue
        seed = (record.get("config") or {}).get("seed")
        if baseline_seed is not None and seed is not None and seed != baseline_seed:
            continue
        rows.append({
            "experiment_id": record.get("experiment_id"),
            "model_name": record.get("model_name"),
            "label": record.get("label"),
            "group": record.get("group"),
            "family": "classical" if record.get("group") == "classical" else "deep",
            "proposed": bool(record.get("proposed")),
            "protocol": record.get("protocol"),
            "notes": record.get("notes"),
            "accuracy": _metric(record, "test.metrics.accuracy"),
            "balanced_accuracy": _metric(record, "test.metrics.balanced_accuracy"),
            "precision": _metric(record, "test.metrics.precision_macro"),
            "recall": _metric(record, "test.metrics.recall_macro"),
            "f1_macro": _metric(record, "test.metrics.f1_macro"),
            "f1_weighted": _metric(record, "test.metrics.f1_weighted"),
            "top1_accuracy": _metric(record, "test.metrics.top1_accuracy"),
            "top3_accuracy": _metric(record, "test.metrics.top3_accuracy"),
            "top5_accuracy": _metric(record, "test.metrics.top5_accuracy"),
            "roc_auc": _metric(record, "test.metrics.roc_auc_macro_ovr"),
            "cohen_kappa": _metric(record, "test.metrics.cohen_kappa"),
            "ece": _metric(record, "test.calibration.ece"),
            "parameters": record.get("total_parameters"),
            "model_size_mb": record.get("model_size_mb"),
            "inference_ms": _metric(record, "test.latency.single_image_ms"),
            "training_time_s": record.get("training_time_s"),
            "epochs_run": record.get("epochs_run"),
            "best_epoch": record.get("best_epoch"),
            "val_accuracy": record.get("best_val_accuracy"),
            "image_size": _metric(record, "config.image_size"),
            "stopped_early": record.get("stopped_early"),
        })
    return sorted(rows, key=lambda r: -(r["accuracy"] or 0))


def ablation_rows() -> list[dict]:
    """Rows for the attention ablation, ordered as A..E in the study."""
    order = ["abl_cnn_none", "abl_cnn_channel", "abl_cnn_spatial", "abl_cnn_se", "abl_cnn_cbam"]
    rows = {r["model_name"]: r for r in benchmark_rows(groups=("ablation",))}
    return [rows[name] for name in order if name in rows]


def placement_rows() -> list[dict]:
    """The CBAM placement variants, fixed-budget runs only.

    place_cbam_last2 was also retrained under the production protocol, and
    including that row put two bars labelled "CBAM on last two stages" in the
    placement chart - four bars for three placements, with the taller one being
    a training-budget difference rather than a placement difference.
    """
    return [r for r in benchmark_rows(groups=("placement",))
            if r.get("protocol") != "production"]


def per_class_performance(experiment_id: str | None = None) -> dict | None:
    """Per-class precision/recall/F1 of one model (default: the production model)."""
    records = experiments()
    if experiment_id:
        record = next((r for r in records if r.get("experiment_id") == experiment_id), None)
    else:
        manifest = production_manifest()
        target = manifest.get("experiment_id") if manifest else None
        record = next((r for r in records if r.get("experiment_id") == target), None)
    if record is None:
        return None
    per_class = _metric(record, "test.metrics.per_class", {}) or {}
    return {
        "experiment_id": record.get("experiment_id"),
        "model_name": record.get("model_name"),
        "label": record.get("label"),
        "split": _metric(record, "test.split"),
        "classes": [
            {"class_name": name, **values}
            for name, values in sorted(per_class.items(), key=lambda kv: kv[1]["f1"])
        ],
    }


def training_curves(experiment_id: str) -> dict | None:
    record = next((r for r in experiments() if r.get("experiment_id") == experiment_id), None)
    if record is None:
        return None
    return {
        "experiment_id": experiment_id,
        "label": record.get("label"),
        "history": record.get("history", {}),
        "best_epoch": record.get("best_epoch"),
        "stopped_early": record.get("stopped_early"),
    }


def dataset_summary() -> dict | None:
    """Compact dataset statistics for the dashboard header cards."""
    report = dataset_report()
    splits = split_config()
    if report is None:
        return None

    train = report["splits"]["train"]
    payload = {
        "source": report["dataset"]["kaggle_slug"],
        "generated_at": report.get("generated_at"),
        "num_classes": report["classes"]["count"],
        "num_plants": report["classes"]["plant_count"],
        "plants": report["classes"]["plants"],
        "healthy_classes": len(report["classes"]["healthy_classes"]),
        "diseased_classes": len(report["classes"]["diseased_classes"]),
        "total_images": sum(s["num_images"] for s in report["splits"].values()),
        "official_splits": {k: v["num_images"] for k, v in report["splits"].items()},
        "image_size": report["splits"]["train"]["image_geometry"]["unique_sizes"][:3],
        "imbalance": train["imbalance"],
        "corrupt_rate_in_sample": train["corrupt_rate_in_sample"],
        "duplicates": report.get("duplicates"),
        "class_counts": train["class_counts"],
    }
    if splits:
        payload["protocol"] = splits.get("protocol")
        payload["imbalance_strategy"] = splits.get("imbalance", {}).get("strategy")
        payload["imbalance_reason"] = splits.get("imbalance", {}).get("reason")
        payload["working_splits"] = {
            name: info["count"] for name, info in splits.get("splits", {}).items()
        }
    return payload


def available_plots() -> dict[str, list[str]]:
    """Every rendered figure, grouped by artefact directory."""
    groups = {
        "plots": settings.plots_dir,
        "confusion_matrices": settings.confusion_dir,
        "explainability": settings.explain_dir,
    }
    return {
        name: sorted(p.name for p in directory.glob("*.png"))
        for name, directory in groups.items()
        if directory.exists()
    }


def plot_path(group: str, filename: str) -> Path | None:
    """Resolve a figure path, rejecting anything outside the artefact directories."""
    directories = {
        "plots": settings.plots_dir,
        "confusion_matrices": settings.confusion_dir,
        "explainability": settings.explain_dir,
    }
    root = directories.get(group)
    if root is None:
        return None
    candidate = (root / filename).resolve()
    if not str(candidate).startswith(str(root.resolve())) or not candidate.is_file():
        return None
    return candidate
