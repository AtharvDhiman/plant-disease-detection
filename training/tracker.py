"""Lightweight experiment tracking (JSON + CSV, no external service).

Each run appends one record to ``artifacts/results/experiments.json`` capturing
the experiment id, model, hyper-parameters, dataset version, timings and every
evaluation metric, plus a flattened CSV for spreadsheet inspection. MLflow would
work too, but a plain JSON ledger keeps the project runnable from a clean clone
with no server to start - and it is what the API reads to populate the research
dashboard.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentTracker:
    """Append-only ledger of experiment records."""

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.results_dir / "experiments.json"
        self.csv_path = self.results_dir / "experiments.csv"
        self.records: list[dict] = []
        if self.json_path.exists():
            try:
                self.records = json.loads(self.json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # A half-written ledger from an interrupted run must not block a
                # rerun; keep the corrupt file for inspection and start fresh.
                self.json_path.rename(self.json_path.with_suffix(".corrupt.json"))
                self.records = []

    # ------------------------------------------------------------------ ids
    # Fields that describe *how* a run was executed rather than *what* it was.
    # Excluding them keeps an experiment's identity stable when the machine's
    # resources change - so lowering the worker count on a memory-constrained
    # host does not invalidate results that are already in the ledger.
    # (More workers do perturb the augmentation RNG stream, but that is
    # seed-level noise, not a different experiment.)
    OPERATIONAL_FIELDS = frozenset({"experiment_id", "num_workers", "notes", "tags"})

    @staticmethod
    def make_experiment_id(model_name: str, protocol_tag: str, config: dict) -> str:
        """Deterministic id: ``<model>__<protocol>__<8-char config hash>``.

        Deterministic rather than random so re-running the same configuration
        overwrites its own record instead of accumulating duplicates.
        """
        payload = json.dumps(
            {k: v for k, v in sorted(config.items())
             if k not in ExperimentTracker.OPERATIONAL_FIELDS},
            sort_keys=True, default=str,
        )
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
        return f"{model_name}__{protocol_tag}__{digest}"

    # --------------------------------------------------------------- record
    def log(self, record: dict[str, Any]) -> dict:
        """Insert or replace a record keyed by ``experiment_id``."""
        record = {**record, "logged_at": datetime.now(timezone.utc).isoformat()}
        experiment_id = record["experiment_id"]
        self.records = [r for r in self.records if r.get("experiment_id") != experiment_id]
        self.records.append(record)
        self.flush()
        return record

    def get(self, experiment_id: str) -> dict | None:
        return next((r for r in self.records if r.get("experiment_id") == experiment_id), None)

    def has(self, experiment_id: str) -> bool:
        return self.get(experiment_id) is not None

    def by_group(self, group: str) -> list[dict]:
        return [r for r in self.records if r.get("group") == group]

    # ---------------------------------------------------------------- write
    def flush(self) -> None:
        """Write both the JSON ledger and the flattened CSV view."""
        self.json_path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")

        columns = [
            "experiment_id", "model_name", "label", "group", "proposed", "protocol",
            "image_size", "batch_size", "learning_rate", "epochs_run", "best_epoch",
            "training_time_s", "total_parameters", "model_size_mb",
            "val_accuracy", "test_accuracy", "test_precision_macro", "test_recall_macro",
            "test_f1_macro", "test_f1_weighted", "test_top3_accuracy", "test_top5_accuracy",
            "test_roc_auc_macro_ovr", "test_ece", "inference_ms_per_image", "checkpoint",
        ]
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for record in self.records:
                writer.writerow(flatten_record(record))


def flatten_record(record: dict) -> dict:
    """Flatten a nested experiment record into scalar columns for CSV/tables."""
    config = record.get("config", {}) or {}
    test = (record.get("test") or {})
    metrics = test.get("metrics", {}) or {}
    calibration = test.get("calibration", {}) or {}
    latency = test.get("latency", {}) or {}

    return {
        "experiment_id": record.get("experiment_id"),
        "model_name": record.get("model_name"),
        "label": record.get("label"),
        "group": record.get("group"),
        "proposed": record.get("proposed"),
        "protocol": record.get("protocol"),
        "image_size": config.get("image_size"),
        "batch_size": config.get("batch_size"),
        "learning_rate": config.get("learning_rate"),
        "epochs_run": record.get("epochs_run"),
        "best_epoch": record.get("best_epoch"),
        "training_time_s": round(record.get("training_time_s") or 0.0, 2),
        "total_parameters": record.get("total_parameters"),
        "model_size_mb": round(record.get("model_size_mb") or 0.0, 3),
        "val_accuracy": record.get("best_val_accuracy"),
        "test_accuracy": metrics.get("accuracy"),
        "test_precision_macro": metrics.get("precision_macro"),
        "test_recall_macro": metrics.get("recall_macro"),
        "test_f1_macro": metrics.get("f1_macro"),
        "test_f1_weighted": metrics.get("f1_weighted"),
        "test_top3_accuracy": metrics.get("top3_accuracy"),
        "test_top5_accuracy": metrics.get("top5_accuracy"),
        "test_roc_auc_macro_ovr": metrics.get("roc_auc_macro_ovr"),
        "test_ece": calibration.get("ece"),
        "inference_ms_per_image": latency.get("single_image_ms"),
        "checkpoint": record.get("checkpoint_path"),
    }


def summarise_for_api(records: list[dict]) -> list[dict]:
    """Compact per-model rows for the benchmark table and charts in the UI."""
    rows = []
    for record in records:
        flat = flatten_record(record)
        rows.append({
            "experiment_id": flat["experiment_id"],
            "model_name": flat["model_name"],
            "label": flat["label"],
            "group": flat["group"],
            "proposed": bool(flat["proposed"]),
            "protocol": flat["protocol"],
            "accuracy": flat["test_accuracy"],
            "precision": flat["test_precision_macro"],
            "recall": flat["test_recall_macro"],
            "f1_macro": flat["test_f1_macro"],
            "f1_weighted": flat["test_f1_weighted"],
            "top3_accuracy": flat["test_top3_accuracy"],
            "top5_accuracy": flat["test_top5_accuracy"],
            "roc_auc": flat["test_roc_auc_macro_ovr"],
            "ece": flat["test_ece"],
            "parameters": flat["total_parameters"],
            "model_size_mb": flat["model_size_mb"],
            "inference_ms": flat["inference_ms_per_image"],
            "training_time_s": flat["training_time_s"],
            "val_accuracy": flat["val_accuracy"],
            "epochs_run": flat["epochs_run"],
        })
    return rows
