"""Select and export the production model from the measured experiment results.

Reads ``artifacts/results/experiments.json``, ranks the trained models by a
configurable criterion and writes ``models/exported/production.json`` - the only
thing the serving code reads to decide which weights to load.

The criterion is explicit and configurable rather than hard-coded to "CBAM
always wins": the brief asks for the production model to be chosen on merit
while CBAM remains the *research* contribution. If a baseline beats the proposed
architecture, this script will say so and pick the baseline, and the README will
report it.

    python training/export_model.py                       # best macro F1
    python training/export_model.py --criterion accuracy
    python training/export_model.py --criterion balanced  # accuracy per MB and ms
    python training/export_model.py --model cnn_cbam      # force a specific model
    python training/export_model.py --torchscript
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import app.core.runtime  # noqa: F401  # isort:skip

import torch  # noqa: E402
from tracker import ExperimentTracker  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.ml.architectures import MODEL_ZOO  # noqa: E402

log = get_logger(__name__)


def _metric(record: dict, path: str, default=None):
    node = record
    for key in path.split("."):
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            return default
    return node


CRITERIA = {
    "f1_macro": {
        "description": "Highest macro F1 on the held-out test split - treats every "
                       "disease class equally regardless of how many images it has.",
        "key": lambda r: _metric(r, "test.metrics.f1_macro", 0.0),
    },
    "accuracy": {
        "description": "Highest top-1 accuracy on the held-out test split.",
        "key": lambda r: _metric(r, "test.metrics.accuracy", 0.0),
    },
    "balanced": {
        "description": "Best accuracy-per-cost: macro F1 penalised by model size and "
                       "single-image latency. Favours a deployable model.",
        "key": lambda r: (
            _metric(r, "test.metrics.f1_macro", 0.0)
            / (1 + (r.get("model_size_mb") or 1) / 100)
            / (1 + (_metric(r, "test.latency.single_image_ms", 10.0) or 10.0) / 100)
        ),
    },
    "fastest": {
        "description": "Lowest single-image inference latency among models within "
                       "2 points of the best macro F1.",
        "key": lambda r: -(_metric(r, "test.latency.single_image_ms", 1e9) or 1e9),
    },
    "smallest": {
        "description": "Smallest on-disk model among models within 2 points of the "
                       "best macro F1.",
        "key": lambda r: -(r.get("model_size_mb") or 1e9),
    },
}

# Criteria that optimise cost must not be allowed to pick a bad model, so they
# only rank models whose macro F1 is within this margin of the leader.
QUALITY_GATE = 0.02


def select(records: list[dict], criterion: str) -> tuple[dict, list[dict]]:
    """Rank eligible records under ``criterion``; return the winner and the ranking."""
    eligible = [r for r in records if _metric(r, "test.metrics.f1_macro") is not None]
    if not eligible:
        raise SystemExit("No completed experiments with test metrics found.")

    pool = eligible
    if criterion in {"fastest", "smallest"}:
        best_f1 = max(_metric(r, "test.metrics.f1_macro", 0.0) for r in eligible)
        pool = [r for r in eligible
                if _metric(r, "test.metrics.f1_macro", 0.0) >= best_f1 - QUALITY_GATE]
        log.info("Quality gate applied",
                 fields={"criterion": criterion, "best_f1": round(best_f1, 4),
                         "eligible": len(pool), "of": len(eligible)})

    ranked = sorted(pool, key=CRITERIA[criterion]["key"], reverse=True)
    return ranked[0], ranked


def summarise(record: dict) -> dict:
    """Compact candidate entry for the manifest."""
    spec = MODEL_ZOO.get(record["model_name"])
    checkpoint = Path(record["checkpoint_path"])
    # A manifest pointing at a file that no longer exists produces an API that
    # starts fine and then 503s on every prediction, which is a confusing way to
    # discover a stale ledger. Fail here instead.
    if not checkpoint.exists():
        raise SystemExit(
            f"Checkpoint for {record['model_name']!r} is missing: {checkpoint}\n"
            "The experiment ledger references a file that is no longer on disk. "
            "Re-run the training suite, or run scripts/migrate_experiment_ids.py "
            "if the id-hashing policy changed."
        )
    # A checkpoint outside the project root keeps its absolute path.
    with contextlib.suppress(ValueError):
        checkpoint = checkpoint.relative_to(settings.project_root)
    return {
        "model_name": record["model_name"],
        "label": record.get("label") or (spec.label if spec else record["model_name"]),
        "experiment_id": record["experiment_id"],
        "protocol": record.get("protocol"),
        "proposed": bool(record.get("proposed")),
        "checkpoint": str(checkpoint),
        "image_size": _metric(record, "config.image_size", settings.image_size),
        "temperature": _metric(record, "test.calibration.temperature") or 1.0,
        "version": "1.0.0",
        "test_metrics": {
            "accuracy": _metric(record, "test.metrics.accuracy"),
            "precision_macro": _metric(record, "test.metrics.precision_macro"),
            "recall_macro": _metric(record, "test.metrics.recall_macro"),
            "f1_macro": _metric(record, "test.metrics.f1_macro"),
            "f1_weighted": _metric(record, "test.metrics.f1_weighted"),
            "top3_accuracy": _metric(record, "test.metrics.top3_accuracy"),
            "top5_accuracy": _metric(record, "test.metrics.top5_accuracy"),
            "roc_auc_macro_ovr": _metric(record, "test.metrics.roc_auc_macro_ovr"),
            "ece": _metric(record, "test.calibration.ece"),
            "ece_after_temperature": _metric(record, "test.calibration.after_temperature.ece"),
            "test_split": _metric(record, "test.split"),
            "test_samples": _metric(record, "test.metrics.num_samples"),
        },
        "parameters": record.get("total_parameters"),
        "model_size_mb": record.get("model_size_mb"),
        "inference_ms": _metric(record, "test.latency.single_image_ms"),
        "training_time_s": record.get("training_time_s"),
    }


def export_torchscript(record: dict, destination: Path) -> dict | None:
    """Trace the model to TorchScript for a Python-free deployment target."""
    from app.ml.architectures import build_model

    checkpoint = Path(record["checkpoint_path"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_names = payload["class_names"]
    model = build_model(record["model_name"], len(class_names), pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    image_size = _metric(record, "config.image_size", settings.image_size)
    example = torch.zeros(1, 3, image_size, image_size)
    try:
        # `trace` rather than `script`: the models contain no data-dependent
        # control flow, and tracing avoids TorchScript's stricter typing rules.
        traced = torch.jit.trace(model, example, strict=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        traced.save(str(destination))
    except Exception as exc:  # noqa: BLE001
        log.warning("TorchScript export failed", fields={"error": str(exc)})
        return None

    size_mb = destination.stat().st_size / 1024**2
    log.info("Exported TorchScript", fields={"path": str(destination), "mb": round(size_mb, 2)})
    return {"path": str(destination.relative_to(settings.project_root)),
            "size_mb": round(size_mb, 3), "format": "torchscript", "image_size": image_size}


def export_onnx(record: dict, destination: Path) -> dict | None:
    """Export to ONNX for runtimes outside Python.

    ONNX is the portable target here for the same reason TensorFlow Lite would
    be in a Keras project: it runs under ONNX Runtime on CPU, mobile and in the
    browser without a PyTorch install. The exported graph is verified by running
    it through ONNX Runtime and comparing against the PyTorch output, because an
    export that silently produces different numbers is worse than no export.
    """
    from app.ml.architectures import build_model

    checkpoint = Path(record["checkpoint_path"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_names = payload["class_names"]
    model = build_model(record["model_name"], len(class_names), pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    image_size = _metric(record, "config.image_size", settings.image_size)
    example = torch.randn(1, 3, image_size, image_size)
    destination.parent.mkdir(parents=True, exist_ok=True)

    common = {
        "input_names": ["image"],
        "output_names": ["logits"],
        # A dynamic batch axis lets the same graph serve one image or a batch.
        "dynamic_axes": {"image": {0: "batch"}, "logits": {0: "batch"}},
        "opset_version": 17,
    }
    exporter = "dynamo"
    try:
        torch.onnx.export(model, (example,), str(destination), **common)
    except Exception as exc:  # noqa: BLE001
        # torch 2.6+ defaults to the dynamo exporter, which needs the optional
        # `onnxscript` package. The TorchScript exporter needs no extra
        # dependency and handles these models fine, so fall back rather than
        # making ONNX export require an install the rest of the project does not.
        log.info("Dynamo ONNX exporter unavailable; falling back to TorchScript-based export",
                 fields={"reason": str(exc)[:160]})
        exporter = "torchscript"
        try:
            torch.onnx.export(model, (example,), str(destination), dynamo=False, **common)
        except Exception as fallback_exc:  # noqa: BLE001
            log.warning("ONNX export failed", fields={"error": str(fallback_exc)})
            return None

    max_difference = None
    try:
        import numpy as np
        import onnxruntime

        session = onnxruntime.InferenceSession(str(destination),
                                               providers=["CPUExecutionProvider"])
        onnx_out = session.run(None, {"image": example.numpy()})[0]
        with torch.no_grad():
            torch_out = model(example).numpy()
        max_difference = float(np.abs(onnx_out - torch_out).max())
        if max_difference > 1e-3:
            log.warning("ONNX output diverges from PyTorch",
                        fields={"max_abs_diff": max_difference})
    except ImportError:
        log.info("onnxruntime not installed; skipping the ONNX verification step")
    except Exception as exc:  # noqa: BLE001
        log.warning("ONNX verification failed", fields={"error": str(exc)})

    size_mb = destination.stat().st_size / 1024**2
    log.info("Exported ONNX", fields={"path": str(destination), "mb": round(size_mb, 2),
                                      "max_abs_diff_vs_torch": max_difference})
    return {
        "path": str(destination.relative_to(settings.project_root)),
        "size_mb": round(size_mb, 3),
        "format": "onnx",
        "exporter": exporter,
        "opset": 17,
        "image_size": image_size,
        "dynamic_batch": True,
        "max_abs_diff_vs_pytorch": max_difference,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criterion", default="f1_macro", choices=list(CRITERIA))
    parser.add_argument("--model", default=None, help="Force a specific model name.")
    parser.add_argument("--protocol", default=None,
                        help="Restrict selection to one protocol (e.g. production).")
    parser.add_argument("--torchscript", action="store_true")
    parser.add_argument("--onnx", action="store_true",
                        help="Also export ONNX for non-Python runtimes.")
    args = parser.parse_args()

    configure_logging(settings.log_level, settings.logs_dir / "export.log")
    settings.ensure_dirs()

    tracker = ExperimentTracker(settings.results_dir)
    records = [r for r in tracker.records if r.get("group") in {"benchmark", "ablation", "placement"}
               or r.get("protocol") == "production"]
    if args.protocol:
        records = [r for r in records if r.get("protocol") == args.protocol]
    if not records:
        raise SystemExit("No experiment records to export from. Run training/run_experiments.py.")

    # Prefer full-data production runs when any exist: a model trained on 63k
    # images is the right thing to ship even if a benchmark-subset run happens
    # to score higher on the smaller subset test split.
    production_records = [r for r in records if r.get("protocol") == "production"]
    pool = production_records or records
    if production_records and not args.protocol:
        log.info("Selecting from full-data production runs",
                 fields={"count": len(production_records)})

    if args.model:
        chosen = next((r for r in pool if r["model_name"] == args.model), None)
        if chosen is None:
            raise SystemExit(f"Model {args.model!r} has no experiment record in the pool.")
        ranking = sorted(pool, key=CRITERIA[args.criterion]["key"], reverse=True)
    else:
        chosen, ranking = select(pool, args.criterion)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **summarise(chosen),
        "selection": {
            "criterion": args.criterion,
            "criterion_description": CRITERIA[args.criterion]["description"],
            "forced": bool(args.model),
            "pool": "production" if production_records and not args.protocol else "all",
            "candidates_considered": len(pool),
            "quality_gate": QUALITY_GATE if args.criterion in {"fastest", "smallest"} else None,
            "note": (
                "The production model is chosen on measured test performance. The "
                "proposed CBAM architecture remains the research contribution and is "
                "reported in the benchmark and ablation tables regardless of whether "
                "it wins this selection."
            ),
        },
        "candidates": [summarise(r) for r in ranking],
        "best_by": {
            name: summarise(select(pool, name)[0])["model_name"] for name in CRITERIA
        },
    }

    if args.torchscript:
        exported = export_torchscript(
            chosen, settings.exported_dir / f"{chosen['model_name']}.torchscript.pt"
        )
        if exported:
            manifest["torchscript"] = exported

    if args.onnx:
        exported = export_onnx(
            chosen, settings.exported_dir / f"{chosen['model_name']}.onnx"
        )
        if exported:
            manifest["onnx"] = exported

    out = settings.exported_dir / "production.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    metrics = manifest["test_metrics"]
    print("\n" + "=" * 78)
    print("PRODUCTION MODEL SELECTED")
    print("=" * 78)
    print(f"Model        : {manifest['label']}  ({manifest['model_name']})")
    print(f"Criterion    : {args.criterion} - {CRITERIA[args.criterion]['description']}")
    print(f"Protocol     : {manifest['protocol']}   Proposed architecture: {manifest['proposed']}")
    print(f"Checkpoint   : {manifest['checkpoint']}")
    print(f"Test split   : {metrics['test_split']} ({metrics['test_samples']} images)")
    print(f"Accuracy     : {metrics['accuracy']:.4f}")
    print(f"Macro F1     : {metrics['f1_macro']:.4f}")
    print(f"Top-3        : {metrics['top3_accuracy']:.4f}")
    print(f"ECE          : {metrics['ece']:.4f}"
          + (f"  ->  {metrics['ece_after_temperature']:.4f} after T={manifest['temperature']:.3f}"
             if metrics.get("ece_after_temperature") is not None else ""))
    print(f"Size / speed : {manifest['model_size_mb']:.1f} MB, "
          f"{manifest['inference_ms']:.2f} ms/image")
    print("-" * 78)
    print("Best model per criterion:")
    for name, model_name in manifest["best_by"].items():
        print(f"  {name:<10} -> {model_name}")
    print("=" * 78)
    print(f"Wrote {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
