"""Model, metrics, dataset and disease-library endpoints.

Everything here is read-only and served from the training artefacts, so the API
can never report a number that no experiment produced.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.core.config import settings
from app.ml.serving import registry
from app.services import research
from app.services.knowledge import knowledge_base

router = APIRouter()


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


@router.get("/models", summary="List available and loaded models")
def list_models():
    manifest = research.production_manifest()
    if manifest is None:
        return {
            "production": None,
            "available": [],
            "loaded": registry.loaded_names(),
            "message": (
                "No model has been exported yet. Run the training pipeline and then "
                "`python training/export_model.py`."
            ),
        }
    return {
        "production": {
            "model_name": manifest["model_name"],
            "label": manifest["label"],
            "version": manifest.get("version"),
            "proposed": manifest.get("proposed"),
            "protocol": manifest.get("protocol"),
            "image_size": manifest.get("image_size"),
            "parameters": manifest.get("parameters"),
            "model_size_mb": manifest.get("model_size_mb"),
            "inference_ms": manifest.get("inference_ms"),
            "temperature": manifest.get("temperature"),
            "test_metrics": manifest.get("test_metrics"),
        },
        "selection": manifest.get("selection"),
        "best_by": manifest.get("best_by"),
        "available": manifest.get("candidates", []),
        "loaded": registry.loaded_names(),
        "torchscript": manifest.get("torchscript"),
    }


@router.get("/health/model", summary="Model readiness probe")
def model_health():
    """Report whether the serving model is loaded and answering."""
    try:
        loaded = registry.load()
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "model_unavailable", "message": str(exc)},
        ) from exc
    return {"status": "ready", **loaded.describe()}


# --------------------------------------------------------------------------- #
# Metrics / benchmark
# --------------------------------------------------------------------------- #


@router.get("/metrics", summary="Full benchmark, ablation and calibration results")
def metrics(
    group: str | None = Query(None, description="benchmark | classical | ablation | placement"),
):
    rows = research.benchmark_rows(
        groups=(group,) if group else ("benchmark", "classical", "ablation", "placement")
    )
    manifest = research.production_manifest()
    return {
        "rows": rows,
        "count": len(rows),
        "production_model": manifest.get("model_name") if manifest else None,
        "best_by": manifest.get("best_by") if manifest else None,
        "protocol_note": (
            "Every deep model in the benchmark and the ablation was trained on an "
            "identical fixed-budget subset with identical hyper-parameters; only the "
            "architecture differs. Production models are additionally trained on the "
            "full training split - see the 'protocol' column."
        ),
    }


@router.get("/metrics/ablation", summary="Attention ablation study results")
def ablation():
    rows = research.ablation_rows()
    summary = research.ablation_summary() or {}
    pairs = summary.get("transfer_pairs") or {}
    # The matched CBAM/no-CBAM backbone pairs come from the benchmark, so they
    # are worth returning even before the dedicated ablation suite has run.
    if not rows and not pairs.get("available"):
        raise HTTPException(
            status_code=404,
            detail={"code": "not_available",
                    "message": "Ablation results not found. Run "
                               "`python training/run_experiments.py --suite ablation`."},
        )
    return {"rows": rows, "summary": summary,
            "transfer_pairs": pairs,
            "placement": research.placement_rows(),
            "research_question": (
                "Does integrating channel and spatial attention through CBAM improve "
                "plant disease classification accuracy, robustness and interpretability "
                "compared with conventional CNN architectures?"
            )}


@router.get("/metrics/per-class", summary="Per-class precision, recall and F1")
def per_class(experiment_id: str | None = Query(None)):
    payload = research.per_class_performance(experiment_id)
    if payload is None:
        raise HTTPException(status_code=404,
                            detail={"code": "not_found", "message": "No matching experiment."})
    return payload


@router.get("/metrics/curves/{experiment_id}", summary="Training/validation curves")
def curves(experiment_id: str):
    payload = research.training_curves(experiment_id)
    if payload is None:
        raise HTTPException(status_code=404,
                            detail={"code": "not_found",
                                    "message": f"No experiment {experiment_id!r}."})
    return payload


@router.get("/metrics/calibration", summary="Confidence calibration and OOD thresholds")
def calibration():
    records = research.experiments()
    manifest = research.production_manifest()
    target = manifest.get("experiment_id") if manifest else None
    record = next((r for r in records if r.get("experiment_id") == target), None)
    calibration_block = (record or {}).get("test", {}).get("calibration")
    return {
        "experiment_id": target,
        "calibration": calibration_block,
        "ood": research.ood_calibration(),
        "confidence_thresholds": {
            "high": settings.confidence_high,
            "medium": settings.confidence_medium,
        },
    }


@router.get("/metrics/performance", summary="Measured serving performance")
def performance():
    """Stage-by-stage latency of the real serving path.

    Produced by ``scripts/benchmark_serving.py`` against the exported model, so
    it reflects what a user waits for rather than a synthetic forward-pass
    benchmark. Also reports the per-architecture forward-pass latency from the
    benchmark suite for comparison.
    """
    measured = research.serving_benchmark()
    rows = research.benchmark_rows()
    return {
        "serving": measured,
        "model_latency": [
            {
                "label": row["label"],
                "model_name": row["model_name"],
                "inference_ms": row["inference_ms"],
                "model_size_mb": row["model_size_mb"],
                "parameters": row["parameters"],
                "accuracy": row["accuracy"],
                "proposed": row["proposed"],
            }
            for row in rows if row["inference_ms"]
        ],
        "note": (
            "Model loading happens once at startup, so no request pays it. "
            "Explainability roughly doubles the response time because Grad-CAM "
            "needs a backward pass; it can be disabled per request with "
            "?explain=false."
        ),
    }


@router.get("/experiments", summary="Raw experiment ledger")
def experiments(limit: int = Query(100, ge=1, le=500)):
    records = research.experiments()
    return {
        "count": len(records),
        "experiments": [
            {
                "experiment_id": r.get("experiment_id"),
                "model_name": r.get("model_name"),
                "label": r.get("label"),
                "group": r.get("group"),
                "protocol": r.get("protocol"),
                "proposed": r.get("proposed"),
                "dataset_version": r.get("dataset_version"),
                "config": r.get("config"),
                "hardware": r.get("hardware"),
                "best_val_accuracy": r.get("best_val_accuracy"),
                "best_epoch": r.get("best_epoch"),
                "epochs_run": r.get("epochs_run"),
                "training_time_s": r.get("training_time_s"),
                "total_parameters": r.get("total_parameters"),
                "model_size_mb": r.get("model_size_mb"),
                "checkpoint_path": r.get("checkpoint_path"),
                "logged_at": r.get("logged_at"),
                "test_accuracy": (r.get("test") or {}).get("metrics", {}).get("accuracy"),
                "test_f1_macro": (r.get("test") or {}).get("metrics", {}).get("f1_macro"),
            }
            for r in records[:limit]
        ],
    }


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


@router.get("/dataset-stats", summary="Dataset statistics and split protocol")
def dataset_stats():
    payload = research.dataset_summary()
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "not_available",
                    "message": "Dataset report not found. Run "
                               "`python training/dataset_inspect.py`."},
        )
    return payload


# --------------------------------------------------------------------------- #
# Disease library
# --------------------------------------------------------------------------- #


@router.get("/diseases", summary="List every class in the knowledge base")
def list_diseases(
    plant: str | None = Query(None),
    category: str | None = Query(None),
    search: str | None = Query(None),
):
    rows = knowledge_base.summaries()
    if plant:
        rows = [r for r in rows if r["plant"].lower() == plant.lower()]
    if category:
        rows = [r for r in rows if r["category"].lower() == category.lower()]
    if search:
        needle = search.lower()
        rows = [r for r in rows
                if needle in r["display_name"].lower()
                or needle in r["common_name"].lower()
                or needle in r["description"].lower()]
    return {
        "items": rows,
        "count": len(rows),
        "plants": knowledge_base.plants(),
        "categories": knowledge_base.categories(),
        "metadata": knowledge_base.metadata,
    }


@router.get("/diseases/{class_name:path}", summary="Fetch one disease entry")
def get_disease(class_name: str):
    entry = knowledge_base.get(class_name)
    if not entry.get("symptoms") and entry.get("category") == "unknown":
        raise HTTPException(
            status_code=404,
            detail={"code": "not_found", "message": f"No knowledge-base entry for {class_name!r}."},
        )
    return entry


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


@router.get("/figures", summary="List rendered training/evaluation figures")
def list_figures():
    return research.available_plots()


@router.get("/figures/{group}/{filename}", summary="Serve one figure",
            response_class=FileResponse)
def get_figure(group: str, filename: str):
    path = research.plot_path(group, filename)
    if path is None:
        raise HTTPException(status_code=404,
                            detail={"code": "not_found", "message": "Figure not found."})
    return FileResponse(path, media_type="image/png",
                        headers={"Cache-Control": "public, max-age=3600"})
