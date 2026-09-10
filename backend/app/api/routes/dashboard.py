"""Dashboard aggregation endpoints - statistics derived from stored predictions."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.prediction import Prediction
from app.services import research
from app.services.knowledge import knowledge_base

router = APIRouter()

CONFIDENCE_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.0)]


@router.get("/dashboard", summary="Aggregated statistics for the main dashboard")
def dashboard(
    days: int = Query(30, ge=1, le=365, description="Window for the activity timeline."),
    recent_limit: int = Query(8, ge=1, le=50),
    db: Session = Depends(get_db),
):
    total = db.execute(select(func.count()).select_from(Prediction)).scalar_one()

    if total == 0:
        return _empty_dashboard(days)

    avg_confidence = db.execute(select(func.avg(Prediction.confidence))).scalar_one()
    avg_latency = db.execute(select(func.avg(Prediction.total_ms))).scalar_one()
    avg_inference = db.execute(select(func.avg(Prediction.inference_ms))).scalar_one()
    avg_quality = db.execute(select(func.avg(Prediction.image_quality_score))).scalar_one()
    rejected_count = db.execute(
        select(func.count()).select_from(Prediction)
        .where(
            Prediction.status.in_(["not_a_plant", "poor_quality"])
            | (Prediction.predicted_class == "Unknown")
        )
    ).scalar_one()

    healthy_count = db.execute(
        select(func.count()).select_from(Prediction)
        .where(
            Prediction.is_healthy.is_(True),
            Prediction.status.notin_(["not_a_plant", "poor_quality"]),
            Prediction.predicted_class != "Unknown",
        )
    ).scalar_one()

    diseased_count = db.execute(
        select(func.count()).select_from(Prediction)
        .where(
            Prediction.is_healthy.is_(False),
            Prediction.status.notin_(["not_a_plant", "poor_quality"]),
            Prediction.predicted_class != "Unknown",
        )
    ).scalar_one()

    by_class = db.execute(
        select(Prediction.predicted_class, func.count().label("n"),
               func.avg(Prediction.confidence))
        .where(
            Prediction.status.notin_(["not_a_plant", "poor_quality"]),
            Prediction.predicted_class != "Unknown",
        )
        .group_by(Prediction.predicted_class)
        .order_by(func.count().desc())
    ).all()

    by_plant = db.execute(
        select(Prediction.predicted_plant, func.count().label("n"))
        .where(
            Prediction.status.notin_(["not_a_plant", "poor_quality"]),
            Prediction.predicted_plant != "Unknown",
        )
        .group_by(Prediction.predicted_plant)
        .order_by(func.count().desc())
    ).all()

    by_status = db.execute(
        select(Prediction.status, func.count()).group_by(Prediction.status)
    ).all()

    by_level = db.execute(
        select(Prediction.confidence_level, func.count()).group_by(Prediction.confidence_level)
    ).all()

    confidence_histogram = []
    for low, high in CONFIDENCE_BINS:
        # The top bin is closed on the right so a perfect 1.0 is counted.
        upper_ok = Prediction.confidence <= high if high == 1.0 else Prediction.confidence < high
        count = db.execute(
            select(func.count()).select_from(Prediction)
            .where(Prediction.confidence >= low, upper_ok)
        ).scalar_one()
        confidence_histogram.append({
            "range": f"{int(low * 100)}-{int(high * 100)}%",
            "low": low, "high": high, "count": count,
        })

    since = datetime.now(timezone.utc) - timedelta(days=days)
    timeline_rows = db.execute(
        select(func.date(Prediction.created_at), func.count(), func.avg(Prediction.confidence))
        .where(Prediction.created_at >= since)
        .group_by(func.date(Prediction.created_at))
        .order_by(func.date(Prediction.created_at))
    ).all()

    recent = db.execute(
        select(Prediction).order_by(Prediction.created_at.desc()).limit(recent_limit)
    ).scalars().all()

    manifest = research.production_manifest()
    benchmark = research.benchmark_rows()
    best_model = max(benchmark, key=lambda r: r["accuracy"] or 0) if benchmark else None

    return {
        "totals": {
            "predictions": total,
            "distinct_classes": len(by_class),
            "distinct_plants": len(by_plant),
            "healthy": healthy_count,
            "diseased": diseased_count,
            "rejected": rejected_count,
            "average_confidence": round(float(avg_confidence or 0), 4),
            "average_quality_score": round(float(avg_quality or 0), 4),
            "average_total_ms": round(float(avg_latency or 0), 2),
            "average_inference_ms": round(float(avg_inference or 0), 2),
        },
        "most_detected": (
            {
                "class_name": by_class[0][0],
                "display_name": knowledge_base.display_name(by_class[0][0]),
                "count": by_class[0][1],
                "average_confidence": round(float(by_class[0][2] or 0), 4),
                "severity": knowledge_base.severity(by_class[0][0]),
            }
            if by_class else None
        ),
        "disease_distribution": [
            {
                "class_name": name,
                "display_name": knowledge_base.display_name(name),
                "plant": knowledge_base.plant_of(name),
                "count": count,
                "average_confidence": round(float(mean or 0), 4),
                "is_healthy": knowledge_base.is_healthy(name),
                "severity": knowledge_base.severity(name),
            }
            for name, count, mean in by_class
        ],
        "plant_distribution": [{"plant": name, "count": count} for name, count in by_plant],
        "status_distribution": [{"status": name, "count": count} for name, count in by_status],
        "confidence_levels": [{"level": name, "count": count} for name, count in by_level],
        "confidence_histogram": confidence_histogram,
        "timeline": [
            {"date": str(day), "count": count, "average_confidence": round(float(mean or 0), 4)}
            for day, count, mean in timeline_rows
        ],
        "recent": [
            {
                "id": row.id,
                "created_at": row.created_at,
                "predicted_class": row.predicted_class,
                "display_name": knowledge_base.display_name(row.predicted_class),
                "plant": row.predicted_plant,
                "confidence": row.confidence,
                "confidence_level": row.confidence_level,
                "status": row.status,
                "is_healthy": row.is_healthy,
                "image_url": f"/api/images/{row.image_filename}",
                "total_ms": row.total_ms,
            }
            for row in recent
        ],
        "model": {
            "production": manifest.get("model_name") if manifest else None,
            "label": manifest.get("label") if manifest else None,
            "accuracy": (manifest or {}).get("test_metrics", {}).get("accuracy"),
            "f1_macro": (manifest or {}).get("test_metrics", {}).get("f1_macro"),
            "inference_ms": manifest.get("inference_ms") if manifest else None,
            "best_benchmark_model": best_model["label"] if best_model else None,
            "best_benchmark_accuracy": best_model["accuracy"] if best_model else None,
        },
        "window_days": days,
    }


def _empty_dashboard(days: int) -> dict:
    """Shape-compatible response for a fresh install with no history yet."""
    manifest = research.production_manifest()
    benchmark = research.benchmark_rows()
    best_model = max(benchmark, key=lambda r: r["accuracy"] or 0) if benchmark else None
    return {
        "totals": {
            "predictions": 0, "distinct_classes": 0, "distinct_plants": 0,
            "healthy": 0, "diseased": 0, "rejected": 0, "average_confidence": 0.0,
            "average_quality_score": 0.0, "average_total_ms": 0.0,
            "average_inference_ms": 0.0,
        },
        "most_detected": None,
        "disease_distribution": [],
        "plant_distribution": [],
        "status_distribution": [],
        "confidence_levels": [],
        "confidence_histogram": [
            {"range": f"{int(low * 100)}-{int(high * 100)}%", "low": low, "high": high, "count": 0}
            for low, high in CONFIDENCE_BINS
        ],
        "timeline": [],
        "recent": [],
        "model": {
            "production": manifest.get("model_name") if manifest else None,
            "label": manifest.get("label") if manifest else None,
            "accuracy": (manifest or {}).get("test_metrics", {}).get("accuracy"),
            "f1_macro": (manifest or {}).get("test_metrics", {}).get("f1_macro"),
            "inference_ms": manifest.get("inference_ms") if manifest else None,
            "best_benchmark_model": best_model["label"] if best_model else None,
            "best_benchmark_accuracy": best_model["accuracy"] if best_model else None,
        },
        "window_days": days,
        "empty": True,
    }
