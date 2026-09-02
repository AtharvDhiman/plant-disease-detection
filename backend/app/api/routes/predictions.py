"""Prediction, image-analysis and history endpoints."""
from __future__ import annotations

import time
import uuid
from enum import Enum

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.logging import get_logger
from app.ml.quality import analyse_image, quality_band
from app.models.prediction import Prediction
from app.schemas.prediction import (
    AnalyseImageOut,
    DeleteOut,
    PredictionDetailOut,
    PredictionListOut,
    PredictionOut,
)
from app.services.knowledge import knowledge_base
from app.services.predictor import PredictionResult, predictor
from app.utils.images import (
    UploadError,
    decode_image,
    resolve_upload_path,
    store_image,
    validate_upload,
)

log = get_logger(__name__)
router = APIRouter()


class ExplainMethod(str, Enum):
    """Explanation methods a caller may request."""

    grad_cam = "grad_cam"
    grad_cam_plus_plus = "grad_cam_plus_plus"
    cbam_spatial = "cbam_spatial"
    integrated_gradients = "integrated_gradients"


# Integrated gradients is excluded by default: it needs ~24 backward passes and
# would roughly triple the response time for a pixel-level map that is noisier
# than Grad-CAM. Callers who want it can ask.
DEFAULT_METHODS = ("grad_cam", "grad_cam_plus_plus", "cbam_spatial")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


async def _read_upload(file: UploadFile):
    """Validate and decode an uploaded image, or raise a 4xx with a useful detail."""
    data = await file.read()
    try:
        validate_upload(file.filename, file.content_type, len(data))
        return decode_image(data), len(data)
    except UploadError as exc:
        code = (
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            if exc.code == "file_too_large"
            else status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
            if exc.code in {"unsupported_content_type", "unsupported_extension",
                            "unsupported_image_format"}
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=code, detail=exc.as_detail()) from exc


def _image_url(filename: str) -> str:
    return f"/api/images/{filename}"


def _serialise(result: PredictionResult, row: Prediction | None) -> dict:
    """Turn a :class:`PredictionResult` into the API response shape."""
    return {
        "id": row.id if row else None,
        "created_at": row.created_at if row else None,
        "predicted_class": result.predicted_class,
        "display_name": result.display_name,
        "plant": result.plant,
        "condition": result.condition,
        "is_healthy": result.is_healthy,
        "confidence": result.confidence,
        "confidence_level": result.confidence_level,
        "confidence_percent": round(result.confidence * 100, 2),
        "status": result.status,
        "status_message": result.status_message,
        "reliable": result.status == "ok" and result.confidence_level != "low",
        "top_predictions": [
            {
                "class_name": s.class_name, "display_name": s.display_name,
                "plant": s.plant, "condition": s.condition,
                "probability": s.probability, "is_healthy": s.is_healthy,
            }
            for s in result.top_predictions
        ],
        "quality": {
            **result.quality.to_dict(),
            "band": quality_band(result.quality.score),
        },
        "ood": result.ood.to_dict(),
        "explanation": {
            "available": [k for k in result.explanations if k != "original"],
            "images": {k: _image_url(v) for k, v in result.explanations.items()},
            "stats": result.attention_stats,
            "errors": result.explanation_errors,
            "channel_attention": result.channel_attention,
        },
        "disease_info": result.disease_info,
        "model_info": result.model,
        "timings": result.timings,
        "image": result.image_meta,
    }


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.post(
    "/predict",
    response_model=PredictionOut,
    summary="Classify a leaf image",
    description=(
        "Runs the full pipeline: image-quality analysis, preprocessing, inference, "
        "confidence calibration, out-of-distribution assessment and explainability. "
        "The result is stored in the prediction history unless `save=false`."
    ),
)
async def predict(
    file: UploadFile = File(..., description="JPG, PNG or WEBP leaf photograph."),
    top_k: int = Query(5, ge=1, le=10),
    explain: bool = Query(True, description="Generate Grad-CAM / CBAM heat maps."),
    methods: list[ExplainMethod] | None = Query(
        None,
        description=(
            "Which explanation methods to run. Defaults to grad_cam, "
            "grad_cam_plus_plus and cbam_spatial. `integrated_gradients` is "
            "available but costs ~24 extra backward passes, so it is opt-in."
        ),
    ),
    save: bool = Query(True, description="Store the result in the prediction history."),
    db: Session = Depends(get_db),
):
    started = time.perf_counter()
    image, byte_size = await _read_upload(file)
    stem = uuid.uuid4().hex

    try:
        result = predictor.predict(
            image, top_k=top_k, explain=explain,
            explain_methods=tuple(m.value for m in methods) if methods else DEFAULT_METHODS,
            save_dir=settings.upload_dir, file_stem=stem,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "model_unavailable", "message": str(exc),
                    "hint": "Train and export a model before using this endpoint."},
        ) from exc

    filename, stored_bytes = store_image(image, settings.upload_dir, stem)

    row: Prediction | None = None
    if save:
        row = Prediction(
            image_path=str(settings.upload_dir / filename),
            image_filename=filename,
            image_width=image.width,
            image_height=image.height,
            image_bytes=stored_bytes,
            predicted_class=result.predicted_class,
            predicted_plant=result.plant,
            predicted_condition=result.condition,
            is_healthy=result.is_healthy,
            confidence=result.confidence,
            confidence_level=result.confidence_level,
            top_predictions=[
                {"class_name": s.class_name, "display_name": s.display_name,
                 "probability": round(s.probability, 6)}
                for s in result.top_predictions
            ],
            status=result.status,
            status_message=result.status_message,
            image_quality_score=result.quality.score,
            image_quality_band=quality_band(result.quality.score),
            quality_issues=[
                {"code": i.code, "severity": i.severity, "message": i.message}
                for i in result.quality.issues
            ],
            ood_flag=result.ood.is_ood,
            ood_scores=result.ood.scores,
            model_name=result.model["name"],
            model_label=result.model["label"],
            model_version=result.model["version"],
            preprocess_ms=result.timings["preprocess_ms"],
            inference_ms=result.timings["inference_ms"],
            explain_ms=result.timings["explain_ms"],
            total_ms=result.timings["total_ms"],
            explanation_files=result.explanations,
            attention_stats=result.attention_stats,
        )
        db.add(row)
        db.commit()
        db.refresh(row)

    log.info(
        "prediction",
        fields={
            "id": row.id if row else None,
            "class": result.predicted_class,
            "confidence": round(result.confidence, 4),
            "status": result.status,
            "model": result.model["name"],
            "upload_bytes": byte_size,
            "inference_ms": result.timings["inference_ms"],
            "request_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )
    return _serialise(result, row)


@router.post(
    "/analyze-image",
    response_model=AnalyseImageOut,
    summary="Check image quality without classifying",
    description=(
        "Runs only the image-quality checks. The frontend calls this immediately "
        "after a file is chosen so the user can fix a bad photo before waiting for "
        "a full analysis."
    ),
)
async def analyze_image(file: UploadFile = File(...)):
    image, byte_size = await _read_upload(file)
    report = analyse_image(image)
    band = quality_band(report.score)

    if band == "good":
        recommendation = "This image is suitable for analysis."
    elif band == "acceptable":
        recommendation = (
            "This image can be analysed, but the result may be less reliable. "
            + " ".join(i.suggestion for i in report.issues) or ""
        ).strip()
    else:
        recommendation = (
            "This image is unlikely to give a reliable result. "
            + " ".join(i.suggestion for i in report.issues)
        ).strip()

    return {
        "quality": {**report.to_dict(), "band": band},
        "image": {"width": image.width, "height": image.height,
                  "mode": image.mode, "bytes": byte_size},
        "recommendation": recommendation,
    }


@router.get("/predictions", response_model=PredictionListOut, summary="List prediction history")
def list_predictions(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    search: str | None = Query(None, description="Substring match on class, plant or condition."),
    disease: str | None = Query(None, description="Exact class name filter."),
    plant: str | None = Query(None),
    status_filter: str | None = Query(None, alias="status"),
    confidence_level: str | None = Query(None),
    sort: str = Query("created_desc",
                      pattern="^(created_desc|created_asc|confidence_desc|confidence_asc)$"),
    db: Session = Depends(get_db),
):
    query = select(Prediction)
    count_query = select(func.count()).select_from(Prediction)

    filters = []
    if search:
        pattern = f"%{search.lower()}%"
        filters.append(
            func.lower(Prediction.predicted_class).like(pattern)
            | func.lower(Prediction.predicted_plant).like(pattern)
            | func.lower(Prediction.predicted_condition).like(pattern)
        )
    if disease:
        filters.append(Prediction.predicted_class == disease)
    if plant:
        filters.append(Prediction.predicted_plant == plant)
    if status_filter:
        filters.append(Prediction.status == status_filter)
    if confidence_level:
        filters.append(Prediction.confidence_level == confidence_level)
    for condition in filters:
        query = query.where(condition)
        count_query = count_query.where(condition)

    ordering = {
        "created_desc": Prediction.created_at.desc(),
        "created_asc": Prediction.created_at.asc(),
        "confidence_desc": Prediction.confidence.desc(),
        "confidence_asc": Prediction.confidence.asc(),
    }[sort]

    total = db.execute(count_query).scalar_one()
    rows = db.execute(query.order_by(ordering).limit(limit).offset(offset)).scalars().all()
    return {"items": rows, "total": total, "limit": limit, "offset": offset}


@router.get("/predictions/{prediction_id}", response_model=PredictionDetailOut,
            summary="Fetch one stored prediction")
def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    row = db.get(Prediction, prediction_id)
    if row is None:
        raise HTTPException(status_code=404, detail={"code": "not_found",
                                                     "message": f"Prediction {prediction_id} not found."})
    payload = {c.name: getattr(row, c.name) for c in row.__table__.columns}
    payload["disease_info"] = knowledge_base.get(row.predicted_class)
    if row.explanation_files:
        payload["explanation_files"] = {k: _image_url(v) for k, v in row.explanation_files.items()}
    return payload


@router.delete("/predictions/{prediction_id}", response_model=DeleteOut,
               summary="Delete one stored prediction")
def delete_prediction(prediction_id: int, db: Session = Depends(get_db)):
    row = db.get(Prediction, prediction_id)
    if row is None:
        raise HTTPException(status_code=404, detail={"code": "not_found",
                                                     "message": f"Prediction {prediction_id} not found."})
    _remove_files(row)
    db.delete(row)
    db.commit()
    return {"deleted": 1, "message": f"Prediction {prediction_id} deleted."}


@router.delete("/predictions", response_model=DeleteOut, summary="Clear the prediction history")
def clear_predictions(
    confirm: bool = Query(False, description="Must be true; guards against accidental clears."),
    db: Session = Depends(get_db),
):
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail={"code": "confirmation_required",
                    "message": "Pass ?confirm=true to clear the entire history."},
        )
    rows = db.execute(select(Prediction)).scalars().all()
    for row in rows:
        _remove_files(row)
    count = db.execute(delete(Prediction)).rowcount
    db.commit()
    log.info("history cleared", fields={"deleted": count})
    return {"deleted": count, "message": f"Deleted {count} predictions."}


@router.get("/images/{filename}", summary="Serve a stored upload or heat map",
            response_class=FileResponse)
def get_image(filename: str):
    try:
        path = resolve_upload_path(filename)
    except UploadError as exc:
        raise HTTPException(status_code=404, detail=exc.as_detail()) from exc
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(
        path,
        media_type=media_type,
        # Filenames are random UUIDs and content is immutable once written.
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _remove_files(row: Prediction) -> None:
    """Delete the stored image and its heat maps; missing files are not an error."""
    names = [row.image_filename] + list((row.explanation_files or {}).values())
    for name in names:
        try:
            path = resolve_upload_path(name)
        except UploadError:
            continue
        try:
            path.unlink()
        except OSError as exc:  # noqa: PERF203
            log.warning("Could not delete file", fields={"file": name, "error": str(exc)})
