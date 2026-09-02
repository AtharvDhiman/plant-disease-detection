"""ORM model for stored predictions."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Prediction(Base):
    """One analysed image and everything the API returned about it.

    Only derived data is stored - never the uploader's identity. The saved image
    is a re-encoded 512 px JPEG under ``uploads/``, referenced by a generated
    filename, so the original file (and any EXIF metadata it carried) never
    reaches the database.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    # ------------------------------------------------------------- image
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    image_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    image_width: Mapped[int | None] = mapped_column(Integer)
    image_height: Mapped[int | None] = mapped_column(Integer)
    image_bytes: Mapped[int | None] = mapped_column(Integer)

    # -------------------------------------------------------- prediction
    predicted_class: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    predicted_plant: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    predicted_condition: Mapped[str] = mapped_column(String(160), nullable=False)
    is_healthy: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    confidence_level: Mapped[str] = mapped_column(String(16), nullable=False)
    # [{"class_name": ..., "display_name": ..., "probability": ...}, ...]
    top_predictions: Mapped[list] = mapped_column(JSON, nullable=False)

    # ---------------------------------------------------------- decision
    # "ok" | "low_confidence" | "out_of_distribution" | "poor_quality"
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    status_message: Mapped[str | None] = mapped_column(Text)

    # ----------------------------------------------------------- quality
    image_quality_score: Mapped[float | None] = mapped_column(Float)
    image_quality_band: Mapped[str | None] = mapped_column(String(16))
    quality_issues: Mapped[list | None] = mapped_column(JSON)

    # --------------------------------------------------------------- OOD
    ood_flag: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    ood_scores: Mapped[dict | None] = mapped_column(JSON)

    # ------------------------------------------------------------- model
    model_name: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    model_label: Mapped[str | None] = mapped_column(String(160))
    model_version: Mapped[str | None] = mapped_column(String(80))

    # -------------------------------------------------------- performance
    preprocess_ms: Mapped[float | None] = mapped_column(Float)
    inference_ms: Mapped[float | None] = mapped_column(Float)
    explain_ms: Mapped[float | None] = mapped_column(Float)
    total_ms: Mapped[float | None] = mapped_column(Float)

    # ---------------------------------------------------- explainability
    # {"grad_cam": "<file>.png", ...} - relative to the uploads directory
    explanation_files: Mapped[dict | None] = mapped_column(JSON)
    attention_stats: Mapped[dict | None] = mapped_column(JSON)

    __table_args__ = (
        Index("ix_predictions_created_class", "created_at", "predicted_class"),
        Index("ix_predictions_status_created", "status", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<Prediction id={self.id} class={self.predicted_class!r} "
            f"conf={self.confidence:.3f} status={self.status!r}>"
        )
