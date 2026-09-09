"""Pydantic response/request models for the prediction endpoints."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ClassScoreOut(BaseModel):
    """One entry of the top-k prediction list."""

    class_name: str = Field(..., description="Raw dataset class label.")
    display_name: str = Field(..., description="Human-readable 'Plant - Condition'.")
    plant: str
    condition: str
    probability: float = Field(..., ge=0.0, le=1.0)
    is_healthy: bool


class QualityIssueOut(BaseModel):
    code: str
    severity: str = Field(..., description="'error' blocks prediction, 'warning' does not.")
    message: str
    suggestion: str


class QualityOut(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    band: str = Field(..., description="good | acceptable | poor")
    passed: bool
    issues: list[QualityIssueOut] = []
    metrics: dict = {}


class OODOut(BaseModel):
    is_out_of_distribution: bool
    reasons: list[str] = []
    scores: dict = {}
    thresholds: dict = {}


class AttentionStatOut(BaseModel):
    focus_ratio: float
    peak_value: float
    mean_value: float
    centroid_x: float | None = None
    centroid_y: float | None = None


class ExplanationOut(BaseModel):
    """URLs of the rendered heat-map overlays, keyed by method."""

    available: list[str] = []
    images: dict[str, str] = Field(
        default_factory=dict,
        description="method -> URL under /api/images/{filename}",
    )
    stats: dict[str, AttentionStatOut] = {}
    errors: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Methods that were requested but could not be produced, with the "
            "reason. A prediction is never failed because a heat map failed."
        ),
    )
    channel_attention: list[float] | None = Field(
        None, description="Per-channel CBAM gate values from the deepest block."
    )
    caveat: str = (
        "Heat maps show where the network pooled evidence for its decision at the "
        "resolution of its final feature map. They are not a disease segmentation "
        "and do not measure lesion extent."
    )


class DiseaseInfoOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    class_name: str
    common_name: str
    display_name: str
    plant: str
    condition: str
    category: str
    severity: str
    is_healthy: bool
    description: str
    symptoms: list[str] = []
    causes: list[str] = []
    favourable_conditions: list[str] = []
    prevention: list[str] = []
    management: list[str] = []
    pathogen: str | None = None
    pathogen_type: str | None = None
    disclaimer: str = ""
    sources: list[str] = []


class TimingsOut(BaseModel):
    quality_ms: float
    preprocess_ms: float
    inference_ms: float
    explain_ms: float
    total_ms: float


class ModelInfoOut(BaseModel):
    name: str
    label: str
    version: str
    image_size: int
    device: str
    temperature: float
    parameters: int


class PredictionOut(BaseModel):
    """Full response of ``POST /api/predict``."""

    id: int | None = Field(None, description="History row id; null when not saved.")
    created_at: datetime | None = None

    predicted_class: str
    display_name: str
    plant: str
    condition: str
    is_healthy: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_level: str = Field(..., description="high | medium | low")
    confidence_percent: float

    status: str = Field(..., description="ok | low_confidence | out_of_distribution | poor_quality | not_a_plant")
    status_message: str | None = None
    reliable: bool = Field(..., description="True when the result is safe to act on.")

    top_predictions: list[ClassScoreOut]
    quality: QualityOut
    ood: OODOut
    explanation: ExplanationOut
    disease_info: DiseaseInfoOut | None = None
    model_info: ModelInfoOut
    timings: TimingsOut
    image: dict = {}


class AnalyseImageOut(BaseModel):
    """Response of ``POST /api/analyze-image`` - quality only, no inference."""

    quality: QualityOut
    image: dict
    recommendation: str


class PredictionListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    predicted_class: str
    predicted_plant: str
    predicted_condition: str
    is_healthy: bool
    confidence: float
    confidence_level: str
    status: str
    model_name: str
    image_filename: str
    image_quality_score: float | None = None
    total_ms: float | None = None


class PredictionListOut(BaseModel):
    items: list[PredictionListItem]
    total: int
    limit: int
    offset: int


class PredictionDetailOut(PredictionListItem):
    top_predictions: list[dict]
    status_message: str | None = None
    quality_issues: list[dict] | None = None
    image_quality_band: str | None = None
    ood_flag: bool
    ood_scores: dict | None = None
    model_label: str | None = None
    model_version: str | None = None
    preprocess_ms: float | None = None
    inference_ms: float | None = None
    explain_ms: float | None = None
    explanation_files: dict | None = None
    attention_stats: dict | None = None
    image_width: int | None = None
    image_height: int | None = None
    disease_info: DiseaseInfoOut | None = None


class DeleteOut(BaseModel):
    deleted: int
    message: str
