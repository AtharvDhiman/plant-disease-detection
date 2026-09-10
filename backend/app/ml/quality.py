"""Pre-flight image quality checks.

The model will happily assign 99% confidence to a photograph of a keyboard, so
the application inspects the upload *before* classifying it and tells the user
when the image is not fit for diagnosis. Every check here is a cheap,
deterministic image statistic - no second neural network - which keeps the
latency budget for the actual prediction.

Checks
------
================  ==========================================================
Check             Signal
================  ==========================================================
Resolution        Images below ~64 px carry too little detail to upscale
Blur              Variance of the Laplacian - low variance means few edges
Exposure          Mean luminance too low (under-exposed) or too high (blown)
Contrast          Standard deviation of luminance
Colour            Fraction of pixels in a plant-like hue range
Saturation        Near-grey images are usually documents or screenshots
Spatial coherence Lag-1 pixel autocorrelation - separates photographs from noise
================  ==========================================================

The plant foliage check ensures the uploaded image contains genuine plant or
leaf tissue: images without vegetation colors are rejected as `not_a_plant`
to prevent classifying non-plant objects (people, cars, animals, electronics).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings


@dataclass
class QualityIssue:
    """One problem found in an uploaded image."""

    code: str
    severity: str      # "error" | "warning"
    message: str
    suggestion: str


@dataclass
class QualityReport:
    """Verdict plus the raw measurements behind it."""

    score: float                       # 0-1, higher is better
    passed: bool
    issues: list[QualityIssue] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "passed": self.passed,
            "issues": [asdict(issue) for issue in self.issues],
            "metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in self.metrics.items()},
        }


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian - the standard no-reference blur estimate.

    A sharp image has strong second derivatives at edges and therefore high
    variance; a blurred one has few. Measured on a fixed 512 px long side so the
    threshold does not depend on the upload's resolution.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _spatial_coherence(gray: np.ndarray) -> float:
    """Lag-1 autocorrelation of the luminance channel.

    Natural photographs obey well-known image statistics: neighbouring pixels are
    highly correlated, typically above 0.9 even for detailed textures. Synthetic
    pixel noise, dithered graphics and random-value images have almost none. This
    single number is what separates "a sharp photograph of something" from "an
    array of random values that happens to be sharp and colourful", which the
    Laplacian blur check by itself cannot do - noise scores *very high* on
    sharpness.

    Returns the mean of the horizontal and vertical lag-1 correlations.
    """
    values = gray.astype(np.float64)
    if values.size < 16:
        return 1.0

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        a = a.ravel()
        b = b.ravel()
        a_centred = a - a.mean()
        b_centred = b - b.mean()
        denominator = np.sqrt((a_centred**2).sum() * (b_centred**2).sum())
        # A perfectly uniform region has zero variance; treat it as fully
        # coherent rather than dividing by zero.
        return 1.0 if denominator < 1e-9 else float((a_centred * b_centred).sum() / denominator)

    horizontal = _corr(values[:, :-1], values[:, 1:])
    vertical = _corr(values[:-1, :], values[1:, :])
    return (horizontal + vertical) / 2


def _green_fraction(hsv: np.ndarray) -> float:
    """Fraction of pixels whose hue lies in the vegetation band."""
    hue = hsv[:, :, 0].astype(np.int32)      # OpenCV hue is 0-179
    sat = hsv[:, :, 1].astype(np.int32)
    val = hsv[:, :, 2].astype(np.int32)
    mask = (hue >= 25) & (hue <= 95) & (sat >= 40) & (val >= 30)
    return float(mask.mean())


def _plant_like_fraction(hsv: np.ndarray) -> float:
    """Fraction of pixels in the combined green + brown/yellow lesion bands."""
    hue = hsv[:, :, 0].astype(np.int32)
    sat = hsv[:, :, 1].astype(np.int32)
    val = hsv[:, :, 2].astype(np.int32)
    green = (hue >= 25) & (hue <= 95)
    lesion = (hue >= 5) & (hue < 25)          # brown/orange necrosis
    return float(((green | lesion) & (sat >= 30) & (val >= 25)).mean())


def analyse_image(image: Image.Image) -> QualityReport:
    """Run every quality check and return a combined report."""
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    if height <= 0 or width <= 0:
        return QualityReport(
            score=0.0,
            passed=False,
            issues=[
                QualityIssue(
                    "too_small",
                    "error",
                    f"Image has invalid dimensions ({width}x{height}).",
                    "Upload an image at least 224x224 pixels; phone photos are ideal.",
                )
            ],
            metrics={"width": width, "height": height, "min_side": 0},
        )

    # Normalise the working resolution so thresholds mean the same thing for a
    # 4000 px phone photo and a 256 px thumbnail.
    scale = 512 / max(height, width, 1)
    working = cv2.resize(rgb, (max(1, int(width * scale)), max(1, int(height * scale))),
                         interpolation=cv2.INTER_AREA) if scale < 1 else rgb

    gray = cv2.cvtColor(working, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(working, cv2.COLOR_RGB2HSV)

    blur = _laplacian_variance(gray)
    coherence = _spatial_coherence(gray)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    saturation = float(hsv[:, :, 1].mean())
    green = _green_fraction(hsv)
    plant_like = _plant_like_fraction(hsv)
    min_side = min(height, width)

    metrics = {
        "width": width,
        "height": height,
        "min_side": min_side,
        "megapixels": round(width * height / 1e6, 3),
        "blur_variance": blur,
        "spatial_coherence": coherence,
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "green_fraction": green,
        "plant_like_fraction": plant_like,
        "aspect_ratio": round(width / height, 3) if height else 0.0,
    }

    issues: list[QualityIssue] = []
    # Each check contributes a factor in [0, 1]; the final score is their product
    # so any single hard failure dominates.
    factors: list[float] = []

    if min_side < settings.quality_min_side:
        issues.append(QualityIssue(
            "too_small", "error",
            f"Image is only {width}x{height} pixels.",
            "Upload an image at least 224x224 pixels; phone photos are ideal.",
        ))
        factors.append(0.15)
    else:
        factors.append(min(1.0, min_side / 224))

    if blur < settings.quality_blur_threshold:
        issues.append(QualityIssue(
            "blurry", "error",
            f"The image looks out of focus (sharpness {blur:.0f}, "
            f"below the {settings.quality_blur_threshold:.0f} threshold).",
            "Hold the camera steady, tap to focus on the leaf, and retake the photo.",
        ))
        factors.append(max(0.15, blur / settings.quality_blur_threshold))
    else:
        factors.append(min(1.0, 0.6 + 0.4 * blur / (settings.quality_blur_threshold * 4)))

    if coherence < settings.quality_min_coherence:
        issues.append(QualityIssue(
            "not_a_photograph", "error",
            f"Neighbouring pixels are almost uncorrelated (spatial coherence "
            f"{coherence:.2f}), which does not occur in camera photographs.",
            "Upload a photograph of a leaf rather than a synthetic, noisy or "
            "heavily processed image.",
        ))
        factors.append(0.2)
    else:
        factors.append(1.0)

    if brightness < settings.quality_dark_threshold:
        issues.append(QualityIssue(
            "too_dark", "error", f"The image is very dark (mean brightness {brightness:.0f}/255).",
            "Photograph the leaf in daylight or add a light source.",
        ))
        factors.append(0.3)
    elif brightness > settings.quality_bright_threshold:
        issues.append(QualityIssue(
            "too_bright", "error",
            f"The image is over-exposed (mean brightness {brightness:.0f}/255).",
            "Avoid direct sunlight and camera flash; use even, diffuse lighting.",
        ))
        factors.append(0.3)
    else:
        factors.append(1.0)

    if contrast < 18:
        issues.append(QualityIssue(
            "low_contrast", "warning", f"Very low contrast (std {contrast:.0f}).",
            "Make sure the leaf is clearly separated from the background.",
        ))
        factors.append(0.65)
    else:
        factors.append(1.0)

    if saturation < 25:
        issues.append(QualityIssue(
            "near_greyscale", "warning",
            "The image has almost no colour, which is unusual for a leaf photograph.",
            "Upload a colour photograph of the leaf rather than a scan or screenshot.",
        ))
        factors.append(0.6)
    else:
        factors.append(1.0)

    if (plant_like < 0.10) or (green < 0.008) or (green < 0.02 and plant_like < 0.22):
        issues.append(QualityIssue(
            "not_a_plant", "error",
            f"No plant or leaf detected. Only {plant_like * 100:.0f}% of the image "
            "contains plant foliage or vegetation colors.",
            "Please upload a clear photograph of a plant leaf from one of the 14 supported crops.",
        ))
        factors.append(0.05)
    elif plant_like < 0.28:
        issues.append(QualityIssue(
            "small_leaf_area", "warning",
            f"The leaf appears to cover only {plant_like * 100:.0f}% of the frame.",
            "Move closer so the leaf fills most of the frame.",
        ))
        factors.append(0.8)
    else:
        factors.append(1.0)

    score = float(np.prod(factors))
    has_error = any(issue.severity == "error" for issue in issues)
    return QualityReport(
        score=score,
        passed=not has_error and score >= settings.quality_min_score,
        issues=issues,
        metrics=metrics,
    )


def quality_band(score: float) -> str:
    """Human-facing bucket for the quality score."""
    if score >= 0.75:
        return "good"
    if score >= settings.quality_min_score:
        return "acceptable"
    return "poor"
