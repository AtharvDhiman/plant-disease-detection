"""Dust and soil film detection.

A leaf coated in dust or dried soil is a common field photograph and a bad
input for the classifier, for two reasons the classifier itself cannot see:

* the film *hides* disease - lesions under a tan haze lose the colour contrast
  the network keys on, so a blighted leaf can come back "healthy";
* the film *imitates* disease - a pale, grainy coating is visually close to
  powdery mildew, so a clean healthy leaf can come back "powdery mildew".

There is no "dusty" class in PlantVillage, so the network cannot say either of
these things. This module does, from image statistics alone, before the
classifier runs.

Signal
------
A dust film is an additive, light, weakly coloured layer. Physically that is
the same thing as atmospheric haze, so the strongest single feature is the
*dark channel* (He, Sun & Tang, CVPR 2009): in a clean photograph almost every
local patch contains some pixel that is dark in at least one colour channel -
leaves are low in blue, shadows are low in everything - while a film lifts that
minimum everywhere. The remaining features describe what the film does to
colour (lower saturation, a warm grey-brown cast) and texture (lower local
contrast, small bright grains).

The features feed a one-hidden-layer network (30 -> 32 -> 1, about a thousand
weights, evaluated in numpy, so it adds nothing to the serving image). It beat
a logistic regression clearly on held-out data - 0.968 vs 0.944 AUC - because
"pale" only means dust when it co-occurs with "low contrast" and "lifted dark
channel", an interaction a linear model cannot express. It is trained by
``training/train_dust_detector.py``
and stored in ``models/metadata/dust_detector.json`` together with the
operating point and the error rates measured on held-out data.

Limitation, stated plainly: there is no public dataset of dusty leaf
photographs, so the positives are synthetic dust films on real leaves. The
training script reports performance on crops withheld from training and on
clean leaves under field-like transforms (soil backgrounds, harsh light, heavy
JPEG), which is where false alarms would come from in practice.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from app.core.config import settings

FEATURE_NAMES: tuple[str, ...] = (
    "dark_p10", "dark_p50", "dark_p90",
    "sat_p10", "sat_p50", "sat_p90",
    "val_p10", "val_p50", "val_p90",
    "local_contrast_rel", "local_contrast_p50",
    "tan_fraction", "green_fraction", "haze_ratio",
    "veg_fraction", "veg_saturation", "veg_dark",
    "log_sharpness",
    "hue_red", "hue_orange", "hue_yellow", "hue_green", "hue_cyan", "hue_other",
    "cast_red_blue", "cast_green_blue",
    "speck_density", "grey_fraction",
    "dark_p50_x_sat", "veg_dark_minus_bg",
)

_WORKING_SIDE = 256


def _resize(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = _WORKING_SIDE / max(height, width, 1)
    if scale >= 1:
        return rgb
    return cv2.resize(rgb, (max(1, int(width * scale)), max(1, int(height * scale))),
                      interpolation=cv2.INTER_AREA)


def dust_features(rgb: np.ndarray) -> np.ndarray:
    """The feature vector, in the order of :data:`FEATURE_NAMES`.

    ``rgb`` is an ``(H, W, 3)`` uint8 array. Work happens at a fixed 256 px long
    side so the statistics mean the same for a phone photo and a thumbnail.
    """
    img = _resize(np.ascontiguousarray(rgb[..., :3], dtype=np.uint8))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.int32)
    sat = hsv[..., 1].astype(np.float32)
    val = hsv[..., 2].astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Dark channel: per-pixel channel minimum, then a local minimum filter.
    dark = cv2.erode(img.min(axis=2), np.ones((7, 7), np.uint8)).astype(np.float32)

    mean = cv2.blur(gray, (7, 7))
    sq_mean = cv2.blur(gray * gray, (7, 7))
    local_std = np.sqrt(np.maximum(sq_mean - mean * mean, 0))

    tan = (hue >= 8) & (hue <= 32) & (sat >= 18) & (sat <= 105) & (val >= 105)
    green = (hue >= 25) & (hue <= 95) & (sat >= 40) & (val >= 30)
    vegetation = (hue >= 15) & (hue <= 95) & (sat >= 18) & (val >= 40)
    grey = (sat < 25) & (val >= 60)

    veg_fraction = float(vegetation.mean())
    if vegetation.sum() >= 50:
        veg_sat = float(np.median(sat[vegetation]))
        veg_dark = float(np.median(dark[vegetation]))
        background = ~vegetation
        bg_dark = float(np.median(dark[background])) if background.sum() >= 50 else veg_dark
    else:
        veg_sat = float(np.median(sat))
        veg_dark = bg_dark = float(np.median(dark))

    coloured = sat >= 30
    if coloured.sum() >= 50:
        counts = np.array([
            ((hue < 8) | (hue >= 170))[coloured].mean(),
            ((hue >= 8) & (hue < 20))[coloured].mean(),
            ((hue >= 20) & (hue < 35))[coloured].mean(),
            ((hue >= 35) & (hue < 85))[coloured].mean(),
            ((hue >= 85) & (hue < 100))[coloured].mean(),
        ])
        hist = np.append(counts, max(0.0, 1.0 - counts.sum()))
    else:
        hist = np.zeros(6)

    tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, np.ones((5, 5), np.uint8))
    speck = (tophat > 22) & (sat < 110)

    channels = img.reshape(-1, 3).astype(np.float32)
    dark_p50 = float(np.median(dark)) / 255
    sat_p50 = float(np.median(sat)) / 255
    features = [
        float(np.percentile(dark, 10)) / 255, dark_p50, float(np.percentile(dark, 90)) / 255,
        float(np.percentile(sat, 10)) / 255, sat_p50, float(np.percentile(sat, 90)) / 255,
        float(np.percentile(val, 10)) / 255, float(np.median(val)) / 255,
        float(np.percentile(val, 90)) / 255,
        float(local_std.mean() / (gray.mean() + 1.0)), float(np.median(local_std)) / 255,
        float(tan.mean()), float(green.mean()),
        float(np.median(dark) / (np.median(val) + 1.0)),
        veg_fraction, veg_sat / 255, veg_dark / 255,
        float(np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var())) / 10,
        *[float(x) for x in hist],
        float((channels[:, 0] - channels[:, 2]).mean()) / 255,
        float((channels[:, 1] - channels[:, 2]).mean()) / 255,
        float(speck.mean()), float(grey.mean()),
        dark_p50 * (1 - sat_p50), (veg_dark - bg_dark) / 255,
    ]
    return np.asarray(features, dtype=np.float64)


@dataclass
class DustVerdict:
    """Probability of a dust film and what the gate makes of it."""

    probability: float
    level: str            # "none" | "possible" | "heavy"
    threshold: float
    heavy_threshold: float


class DustDetector:
    """Standardise, run the small ReLU network, squash. Loaded once from JSON."""

    def __init__(self, payload: dict) -> None:
        if tuple(payload["features"]) != FEATURE_NAMES:
            raise ValueError("dust_detector.json was trained on a different feature set; "
                             "re-run training/train_dust_detector.py")
        self.mean = np.asarray(payload["mean"], dtype=np.float64)
        self.scale = np.asarray(payload["scale"], dtype=np.float64)
        self.layers = [(np.asarray(layer["weights"], dtype=np.float64),
                        np.asarray(layer["bias"], dtype=np.float64))
                       for layer in payload["layers"]]
        self.threshold = float(payload["threshold"])
        self.heavy_threshold = float(payload["heavy_threshold"])

    def logit(self, features: np.ndarray) -> float:
        activation = (features - self.mean) / self.scale
        for weights, bias in self.layers[:-1]:
            activation = np.maximum(activation @ weights + bias, 0.0)
        weights, bias = self.layers[-1]
        return float((activation @ weights + bias).ravel()[0])

    def probability(self, rgb: np.ndarray) -> float:
        logit = self.logit(dust_features(rgb))
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40))))

    def assess(self, rgb: np.ndarray) -> DustVerdict:
        p = self.probability(rgb)
        level = ("heavy" if p >= self.heavy_threshold
                 else "possible" if p >= self.threshold else "none")
        return DustVerdict(p, level, self.threshold, self.heavy_threshold)


def detector_path() -> Path:
    return settings.metadata_dir / "dust_detector.json"


@lru_cache(maxsize=1)
def load_detector() -> DustDetector | None:
    """The trained detector, or ``None`` when it has not been trained.

    A missing file disables the check rather than failing uploads: the rest of
    the quality gate still works without it.
    """
    path = detector_path()
    if not path.is_file():
        return None
    return DustDetector(json.loads(path.read_text(encoding="utf-8")))
