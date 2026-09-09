"""Heat-map rendering and statistics - pure numpy, no deep-learning framework.

Split out of :mod:`app.ml.explain` so the ONNX serving backend can render and
summarise heat maps without importing torch. ``explain.py`` re-exports
everything here, so the torch path and its callers are unaffected.

Nothing in this module knows how a heat map was produced. Grad-CAM from an
autograd backward pass, Grad-CAM read out of a baked CAM head, or an occlusion
sweep all arrive as the same ``(H, W)`` float array in ``[0, 1]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CamResult:
    """One heat map plus the class it explains."""

    heatmap: np.ndarray      # (H, W) float32 in [0, 1]
    class_index: int
    method: str


# A jet-style ramp, sampled at nine anchors and linearly interpolated between
# them. Kept as data rather than pulled from matplotlib so the inference path
# never imports a plotting backend - which matters both for startup time and,
# on a 512 MB host, for resident memory.
JET_ANCHORS = np.array([
    [0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.0, 0.5, 1.0], [0.0, 1.0, 1.0],
    [0.5, 1.0, 0.5], [1.0, 1.0, 0.0], [1.0, 0.5, 0.0], [1.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
], dtype=np.float32)


def colorize(heatmap: np.ndarray) -> np.ndarray:
    """Map a ``[0, 1]`` heat map to an RGB uint8 jet-style image."""
    heat = np.clip(heatmap, 0.0, 1.0)
    positions = np.linspace(0.0, 1.0, len(JET_ANCHORS))
    rgb = np.stack(
        [np.interp(heat, positions, JET_ANCHORS[:, channel]) for channel in range(3)],
        axis=-1,
    )
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45, threshold: float = 0.15
) -> np.ndarray:
    """Blend a heat map over an RGB image.

    Values below ``threshold`` stay fully transparent so the leaf remains legible
    where the model found nothing of interest.
    """
    if image.shape[:2] != heatmap.shape[:2]:
        raise ValueError(f"Shape mismatch: image {image.shape[:2]} vs heatmap {heatmap.shape[:2]}")
    colored = colorize(heatmap).astype(np.float32)
    base = image.astype(np.float32)
    weight = (alpha * np.clip((heatmap - threshold) / (1 - threshold), 0, 1))[..., None]
    return np.clip(base * (1 - weight) + colored * weight, 0, 255).astype(np.uint8)


def attention_coverage(heatmap: np.ndarray, threshold: float = 0.5) -> dict:
    """Summary statistics of a heat map, shown alongside the visualisation.

    ``focus_ratio`` is the fraction of the image above ``threshold``; a very
    small value means the model keyed on a tiny patch, a very large one means it
    used most of the frame (often a sign of background reliance).
    """
    mask = heatmap >= threshold
    coords = np.argwhere(mask)
    centroid = coords.mean(axis=0) if len(coords) else np.array([np.nan, np.nan])
    return {
        "focus_ratio": float(mask.mean()),
        "peak_value": float(heatmap.max()),
        "mean_value": float(heatmap.mean()),
        "centroid_y": None if np.isnan(centroid[0]) else float(centroid[0] / heatmap.shape[0]),
        "centroid_x": None if np.isnan(centroid[1]) else float(centroid[1] / heatmap.shape[1]),
    }


def normalise(heatmap: np.ndarray) -> np.ndarray:
    """Rescale to ``[0, 1]``, tolerating a constant map without producing NaNs."""
    shifted = heatmap - heatmap.min()
    return shifted / max(float(shifted.max()), 1e-8)
