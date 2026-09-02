"""Hand-crafted image features for the classical machine-learning baselines.

These are the descriptors a computer-vision practitioner would have reached for
before deep learning: colour distribution, colour moments, texture statistics and
gradient-orientation histograms. They give the deep models something honest to be
compared against, and they make the "why CNN?" argument empirical rather than
assumed.

Feature groups
--------------
=================  =====  ==========================================================
Group              Dims   Rationale
=================  =====  ==========================================================
RGB histogram         96  Overall colour distribution; chlorosis shifts green->yellow
HSV histogram         96  Hue/saturation separate pigment change from illumination
Colour moments        18  Mean/std/skew per channel - compact colour summary
LBP histogram         26  Local Binary Patterns: micro-texture of lesions/mildew
GLCM properties       48  Haralick co-occurrence statistics: coarse texture
HOG                  900  Edge/gradient structure of lesion borders and veins
=================  =====  ==========================================================
Total: 1184 dimensions per image (verified by ``feature_group_slices``).
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

FEATURE_IMAGE_SIZE = 96
HIST_BINS = 32
LBP_POINTS = 24
LBP_RADIUS = 3
GLCM_DISTANCES = (1, 3)
GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


def _histogram(channel: np.ndarray, bins: int = HIST_BINS) -> np.ndarray:
    """L1-normalised histogram of one channel in ``[0, 1]``."""
    counts, _ = np.histogram(channel, bins=bins, range=(0.0, 1.0))
    total = counts.sum()
    return counts / total if total else counts.astype(np.float64)


def _moments(channel: np.ndarray) -> np.ndarray:
    """Mean, standard deviation and (signed cube-root) skewness."""
    mean = float(channel.mean())
    std = float(channel.std())
    centred = channel - mean
    third = float(np.mean(centred**3))
    skew = float(np.sign(third) * abs(third) ** (1 / 3))
    return np.array([mean, std, skew], dtype=np.float64)


def color_features(rgb: np.ndarray) -> np.ndarray:
    """RGB + HSV histograms and per-channel colour moments."""
    hsv = rgb2hsv(rgb)
    parts = [_histogram(rgb[:, :, i]) for i in range(3)]
    parts += [_histogram(hsv[:, :, i]) for i in range(3)]
    parts += [_moments(rgb[:, :, i]) for i in range(3)]
    parts += [_moments(hsv[:, :, i]) for i in range(3)]
    return np.concatenate(parts)


def texture_features(gray: np.ndarray) -> np.ndarray:
    """Uniform LBP histogram plus GLCM Haralick properties."""
    # LBP thresholds neighbours against the centre pixel; on floating-point input
    # sub-quantum differences flip bits arbitrarily, so quantise to uint8 first.
    gray_u8 = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_u8, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2
    lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    lbp_hist = lbp_hist / (lbp_hist.sum() or 1)

    quantised = gray_u8 // 4  # 64 grey levels keeps the GLCM small
    glcm = graycomatrix(quantised, distances=list(GLCM_DISTANCES), angles=list(GLCM_ANGLES),
                        levels=64, symmetric=True, normed=True)
    glcm_feats = np.concatenate([graycoprops(glcm, prop).ravel() for prop in GLCM_PROPS])
    return np.concatenate([lbp_hist, glcm_feats])


def hog_features(gray: np.ndarray) -> np.ndarray:
    """Histogram of oriented gradients over a coarse cell grid."""
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def extract_features(image: Image.Image, size: int = FEATURE_IMAGE_SIZE) -> np.ndarray:
    """Full feature vector for one PIL image."""
    rgb = np.asarray(image.convert("RGB").resize((size, size), Image.BILINEAR),
                     dtype=np.float64) / 255.0
    gray = rgb2gray(rgb)
    return np.concatenate([
        color_features(rgb),
        texture_features(gray),
        hog_features(gray),
    ]).astype(np.float32)


def extract_from_path(path: str, size: int = FEATURE_IMAGE_SIZE) -> np.ndarray | None:
    """Feature vector for an image on disk; ``None`` when it cannot be read."""
    try:
        with Image.open(path) as image:
            return extract_features(image, size)
    except Exception:  # noqa: BLE001 - unreadable images are reported by the caller
        return None


def feature_group_slices(size: int = FEATURE_IMAGE_SIZE) -> dict[str, slice]:
    """Index ranges of each feature group, for group-wise importance analysis."""
    probe = Image.new("RGB", (size, size))
    rgb = np.asarray(probe, dtype=np.float64) / 255.0
    gray = rgb2gray(rgb)
    n_color = len(color_features(rgb))
    n_texture = len(texture_features(gray))
    n_hog = len(hog_features(gray))
    return {
        "color": slice(0, n_color),
        "texture": slice(n_color, n_color + n_texture),
        "hog": slice(n_color + n_texture, n_color + n_texture + n_hog),
    }


def feature_names(size: int = FEATURE_IMAGE_SIZE) -> list[str]:
    """Human-readable name per feature dimension."""
    names: list[str] = []
    for space in ("R", "G", "B", "H", "S", "V"):
        names += [f"hist_{space}_{i}" for i in range(HIST_BINS)]
    for space in ("R", "G", "B", "H", "S", "V"):
        names += [f"moment_{space}_{m}" for m in ("mean", "std", "skew")]
    names += [f"lbp_{i}" for i in range(LBP_POINTS + 2)]
    for prop in GLCM_PROPS:
        for d in GLCM_DISTANCES:
            for a_idx in range(len(GLCM_ANGLES)):
                names.append(f"glcm_{prop}_d{d}_a{a_idx}")
    slices = feature_group_slices(size)
    n_hog = slices["hog"].stop - slices["hog"].start
    names += [f"hog_{i}" for i in range(n_hog)]
    return names
