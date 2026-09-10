"""Torch-free preprocessing for the ONNX serving path.

Why this module exists
----------------------
``transforms.py`` builds the eval pipeline out of ``torchvision.transforms``,
which means importing it drags in torch. On the ONNX serving backend that is the
whole cost we are trying to avoid: torch accounts for roughly 600 MB of resident
memory against 100 MB for the entire ONNX Runtime stack, which is the difference
between fitting in a 512 MB container and being OOM-killed at startup.

Correctness, not approximation
------------------------------
A preprocessing mismatch between training and serving is the classic silent
accuracy killer, so this is not a "close enough" reimplementation. torchvision's
``Resize``/``CenterCrop`` delegate to Pillow when handed a PIL image, so doing
the same Pillow calls in the same order reproduces the reference pipeline
exactly. Verified against ``build_eval_transform(224)`` on real uploads and on
non-square synthetic images (640x480, 480x640, 1000x300, 300x1000): maximum
absolute difference 0.0, i.e. bit-identical, not merely close.

The two details that a naive reimplementation gets wrong, and that the
verification above exists to catch:

* ``Resize(int)`` scales the *shorter* side to that value and preserves aspect
  ratio. Resizing straight to ``(224, 224)`` squashes non-square photographs and
  shifts every feature the model was trained on.
* ``CenterCrop`` rounds its offsets with ``round((size - crop) / 2)``, so an odd
  size difference lands one pixel off if you use floor division.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from app.core.config import settings

# Matching build_eval_transform's default: resize the short side to 1.14x the
# crop, then centre-crop. Preserving aspect ratio this way is how the ImageNet
# backbones were evaluated, and the ratio is baked into the trained weights.
RESIZE_RATIO = 1.14


def eval_preprocess(
    image: Image.Image,
    image_size: int | None = None,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Return a normalised ``(1, 3, S, S)`` float32 array ready for ONNX Runtime.

    Equivalent to ``Resize -> CenterCrop -> ToTensor -> Normalize`` from
    :func:`app.ml.transforms.build_eval_transform`, without importing torch.
    """
    size = image_size or settings.image_size
    mean = mean or settings.normalize_mean
    std = std or settings.normalize_std

    cropped = eval_crop(image, size)

    array = np.asarray(cropped, dtype=np.uint8).astype(np.float32) / 255.0
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    # HWC -> CHW, then add the batch axis. ascontiguousarray because ONNX Runtime
    # copies a non-contiguous input anyway; doing it once here is cheaper.
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def eval_crop(image: Image.Image, image_size: int | None = None) -> Image.Image:
    """The exact RGB crop the model sees, for heat-map overlays.

    The torch path's :func:`app.ml.transforms.resized_rgb` does the same thing;
    the geometry must stay identical or a heat map would be drawn over a
    differently framed image.
    """
    size = image_size or settings.image_size
    resize_to = int(round(size * RESIZE_RATIO))
    img = image.convert("RGB")

    # torchvision Resize(int): shorter side -> resize_to, aspect ratio preserved.
    width, height = img.size
    short, long_side = min(width, height), max(width, height)
    if short != resize_to:
        new_long = int(resize_to * long_side / short)
        if width <= height:
            new_width, new_height = resize_to, new_long
        else:
            new_width, new_height = new_long, resize_to
        resample = getattr(Image, "Resampling", Image).BILINEAR
        img = img.resize((new_width, new_height), resample)

    # torchvision CenterCrop, including its rounding behaviour.
    width, height = img.size
    top = int(round((height - size) / 2.0))
    left = int(round((width - size) / 2.0))
    return img.crop((left, top, left + size, top + size))


def denormalize(batch: np.ndarray,
                mean: tuple[float, float, float] | None = None,
                std: tuple[float, float, float] | None = None) -> np.ndarray:
    """Invert :func:`eval_preprocess`'s normalisation, for visualisation."""
    mean = mean or settings.normalize_mean
    std = std or settings.normalize_std
    m = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    s = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)
    return np.clip(batch * s + m, 0.0, 1.0)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with optional temperature scaling.

    The torch path applies temperature to the logits before softmax so the
    reported confidence is calibrated; this reproduces that on the ONNX path.
    Subtracting the max first prevents ``exp`` overflowing on confident logits.
    """
    scaled = np.asarray(logits, dtype=np.float32) / max(temperature, 1e-6)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)
