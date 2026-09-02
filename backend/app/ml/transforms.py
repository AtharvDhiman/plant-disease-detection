"""Preprocessing and augmentation pipelines shared by training and inference.

Keeping both pipelines in one module - imported by the training scripts *and* by
the serving code - is what guarantees that an image is normalised at inference
exactly the way it was during training. A mismatch here is one of the most
common silent accuracy killers in deployed vision systems.

Augmentation policy
-------------------
Only transforms that produce a *biologically plausible* leaf photograph are
used. Leaves are near-symmetric and are photographed from arbitrary angles, so
flips and rotations are safe. Colour jitter is deliberately mild: hue is the
signal for chlorosis and necrosis, and shifting it aggressively would teach the
model to ignore the very cue a plant pathologist uses. No vertical shear, no
channel shuffling, no cutout over large areas.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from torchvision import transforms as T

from app.core.config import settings


@dataclass(frozen=True)
class NormalizationStats:
    """Channel mean/std used for input normalisation."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def as_tensor(self, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tensor(self.mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=device).view(1, 3, 1, 1)
        return mean, std


IMAGENET_STATS = NormalizationStats(settings.normalize_mean, settings.normalize_std)


def build_train_transform(
    image_size: int,
    stats: NormalizationStats = IMAGENET_STATS,
    strength: str = "medium",
) -> T.Compose:
    """Augmentation pipeline applied to the training split only.

    ``strength`` selects a preset: ``light`` for short benchmark runs where we
    want models to converge under a fixed epoch budget, ``medium`` for the
    production run, ``strong`` when over-fitting is observed.
    """
    if strength not in {"light", "medium", "strong"}:
        raise ValueError(f"Unknown augmentation strength {strength!r}")

    scale = {"light": (0.85, 1.0), "medium": (0.7, 1.0), "strong": (0.55, 1.0)}[strength]
    rotation = {"light": 15, "medium": 25, "strong": 35}[strength]
    jitter = {"light": 0.12, "medium": 0.22, "strong": 0.32}[strength]

    steps: list = [
        # RandomResizedCrop gives scale + translation + aspect jitter in one op.
        T.RandomResizedCrop(image_size, scale=scale, ratio=(0.85, 1.18)),
        T.RandomHorizontalFlip(p=0.5),
        # Leaves have no canonical "up" in these photographs, so a vertical flip
        # is a genuine viewpoint change rather than an impossible image.
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(rotation, expand=False),
        T.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter * 0.8, hue=jitter * 0.15),
    ]
    if strength == "strong":
        # Mild blur stands in for out-of-focus phone photographs.
        steps.append(T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.2))

    steps += [T.ToTensor(), T.Normalize(stats.mean, stats.std)]
    if strength == "strong":
        steps.append(T.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.4, 2.5)))
    return T.Compose(steps)


def build_eval_transform(
    image_size: int,
    stats: NormalizationStats = IMAGENET_STATS,
    resize_ratio: float = 1.14,
) -> T.Compose:
    """Deterministic pipeline for validation, test and production inference.

    Resize-then-centre-crop (rather than a direct squash to ``image_size``)
    preserves aspect ratio, matching how the ImageNet backbones were trained.
    """
    resize_to = int(round(image_size * resize_ratio))
    return T.Compose([
        T.Resize(resize_to),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(stats.mean, stats.std),
    ])


def preprocess_pil(
    image: Image.Image,
    image_size: int | None = None,
    stats: NormalizationStats = IMAGENET_STATS,
) -> torch.Tensor:
    """Turn a PIL image into a normalised ``(1, 3, H, W)`` batch for inference."""
    size = image_size or settings.image_size
    tensor = build_eval_transform(size, stats)(image.convert("RGB"))
    return tensor.unsqueeze(0)


def denormalize(batch: torch.Tensor, stats: NormalizationStats = IMAGENET_STATS) -> torch.Tensor:
    """Invert :func:`build_eval_transform`'s normalisation for visualisation."""
    mean, std = stats.as_tensor(batch.device)
    return (batch * std + mean).clamp(0, 1)


def resized_rgb(image: Image.Image, image_size: int | None = None) -> Image.Image:
    """Return the RGB crop the model actually sees, for heat-map overlays.

    The geometry must match :func:`build_eval_transform` exactly, otherwise the
    heat map would be drawn over a differently framed image.
    """
    size = image_size or settings.image_size
    resize_to = int(round(size * 1.14))
    pipeline = T.Compose([T.Resize(resize_to), T.CenterCrop(size)])
    return pipeline(image.convert("RGB"))
