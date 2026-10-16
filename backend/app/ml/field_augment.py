"""Field-condition augmentation: make a lab leaf photo look like a field one.

Why
---
PlantVillage photographs are a single leaf, flat on grey card, evenly lit,
filling the frame. Measured through the live pipeline, the production model
keeps ~97% on those but falls to ~77% when the same leaves are given a soil
backdrop, harsh sun and heavy JPEG together, and to ~10% when the leaf is
small in the frame. The model has simply never seen those conditions.

This transform shows them to it during training. Each condition is applied
independently with its own probability, so the model sees every combination,
including the untouched lab photo:

* **background** - the leaf is cut out (PlantVillage's grey card makes that a
  saturation threshold) and placed on a procedural backdrop: soil, mulch,
  out-of-focus foliage or mixed clutter;
* **framing**   - the leaf shrinks to 35-80% of the frame at a random position,
  as when the photo is taken from further away;
* **lighting**  - a directional brightness gradient at a random angle, and
  sometimes a soft shadow, as in direct sun;
* **blur / noise / JPEG** - focus error, sensor noise and re-compression.

What it does not touch: hue. Colour is the diagnostic signal for chlorosis and
necrosis, so backgrounds and lighting change brightness and context but never
recolour the leaf itself.

Independence from the evaluation
--------------------------------
``scripts/domain_shift_probe.py`` measures the same *kinds* of condition, so a
model trained with this transform is not tested on a fully independent
distribution there. The generators deliberately differ (multi-scale textures
and four backdrop families here, against one fixed-colour Gaussian soil and a
fixed 36% centred leaf in the probe), but the honest independent test is real
field photographs - see the README.

Randomness comes from Python's ``random`` module only. PyTorch seeds it per
DataLoader worker, so runs are reproducible and workers do not repeat draws.
"""
from __future__ import annotations

import io
import math
import random

import numpy as np
from PIL import Image, ImageFilter


def _rng() -> np.random.Generator:
    # Derived from Python's per-worker-seeded RNG so numpy draws stay reproducible.
    return np.random.default_rng(random.getrandbits(32))


def leaf_mask(image: Image.Image) -> Image.Image:
    """Soft leaf mask for a PlantVillage photograph, as an ``L`` image.

    The card behind the leaf is grey or faintly blue-lavender; leaves are green,
    yellow, brown or red. So the mask is "coloured, and not in the blue-purple
    band", followed by:

    * a morphological close and open to join veins and drop speckle;
    * keeping only sizeable connected components (the leaf, not card texture);
    * filling holes, so pale lesions and white mildew stay part of the leaf -
      cutting out the disease would defeat the purpose;
    * a slight blur, so the paste has no hard cut-out edge.
    """
    import cv2

    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.int32)        # 0-179
    sat = hsv[..., 1].astype(np.int32)
    val = hsv[..., 2].astype(np.int32)
    not_blue = (hue < 98) | (hue > 165)
    mask = (((sat > 38) & not_blue) | ((sat > 22) & (hue >= 8) & (hue <= 95) & (val < 215)))
    mask = mask.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.flatnonzero(areas >= max(0.02 * mask.size, 0.15 * areas.max()))
        mask = np.isin(labels, keep).astype(np.uint8) * 255

    # Fill holes: flood the background from the border; whatever it cannot
    # reach and is not already leaf is an interior hole.
    flood = mask.copy()
    h, w = mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if flood[seed[1], seed[0]] == 0:
            cv2.floodFill(flood, canvas, seed, 255)
    mask = mask | cv2.bitwise_not(flood)

    return Image.fromarray(mask, "L").filter(ImageFilter.GaussianBlur(1.2))


def _multiscale_noise(size: int, rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    """Fractal value noise in [0, 1]: coarse clumps plus fine grain."""
    field = np.zeros((size, size), dtype=np.float64)
    weight_total = 0.0
    for octave in range(octaves):
        cells = 2 ** (octave + 2)
        coarse = rng.random((cells, cells))
        layer = np.asarray(Image.fromarray((coarse * 255).astype(np.uint8))
                           .resize((size, size), Image.BICUBIC), dtype=np.float64) / 255
        weight = 0.55 ** octave
        field += layer * weight
        weight_total += weight
    field /= weight_total
    return (field - field.min()) / (np.ptp(field) + 1e-9)


def background(size: int) -> Image.Image:
    """One procedural field backdrop, drawn from four families."""
    rng = _rng()
    kind = rng.choice(["soil", "mulch", "foliage", "clutter"], p=[0.4, 0.2, 0.25, 0.15])
    noise = _multiscale_noise(size, rng)[..., None]
    grain = rng.normal(0, 1, (size, size, 1))

    if kind == "soil":
        base = np.array([rng.uniform(70, 160), rng.uniform(55, 125), rng.uniform(35, 95)])
        image = base * (0.65 + 0.6 * noise) + grain * rng.uniform(6, 18)
    elif kind == "mulch":
        dark = np.array([rng.uniform(40, 80), rng.uniform(30, 60), rng.uniform(20, 45)])
        light = np.array([rng.uniform(120, 170), rng.uniform(95, 135), rng.uniform(60, 100)])
        streaks = _multiscale_noise(size, rng, octaves=5)[..., None] ** 2
        image = dark + (light - dark) * streaks + grain * 10
    elif kind == "foliage":
        # Out-of-focus vegetation behind the leaf. Kept darker and blurrier
        # than a subject leaf, which is how depth of field renders it.
        base = np.array([rng.uniform(25, 70), rng.uniform(55, 120), rng.uniform(20, 60)])
        image = base * (0.5 + 0.9 * noise)
    else:
        # Stones, stems, dead leaves, a hand or a pot rim: muted, earthy and
        # grey tones. Saturated colours would be easier to separate from the
        # leaf than anything a field actually contains.
        def muted() -> np.ndarray:
            grey = rng.uniform(40, 190)
            tint = np.array([rng.uniform(0, 30), rng.uniform(-5, 20), rng.uniform(-25, 5)])
            return grey + tint

        image = np.broadcast_to(muted(), (size, size, 3)).copy()
        for _ in range(int(rng.integers(4, 9))):
            blob = _multiscale_noise(size, rng, octaves=2)[..., None] > rng.uniform(0.55, 0.75)
            image = np.where(blob, muted(), image)
        image = image * (0.8 + 0.4 * noise) + grain * 6

    out = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    blur = 2.5 if kind == "foliage" else rng.uniform(0.3, 1.2)
    return out.filter(ImageFilter.GaussianBlur(blur))


def place_on_background(image: Image.Image, scale: float) -> Image.Image:
    """Cut the leaf out and paste it at ``scale`` of the frame on a backdrop."""
    size = image.width
    canvas = background(size)
    target = max(16, int(size * scale))
    leaf = image.resize((target, target), Image.BILINEAR)
    mask = leaf_mask(leaf)
    x = random.randint(0, size - target)
    y = random.randint(0, size - target)
    canvas.paste(leaf, (x, y), mask)
    return canvas


def directional_light(image: Image.Image) -> Image.Image:
    """Brightness ramp at a random angle, optionally with a soft shadow."""
    rng = _rng()
    array = np.asarray(image, dtype=np.float64)
    height, width = array.shape[:2]
    angle = rng.uniform(0, 2 * math.pi)
    yy, xx = np.mgrid[0:height, 0:width]
    ramp = (np.cos(angle) * (xx / width - 0.5) + np.sin(angle) * (yy / height - 0.5))
    ramp = (ramp - ramp.min()) / (np.ptp(ramp) + 1e-9)
    bright, dim = rng.uniform(1.15, 1.5), rng.uniform(0.5, 0.85)
    gain = dim + (bright - dim) * ramp
    if rng.random() < 0.35:
        shadow = _multiscale_noise(width, rng, octaves=2)[:height, :width] > rng.uniform(0.55, 0.7)
        shadow = np.asarray(Image.fromarray((shadow * 255).astype(np.uint8))
                            .filter(ImageFilter.GaussianBlur(6)), dtype=np.float64) / 255
        gain = gain * (1 - shadow * rng.uniform(0.25, 0.5))
    # Gain is applied equally to all three channels: brighter or darker, never
    # a different hue.
    return Image.fromarray(np.clip(array * gain[..., None], 0, 255).astype(np.uint8))


def sensor_noise(image: Image.Image) -> Image.Image:
    rng = _rng()
    array = np.asarray(image, dtype=np.float64)
    # Luminance noise shared across channels, so it does not tint the leaf.
    noise = rng.normal(0, rng.uniform(3, 10), array.shape[:2])[..., None]
    return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))


def jpeg(image: Image.Image) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=random.randint(18, 75))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


class FieldConditions:
    """Apply field conditions to a PIL image; see the module docstring.

    ``p`` is the chance that an image is altered at all, so ``1 - p`` of the
    training stream stays lab-clean and lab accuracy is not traded away.
    """

    def __init__(self, p: float = 0.7, background_p: float = 0.55, framing_p: float = 0.4,
                 lighting_p: float = 0.35, blur_p: float = 0.2, noise_p: float = 0.2,
                 jpeg_p: float = 0.3) -> None:
        self.p = p
        self.background_p = background_p
        self.framing_p = framing_p
        self.lighting_p = lighting_p
        self.blur_p = blur_p
        self.noise_p = noise_p
        self.jpeg_p = jpeg_p

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if random.random() >= self.p:
            return image
        # Square working canvas: placement maths assumes it, and PlantVillage
        # images already are 256x256.
        side = min(image.size)
        if image.width != image.height:
            image = image.crop((0, 0, side, side))

        if random.random() < self.framing_p:
            image = place_on_background(image, random.uniform(0.35, 0.8))
        elif random.random() < self.background_p:
            image = place_on_background(image, random.uniform(0.8, 0.98))
        if random.random() < self.lighting_p:
            image = directional_light(image)
        if random.random() < self.blur_p:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.6)))
        if random.random() < self.noise_p:
            image = sensor_noise(image)
        if random.random() < self.jpeg_p:
            image = jpeg(image)
        return image

    def __repr__(self) -> str:
        return f"{type(self).__name__}(p={self.p})"
