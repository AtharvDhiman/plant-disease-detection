"""Measure how far accuracy falls on field-like photographs.

Why this exists
---------------
Every accuracy figure in this project comes from PlantVillage: a single leaf,
flat on a uniform background, evenly lit. A photograph taken in a field - or
pulled off a web search - is none of those things. The documentation says this
gap is the largest limitation of the work, but saying so is cheap. This measures
it.

Each transformation below imitates one way a real photograph differs from the
training distribution, applied to leaves the model already classifies correctly:

* background   - the leaf pasted on soil-like texture instead of grey card
* angle        - photographed off-axis rather than flat
* distance     - leaf occupying a third of the frame, not filling it
* lighting     - harsh directional sun with a bright side and a shadow side
* compression  - heavy JPEG artefacts, as from a re-shared web image
* combined     - all of the above at once, the closest to a real field photo

The result is a lower bound on the drop, not an upper one: real photographs also
bring occlusion, motion blur, multiple overlapping leaves and species the model
has never seen. What this establishes is a floor - if accuracy falls this far on
imitations, it falls at least this far on the real thing.

    python scripts/domain_shift_probe.py --per-class 3
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8000/api/predict"


def soil_background(size: int, rng: random.Random) -> Image.Image:
    """A brown, grainy field-like backdrop.

    Real backgrounds are textured and mid-brown; the training images are flat
    grey. The texture matters as much as the colour, because a uniform fill is
    still unlike anything in the field.
    """
    noise = np.zeros((size, size, 3), dtype=np.uint8)
    base = np.array([104, 82, 58], dtype=np.float64)
    for channel in range(3):
        field = np.array([[rng.gauss(base[channel], 26) for _ in range(size)]
                          for _ in range(size)])
        noise[..., channel] = np.clip(field, 0, 255).astype(np.uint8)
    return Image.fromarray(noise).filter(ImageFilter.GaussianBlur(1.2))


def with_background(image: Image.Image, rng: random.Random) -> Image.Image:
    canvas = soil_background(image.width, rng)
    leaf = image.resize((int(image.width * 0.82), int(image.height * 0.82)))
    canvas.paste(leaf, (int(image.width * 0.09), int(image.height * 0.09)))
    return canvas


def off_axis(image: Image.Image, rng: random.Random) -> Image.Image:
    """Rotate and perspective-squash, as when shooting a leaf from the side."""
    rotated = image.rotate(rng.uniform(-28, 28), resample=Image.BICUBIC,
                           fillcolor=(96, 78, 56), expand=False)
    squashed = rotated.resize((image.width, int(image.height * 0.78)))
    canvas = Image.new("RGB", image.size, (96, 78, 56))
    canvas.paste(squashed, (0, int(image.height * 0.11)))
    return canvas


def far_away(image: Image.Image, rng: random.Random) -> Image.Image:
    """Leaf occupying a third of the frame, the rest background."""
    canvas = soil_background(image.width, rng)
    small = image.resize((int(image.width * 0.36), int(image.height * 0.36)))
    offset = int(image.width * 0.32)
    canvas.paste(small, (offset, offset))
    return canvas


def harsh_light(image: Image.Image, rng: random.Random) -> Image.Image:
    """A bright side and a shadowed side, as in direct sun."""
    array = np.asarray(image).astype(np.float64)
    height, width = array.shape[:2]
    gradient = np.linspace(1.55, 0.5, width)[None, :, None]
    lit = np.clip(array * gradient, 0, 255).astype(np.uint8)
    out = Image.fromarray(lit)
    return ImageEnhance.Contrast(out).enhance(rng.uniform(1.15, 1.4))


def recompressed(image: Image.Image, rng: random.Random) -> Image.Image:
    """Heavy JPEG artefacts, as from an image re-saved several times."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=rng.randint(14, 24))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def combined(image: Image.Image, rng: random.Random) -> Image.Image:
    return recompressed(harsh_light(off_axis(with_background(image, rng), rng), rng), rng)


TRANSFORMS = {
    "original": lambda img, rng: img,
    "background": with_background,
    "angle": off_axis,
    "distance": far_away,
    "lighting": harsh_light,
    "compression": recompressed,
    "combined": combined,
}


def predict(image: Image.Image) -> dict:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    body = buffer.getvalue()

    boundary = "----probe"
    payload = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="probe.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + body + f"\r\n--{boundary}--\r\n".encode()

    request = urllib.request.Request(
        API, data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    record = json.loads((PROJECT_ROOT / "data" / "dataset_path.json").read_text(encoding="utf-8"))
    valid = Path(record["path"]) / "New Plant Diseases Dataset(Augmented)" \
        / "New Plant Diseases Dataset(Augmented)" / "valid"
    if not valid.is_dir():
        raise SystemExit(f"Validation split not found at {valid}")

    diseased = sorted(d for d in valid.iterdir() if d.is_dir() and "healthy" not in d.name)
    chosen = diseased[:: max(1, len(diseased) // args.classes)][: args.classes]

    stats = {name: {"correct": 0, "total": 0, "refused": 0} for name in TRANSFORMS}
    for class_dir in chosen:
        images = sorted(class_dir.glob("*.JPG"))[: args.per_class]
        for path in images:
            original = Image.open(path).convert("RGB")
            for name, fn in TRANSFORMS.items():
                try:
                    result = predict(fn(original, rng))
                except Exception as exc:  # noqa: BLE001
                    print(f"  {name}: request failed: {exc}", flush=True)
                    continue
                stats[name]["total"] += 1
                if result.get("predicted_class") == class_dir.name:
                    stats[name]["correct"] += 1
                if result.get("status") != "ok":
                    stats[name]["refused"] += 1
        print(f"  done {class_dir.name}", flush=True)

    print(f"\n{'CONDITION':<14}{'ACCURACY':>10}{'vs LAB':>10}{'REFUSED':>10}")
    print("-" * 46)
    baseline = None
    for name in TRANSFORMS:
        entry = stats[name]
        if not entry["total"]:
            continue
        accuracy = entry["correct"] / entry["total"]
        refused = entry["refused"] / entry["total"]
        if baseline is None:
            baseline = accuracy
        delta = "" if name == "original" else f"{(accuracy - baseline) * 100:+.1f}pt"
        print(f"{name:<14}{accuracy * 100:>9.1f}%{delta:>10}{refused * 100:>9.1f}%")

    out = PROJECT_ROOT / "artifacts" / "results" / "domain_shift.json"
    out.write_text(json.dumps({
        "images_per_condition": stats["original"]["total"],
        "conditions": {
            name: {
                "accuracy": (e["correct"] / e["total"]) if e["total"] else None,
                "refused_rate": (e["refused"] / e["total"]) if e["total"] else None,
                **e,
            }
            for name, e in stats.items()
        },
        "caveat": (
            "Transformations imitate field conditions applied to laboratory "
            "images. Real photographs additionally bring occlusion, motion blur, "
            "overlapping leaves and unseen species, so this is a lower bound on "
            "the accuracy drop rather than an estimate of it."
        ),
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
