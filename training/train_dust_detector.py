"""Train and calibrate the dust/soil film detector used by the quality gate.

Why synthetic positives
-----------------------
No public dataset photographs dusty leaves, so positives are made: a film of
dust is laid over real PlantVillage leaves. To stop the detector learning one
generator's fingerprint the film is varied on every axis that varies in the
field - colour (grey road dust to red-brown soil), thickness, patchiness,
coverage (leaf only or whole frame), grain density and size, and blur.

Why field-like negatives
------------------------
A detector that only ever saw clean grey-card leaves would learn "anything
unusual is dust" and fire on every real photograph. So clean leaves are also
passed through the transforms that make a field photo unlike a lab photo -
soil backgrounds, harsh directional light, distance, heavy JPEG, white-balance
drift - and must *not* be flagged. Powdery mildew leaves, which genuinely carry
a pale coating, are clean negatives too and their false-alarm rate is
reported separately.

Evaluation
----------
* fitted on the dataset's ``train`` folder, evaluated on ``valid``;
* a leave-crops-out check refits without four crops (including both powdery
  mildew crops) and evaluates on those crops only, which measures whether the
  detector transfers to leaves it has never seen.

The operating points are chosen on held-out clean leaves (field transforms
included): ``threshold`` at a 3% false-alarm rate, ``heavy_threshold`` at 0.5%.

    python training/train_dust_detector.py --per-class 40
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import settings  # noqa: E402
from app.ml.dust import FEATURE_NAMES, DustDetector, dust_features  # noqa: E402

HELD_OUT_CROPS = ("Squash", "Cherry_(including_sour)", "Corn_(maize)", "Grape")
TARGET_FPR = 0.03
HEAVY_FPR = 0.005


# --------------------------------------------------------------------------- #
# Image generators
# --------------------------------------------------------------------------- #


def leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """Rough leaf segmentation for grey-card PlantVillage images."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 1] > 45) | ((hsv[..., 0] >= 15) & (hsv[..., 0] <= 95)
                                  & (hsv[..., 1] > 25))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2)


def soil_texture(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    height, width = shape
    base = np.array([rng.uniform(70, 150), rng.uniform(55, 120), rng.uniform(35, 95)])
    field = rng.normal(0, rng.uniform(12, 30), (height, width, 3))
    field = cv2.GaussianBlur(field, (0, 0), rng.uniform(0.8, 2.0))
    clumps = cv2.GaussianBlur(rng.normal(0, 25, (height, width)), (0, 0), 6)[..., None]
    return np.clip(base + field + clumps, 0, 255)


def on_soil(rgb: np.ndarray, rng: np.random.Generator, scale: float | None = None) -> np.ndarray:
    """Leaf cut out and placed on a soil backdrop, optionally smaller in frame."""
    height, width = rgb.shape[:2]
    canvas = soil_texture((height, width), rng)
    scale = scale or rng.uniform(0.55, 1.0)
    small = cv2.resize(rgb, (int(width * scale), int(height * scale)))
    mask = leaf_mask(small)[..., None]
    y = int(rng.integers(0, height - small.shape[0] + 1))
    x = int(rng.integers(0, width - small.shape[1] + 1))
    region = canvas[y:y + small.shape[0], x:x + small.shape[1]]
    canvas[y:y + small.shape[0], x:x + small.shape[1]] = region * (1 - mask) + small * mask
    return canvas.astype(np.uint8)


def harsh_light(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    width = rgb.shape[1]
    gradient = np.linspace(rng.uniform(1.2, 1.6), rng.uniform(0.45, 0.8), width)
    if rng.random() < 0.5:
        gradient = gradient[::-1]
    out = rgb.astype(np.float64) * gradient[None, :, None]
    mean = out.mean()
    out = (out - mean) * rng.uniform(1.0, 1.35) + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=int(rng.integers(12, 60)))
    return np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))


def photometric(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Exposure, contrast, white balance and mild saturation drift."""
    out = rgb.astype(np.float64)
    out *= rng.uniform(0.7, 1.3)
    mean = out.mean()
    out = (out - mean) * rng.uniform(0.75, 1.3) + mean
    out *= np.array([rng.uniform(0.9, 1.12), 1.0, rng.uniform(0.85, 1.1)])
    gray = out.mean(axis=2, keepdims=True)
    out = gray + (out - gray) * rng.uniform(0.75, 1.25)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 1.4))
    return out


def field_transform(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One random departure from lab conditions - the false-alarm stress test."""
    choice = int(rng.integers(0, 6))
    if choice == 0:
        return on_soil(rgb, rng)
    if choice == 1:
        return on_soil(rgb, rng, scale=rng.uniform(0.35, 0.55))
    if choice == 2:
        return harsh_light(rgb, rng)
    if choice == 3:
        return jpeg(rgb, rng)
    if choice == 4:
        return photometric(rgb, rng)
    return jpeg(harsh_light(on_soil(rgb, rng), rng), rng)


def add_dust(rgb: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """A film of dust over the leaf (or the whole frame), plus grains."""
    height, width = rgb.shape[:2]
    out = rgb.astype(np.float64)
    tone = rng.uniform(0, 1)
    # Interpolate between grey road dust and warm field soil, then jitter.
    grey_dust = np.array([170, 165, 155])
    soil_dust = np.array([185, 150, 110])
    colour = grey_dust * (1 - tone) + soil_dust * tone + rng.normal(0, 10, 3)
    colour *= rng.uniform(0.8, 1.15)

    patch = cv2.GaussianBlur(rng.uniform(0, 1, (height, width)), (0, 0), rng.uniform(6, 30))
    patch = (patch - patch.min()) / (np.ptp(patch) + 1e-9)
    unevenness = rng.uniform(0.2, 0.8)
    alpha = strength * rng.uniform(0.45, 0.75) * ((1 - unevenness) + unevenness * patch)
    if rng.random() < 0.5:
        alpha = alpha * leaf_mask(rgb)
    alpha = alpha[..., None]
    out = out * (1 - alpha) + colour * alpha

    density = strength * rng.uniform(0.01, 0.08)
    grains = rng.random((height, width)) < density
    size = int(rng.integers(1, 4))
    if size > 1:
        grains = cv2.dilate(grains.astype(np.uint8), np.ones((size, size), np.uint8)) > 0
    grain_colour = colour * rng.uniform(0.85, 1.25)
    out[grains] = out[grains] * 0.3 + grain_colour * 0.7
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.0))


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #


def dataset_split_dir(split: str) -> Path:
    record = json.loads((PROJECT_ROOT / "data" / "dataset_path.json").read_text(encoding="utf-8"))
    root = Path(record["path"])
    for candidate in root.rglob(split):
        if candidate.is_dir() and any(child.is_dir() for child in candidate.iterdir()):
            return candidate
    raise SystemExit(f"No '{split}' folder under {root}")


def build(split: str, per_class: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    root = dataset_split_dir(split)
    feats, labels, crops, kinds, classes = [], [], [], [], []

    def add(rgb, label, crop, kind, cls):
        feats.append(dust_features(rgb))
        labels.append(label)
        crops.append(crop)
        kinds.append(kind)
        classes.append(cls)

    for class_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        crop = class_dir.name.split("___")[0]
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        picks = rng.choice(len(files), size=min(per_class, len(files)), replace=False)
        for index in picks:
            rgb = np.asarray(Image.open(files[index]).convert("RGB"))
            add(rgb, 0, crop, "clean", class_dir.name)
            add(field_transform(rgb, rng), 0, crop, "clean_field", class_dir.name)
            strength = float(rng.uniform(0.25, 1.0))
            kind = "dust_light" if strength < 0.45 else "dust_moderate" if strength < 0.7 else "dust_heavy"
            add(add_dust(rgb, rng, strength), 1, crop, kind, class_dir.name)
            # Dust photographed in the field: the film plus a field transform.
            strength = float(rng.uniform(0.25, 1.0))
            kind = "dust_light" if strength < 0.45 else "dust_moderate" if strength < 0.7 else "dust_heavy"
            add(field_transform(add_dust(rgb, rng, strength), rng), 1, crop, kind + "_field",
                class_dir.name)
    return {"X": np.asarray(feats), "y": np.asarray(labels), "crop": np.asarray(crops),
            "kind": np.asarray(kinds), "cls": np.asarray(classes)}


def fit(X: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, MLPClassifier]:
    # Classes are balanced by construction (two clean and two dusty rows per
    # source image), so no class weighting is needed.
    scaler = StandardScaler().fit(X)
    model = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=3000, random_state=0)
    model.fit(scaler.transform(X), y)
    return scaler, model


def threshold_at_fpr(negative_scores: np.ndarray, fpr: float) -> float:
    return float(np.quantile(negative_scores, 1 - fpr))


def rates(scores: np.ndarray, data: dict, threshold: float) -> dict:
    out = {}
    for kind in sorted(set(data["kind"])):
        mask = data["kind"] == kind
        out[kind] = {"n": int(mask.sum()), "flagged": round(float((scores[mask] >= threshold).mean()), 4)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--eval-per-class", type=int, default=20)
    parser.add_argument("--rebuild", action="store_true", help="Ignore cached features.")
    args = parser.parse_args()

    started = time.time()
    # Feature extraction is the slow part (~4 minutes); cache it so changing the
    # model or the operating points does not regenerate every image.
    cache = settings.project_root / "artifacts" / "features" / (
        f"dust_features_{args.per_class}_{args.eval_per_class}.npz")
    if cache.is_file() and not args.rebuild:
        stored = np.load(cache)
        train = {k[6:]: stored[k] for k in stored.files if k.startswith("train_")}
        test = {k[5:]: stored[k] for k in stored.files if k.startswith("test_")}
        print(f"Loaded cached features from {cache.name}")
    else:
        print("Building training set ...", flush=True)
        train = build("train", args.per_class, seed=11)
        print(f"  {len(train['y'])} rows in {time.time() - started:.0f}s", flush=True)
        print("Building evaluation set ...", flush=True)
        test = build("valid", args.eval_per_class, seed=23)
        print(f"  {len(test['y'])} rows", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **{f"train_{k}": v for k, v in train.items()},
                 **{f"test_{k}": v for k, v in test.items()})

    # ---- leave-crops-out transfer check ---------------------------------
    keep = ~np.isin(train["crop"], HELD_OUT_CROPS)
    scaler_lco, model_lco = fit(train["X"][keep], train["y"][keep])
    held = np.isin(test["crop"], HELD_OUT_CROPS)
    lco_scores = model_lco.predict_proba(scaler_lco.transform(test["X"][held]))[:, 1]
    lco_auc = roc_auc_score(test["y"][held], lco_scores)

    # ---- final model ----------------------------------------------------
    scaler, model = fit(train["X"], train["y"])
    scores = model.predict_proba(scaler.transform(test["X"]))[:, 1]
    auc = roc_auc_score(test["y"], scores)
    negatives = scores[test["y"] == 0]
    threshold = threshold_at_fpr(negatives, TARGET_FPR)
    heavy = max(threshold_at_fpr(negatives, HEAVY_FPR), threshold + 1e-6)

    powdery = (test["y"] == 0) & (np.char.find(test["cls"].astype(str), "Powdery") >= 0)
    report = {
        "auc": round(float(auc), 4),
        "leave_crops_out": {"crops": list(HELD_OUT_CROPS), "auc": round(float(lco_auc), 4)},
        "threshold": {"value": threshold, "target_false_alarm_rate": TARGET_FPR,
                      "rates": rates(scores, test, threshold)},
        "heavy_threshold": {"value": heavy, "target_false_alarm_rate": HEAVY_FPR,
                            "rates": rates(scores, test, heavy)},
        "powdery_mildew_false_alarm": {
            "n": int(powdery.sum()),
            "at_threshold": round(float((scores[powdery] >= threshold).mean()), 4),
            "at_heavy": round(float((scores[powdery] >= heavy).mean()), 4),
        },
        "train_rows": int(len(train["y"])),
        "eval_rows": int(len(test["y"])),
    }

    payload = {
        "description": "30-32-1 ReLU network over dust_features(); see app/ml/dust.py.",
        "features": list(FEATURE_NAMES),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "layers": [{"weights": w.tolist(), "bias": b.tolist()}
                   for w, b in zip(model.coefs_, model.intercepts_, strict=False)],
        "threshold": threshold,
        "heavy_threshold": heavy,
        "evaluation": report,
        "caveat": ("Positives are synthetic dust films on real leaves; no dataset of "
                   "photographed dusty leaves exists. Rates are on held-out images."),
    }
    # The server evaluates the network in numpy, not sklearn. Prove the two agree
    # before writing a file the server will trust.
    served = DustDetector(payload)
    served_p = 1 / (1 + np.exp(-np.array([served.logit(x) for x in test["X"]])))
    gap = float(np.abs(served_p - scores).max())
    if gap > 1e-6:
        raise SystemExit(f"numpy evaluation disagrees with sklearn by {gap:.2e}")
    report["numpy_vs_sklearn_max_gap"] = gap
    payload["evaluation"] = report

    out = settings.metadata_dir / "dust_detector.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}  ({time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
