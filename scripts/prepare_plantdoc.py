"""Prepare PlantDoc real-world photographs for evaluation and fine-tuning.

PlantDoc (Singh et al., CODS-COMAD 2020; CC-BY-4.0) is 2,598 photographs of
diseased and healthy leaves taken in fields or collected from the web - the
kind of picture a user actually uploads, and the thing PlantVillage's
laboratory photographs are not. Every one of its classes corresponds to a
PlantVillage class, so the model's 38 outputs do not change.

This script:

1. maps PlantDoc's folder names onto the model's class names;
2. drops unreadable files;
3. **removes leakage**: any training photo whose perceptual hash is within a
   small Hamming distance of a test photo is dropped from training (PlantDoc
   was assembled from web searches, and near-identical copies land on both
   sides of its split). The official test split is kept intact;
4. holds out a stratified validation split from PlantDoc's training photos;
5. builds a mixed training set - a stratified PlantVillage subset (so the ten
   classes PlantDoc lacks are not forgotten) plus the PlantDoc training photos
   repeated ``--oversample`` times - and a matching validation set;
6. writes the manifests and registers the splits in ``data/splits.json``.

    python scripts/prepare_plantdoc.py --root _datasets/plantdoc
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.ml.datasets import ManifestEntry, read_manifest, write_manifest  # noqa: E402

# PlantDoc folder -> PlantVillage class. PlantDoc names a healthy leaf by the
# plant alone ("Apple leaf"); "Corn leaf blight" is northern leaf blight and
# "Tomato leaf yellow virus" is yellow leaf curl virus, as in the PlantDoc paper.
MAPPING = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf": "Tomato___healthy",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def perceptual_hash(path: Path) -> int | None:
    """64-bit DCT perceptual hash (pHash), or None if the file is unreadable."""
    try:
        with Image.open(path) as image:
            gray = np.asarray(image.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float32)
    except Exception:  # noqa: BLE001 - any decode failure means "skip this file"
        return None
    low = cv2.dct(gray)[:8, :8]
    bits = (low > np.median(low)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: np.ndarray, b: int) -> np.ndarray:
    x = np.bitwise_xor(a, np.uint64(b))
    # popcount of uint64 via byte view
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def collect(split_dir: Path, class_to_idx: dict) -> list[tuple[Path, str, int]]:
    rows = []
    for folder in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        target = MAPPING.get(folder.name)
        if target is None:
            raise SystemExit(f"Unmapped PlantDoc folder: {folder.name!r}")
        if target not in class_to_idx:
            raise SystemExit(f"Mapped class {target!r} is not one of the model's classes")
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                rows.append((path, target, class_to_idx[target]))
    return rows


def stratified_take(entries: list[ManifestEntry], per_class: int, rng: random.Random) -> list[ManifestEntry]:
    by_class = defaultdict(list)
    for entry in entries:
        by_class[entry.label].append(entry)
    taken = []
    for label in sorted(by_class):
        rows = by_class[label]
        taken += rng.sample(rows, min(per_class, len(rows)))
    return taken


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "_datasets" / "plantdoc")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--dedup-distance", type=int, default=6,
                        help="Hamming distance on the 64-bit pHash treated as the same photo.")
    parser.add_argument("--pv-train-per-class", type=int, default=300,
                        help="PlantVillage training photos per class in the mixed training set.")
    # Small on purpose: early stopping and temperature fitting use this split,
    # and with 40 lab photos per class the ~300 real photos would barely move
    # the score that picks the best epoch. Lab accuracy is still checked on the
    # full 17,572-image PlantVillage test split.
    parser.add_argument("--pv-val-per-class", type=int, default=10)
    parser.add_argument("--oversample", type=int, default=4,
                        help="Times each PlantDoc training photo appears per epoch.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    split_config = json.loads(settings.splits_path.read_text(encoding="utf-8"))
    class_to_idx = split_config["class_to_idx"]
    root = args.root.resolve()

    raw_train = collect(root / "train", class_to_idx)
    raw_test = collect(root / "test", class_to_idx)
    print(f"PlantDoc files: train {len(raw_train)}, test {len(raw_test)}")

    def hashed(rows):
        kept, unreadable = [], 0
        for path, name, label in rows:
            h = perceptual_hash(path)
            if h is None:
                unreadable += 1
                continue
            kept.append((path, name, label, h))
        return kept, unreadable

    train, bad_train = hashed(raw_train)
    test, bad_test = hashed(raw_test)

    # Exact duplicates inside the test split would count the same photo twice.
    seen, test_unique, test_dupes = set(), [], 0
    for row in test:
        if row[3] in seen:
            test_dupes += 1
            continue
        seen.add(row[3])
        test_unique.append(row)
    test = test_unique

    test_hashes = np.array([row[3] for row in test], dtype=np.uint64)
    clean_train, leaked, conflicting = [], 0, 0
    for path, name, label, h in train:
        close = hamming(test_hashes, h) <= args.dedup_distance
        if close.any():
            leaked += 1
            if any(test[i][2] != label for i in np.flatnonzero(close)):
                conflicting += 1
            continue
        clean_train.append((path, name, label, h))
    print(f"unreadable: train {bad_train}, test {bad_test}; duplicate test photos removed: "
          f"{test_dupes}; training photos removed as near-duplicates of test photos: {leaked} "
          f"({conflicting} of them with a different label)")

    # Stratified validation hold-out from the clean training photos.
    by_class = defaultdict(list)
    for row in clean_train:
        by_class[row[2]].append(row)
    pd_train, pd_val = [], []
    for label in sorted(by_class):
        rows = by_class[label][:]
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * args.val_fraction)) if len(rows) >= 4 else 0
        pd_val += rows[:n_val]
        pd_train += rows[n_val:]

    def entries(rows):
        return [ManifestEntry(path=str(p), label=label, class_name=name, source="plantdoc")
                for p, name, label, _ in rows]

    pd_train_e, pd_val_e, pd_test_e = entries(pd_train), entries(pd_val), entries(test)

    pv_train = read_manifest(Path(split_config["splits"]["train"]["manifest"]))
    pv_val = read_manifest(Path(split_config["splits"]["val"]["manifest"]))
    mixed_train = stratified_take(pv_train, args.pv_train_per_class, rng) + pd_train_e * args.oversample
    mixed_val = stratified_take(pv_val, args.pv_val_per_class, rng) + pd_val_e
    rng.shuffle(mixed_train)

    manifests = settings.project_root / "data" / "manifests"
    new_splits = {
        "plantdoc_train": pd_train_e,
        "plantdoc_val": pd_val_e,
        "plantdoc_test": pd_test_e,
        "realmix_train": mixed_train,
        "realmix_val": mixed_val,
    }
    for name, rows in new_splits.items():
        path = manifests / f"{name}.csv"
        write_manifest(rows, path)
        counts = Counter(r.class_name for r in rows)
        split_config["splits"][name] = {"manifest": str(path), "count": len(rows),
                                        "class_counts": dict(sorted(counts.items()))}
        print(f"  {name:15s} {len(rows):6d} images, {len(counts)} classes")

    split_config.setdefault("external_sources", {})["plantdoc"] = {
        "citation": "Singh et al., PlantDoc: A Dataset for Visual Plant Disease Detection, "
                    "CODS-COMAD 2020",
        "url": "https://github.com/pratikkayal/PlantDoc-Dataset",
        "licence": "CC-BY-4.0",
        "mapping": MAPPING,
    }
    settings.splits_path.write_text(json.dumps(split_config, indent=2), encoding="utf-8")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": split_config["external_sources"]["plantdoc"],
        "files": {"train": len(raw_train), "test": len(raw_test)},
        "unreadable": {"train": bad_train, "test": bad_test},
        "duplicate_test_photos_removed": test_dupes,
        "train_photos_removed_as_test_near_duplicates": leaked,
        "of_which_labelled_differently": conflicting,
        "dedup_hamming_distance": args.dedup_distance,
        "splits": {name: len(rows) for name, rows in new_splits.items()},
        "plantdoc_classes_in_test": len({r.label for r in pd_test_e}),
        "mixed_training": {"plantvillage_per_class": args.pv_train_per_class,
                           "plantdoc_oversample": args.oversample},
    }
    out = settings.results_dir / "plantdoc_preparation.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Registered {len(new_splits)} splits in {settings.splits_path.name}; summary {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
