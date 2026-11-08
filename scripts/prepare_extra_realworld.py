"""Add two more real-world photo sources to the mixed fine-tuning set.

Run after ``scripts/prepare_plantdoc.py``. Sources, and how their labels map:

* **Plant Pathology 2021** (Thapa et al., Cornell; FGVC8 challenge; the
  Hugging Face mirror ``timm/plant-pathology-2021``, Apache-2.0 there). Real
  orchard photographs of apple leaves. Only single-label images are used:
  ``scab`` -> apple scab, ``rust`` -> cedar apple rust, ``healthy`` -> healthy
  apple, ``frog_eye_leaf_spot`` -> apple black rot (frogeye leaf spot is the
  leaf phase of black rot; both are *Botryosphaeria obtusa*). ``complex`` and
  ``powdery_mildew`` have no counterpart among the model's classes and are
  left out.
* **CCMT** (Mensah Kwabena et al., Mendeley Data 2023, CC BY 4.0). Photographs
  from farms in Ghana, validated by plant virologists. Only classes that map
  unambiguously are used: tomato healthy, tomato Septoria leaf spot, maize
  healthy, and tomato leaf curl -> tomato yellow leaf curl virus (the
  begomovirus disease the name describes; an assumption stated in the docs).
  CCMT's "leaf blight" and "leaf spot" could each be more than one of the
  model's classes and are left out rather than guessed.

Each photo is resized to at most 512 px on its long side - training crops to
224 px, and decoding 4000-px orchard photos every epoch would starve the GPU.
Photos whose perceptual hash matches a PlantDoc test photo are dropped, so the
real-world benchmark stays unseen. Each source gets its own stratified
train/val/test split (70/10/20); its test split is reported separately as a
*same-source* score, which is easier than PlantDoc's cross-source test.

    python scripts/prepare_extra_realworld.py
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402
from prepare_plantdoc import hamming, perceptual_hash  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.ml.datasets import ManifestEntry, read_manifest, write_manifest  # noqa: E402

PLANT_PATHOLOGY = {
    "scab": "Apple___Apple_scab",
    "rust": "Apple___Cedar_apple_rust",
    "healthy": "Apple___healthy",
    "frog_eye_leaf_spot": "Apple___Black_rot",
}
CCMT_ROOT = "Dataset for Crop Pest and Disease Detection/Raw Data/CCMT Dataset"
CCMT = {
    "Tomato/healthy": "Tomato___healthy",
    "Tomato/septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato/leaf curl": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Maize/healthy": "Corn_(maize)___healthy",
}
# PlantWild (Wei et al., ACM MM 2024; huggingface.co/datasets/uqtwei2/PlantWild,
# CC BY-NC-ND 4.0): photographs gathered from search engines and curated against
# plant-pathology references. Its "<plant> leaf" folders are healthy leaves.
# "grape leaf spot" is left out: it could be Isariopsis leaf spot or another.
PLANTWILD_ROOT = "plantwild/images"
PLANTWILD = {
    "apple black rot": "Apple___Black_rot",
    "apple leaf": "Apple___healthy",
    "apple rust": "Apple___Cedar_apple_rust",
    "apple scab": "Apple___Apple_scab",
    "bell pepper leaf": "Pepper,_bell___healthy",
    "bell pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "blueberry leaf": "Blueberry___healthy",
    "cherry leaf": "Cherry_(including_sour)___healthy",
    "cherry powdery mildew": "Cherry_(including_sour)___Powdery_mildew",
    "citrus greening disease": "Orange___Haunglongbing_(Citrus_greening)",
    "corn gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "corn leaf": "Corn_(maize)___healthy",
    "corn northern leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "corn rust": "Corn_(maize)___Common_rust_",
    "grape black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
    "peach leaf": "Peach___healthy",
    "potato early blight": "Potato___Early_blight",
    "potato late blight": "Potato___Late_blight",
    "potato leaf": "Potato___healthy",
    "raspberry leaf": "Raspberry___healthy",
    "soybean leaf": "Soybean___healthy",
    "squash powdery mildew": "Squash___Powdery_mildew",
    "strawberry leaf": "Strawberry___healthy",
    "strawberry leaf scorch": "Strawberry___Leaf_scorch",
    "tomato bacterial leaf spot": "Tomato___Bacterial_spot",
    "tomato early blight": "Tomato___Early_blight",
    "tomato late blight": "Tomato___Late_blight",
    "tomato leaf": "Tomato___healthy",
    "tomato leaf mold": "Tomato___Leaf_Mold",
    "tomato mosaic virus": "Tomato___Tomato_mosaic_virus",
    "tomato septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "tomato yellow leaf curl virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}
# Potato leaves photographed on farms in Central Java (Shabrina et al., Data in
# Brief 2024; Mendeley ptz377bwb8, CC BY 4.0). "Fungi" spans several diseases
# and is left out; Phytophthora is late blight.
POTATO_ROOT = "Potato Leaf Disease Dataset in Uncontrolled Environment"
POTATO = {
    "Healthy": "Potato___healthy",
    "Phytopthora": "Potato___Late_blight",
}
MAX_SIDE = 512


def save_resized(image: Image.Image, destination: Path) -> None:
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=92)


def plant_pathology(source_dir: Path, out_dir: Path) -> tuple[list[tuple[Path, str]], Counter]:
    rows, skipped = [], Counter()
    for shard in sorted(source_dir.glob("*.parquet")):
        table = pq.read_table(shard, columns=["image", "label_names", "image_id"]).to_pylist()
        for row in table:
            names = row["label_names"]
            if len(names) != 1 or names[0] not in PLANT_PATHOLOGY:
                skipped["multi-label" if len(names) != 1 else names[0]] += 1
                continue
            target = PLANT_PATHOLOGY[names[0]]
            destination = out_dir / target / f"{Path(row['image_id']).stem}.jpg"
            if not destination.exists():
                save_resized(Image.open(io.BytesIO(row["image"]["bytes"])), destination)
            rows.append((destination, target))
    return rows, skipped


def folders(source_dir: Path, root: str, mapping: dict, out_dir: Path) -> list[tuple[Path, str]]:
    """Photos laid out as <root>/<folder>/<image>, one folder per class."""
    rows = []
    for folder, target in mapping.items():
        prefix = folder.replace("/", "_").replace(" ", "_")
        for path in sorted((source_dir / root / folder).glob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            destination = out_dir / target / f"{prefix}__{path.stem}.jpg"
            if not destination.exists():
                try:
                    with Image.open(path) as image:
                        save_resized(image, destination)
                except Exception:  # noqa: BLE001 - unreadable file: skip it
                    continue
            rows.append((destination, target))
    return rows


def split(rows, class_to_idx, source, rng, cap):
    """Stratified 70/10/20 split, with the training part capped per class."""
    by_class = defaultdict(list)
    for path, name in rows:
        by_class[name].append(path)
    parts = {"train": [], "val": [], "test": []}
    for name in sorted(by_class):
        paths = by_class[name][:]
        rng.shuffle(paths)
        n_test, n_val = round(len(paths) * 0.2), round(len(paths) * 0.1)
        groups = {"test": paths[:n_test], "val": paths[n_test:n_test + n_val],
                  "train": paths[n_test + n_val:][:cap]}
        for part, chosen in groups.items():
            parts[part] += [ManifestEntry(str(p), class_to_idx[name], name, source) for p in chosen]
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plantpathology", type=Path, default=PROJECT_ROOT / "_datasets" / "plantpathology2021")
    parser.add_argument("--ccmt", type=Path, default=PROJECT_ROOT / "_datasets" / "ccmt")
    parser.add_argument("--plantwild", type=Path, default=PROJECT_ROOT / "_datasets" / "plantwild")
    parser.add_argument("--potato", type=Path, default=PROJECT_ROOT / "_datasets" / "potato_field")
    parser.add_argument("--mix-name", default="realmix3",
                        help="Name of the combined training/validation splits to write.")
    parser.add_argument("--train-cap", type=int, default=500,
                        help="Most training photos per class per source, so one large class "
                             "(CCMT Septoria: 2,743) cannot dominate.")
    parser.add_argument("--dedup-distance", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    split_config = json.loads(settings.splits_path.read_text(encoding="utf-8"))
    if "plantdoc_test" not in split_config["splits"]:
        raise SystemExit("Run scripts/prepare_plantdoc.py first.")
    class_to_idx = split_config["class_to_idx"]

    plantdoc_test = read_manifest(Path(split_config["splits"]["plantdoc_test"]["manifest"]))
    guard = np.array([perceptual_hash(Path(e.path)) for e in plantdoc_test], dtype=np.uint64)

    sources = {}
    pp_rows, pp_skipped = plant_pathology(args.plantpathology, args.plantpathology / "images")
    sources["plantpathology2021"] = pp_rows
    sources["ccmt"] = folders(args.ccmt, CCMT_ROOT, CCMT, args.ccmt / "images")
    if (args.plantwild / PLANTWILD_ROOT).is_dir():
        sources["plantwild"] = folders(args.plantwild, PLANTWILD_ROOT, PLANTWILD,
                                       args.plantwild / "resized")
    if (args.potato / POTATO_ROOT).is_dir():
        sources["potato_field"] = folders(args.potato, POTATO_ROOT, POTATO, args.potato / "resized")

    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "sources": {}}
    new_splits: dict[str, list[ManifestEntry]] = {}
    for source, rows in sources.items():
        kept, seen, exact_dupes, leaked = [], set(), 0, 0
        for path, name in rows:
            h = perceptual_hash(path)
            if h is None:
                continue
            if h in seen:
                exact_dupes += 1
                continue
            if (hamming(guard, h) <= args.dedup_distance).any():
                leaked += 1
                continue
            seen.add(h)
            kept.append((path, name))
        parts = split(kept, class_to_idx, source, rng, args.train_cap)
        for part, entries in parts.items():
            new_splits[f"{source}_{part}"] = entries
        summary["sources"][source] = {
            "photos_used": len(kept),
            "duplicates_removed": exact_dupes,
            "removed_as_plantdoc_test_near_duplicates": leaked,
            "per_class": dict(sorted(Counter(n for _, n in kept).items())),
            "splits": {part: len(entries) for part, entries in parts.items()},
        }
        print(f"{source}: {len(kept)} photos ({exact_dupes} duplicates, {leaked} matching PlantDoc "
              f"test removed) -> " + ", ".join(f"{k} {len(v)}" for k, v in parts.items()))
    summary["sources"]["plantpathology2021"]["skipped_labels"] = dict(pp_skipped)

    base_train = read_manifest(Path(split_config["splits"]["realmix_train"]["manifest"]))
    base_val = read_manifest(Path(split_config["splits"]["realmix_val"]["manifest"]))
    mix_train, mix_val = f"{args.mix_name}_train", f"{args.mix_name}_val"
    new_splits[mix_train] = base_train + [e for s in sources for e in new_splits[f"{s}_train"]]
    new_splits[mix_val] = base_val + [e for s in sources for e in new_splits[f"{s}_val"]]
    rng.shuffle(new_splits[mix_train])

    manifests = settings.project_root / "data" / "manifests"
    for name, rows in new_splits.items():
        path = manifests / f"{name}.csv"
        write_manifest(rows, path)
        counts = Counter(r.class_name for r in rows)
        split_config["splits"][name] = {"manifest": str(path), "count": len(rows),
                                        "class_counts": dict(sorted(counts.items()))}
    print(f"{mix_train} {len(new_splits[mix_train])}, {mix_val} {len(new_splits[mix_val])}")

    split_config.setdefault("external_sources", {}).update({
        "plantpathology2021": {
            "citation": "Thapa et al., Plant Pathology 2021 - FGVC8 (Cornell)",
            "url": "https://huggingface.co/datasets/timm/plant-pathology-2021",
            "licence": "Apache-2.0 (Hugging Face mirror)", "mapping": PLANT_PATHOLOGY},
        "ccmt": {
            "citation": "Mensah Kwabena et al., Dataset for Crop Pest and Disease Detection, "
                        "Mendeley Data v1, 2023 (doi:10.17632/bwh3zbpkpv.1)",
            "url": "https://data.mendeley.com/datasets/bwh3zbpkpv/1",
            "licence": "CC BY 4.0", "mapping": CCMT},
        "plantwild": {
            "citation": "Wei et al., Benchmarking In-the-Wild Multimodal Plant Disease "
                        "Recognition and A Versatile Baseline, ACM MM 2024",
            "url": "https://huggingface.co/datasets/uqtwei2/PlantWild",
            "licence": "CC BY-NC-ND 4.0 (non-commercial)", "mapping": PLANTWILD},
        "potato_field": {
            "citation": "Shabrina et al., A novel dataset of potato leaf disease in uncontrolled "
                        "environment, Data in Brief 2024 (doi:10.17632/ptz377bwb8.1)",
            "url": "https://data.mendeley.com/datasets/ptz377bwb8/1",
            "licence": "CC BY 4.0", "mapping": POTATO},
    })
    settings.splits_path.write_text(json.dumps(split_config, indent=2), encoding="utf-8")
    summary[args.mix_name] = {"train": len(new_splits[mix_train]), "val": len(new_splits[mix_val]),
                              "train_cap_per_class": args.train_cap, "sources": sorted(sources)}
    (settings.results_dir / "realworld_extra_preparation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
