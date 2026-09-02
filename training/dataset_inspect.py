"""Discover the downloaded dataset's structure and write ``artifacts/dataset_report.json``.

Nothing about the dataset layout is assumed. The script walks the download
directory, finds every folder that looks like an ImageFolder-style split
(``<split>/<class>/<image>``), and derives class names, per-class counts, image
dimensions, corrupt files and class imbalance from the files that are actually
present.

Run:
    python training/dataset_inspect.py
    python training/dataset_inspect.py --sample-per-class 40 --hash-per-class 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import app.core.runtime  # noqa: F401  (sets OMP env before numpy/torch)  # isort:skip

import numpy as np
from PIL import Image, ImageFile

from app.core.config import settings
from app.core.logging import configure_logging, get_logger

# Truncated JPEGs should raise during verification, not be silently padded.
ImageFile.LOAD_TRUNCATED_IMAGES = False

log = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Structure discovery
# --------------------------------------------------------------------------- #


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def find_split_dirs(root: Path) -> dict[str, Path]:
    """Locate ImageFolder-style split directories under ``root``.

    A directory qualifies as a split when at least two of its immediate
    subdirectories contain image files. Kaggle archives frequently nest the real
    data one or two levels deep (and sometimes duplicate it), so we score every
    candidate and keep the deepest, largest one per split name.
    """
    candidates: dict[str, list[tuple[int, Path]]] = defaultdict(list)

    for dirpath, dirnames, _filenames in os.walk(root):
        current = Path(dirpath)
        # Skip macOS resource forks that Kaggle archives often carry.
        dirnames[:] = [d for d in dirnames if d not in {"__MACOSX", ".ipynb_checkpoints"}]
        if not dirnames:
            continue

        class_like = 0
        image_total = 0
        for name in dirnames:
            child = current / name
            try:
                found = sum(1 for entry in os.scandir(child) if _is_image(Path(entry.name)))
            except OSError:
                continue
            if found:
                class_like += 1
                image_total += found
        if class_like >= 2:
            candidates[current.name.lower()].append((image_total, current))

    splits: dict[str, Path] = {}
    for name, entries in candidates.items():
        entries.sort(key=lambda item: (item[0], len(item[1].parts)))
        splits[name] = entries[-1][1]

    # Normalise the split names we care about.
    normalised: dict[str, Path] = {}
    for key, path in splits.items():
        if key.startswith("train"):
            normalised["train"] = path
        elif key.startswith("valid") or key.startswith("val"):
            normalised["valid"] = path
        elif key.startswith("test"):
            normalised["test"] = path
        else:
            normalised[key] = path
    return normalised


def find_flat_image_dirs(root: Path, exclude: list[Path]) -> dict[str, int]:
    """Find directories holding loose images that are *not* part of a labelled split.

    The Kaggle archive ships a small unlabelled ``test`` folder in this form; it
    is useful as a smoke-test corpus for the API but must never be treated as a
    labelled split. Class directories inside a discovered split also look "flat",
    so they are excluded explicitly.
    """
    excluded = [str(p.resolve()) for p in exclude]
    flat: dict[str, int] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        if "__MACOSX" in dirpath:
            continue
        resolved = str(Path(dirpath).resolve())
        if any(resolved.startswith(e) for e in excluded):
            continue
        images = [f for f in filenames if _is_image(Path(f))]
        if not images:
            continue
        has_image_subdirs = False
        for name in dirnames:
            child = Path(dirpath) / name
            try:
                if any(_is_image(Path(entry.name)) for entry in os.scandir(child)):
                    has_image_subdirs = True
                    break
            except OSError:
                continue
        if not has_image_subdirs:
            flat[str(Path(dirpath))] = len(images)
    return flat


# --------------------------------------------------------------------------- #
# Per-split statistics
# --------------------------------------------------------------------------- #


def _probe_image(path: Path) -> dict | None:
    """Open one image and return its geometry, or ``None`` if it is unreadable."""
    try:
        with Image.open(path) as img:
            img.verify()  # cheap structural check; invalidates the handle
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
            fmt = img.format
        return {
            "width": width,
            "height": height,
            "mode": mode,
            "format": fmt,
            "bytes": path.stat().st_size,
        }
    except Exception:  # noqa: BLE001 - any failure means "unusable image"
        return None


def scan_split(split_dir: Path, sample_per_class: int, workers: int) -> dict:
    """Count images per class and probe a sample of them for geometry/corruption."""
    classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    per_class: dict[str, int] = {}
    sampled: list[Path] = []
    extensions: Counter = Counter()

    for class_name in classes:
        files = [p for p in (split_dir / class_name).iterdir() if p.is_file() and _is_image(p)]
        per_class[class_name] = len(files)
        extensions.update(p.suffix.lower() for p in files)
        step = max(1, len(files) // sample_per_class) if sample_per_class else 1
        sampled.extend(files[::step][:sample_per_class])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        probes = list(pool.map(_probe_image, sampled))

    corrupt = [str(p) for p, probe in zip(sampled, probes, strict=False) if probe is None]
    ok = [p for p in probes if p is not None]
    widths = [p["width"] for p in ok]
    heights = [p["height"] for p in ok]
    sizes = [p["bytes"] for p in ok]

    total = sum(per_class.values())
    counts = np.array(list(per_class.values()), dtype=float)

    return {
        "path": str(split_dir),
        "num_classes": len(classes),
        "num_images": total,
        "class_counts": per_class,
        "extensions": dict(extensions),
        "sampled_images": len(sampled),
        "corrupt_sampled": corrupt,
        "corrupt_rate_in_sample": (len(corrupt) / len(sampled)) if sampled else 0.0,
        "image_geometry": {
            "unique_sizes": sorted({(w, h) for w, h in zip(widths, heights, strict=False)})[:20],
            "width": {"min": min(widths, default=0), "max": max(widths, default=0),
                      "mean": float(np.mean(widths)) if widths else 0.0},
            "height": {"min": min(heights, default=0), "max": max(heights, default=0),
                       "mean": float(np.mean(heights)) if heights else 0.0},
            "modes": dict(Counter(p["mode"] for p in ok)),
            "formats": dict(Counter(p["format"] for p in ok)),
            "file_bytes": {"min": int(min(sizes, default=0)), "max": int(max(sizes, default=0)),
                           "mean": float(np.mean(sizes)) if sizes else 0.0},
        },
        "imbalance": {
            "min_class": min(per_class, key=per_class.get) if per_class else None,
            "min_count": int(counts.min()) if counts.size else 0,
            "max_class": max(per_class, key=per_class.get) if per_class else None,
            "max_count": int(counts.max()) if counts.size else 0,
            "mean_count": float(counts.mean()) if counts.size else 0.0,
            "std_count": float(counts.std()) if counts.size else 0.0,
            # >1.5 is commonly treated as meaningful imbalance for CV datasets.
            "imbalance_ratio": float(counts.max() / counts.min()) if counts.size and counts.min() else 0.0,
            "coefficient_of_variation": float(counts.std() / counts.mean()) if counts.size and counts.mean() else 0.0,
        },
    }


# --------------------------------------------------------------------------- #
# Duplicate / leakage detection
# --------------------------------------------------------------------------- #


def _phash(path: Path) -> str | None:
    """Perceptual hash, robust to re-encoding but not to rotation/flip."""
    try:
        import imagehash

        with Image.open(path) as img:
            return str(imagehash.phash(img.convert("RGB"), hash_size=8))
    except Exception:  # noqa: BLE001
        return None


def hash_split(split_dir: Path, per_class: int, workers: int) -> dict[str, list[str]]:
    """Return ``{phash: [relative paths]}`` for a stratified sample of a split."""
    files: list[Path] = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        class_files = sorted(p for p in class_dir.iterdir() if p.is_file() and _is_image(p))
        step = max(1, len(class_files) // per_class) if per_class else 1
        files.extend(class_files[::step][:per_class])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashes = list(pool.map(_phash, files))

    table: dict[str, list[str]] = defaultdict(list)
    for path, digest in zip(files, hashes, strict=False):
        if digest:
            table[digest].append(f"{path.parent.name}/{path.name}")
    return table


def duplicate_analysis(train_dir: Path, valid_dir: Path | None, per_class: int, workers: int) -> dict:
    """Estimate within-split duplicates and cross-split leakage from pHash collisions.

    This is a *sampled* estimate. Reporting it as such matters: an exact
    all-pairs comparison over 87k images is not affordable here, and claiming
    "zero leakage" from a sample would be dishonest.
    """
    train_hashes = hash_split(train_dir, per_class, workers)
    train_sampled = sum(len(v) for v in train_hashes.values())
    train_dupes = {h: v for h, v in train_hashes.items() if len(v) > 1}

    result = {
        "method": "perceptual hash (pHash, 8x8 DCT), exact-hash collisions only",
        "sampled_per_class": per_class,
        "train": {
            "sampled": train_sampled,
            "duplicate_groups": len(train_dupes),
            "duplicate_images": sum(len(v) for v in train_dupes.values()) - len(train_dupes),
            "examples": {h: v[:4] for h, v in list(train_dupes.items())[:5]},
        },
    }

    if valid_dir is not None:
        valid_hashes = hash_split(valid_dir, per_class, workers)
        valid_sampled = sum(len(v) for v in valid_hashes.values())
        valid_dupes = {h: v for h, v in valid_hashes.items() if len(v) > 1}
        overlap = set(train_hashes) & set(valid_hashes)
        result["valid"] = {
            "sampled": valid_sampled,
            "duplicate_groups": len(valid_dupes),
            "duplicate_images": sum(len(v) for v in valid_dupes.values()) - len(valid_dupes),
        }
        result["cross_split_leakage"] = {
            "colliding_hashes": len(overlap),
            "leak_rate_in_sample": len(overlap) / valid_sampled if valid_sampled else 0.0,
            "examples": [
                {"hash": h, "train": train_hashes[h][:2], "valid": valid_hashes[h][:2]}
                for h in list(overlap)[:5]
            ],
        }
    return result


# --------------------------------------------------------------------------- #
# Class-name parsing
# --------------------------------------------------------------------------- #


def parse_class_name(raw: str) -> dict[str, str]:
    """Split ``Tomato___Late_blight`` into plant/condition components.

    Falls back to treating the whole string as the plant when the separator is
    absent, so an unfamiliar dataset layout degrades gracefully.
    """
    if "___" in raw:
        plant, condition = raw.split("___", 1)
    elif "__" in raw:
        plant, condition = raw.split("__", 1)
    else:
        plant, condition = raw, "unknown"

    plant_clean = plant.replace("_", " ").replace(",", ",").strip()
    condition_clean = condition.replace("_", " ").strip()
    healthy = condition_clean.lower().startswith("healthy")
    return {
        "raw": raw,
        "plant": plant_clean,
        "condition": "Healthy" if healthy else condition_clean,
        "is_healthy": healthy,
        "display": f"{plant_clean} - {'Healthy' if healthy else condition_clean}",
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-per-class", type=int, default=40,
                        help="Images probed per class for geometry/corruption checks.")
    parser.add_argument("--hash-per-class", type=int, default=60,
                        help="Images hashed per class for duplicate/leakage estimation.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dataset-path", type=str, default=None)
    args = parser.parse_args()

    configure_logging(settings.log_level, settings.logs_dir / "dataset_inspect.log")
    settings.ensure_dirs()

    root = Path(args.dataset_path) if args.dataset_path else settings.resolve_dataset_path()
    log.info("Inspecting dataset", fields={"root": str(root)})
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    splits = find_split_dirs(root)
    log.info("Discovered splits", fields={k: str(v) for k, v in splits.items()})
    if "train" not in splits:
        raise RuntimeError(
            f"No train-like split found under {root}. Discovered: {list(splits)}"
        )

    split_reports = {}
    for name, path in splits.items():
        log.info(f"Scanning split '{name}'", fields={"path": str(path)})
        split_reports[name] = scan_split(path, args.sample_per_class, args.workers)
        log.info(
            f"Split '{name}' scanned",
            fields={
                "classes": split_reports[name]["num_classes"],
                "images": split_reports[name]["num_images"],
                "corrupt_in_sample": len(split_reports[name]["corrupt_sampled"]),
            },
        )

    train_classes = sorted(split_reports["train"]["class_counts"])
    class_details = [parse_class_name(c) for c in train_classes]
    plants = sorted({d["plant"] for d in class_details})

    consistency = {}
    for name, report in split_reports.items():
        if name == "train":
            continue
        other = set(report["class_counts"])
        consistency[name] = {
            "same_class_set": other == set(train_classes),
            "missing_vs_train": sorted(set(train_classes) - other),
            "extra_vs_train": sorted(other - set(train_classes)),
        }

    log.info("Estimating duplicates and cross-split leakage (sampled pHash)")
    dupes = duplicate_analysis(
        splits["train"], splits.get("valid"), args.hash_per_class, args.workers
    )

    flat_dirs = find_flat_image_dirs(root, exclude=list(splits.values()))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "kaggle_slug": "vipoooool/new-plant-diseases-dataset",
            "root": str(root),
        },
        "splits": split_reports,
        "split_consistency": consistency,
        "classes": {
            "count": len(train_classes),
            "names": train_classes,
            "details": class_details,
            "plants": plants,
            "plant_count": len(plants),
            "healthy_classes": [d["raw"] for d in class_details if d["is_healthy"]],
            "diseased_classes": [d["raw"] for d in class_details if not d["is_healthy"]],
        },
        "duplicates": dupes,
        "unlabelled_flat_dirs": flat_dirs,
    }

    settings.dataset_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Wrote dataset report", fields={"path": str(settings.dataset_report_path)})

    train = split_reports["train"]
    print("\n" + "=" * 72)
    print("DATASET REPORT")
    print("=" * 72)
    print(f"Root                : {root}")
    print(f"Splits found        : {', '.join(split_reports)}")
    for name, rep in split_reports.items():
        print(f"  {name:<8} {rep['num_images']:>7,} images across {rep['num_classes']:>3} classes")
    print(f"Classes             : {len(train_classes)}  ({len(plants)} distinct plants)")
    print(f"Healthy classes     : {len(report['classes']['healthy_classes'])}")
    print(f"Diseased classes    : {len(report['classes']['diseased_classes'])}")
    imb = train["imbalance"]
    print(f"Imbalance ratio     : {imb['imbalance_ratio']:.2f}  (min {imb['min_count']} / max {imb['max_count']})")
    print(f"Coeff. of variation : {imb['coefficient_of_variation']:.3f}")
    print(f"Image sizes (sample): {train['image_geometry']['unique_sizes'][:5]}")
    print(f"Corrupt in sample   : {len(train['corrupt_sampled'])} / {train['sampled_images']}")
    leak = dupes.get("cross_split_leakage", {})
    if leak:
        print(f"Train/valid pHash collisions: {leak['colliding_hashes']} "
              f"({leak['leak_rate_in_sample'] * 100:.2f}% of sampled valid images)")
    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
