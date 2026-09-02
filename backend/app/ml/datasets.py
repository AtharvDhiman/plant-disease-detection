"""Manifest-driven datasets and the split protocol.

Why manifests instead of ``ImageFolder``
----------------------------------------
Every experiment in this project - the classical-ML baselines, the eight deep
architectures, the ablation arms - must see *exactly* the same images in the
same splits, otherwise the comparison is not a comparison. Materialising the
splits once into CSV manifests (``data/manifests/*.csv``) and having every
script read them removes any chance of a script re-shuffling the data with a
different seed and quietly invalidating the benchmark.

Split protocol
--------------
The Kaggle archive ships an official ``train/`` and ``valid/`` directory. We:

* carve a stratified **validation** set out of the official ``train/`` directory
  (used for early stopping, LR scheduling and model selection), and
* keep the official ``valid/`` directory as an untouched **test** set, used
  once per model to produce the reported numbers.

That gives a genuine three-way split in which nothing that influences model
selection is ever scored as test data.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, WeightedRandomSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True  # inference must survive slightly damaged uploads


# --------------------------------------------------------------------------- #
# Manifests
# --------------------------------------------------------------------------- #


@dataclass
class ManifestEntry:
    """One row of a split manifest."""

    path: str          # absolute path to the image
    label: int         # class index
    class_name: str    # raw class directory name


def write_manifest(rows: list[ManifestEntry], destination: Path) -> None:
    """Persist manifest rows as CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "class_name"])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def read_manifest(source: Path) -> list[ManifestEntry]:
    """Load manifest rows from CSV."""
    with Path(source).open("r", newline="", encoding="utf-8") as handle:
        return [
            ManifestEntry(path=row["path"], label=int(row["label"]), class_name=row["class_name"])
            for row in csv.DictReader(handle)
        ]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class ManifestImageDataset(Dataset):
    """Image classification dataset backed by a manifest.

    Unreadable images are skipped by substituting the next readable sample and
    recording the failure in :attr:`failed_paths`, so one corrupt JPEG cannot
    abort a multi-hour training run. The count is reported at the end of
    training rather than silently swallowed.
    """

    def __init__(self, entries: list[ManifestEntry], transform=None, class_names: list[str] | None = None):
        self.entries = entries
        self.transform = transform
        self.class_names = class_names or sorted({e.class_name for e in entries})
        self.failed_paths: list[str] = []

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        attempts = 0
        while attempts < 8:
            entry = self.entries[(index + attempts) % len(self.entries)]
            try:
                with Image.open(entry.path) as img:
                    image = img.convert("RGB")
                break
            except Exception:  # noqa: BLE001 - any decode failure
                self.failed_paths.append(entry.path)
                attempts += 1
        else:  # pragma: no cover - would mean the dataset directory is gone
            raise RuntimeError(f"8 consecutive unreadable images starting at index {index}")

        if self.transform is not None:
            image = self.transform(image)
        return image, entry.label

    @property
    def labels(self) -> np.ndarray:
        return np.array([e.label for e in self.entries], dtype=np.int64)

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=len(self.class_names))


# --------------------------------------------------------------------------- #
# Split construction
# --------------------------------------------------------------------------- #


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_class_files(split_dir: Path, class_to_idx: dict[str, int]) -> list[ManifestEntry]:
    """Enumerate every image under ``split_dir`` in a deterministic order."""
    rows: list[ManifestEntry] = []
    for class_name, index in sorted(class_to_idx.items(), key=lambda kv: kv[1]):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append(ManifestEntry(str(path), index, class_name))
    return rows


def stratified_split(
    rows: list[ManifestEntry],
    fraction: float,
    seed: int,
) -> tuple[list[ManifestEntry], list[ManifestEntry]]:
    """Split ``rows`` into ``(major, minor)`` preserving the class distribution."""
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[ManifestEntry]] = {}
    for row in rows:
        by_class.setdefault(row.label, []).append(row)

    major: list[ManifestEntry] = []
    minor: list[ManifestEntry] = []
    for label in sorted(by_class):
        items = by_class[label]
        order = rng.permutation(len(items))
        cut = int(round(len(items) * fraction))
        # Guarantee at least one sample per class on each side of the split.
        cut = min(max(cut, 1), len(items) - 1) if len(items) > 1 else 0
        minor.extend(items[i] for i in order[:cut])
        major.extend(items[i] for i in order[cut:])
    return major, minor


def stratified_subsample(
    rows: list[ManifestEntry],
    per_class: int,
    seed: int,
) -> list[ManifestEntry]:
    """Take up to ``per_class`` images from each class, deterministically."""
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[ManifestEntry]] = {}
    for row in rows:
        by_class.setdefault(row.label, []).append(row)

    picked: list[ManifestEntry] = []
    for label in sorted(by_class):
        items = by_class[label]
        take = min(per_class, len(items))
        order = rng.permutation(len(items))[:take]
        picked.extend(items[i] for i in sorted(order))
    return picked


# --------------------------------------------------------------------------- #
# Imbalance handling
# --------------------------------------------------------------------------- #


def compute_class_weights(counts: np.ndarray, scheme: str = "inverse_sqrt") -> torch.Tensor:
    """Loss weights derived from class frequencies.

    ``inverse``       - w_c = N / (K * n_c); the textbook balanced weighting.
    ``inverse_sqrt``  - a softer version that avoids over-amplifying the rarest
                        class when the imbalance is mild.
    ``effective``     - Cui et al. (2019) effective-number weighting.
    """
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.where(counts == 0, 1.0, counts)

    if scheme == "inverse":
        weights = counts.sum() / (len(counts) * counts)
    elif scheme == "inverse_sqrt":
        weights = np.sqrt(counts.sum() / (len(counts) * counts))
    elif scheme == "effective":
        beta = 0.999
        effective = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        weights = (1.0 / effective) * len(counts) / np.sum(1.0 / effective)
    else:
        raise ValueError(f"Unknown class-weight scheme {scheme!r}")

    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def build_balanced_sampler(labels: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    """Sampler that draws each class with equal probability."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    sample_weights = (1.0 / counts)[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


# --------------------------------------------------------------------------- #
# Split file I/O
# --------------------------------------------------------------------------- #


def load_split_config(path: Path) -> dict:
    """Read ``data/splits.json``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_split(split_config: dict, name: str, transform=None) -> ManifestImageDataset:
    """Materialise one named split from a split configuration."""
    if name not in split_config["splits"]:
        raise KeyError(f"Split {name!r} not in {sorted(split_config['splits'])}")
    manifest = Path(split_config["splits"][name]["manifest"])
    return ManifestImageDataset(
        read_manifest(manifest),
        transform=transform,
        class_names=split_config["class_names"],
    )
