"""Materialise the train/validation/test manifests and the dataset plots.

Reads ``artifacts/dataset_report.json`` (produced by ``dataset_inspect.py``),
applies the split protocol documented in ``app.ml.datasets`` and writes:

* ``data/manifests/{train,val,test}.csv``            - full-data protocol
* ``data/manifests/{train,val,test}_bench.csv``      - fixed-budget benchmark subset
* ``data/splits.json``                               - the index every script reads
* ``artifacts/plots/class_distribution.png`` etc.    - dataset visualisations

Run:
    python training/prepare_splits.py
    python training/prepare_splits.py --val-fraction 0.1 --bench-train-per-class 300
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import app.core.runtime  # noqa: F401  # isort:skip

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.ml.datasets import (  # noqa: E402
    ManifestEntry,
    compute_class_weights,
    list_class_files,
    stratified_split,
    stratified_subsample,
    write_manifest,
)

log = get_logger(__name__)


def choose_imbalance_strategy(counts: np.ndarray) -> dict:
    """Pick *one* imbalance remedy based on the measured distribution.

    The brief explicitly asks not to stack every technique at once. The decision
    rule below is applied to the real counts and the reasoning is stored in
    ``data/splits.json`` so the choice is auditable rather than arbitrary.
    """
    ratio = float(counts.max() / counts.min())
    cv = float(counts.std() / counts.mean())

    if ratio < 1.5 and cv < 0.20:
        return {
            "strategy": "none",
            "reason": (
                f"Imbalance ratio {ratio:.2f} and coefficient of variation {cv:.3f} are both "
                "small. Re-weighting a near-uniform distribution adds gradient noise without "
                "improving minority-class recall, so plain cross-entropy is used."
            ),
            "class_weight_scheme": None,
        }
    if ratio < 4.0:
        return {
            "strategy": "class_weights",
            "reason": (
                f"Imbalance ratio {ratio:.2f} (CV {cv:.3f}) is moderate. Weighting the loss by "
                "inverse square-root frequency corrects the bias without the variance cost of "
                "oversampling or the extra hyper-parameters of focal loss."
            ),
            "class_weight_scheme": "inverse_sqrt",
        }
    if ratio < 15.0:
        return {
            "strategy": "class_weights",
            "reason": (
                f"Imbalance ratio {ratio:.2f} (CV {cv:.3f}) is substantial. Inverse-frequency "
                "loss weighting is applied so rare classes contribute proportionally to the "
                "gradient."
            ),
            "class_weight_scheme": "inverse",
        }
    return {
        "strategy": "balanced_sampler",
        "reason": (
            f"Imbalance ratio {ratio:.2f} (CV {cv:.3f}) is severe. Loss weighting alone leaves "
            "rare classes under-represented in every mini-batch, so a class-balanced sampler is "
            "used instead."
        ),
        "class_weight_scheme": None,
    }


def plot_class_distribution(counts: dict[str, int], destination: Path, title: str) -> None:
    """Horizontal bar chart of images per class."""
    names = list(counts)
    values = [counts[n] for n in names]
    order = np.argsort(values)
    names = [names[i].replace("___", " / ").replace("_", " ") for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, max(6, len(names) * 0.28)))
    colors = plt.cm.YlGn(np.linspace(0.35, 0.85, len(names)))
    ax.barh(names, values, color=colors, edgecolor="#2f5d3a", linewidth=0.4)
    ax.set_xlabel("Number of images")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for y, v in enumerate(values):
        ax.text(v + max(values) * 0.005, y, f"{v:,}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def plot_split_sizes(sizes: dict[str, int], destination: Path) -> None:
    """Bar chart of train/val/test sizes."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    names = list(sizes)
    values = [sizes[n] for n in names]
    bars = ax.bar(names, values, color=["#2f7a4d", "#5aa469", "#a5c882"], edgecolor="#1f3d2b")
    ax.set_ylabel("Images")
    ax.set_title("Dataset split sizes")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for bar, value in zip(bars, values, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}",
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def plot_plant_breakdown(class_details: list[dict], counts: dict[str, int], destination: Path) -> None:
    """Stacked healthy/diseased image counts per plant species."""
    per_plant: dict[str, dict[str, int]] = {}
    for detail in class_details:
        bucket = per_plant.setdefault(detail["plant"], {"healthy": 0, "diseased": 0})
        key = "healthy" if detail["is_healthy"] else "diseased"
        bucket[key] += counts.get(detail["raw"], 0)

    plants = sorted(per_plant, key=lambda p: -(per_plant[p]["healthy"] + per_plant[p]["diseased"]))
    healthy = [per_plant[p]["healthy"] for p in plants]
    diseased = [per_plant[p]["diseased"] for p in plants]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(plants, healthy, label="Healthy", color="#4fa96a")
    ax.bar(plants, diseased, bottom=healthy, label="Diseased", color="#c1553b")
    ax.set_ylabel("Images")
    ax.set_title("Images per plant species (training split)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    plt.setp(ax.get_xticklabels(), rotation=38, ha="right", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def plot_sample_grid(rows: list[ManifestEntry], class_names: list[str], destination: Path, per_row: int = 8) -> None:
    """Contact sheet with one example per class."""
    from PIL import Image

    by_class: dict[int, ManifestEntry] = {}
    for row in rows:
        by_class.setdefault(row.label, row)

    labels = sorted(by_class)
    n_rows = int(np.ceil(len(labels) / per_row))
    fig, axes = plt.subplots(n_rows, per_row, figsize=(per_row * 1.9, n_rows * 2.15))
    axes = np.atleast_2d(axes)

    for ax in axes.ravel():
        ax.axis("off")
    for position, label in enumerate(labels):
        ax = axes[position // per_row, position % per_row]
        with Image.open(by_class[label].path) as img:
            ax.imshow(img.convert("RGB").resize((128, 128)))
        pretty = class_names[label].replace("___", "\n").replace("_", " ")
        ax.set_title(pretty, fontsize=6.2, pad=2)
    fig.suptitle("One sample per class", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-fraction", type=float, default=0.10,
                        help="Fraction of the official train split held out for validation.")
    parser.add_argument("--bench-train-per-class", type=int, default=300,
                        help="Training images per class in the fixed-budget benchmark subset.")
    parser.add_argument("--bench-val-per-class", type=int, default=60)
    parser.add_argument("--bench-test-per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    configure_logging(settings.log_level, settings.logs_dir / "prepare_splits.log")
    settings.ensure_dirs()
    seed = args.seed if args.seed is not None else settings.seed

    if not settings.dataset_report_path.exists():
        raise FileNotFoundError(
            "artifacts/dataset_report.json missing - run training/dataset_inspect.py first."
        )
    report = json.loads(settings.dataset_report_path.read_text(encoding="utf-8"))

    class_names: list[str] = report["classes"]["names"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    log.info("Loaded dataset report", fields={"classes": len(class_names)})

    train_dir = Path(report["splits"]["train"]["path"])
    official_valid = report["splits"].get("valid")
    if official_valid is None:
        raise RuntimeError(
            "The dataset report contains no 'valid' split; the three-way protocol needs one. "
            "Re-run dataset_inspect.py or supply a test directory explicitly."
        )
    test_dir = Path(official_valid["path"])

    log.info("Enumerating images")
    train_pool = list_class_files(train_dir, class_to_idx)
    test_rows = list_class_files(test_dir, class_to_idx)
    log.info("Enumerated", fields={"train_pool": len(train_pool), "official_valid": len(test_rows)})

    train_rows, val_rows = stratified_split(train_pool, args.val_fraction, seed)
    log.info("Split train pool", fields={"train": len(train_rows), "val": len(val_rows)})

    bench_train = stratified_subsample(train_rows, args.bench_train_per_class, seed)
    bench_val = stratified_subsample(val_rows, args.bench_val_per_class, seed)
    bench_test = stratified_subsample(test_rows, args.bench_test_per_class, seed)

    manifest_dir = settings.data_dir / "manifests"
    written = {
        "train": (train_rows, manifest_dir / "train.csv"),
        "val": (val_rows, manifest_dir / "val.csv"),
        "test": (test_rows, manifest_dir / "test.csv"),
        "train_bench": (bench_train, manifest_dir / "train_bench.csv"),
        "val_bench": (bench_val, manifest_dir / "val_bench.csv"),
        "test_bench": (bench_test, manifest_dir / "test_bench.csv"),
    }
    for name, (rows, path) in written.items():
        write_manifest(rows, path)
        log.info("Wrote manifest", fields={"split": name, "rows": len(rows), "path": str(path)})

    counts = np.bincount([r.label for r in train_rows], minlength=len(class_names))
    imbalance = choose_imbalance_strategy(counts.astype(float))
    weights = compute_class_weights(counts, imbalance["class_weight_scheme"] or "inverse_sqrt")
    log.info("Imbalance strategy", fields={"strategy": imbalance["strategy"]})

    split_config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "num_classes": len(class_names),
        "protocol": {
            "description": (
                "The official Kaggle train/ directory is split stratified into our train and "
                "validation sets. The official valid/ directory is never used for tuning and "
                "serves as the held-out test set for all reported numbers."
            ),
            "val_fraction_of_official_train": args.val_fraction,
            "test_source": "official valid/ directory",
            "stratified": True,
            "benchmark_subset": {
                "why": (
                    "Training 15+ architectures on 70k images is not feasible on the available "
                    "4 GB GPU. Every model in the benchmark and the ablation study is therefore "
                    "trained on this identical fixed-budget subset under an identical schedule, "
                    "which keeps the comparison fair. The selected architectures are then "
                    "retrained on the full split for the production model."
                ),
                "train_per_class": args.bench_train_per_class,
                "val_per_class": args.bench_val_per_class,
                "test_per_class": args.bench_test_per_class,
            },
        },
        "imbalance": {
            **imbalance,
            "counts": {class_names[i]: int(counts[i]) for i in range(len(class_names))},
            "ratio": float(counts.max() / counts.min()),
            "coefficient_of_variation": float(counts.std() / counts.mean()),
            "class_weights": [round(float(w), 5) for w in weights],
        },
        "splits": {
            name: {
                "manifest": str(path),
                "count": len(rows),
                "class_counts": {
                    class_names[i]: int(c)
                    for i, c in enumerate(np.bincount([r.label for r in rows], minlength=len(class_names)))
                },
            }
            for name, (rows, path) in written.items()
        },
    }
    settings.splits_path.write_text(json.dumps(split_config, indent=2), encoding="utf-8")
    log.info("Wrote split config", fields={"path": str(settings.splits_path)})

    log.info("Rendering dataset plots")
    plot_class_distribution(
        split_config["splits"]["train"]["class_counts"],
        settings.plots_dir / "class_distribution_train.png",
        "Training images per class",
    )
    plot_class_distribution(
        split_config["splits"]["test"]["class_counts"],
        settings.plots_dir / "class_distribution_test.png",
        "Held-out test images per class",
    )
    plot_split_sizes(
        {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)},
        settings.plots_dir / "split_sizes.png",
    )
    plot_plant_breakdown(
        report["classes"]["details"],
        split_config["splits"]["train"]["class_counts"],
        settings.plots_dir / "plant_breakdown.png",
    )
    plot_sample_grid(train_rows, class_names, settings.plots_dir / "class_samples.png")

    print("\n" + "=" * 72)
    print("SPLIT SUMMARY")
    print("=" * 72)
    print(f"Classes            : {len(class_names)}")
    print(f"Train              : {len(train_rows):,}")
    print(f"Validation         : {len(val_rows):,}   (from official train/, stratified)")
    print(f"Test               : {len(test_rows):,}   (official valid/, untouched)")
    print(f"Benchmark subset   : {len(bench_train):,} / {len(bench_val):,} / {len(bench_test):,}")
    print(f"Imbalance ratio    : {split_config['imbalance']['ratio']:.2f}")
    print(f"Imbalance strategy : {imbalance['strategy']}")
    print(f"  reason           : {imbalance['reason']}")
    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
