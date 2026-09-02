"""Calibrate the out-of-distribution thresholds on real model outputs.

Method
------
1. Collect logits for the held-out **test** split - genuine leaf photographs the
   model has never seen. These are the in-distribution samples.
2. Build an out-of-distribution set and collect logits for it too.
3. Choose thresholds that retain a target fraction of in-distribution inputs,
   and report how many OOD inputs each threshold catches.

Honesty note on the OOD set
---------------------------
This project ships with no external non-leaf image corpus, so the OOD set is
built from *synthetic proxies*: uniform-noise images, flat colour fields,
patch-shuffled leaves (which destroy global structure while keeping leaf
texture and colour) and heavily degraded leaves. These are genuinely
out-of-distribution but they are **easier** than a real photograph of, say, a
brick wall or a face. The detection rates reported here should therefore be read
as an upper bound, and the JSON records this caveat so it travels with the
numbers. Pass ``--ood-dir`` to calibrate against a real corpus instead.

    python training/calibrate_ood.py
    python training/calibrate_ood.py --target-tpr 0.97 --ood-dir path/to/non_leaf_images
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import app.core.runtime  # noqa: F401  # isort:skip

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from evaluate import collect_predictions, load_checkpoint  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.ml.datasets import load_split_config, read_manifest  # noqa: E402
from app.ml.ood import (  # noqa: E402
    auroc,
    energy_score,
    msp_score,
    normalised_entropy,
    select_thresholds,
    softmax,
)
from app.ml.transforms import build_eval_transform  # noqa: E402

log = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Synthetic OOD proxies
# --------------------------------------------------------------------------- #


class SyntheticOODDataset(Dataset):
    """Generates out-of-distribution images from four families.

    ``noise``   - uniform RGB noise: no structure at all
    ``flat``    - single-colour fields: structure-free and low-entropy
    ``shuffle`` - the leaf cut into 8x8 tiles and permuted, which keeps local
                  texture and colour statistics but destroys the global shape a
                  classifier relies on
    ``extreme`` - real leaves pushed far outside the training distribution by
                  heavy blur, inversion or extreme exposure
    """

    FAMILIES = ("noise", "flat", "shuffle", "extreme")

    def __init__(self, leaf_paths: list[str], count: int, image_size: int, transform, seed: int = 0):
        self.leaf_paths = leaf_paths
        self.count = count
        self.image_size = image_size
        self.transform = transform
        self.seed = seed

    def __len__(self) -> int:
        return self.count

    def _family(self, index: int) -> str:
        return self.FAMILIES[index % len(self.FAMILIES)]

    def families(self) -> list[str]:
        """Family label of every index, for per-family reporting."""
        return [self._family(i) for i in range(self.count)]

    def __getitem__(self, index: int):
        return self.transform(self.raw_image(index)), 0

    def raw_image(self, index: int) -> Image.Image:
        """Build the untransformed PIL image, so the quality gate can see it too."""
        rng = np.random.default_rng(self.seed + index)
        family = self._family(index)
        side = max(self.image_size, 256)

        if family == "noise":
            array = rng.integers(0, 256, (side, side, 3), dtype=np.uint8)
            image = Image.fromarray(array)
        elif family == "flat":
            colour = rng.integers(30, 226, 3, dtype=np.uint8)
            array = np.full((side, side, 3), colour, dtype=np.uint8)
            # A little noise, otherwise the JPEG-free flat field is degenerate.
            array = np.clip(array.astype(int) + rng.integers(-8, 9, array.shape), 0, 255)
            image = Image.fromarray(array.astype(np.uint8))
        else:
            path = self.leaf_paths[int(rng.integers(0, len(self.leaf_paths)))]
            with Image.open(path) as handle:
                leaf = handle.convert("RGB").resize((side, side))
            if family == "shuffle":
                array = np.asarray(leaf)
                tile = side // 8
                tiles = [
                    array[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile]
                    for r in range(8) for c in range(8)
                ]
                order = rng.permutation(len(tiles))
                shuffled = np.zeros_like(array)
                for position, source in enumerate(order):
                    r, c = divmod(position, 8)
                    shuffled[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = tiles[source]
                image = Image.fromarray(shuffled)
            else:
                choice = int(rng.integers(0, 3))
                array = np.asarray(leaf).astype(np.float32)
                if choice == 0:
                    from PIL import ImageFilter

                    image = leaf.filter(ImageFilter.GaussianBlur(12))
                elif choice == 1:
                    image = Image.fromarray((255 - array).astype(np.uint8))
                else:
                    image = Image.fromarray(np.clip(array * 3.2, 0, 255).astype(np.uint8))

        return image


class PathDataset(Dataset):
    """Loads images from an explicit list of paths (used for ``--ood-dir``)."""

    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        return self.transform(self.raw_image(index)), 0

    def raw_image(self, index: int) -> Image.Image:
        with Image.open(self.paths[index]) as handle:
            return handle.convert("RGB")

    def families(self) -> list[str]:
        return ["external"] * len(self.paths)


# --------------------------------------------------------------------------- #
# Whole-pipeline evaluation
# --------------------------------------------------------------------------- #


def quality_rejects(dataset, indices: list[int]) -> np.ndarray:
    """Which images the pre-flight quality gate rejects, before any inference.

    The deployed system runs the quality checks *first*, so the OOD score's
    stand-alone AUROC understates what actually protects the user. Measuring
    both stages is the only way to report the pipeline honestly.
    """
    from app.ml.quality import analyse_image

    rejected = np.zeros(len(indices), dtype=bool)
    for position, index in enumerate(indices):
        image = dataset.raw_image(index)
        report = analyse_image(image)
        rejected[position] = not report.passed
    return rejected


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def plot_score_distributions(scores: dict, destination: Path) -> None:
    """Overlaid in-distribution vs OOD histograms for each score."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (name, (in_scores, ood_scores, higher_is_in)) in zip(axes, scores.items(), strict=False):
        lo = min(in_scores.min(), ood_scores.min())
        hi = max(in_scores.max(), ood_scores.max())
        bins = np.linspace(lo, hi, 45)
        ax.hist(in_scores, bins=bins, alpha=0.75, label="In-distribution (test leaves)",
                color="#2f7a4d", density=True)
        ax.hist(ood_scores, bins=bins, alpha=0.75, label="Out-of-distribution proxies",
                color="#c1553b", density=True)
        area = auroc(in_scores, ood_scores, higher_is_in)
        ax.set_title(f"{name}   AUROC = {area:.4f}")
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25, linestyle="--")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Out-of-distribution score separation")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None,
                        help="Defaults to the exported production checkpoint.")
    parser.add_argument("--split", default="test_bench",
                        help="In-distribution split to calibrate against.")
    parser.add_argument("--target-tpr", type=float, default=0.95)
    parser.add_argument("--ood-count", type=int, default=1200)
    parser.add_argument("--ood-dir", default=None,
                        help="Directory of real non-leaf images; overrides the synthetic proxies.")
    parser.add_argument("--max-in-dist", type=int, default=4000)
    args = parser.parse_args()

    configure_logging(settings.log_level, settings.logs_dir / "calibrate_ood.log")
    settings.ensure_dirs()

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint is None:
        manifest_path = settings.exported_dir / "production.json"
        if not manifest_path.exists():
            raise SystemExit("No exported model. Run training/export_model.py first.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = Path(manifest["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = settings.project_root / checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, class_names = load_checkpoint(checkpoint, device)
    log.info("Loaded model", fields={"model": config.model_name, "checkpoint": str(checkpoint)})

    split_config = load_split_config(settings.splits_path)
    transform = build_eval_transform(config.image_size)

    # ---------------------------------------------------------- in-distribution
    from app.ml.datasets import ManifestImageDataset

    entries = read_manifest(Path(split_config["splits"][args.split]["manifest"]))
    if len(entries) > args.max_in_dist:
        step = len(entries) // args.max_in_dist
        entries = entries[::step][: args.max_in_dist]
    in_dataset = ManifestImageDataset(entries, transform, class_names)
    in_loader = DataLoader(in_dataset, batch_size=48, shuffle=False, num_workers=0)
    in_logits, _, in_labels = collect_predictions(model, in_loader, device, amp=False)
    log.info("Collected in-distribution logits", fields={"n": len(in_logits)})

    # ------------------------------------------------------------------- OOD
    if args.ood_dir:
        root = Path(args.ood_dir)
        paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            raise SystemExit(f"No images found under {root}")
        ood_dataset = PathDataset(paths[: args.ood_count], transform)
        ood_source = {"kind": "external_directory", "path": str(root), "count": len(ood_dataset)}
    else:
        ood_dataset = SyntheticOODDataset(
            [e.path for e in entries], args.ood_count, config.image_size, transform,
            seed=split_config.get("seed", 42),
        )
        ood_source = {
            "kind": "synthetic_proxies",
            "families": list(SyntheticOODDataset.FAMILIES),
            "count": args.ood_count,
            "caveat": (
                "Synthetic proxies (uniform noise, flat colour fields, tile-shuffled "
                "leaves, heavily degraded leaves) are genuinely out of distribution but "
                "are easier to reject than real photographs of unrelated subjects. The "
                "detection rates below are therefore an upper bound. Re-run with "
                "--ood-dir pointing at a real non-leaf corpus for a tighter estimate."
            ),
        }

    ood_loader = DataLoader(ood_dataset, batch_size=48, shuffle=False, num_workers=0)
    ood_logits, _, _ = collect_predictions(model, ood_loader, device, amp=False)
    log.info("Collected OOD logits", fields={"n": len(ood_logits)})

    # ------------------------------------------------------------ thresholds
    selected = select_thresholds(in_logits, ood_logits, args.target_tpr)
    selected["method"] = "msp+entropy"

    in_probs, ood_probs = softmax(in_logits), softmax(ood_logits)
    score_sets = {
        "Max softmax probability": (msp_score(in_probs), msp_score(ood_probs), True),
        "Normalised entropy": (normalised_entropy(in_probs), normalised_entropy(ood_probs), False),
        "Free energy": (energy_score(in_logits), energy_score(ood_logits), False),
    }
    aurocs = {name: auroc(a, b, higher) for name, (a, b, higher) in score_sets.items()}

    plot_path = settings.plots_dir / "ood_score_distributions.png"
    plot_score_distributions(score_sets, plot_path)

    # ------------------------------------------------- whole-pipeline rates
    # The OOD score never runs alone in production: the quality gate rejects
    # first. Measuring the two stages together is what the user actually gets.
    sample_size = min(len(ood_dataset), 400)
    sample_indices = list(range(0, len(ood_dataset), max(1, len(ood_dataset) // sample_size)))[:sample_size]
    ood_quality_rejected = quality_rejects(ood_dataset, sample_indices)

    in_sample_size = min(len(in_dataset), 300)
    in_sample_indices = list(range(0, len(in_dataset), max(1, len(in_dataset) // in_sample_size)))[:in_sample_size]

    from app.ml.quality import analyse_image as _analyse

    in_quality_rejected = np.array([
        not _analyse(Image.open(in_dataset.entries[i].path).convert("RGB")).passed
        for i in in_sample_indices
    ])

    ood_msp = msp_score(ood_probs)
    ood_ent = normalised_entropy(ood_probs)
    ood_score_rejected = (ood_msp < selected["msp_min"]) | (ood_ent > selected["entropy_max"])
    combined_rejected = ood_quality_rejected | ood_score_rejected[sample_indices]

    families = ood_dataset.families()
    per_family = {}
    for family in sorted(set(families)):
        positions = [p for p, i in enumerate(sample_indices) if families[i] == family]
        if not positions:
            continue
        per_family[family] = {
            "count": len(positions),
            "quality_gate_rejected": float(ood_quality_rejected[positions].mean()),
            "ood_score_rejected": float(ood_score_rejected[[sample_indices[p] for p in positions]].mean()),
            "pipeline_rejected": float(combined_rejected[positions].mean()),
        }

    pipeline = {
        "sampled_ood": len(sample_indices),
        "sampled_in_distribution": len(in_sample_indices),
        "quality_gate_rejects_ood": float(ood_quality_rejected.mean()),
        "ood_score_rejects_ood": float(ood_score_rejected.mean()),
        "pipeline_rejects_ood": float(combined_rejected.mean()),
        "quality_gate_rejects_real_leaves": float(in_quality_rejected.mean()),
        "per_family": per_family,
        "interpretation": (
            "The deployed pipeline runs the image-quality gate before the model, so "
            "the end-to-end rejection rate is what a user experiences. The "
            "'quality_gate_rejects_real_leaves' figure is the cost of that gate: the "
            "fraction of genuine test photographs it would ask the user to retake."
        ),
    }

    # The API gates on MSP and entropy; energy is reported but not enforced by
    # default because a single scalar threshold on it is the least stable of the
    # three across retraining.
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config.model_name,
        "checkpoint": str(checkpoint),
        "in_distribution": {
            "split": args.split,
            "count": int(len(in_logits)),
            "accuracy": float((in_logits.argmax(axis=1) == in_labels).mean()),
        },
        "ood_source": ood_source,
        "selected": selected,
        "metrics": {
            "auroc": aurocs,
            "detection_rates": {
                name: selected["per_score"][key]["ood_detection_rate"]
                for name, key in (("Max softmax probability", "msp"),
                                  ("Normalised entropy", "entropy"),
                                  ("Free energy", "energy"))
            },
            "pipeline": pipeline,
        },
        "plot": str(plot_path),
        "note": (
            "The serving path enforces the MSP and entropy thresholds. The energy "
            "threshold is recorded for reference; it is the least stable of the three "
            "across retraining because it depends on the absolute logit scale."
        ),
    }

    out = settings.metadata_dir / "ood_thresholds.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    model_specific = settings.metadata_dir / f"ood_{config.model_name}.json"
    model_specific.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("OOD CALIBRATION")
    print("=" * 78)
    print(f"Model              : {config.model_name}")
    print(f"In-distribution    : {args.split} ({len(in_logits)} images, "
          f"accuracy {payload['in_distribution']['accuracy']:.4f})")
    print(f"OOD set            : {ood_source['kind']} ({len(ood_logits)} images)")
    print(f"Target TPR         : {args.target_tpr:.2f}")
    print("-" * 78)
    print(f"{'SCORE':<28}{'AUROC':>9}{'THRESHOLD':>13}{'TPR':>8}{'OOD CAUGHT':>13}")
    for name, key in (("Max softmax probability", "msp"), ("Normalised entropy", "entropy"),
                      ("Free energy", "energy")):
        rates = selected["per_score"][key]
        threshold = selected[f"{key}_min" if key == "msp" else f"{key}_max"]
        print(f"{name:<28}{aurocs[name]:>9.4f}{threshold:>13.4f}"
              f"{rates['tpr']:>8.3f}{rates['ood_detection_rate']:>13.3f}")
    print("-" * 78)
    print("WHOLE-PIPELINE REJECTION (quality gate runs before the model)")
    print(f"  quality gate catches      : {pipeline['quality_gate_rejects_ood'] * 100:5.1f}% of OOD")
    print(f"  OOD score catches         : {pipeline['ood_score_rejects_ood'] * 100:5.1f}% of OOD")
    print(f"  combined pipeline catches : {pipeline['pipeline_rejects_ood'] * 100:5.1f}% of OOD")
    print(f"  cost: real leaves rejected: {pipeline['quality_gate_rejects_real_leaves'] * 100:5.1f}%")
    if pipeline["per_family"]:
        print(f"  {'family':<12}{'quality':>10}{'ood score':>12}{'pipeline':>11}")
        for family, rates in pipeline["per_family"].items():
            print(f"  {family:<12}{rates['quality_gate_rejected'] * 100:>9.1f}%"
                  f"{rates['ood_score_rejected'] * 100:>11.1f}%"
                  f"{rates['pipeline_rejected'] * 100:>10.1f}%")
    print("=" * 78)
    if ood_source["kind"] == "synthetic_proxies":
        print("NOTE: " + ood_source["caveat"])
    print(f"\nWrote {out}")
    print(f"Wrote {plot_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
