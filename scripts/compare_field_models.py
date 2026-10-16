"""Compare candidate models on lab and field-like photos, through the full pipeline.

The ledger already holds each model's accuracy on the held-out lab test split.
What it cannot say is how a model behaves on the photos the application will
actually receive. This script runs every candidate through the *serving*
pipeline - quality gate, dust check, classifier, OOD check - on the same
held-out leaves under the field conditions of ``scripts/domain_shift_probe.py``,
and reports, per condition:

* ``correct``  - top-1 class right, whatever status the app attached;
* ``ok_right`` - the app said "ok" *and* was right: a diagnosis a user can act on;
* ``ok_wrong`` - the app said "ok" and was wrong: the harmful outcome.

Fairness: every model sees the identical images and the identical transformed
versions (the transforms are generated once, then shared), and every model is
judged with the same OOD thresholds - the deployed model's - so a difference
is the network, not the operating point.

Caveat, stated in the output too: the field-trained models were trained with
augmentation that imitates the same kinds of condition (by different
generators), so this is not a fully independent test for them. Real field
photographs are.

    python scripts/compare_field_models.py --per-class 4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import app.core.runtime  # noqa: F401,E402  # isort:skip

import domain_shift_probe as probe  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.ml.architectures import MODEL_ZOO, build_model  # noqa: E402
from app.ml.ood import OODThresholds  # noqa: E402
from app.ml.registry import LoadedModel, registry  # noqa: E402
from app.services.predictor import Predictor  # noqa: E402


def load_candidate(record: dict, alias: str, thresholds: OODThresholds,
                   device: torch.device) -> str:
    """Register a ledger checkpoint in the serving registry under ``alias``."""
    path = Path(record["checkpoint_path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(record["model_name"], len(payload["class_names"]), pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    spec = MODEL_ZOO[record["model_name"]]
    temperature = ((record.get("test") or {}).get("calibration") or {}).get("temperature", 1.0)
    registry._models[alias] = LoadedModel(  # noqa: SLF001 - evaluation-only injection
        name=alias, label=f"{spec.label} [{record['protocol']}]", model=model,
        class_names=payload["class_names"], image_size=record["config"]["image_size"],
        device=device, temperature=float(temperature), ood_thresholds=thresholds,
        checkpoint_path=str(path), parameters=record.get("total_parameters", 0),
        size_mb=record.get("model_size_mb", 0.0), load_time_ms=0.0,
        metrics=(record.get("test") or {}).get("metrics", {}), has_cbam="cbam" in alias,
        version="eval", selection={},
    )
    return alias


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=4,
                        help="Images per disease class (26 disease classes).")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--protocols", nargs="+", default=["production", "field"])
    parser.add_argument("--models", nargs="+",
                        default=["efficientnet_b0", "efficientnet_b0_cbam"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Use cpu while a training run holds the GPU.")
    parser.add_argument("--out", default="field_comparison.json")
    args = parser.parse_args()

    ledger = json.loads((settings.results_dir / "experiments.json").read_text(encoding="utf-8"))
    candidates = [r for r in ledger
                  if r.get("protocol") in args.protocols and r["model_name"] in args.models
                  and (r.get("test") or {}).get("metrics")]
    if not candidates:
        raise SystemExit("No matching models in the ledger.")

    deployed = json.loads((settings.exported_dir / "production.json").read_text(encoding="utf-8"))
    thresholds = (OODThresholds.from_file(settings.metadata_dir / f"ood_{deployed['model_name']}.json",
                                          deployed["model_name"])
                  or OODThresholds.defaults(settings.ood_confidence_floor,
                                            settings.ood_entropy_ceiling))
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)

    record = json.loads((PROJECT_ROOT / "data" / "dataset_path.json").read_text(encoding="utf-8"))
    valid = (Path(record["path"]) / "New Plant Diseases Dataset(Augmented)"
             / "New Plant Diseases Dataset(Augmented)" / "valid")
    diseased = sorted(d for d in valid.iterdir() if d.is_dir() and "healthy" not in d.name)

    # Build every test image once, so each model sees exactly the same pixels.
    rng = random.Random(args.seed)
    samples: list[tuple[str, str, Image.Image]] = []
    for class_dir in diseased:
        files = sorted(class_dir.glob("*.JPG"))
        for path in rng.sample(files, min(args.per_class, len(files))):
            original = Image.open(path).convert("RGB")
            for condition, transform in probe.TRANSFORMS.items():
                samples.append((condition, class_dir.name, transform(original, rng)))
    print(f"{len(samples)} images ({len(probe.TRANSFORMS)} conditions x "
          f"{len(samples) // len(probe.TRANSFORMS)} leaves)", flush=True)

    results: dict = {}
    for candidate in candidates:
        alias = f"{candidate['model_name']}@{candidate['protocol']}"
        load_candidate(candidate, alias, thresholds, device)
        predictor = Predictor(model_name=alias)
        stats: dict = {}
        for condition, truth, image in samples:
            outcome = predictor.predict(image, explain=False)
            entry = stats.setdefault(condition, {"n": 0, "correct": 0, "ok_right": 0, "ok_wrong": 0})
            entry["n"] += 1
            right = outcome.predicted_class == truth
            entry["correct"] += right
            if outcome.status == "ok":
                entry["ok_right" if right else "ok_wrong"] += 1
        metrics = candidate["test"]["metrics"]
        results[alias] = {
            "label": registry._models[alias].label,  # noqa: SLF001
            "experiment_id": candidate["experiment_id"],
            "lab_test_accuracy": metrics["accuracy"],
            "lab_test_f1_macro": metrics["f1_macro"],
            "conditions": stats,
        }
        del registry._models[alias]  # noqa: SLF001 - free GPU memory between models
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  done {alias}", flush=True)

    conditions = list(probe.TRANSFORMS)
    print("\nCorrect class (ok-and-right / ok-and-WRONG) per condition")
    print(f"{'model':<34}{'lab test':>10}" + "".join(f"{c[:11]:>13}" for c in conditions))
    for alias, result in results.items():
        cells = ""
        for condition in conditions:
            s = result["conditions"][condition]
            cells += f"{s['correct'] / s['n'] * 100:>6.1f}%({s['ok_wrong']:>2}w)"[:13].rjust(13)
        print(f"{alias:<34}{result['lab_test_accuracy'] * 100:>9.2f}%{cells}")

    out = settings.results_dir / args.out
    out.write_text(json.dumps({
        "per_class": args.per_class,
        "disease_classes": len(diseased),
        "seed": args.seed,
        "ood_thresholds_from": deployed["model_name"],
        "results": results,
        "caveat": ("Field-trained models were trained with augmentation imitating the same "
                   "kinds of condition, through different generators, so this is not a fully "
                   "independent test for them. Real field photographs are."),
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
