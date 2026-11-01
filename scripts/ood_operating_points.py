"""Measure what each out-of-distribution operating point does to real outcomes.

``training/calibrate_ood.py`` keeps 95% of *laboratory* leaves, which makes the
thresholds strict for field photographs. This script runs the deployed model's
full serving pipeline (quality gate, dust check, classifier, OOD check) twice
over the same lab and field-like images - once with the calibrated thresholds,
once with the operating point actually in force - and records, per condition:

* ``ok_right``  - status "ok" and the class is right: a diagnosis a user can act on;
* ``ok_wrong``  - status "ok" and the class is wrong: the harmful outcome;
* ``not_identified`` - flagged out-of-distribution (the closest match is still shown);
* the remaining statuses (dust, poor quality, not a plant, low confidence).

It exists because the choice between the two is a trade-off between refusing
photographs the model gets right and letting wrong answers through, and that
trade-off should be read off numbers rather than argued.

    python scripts/ood_operating_points.py --per-class 4
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import app.core.runtime  # noqa: F401,E402  # isort:skip

import domain_shift_probe as probe  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.ml.ood import OODThresholds  # noqa: E402
from app.ml.registry import registry  # noqa: E402
from app.services.predictor import Predictor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=4, help="Images per class (38 classes).")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    loaded = registry.load()
    in_force = loaded.ood_thresholds
    calibrated = OODThresholds.from_file(settings.metadata_dir / "ood_thresholds.json", loaded.name)
    if calibrated is None:
        raise SystemExit("No calibrated thresholds for the deployed model; run "
                         "training/calibrate_ood.py first.")
    points = {"calibrated_95tpr": calibrated,
              "in_force": dataclasses.replace(in_force, calibrated=in_force.calibrated)}

    record = json.loads((PROJECT_ROOT / "data" / "dataset_path.json").read_text(encoding="utf-8"))
    valid = (Path(record["path"]) / "New Plant Diseases Dataset(Augmented)"
             / "New Plant Diseases Dataset(Augmented)" / "valid")
    rng = random.Random(args.seed)
    samples = []
    for class_dir in sorted(d for d in valid.iterdir() if d.is_dir()):
        files = sorted(class_dir.glob("*.JPG"))
        for path in rng.sample(files, min(args.per_class, len(files))):
            original = Image.open(path).convert("RGB")
            for condition, transform in probe.TRANSFORMS.items():
                samples.append((condition, class_dir.name, transform(original, rng)))
    print(f"{len(samples)} images, model {loaded.name}", flush=True)

    predictor = Predictor()
    results: dict = {}
    for name, thresholds in points.items():
        loaded.ood_thresholds = thresholds
        table: dict = defaultdict(Counter)
        for condition, truth, image in samples:
            outcome = predictor.predict(image, explain=False)
            row = table[condition]
            row["n"] += 1
            if outcome.status == "ok":
                row["ok_right" if outcome.predicted_class == truth else "ok_wrong"] += 1
            elif outcome.status == "out_of_distribution":
                row["not_identified"] += 1
            else:
                row["other"] += 1
        results[name] = {
            "msp_min": thresholds.msp_min,
            "entropy_max": thresholds.entropy_max,
            "conditions": {c: dict(t) for c, t in table.items()},
        }
        print(f"  done {name}", flush=True)
    loaded.ood_thresholds = in_force

    print(f"\n{'condition':<13}" + "".join(f"{p:>34}" for p in points))
    print(f"{'':<13}" + "".join(f"{'right / wrong / not identified':>34}" for _ in points))
    for condition in probe.TRANSFORMS:
        cells = ""
        for name in points:
            t = results[name]["conditions"][condition]
            n = t["n"]
            cells += (f"{t.get('ok_right', 0) / n * 100:>14.0f}% /{t.get('ok_wrong', 0) / n * 100:>4.0f}% "
                      f"/{t.get('not_identified', 0) / n * 100:>4.0f}%").rjust(34)
        print(f"{condition:<13}{cells}")

    out = settings.results_dir / "ood_operating_points.json"
    out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": loaded.name,
        "images_per_condition": len(samples) // len(probe.TRANSFORMS),
        "seed": args.seed,
        "points": results,
        "caveat": ("Transformations imitate field conditions applied to laboratory images; the "
                   "model was trained with augmentation of the same kinds, so this is not a fully "
                   "independent test. 'other' groups dust, quality, not-a-plant and low-confidence "
                   "outcomes, which do not depend on the OOD thresholds."),
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
