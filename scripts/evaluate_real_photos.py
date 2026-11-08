"""Measure models on real-world photographs (PlantDoc's held-out test split).

Two numbers per model, because they answer different questions:

* **model accuracy** - top-1 and top-3 of the network alone, on every test
  photo. This is the honest measure of how well it recognises a real leaf.
* **what the application shows** - the full serving pipeline (quality gate,
  dust check, classifier, OOD check): how often a user is shown the right
  diagnosis with status ``ok``, how often a *wrong* one with status ``ok`` (the
  harmful case), and how often the photo is flagged or refused instead.

The test split has a few hundred photos, so every accuracy is reported with a
95% Wilson interval: a difference smaller than the intervals' overlap is not a
difference this test can see.

    python scripts/evaluate_real_photos.py --experiments <id> [<id> ...]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import app.core.runtime  # noqa: F401,E402  # isort:skip

import numpy as np  # noqa: E402
import torch  # noqa: E402
from compare_field_models import load_candidate  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.ml.datasets import read_manifest  # noqa: E402
from app.ml.ood import OODThresholds  # noqa: E402
from app.ml.preprocess import eval_preprocess  # noqa: E402
from app.ml.registry import registry  # noqa: E402
from app.services.predictor import Predictor  # noqa: E402


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--split", default="plantdoc_test")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", default="real_photo_eval.json")
    args = parser.parse_args()

    split_config = json.loads(settings.splits_path.read_text(encoding="utf-8"))
    if args.split not in split_config["splits"]:
        raise SystemExit(f"Split {args.split!r} missing - run scripts/prepare_plantdoc.py first.")
    entries = read_manifest(Path(split_config["splits"][args.split]["manifest"]))
    class_names = split_config["class_names"]

    ledger = {r["experiment_id"]: r
              for r in json.loads((settings.results_dir / "experiments.json").read_text(encoding="utf-8"))}
    deployed = json.loads((settings.exported_dir / "production.json").read_text(encoding="utf-8"))
    # One OOD operating point for every model - the one the site is served with -
    # so a difference in outcomes is the network, not the threshold.
    thresholds = (OODThresholds.from_file(settings.metadata_dir / f"ood_{deployed['model_name']}.json",
                                          deployed["model_name"])
                  or OODThresholds.defaults(settings.ood_confidence_floor, settings.ood_entropy_ceiling))
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)

    images = []
    for entry in entries:
        with Image.open(entry.path) as image:
            images.append(image.convert("RGB"))
    print(f"{len(images)} real photographs, {len({e.label for e in entries})} classes", flush=True)

    results = {}
    for experiment_id in args.experiments:
        record = ledger.get(experiment_id)
        if record is None:
            raise SystemExit(f"{experiment_id} is not in the ledger.")
        alias = f"{experiment_id}@real"
        load_candidate(record, alias, thresholds, device)
        loaded = registry._models[alias]  # noqa: SLF001 - evaluation-only
        predictor = Predictor(model_name=alias)

        top1 = top3 = 0
        per_class = defaultdict(lambda: [0, 0])
        confusions = Counter()
        outcomes = Counter()
        for entry, image in zip(entries, images, strict=True):
            batch = torch.from_numpy(eval_preprocess(image, loaded.image_size)).to(device)
            with torch.no_grad():
                logits = loaded.model(batch).float().cpu().numpy()[0]
            ranked = np.argsort(logits)[::-1]
            correct = int(ranked[0]) == entry.label
            top1 += correct
            top3 += entry.label in ranked[:3]
            per_class[entry.class_name][0] += correct
            per_class[entry.class_name][1] += 1
            if not correct:
                confusions[(entry.class_name, class_names[int(ranked[0])])] += 1

            outcome = predictor.predict(image, explain=False)
            if outcome.status == "ok":
                outcomes["ok_right" if outcome.predicted_class == entry.class_name else "ok_wrong"] += 1
            else:
                outcomes[outcome.status] += 1

        n = len(entries)
        low, high = wilson(top1, n)
        results[experiment_id] = {
            "label": record["label"],
            "protocol": record["protocol"],
            "lab_test_accuracy": record["test"]["metrics"]["accuracy"],
            "n": n,
            "top1": top1 / n,
            "top1_ci95": [low, high],
            "top3": top3 / n,
            "macro_class_accuracy": float(np.mean([c / t for c, t in per_class.values()])),
            "per_class": {k: {"correct": c, "total": t} for k, (c, t) in sorted(per_class.items())},
            "pipeline": {k: v / n for k, v in sorted(outcomes.items())},
            "pipeline_counts": dict(sorted(outcomes.items())),
            "top_confusions": [{"true": t, "predicted": p, "count": c}
                               for (t, p), c in confusions.most_common(8)],
        }
        del registry._models[alias]  # noqa: SLF001
        if device.type == "cuda":
            torch.cuda.empty_cache()
        r = results[experiment_id]
        print(f"{record['label']:28s} {record['protocol']:20s} real top-1 {r['top1'] * 100:5.1f}% "
              f"[{low * 100:4.1f}-{high * 100:4.1f}]  top-3 {r['top3'] * 100:5.1f}%  "
              f"shown right {r['pipeline'].get('ok_right', 0) * 100:5.1f}%  "
              f"shown WRONG {r['pipeline'].get('ok_wrong', 0) * 100:5.1f}%", flush=True)

    out = settings.results_dir / args.out
    out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "n": len(entries),
        "ood_thresholds_from": deployed["model_name"],
        "results": results,
        "note": ("PlantDoc test split (real field and web photographs), after removing training "
                 "photos that near-duplicate test photos. Labels are PlantDoc's, mapped onto the "
                 "model's classes; PlantDoc is known to contain some label noise."),
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
