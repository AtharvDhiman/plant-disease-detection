"""Would averaging two models' predictions beat either one on real photographs?

Averages the temperature-scaled class probabilities of two ledger runs and
scores each model and the average on real-photo splits (the PlantDoc test
split, plus validation splits for a larger sample). Answers whether serving an
ensemble - roughly twice the memory and latency - is worth it.

    python scripts/evaluate_ensemble.py --experiments <id-a> <id-b>
"""
from __future__ import annotations

import argparse
import json
import sys
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
from app.ml.ood import OODThresholds, softmax  # noqa: E402
from app.ml.preprocess import eval_preprocess  # noqa: E402
from app.ml.registry import registry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs=2, required=True)
    parser.add_argument("--splits", nargs="+", default=["plantdoc_test", "plantdoc_val", "plantwild_val"])
    args = parser.parse_args()

    ledger = {r["experiment_id"]: r
              for r in json.loads((settings.results_dir / "experiments.json").read_text(encoding="utf-8"))}
    split_config = json.loads(settings.splits_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels_of = {e: ledger[e]["label"] for e in args.experiments}

    results = {}
    for split in args.splits:
        entries = read_manifest(Path(split_config["splits"][split]["manifest"]))
        images = []
        for entry in entries:
            with Image.open(entry.path) as image:
                images.append(image.convert("RGB"))
        truth = np.array([e.label for e in entries])
        probabilities = {}
        for experiment_id in args.experiments:
            load_candidate(ledger[experiment_id], experiment_id, OODThresholds.defaults(0.0, 1.0), device)
            loaded = registry._models[experiment_id]  # noqa: SLF001 - evaluation-only
            logits = []
            for image in images:
                batch = torch.from_numpy(eval_preprocess(image, loaded.image_size)).to(device)
                with torch.no_grad():
                    logits.append(loaded.model(batch).float().cpu().numpy()[0] / loaded.temperature)
            probabilities[experiment_id] = softmax(np.array(logits))
            del registry._models[experiment_id]  # noqa: SLF001
            if device.type == "cuda":
                torch.cuda.empty_cache()
        probabilities["ensemble"] = np.mean([probabilities[e] for e in args.experiments], axis=0)
        results[split] = {"n": len(entries), "scores": {}}
        for key, p in probabilities.items():
            ranked = np.argsort(p, axis=1)[:, ::-1]
            results[split]["scores"][key] = {
                "top1": float(np.mean(ranked[:, 0] == truth)),
                "top3": float(np.mean([t in r[:3] for t, r in zip(truth, ranked, strict=True)])),
            }
            print(f"{split:15s} {labels_of.get(key, 'average of both'):26s} "
                  f"top-1 {results[split]['scores'][key]['top1'] * 100:5.1f}%  "
                  f"top-3 {results[split]['scores'][key]['top3'] * 100:5.1f}%", flush=True)

    out = settings.results_dir / "ensemble_eval.json"
    out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiments": args.experiments, "labels": labels_of, "splits": results,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
