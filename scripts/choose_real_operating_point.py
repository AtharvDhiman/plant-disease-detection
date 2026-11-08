"""Choose the "not sure" threshold from real photographs, not laboratory ones.

The out-of-distribution check refuses to give a plain diagnosis when the
model's top-class probability is low. The threshold used to come from
laboratory photos (strict: it refused many real photos the model got right) or
from a hand-set permissive value (which let through many confidently wrong
answers on real photos - the failure users reported).

This script picks it from **real validation photographs** (the ``*_val`` splits
of PlantDoc, PlantWild, Plant Pathology 2021, CCMT and the potato field set -
never the PlantDoc test split). It sweeps the threshold and takes the most
permissive one at which the answers that pass are right at least
``--target-precision`` of the time. Photos the quality gate would refuse are
left out, since they never reach the threshold.

    python scripts/choose_real_operating_point.py --experiment <id> --target-precision 0.9
    python scripts/choose_real_operating_point.py --experiment <id> --apply
"""
from __future__ import annotations

import argparse
import json
import subprocess
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
from app.ml.quality import analyse_image  # noqa: E402
from app.ml.registry import registry  # noqa: E402

VALIDATION = ("plantdoc_val", "plantwild_val", "plantpathology2021_val", "ccmt_val",
              "potato_field_val")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--apply", action="store_true",
                        help="Also record the chosen threshold with set_ood_operating_point.py.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    ledger = {r["experiment_id"]: r
              for r in json.loads((settings.results_dir / "experiments.json").read_text(encoding="utf-8"))}
    record = ledger[args.experiment]
    split_config = json.loads(settings.splits_path.read_text(encoding="utf-8"))
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    alias = f"{args.experiment}@op"
    load_candidate(record, alias, OODThresholds.defaults(0.0, 1.0), device)
    loaded = registry._models[alias]  # noqa: SLF001 - evaluation-only

    msp, correct, source = [], [], []
    refused_by_gate = 0
    for name in VALIDATION:
        if name not in split_config["splits"]:
            continue
        for entry in read_manifest(Path(split_config["splits"][name]["manifest"])):
            with Image.open(entry.path) as image:
                image = image.convert("RGB")
            report = analyse_image(image)
            if not report.passed:
                refused_by_gate += 1
                continue
            batch = torch.from_numpy(eval_preprocess(image, loaded.image_size)).to(device)
            with torch.no_grad():
                logits = loaded.model(batch).float().cpu().numpy()
            probabilities = softmax(logits)[0]          # raw, as the OOD check uses
            msp.append(float(probabilities.max()))
            correct.append(int(probabilities.argmax()) == entry.label)
            source.append(name)
    msp, correct = np.array(msp), np.array(correct)
    n = len(msp)
    print(f"{n} real validation photos passed the quality gate ({refused_by_gate} refused); "
          f"model accuracy on them {correct.mean() * 100:.1f}%")

    sweep = []
    for threshold in np.round(np.arange(0.05, 0.96, 0.01), 2):
        accepted = msp >= threshold
        coverage = accepted.mean()
        precision = correct[accepted].mean() if accepted.any() else 1.0
        sweep.append({"msp_min": float(threshold), "coverage": float(coverage),
                      "precision": float(precision),
                      "shown_right": float((accepted & correct).mean()),
                      "shown_wrong": float((accepted & ~correct).mean())})
    eligible = [row for row in sweep if row["precision"] >= args.target_precision]
    if not eligible:
        raise SystemExit("No threshold reaches the target precision.")
    chosen = min(eligible, key=lambda row: row["msp_min"])
    print(f"chosen msp_min {chosen['msp_min']:.2f}: shown {chosen['coverage'] * 100:.1f}% of photos, "
          f"{chosen['precision'] * 100:.1f}% of them right; shown wrong {chosen['shown_wrong'] * 100:.1f}% "
          f"of all photos (vs {sweep[10]['shown_wrong'] * 100:.1f}% at 0.15)")

    out = settings.results_dir / "real_operating_point.json"
    out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment,
        "validation_splits": [s for s in VALIDATION if s in split_config["splits"]],
        "photos": n, "refused_by_quality_gate": refused_by_gate,
        "target_precision": args.target_precision,
        "chosen": chosen,
        "sweep": sweep,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}")

    if args.apply:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "set_ood_operating_point.py"),
                        "--msp-min", f"{chosen['msp_min']:.2f}", "--entropy-max", "1.0",
                        "--method", "msp", "--chosen-by", "real-validation-sweep",
                        "--reason", (f"Chosen on {n} real validation photographs (PlantDoc, PlantWild, "
                                     f"Plant Pathology 2021, CCMT, potato field): the most permissive "
                                     f"threshold at which answers shown as diagnoses are right at least "
                                     f"{args.target_precision:.0%} of the time "
                                     "(scripts/choose_real_operating_point.py).")],
                       check=True, cwd=PROJECT_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
