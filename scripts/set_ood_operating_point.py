"""Choose the OOD operating point a model is served with, and record why.

``training/calibrate_ood.py`` derives thresholds that keep 95% of *laboratory*
leaves - strict, by construction: a field photograph is less confident than a
PlantVillage one, so the calibrated floor refuses many photographs the model
classifies correctly. The application has been served with a deliberately more
permissive setting since commit 26882f4, after real-world uploads were being
refused; for the deployed model that choice lived only as hand-edited numbers
in a JSON file, indistinguishable from calibrated ones.

This script makes the choice explicit. It reads the calibrated file, writes a
per-model file (``ood_<model>.json``, which the registry prefers over the
generic one) whose ``selected`` thresholds are the chosen operating point, and
keeps both the calibrated values and the reason beside them:

    python scripts/set_ood_operating_point.py --msp-min 0.15 --entropy-max 0.85 \
        --reason "Matches the permissive setting the site has run with since 26882f4."

The measured trade-off between the two settings is produced by
``scripts/ood_operating_points.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msp-min", type=float, required=True)
    parser.add_argument("--entropy-max", type=float, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--source", default=str(settings.metadata_dir / "ood_thresholds.json"),
                        help="The calibrated file to start from.")
    args = parser.parse_args()

    source = Path(args.source)
    calibrated = json.loads(source.read_text(encoding="utf-8"))
    model = calibrated.get("model")
    if not model:
        raise SystemExit(f"{source} does not name the model it was calibrated for.")
    if not 0.0 < args.msp_min < 1.0 or not 0.0 < args.entropy_max <= 1.0:
        raise SystemExit("msp-min must be in (0, 1) and entropy-max in (0, 1].")

    selected = dict(calibrated["selected"])
    calibrated_values = {key: selected[key] for key in ("msp_min", "entropy_max", "energy_max")
                         if key in selected}
    selected["msp_min"] = args.msp_min
    selected["entropy_max"] = args.entropy_max
    # per_score holds the detection/false-alarm rates measured AT the calibrated
    # thresholds. They do not describe the operating point now in force, so they
    # are moved aside rather than left looking as though they do.
    calibrated_values["per_score"] = selected.pop("per_score", None)
    selected["method"] = selected.get("method", "msp+entropy")

    calibrated["selected"] = selected
    calibrated["operating_point"] = {
        "msp_min": args.msp_min,
        "entropy_max": args.entropy_max,
        "chosen_by": "hand",
        "reason": args.reason,
        "set_at": datetime.now(timezone.utc).isoformat(),
        "calibrated_at_target_tpr": {**calibrated_values, "target_tpr": selected.get("target_tpr")},
        "note": ("`selected` holds the operating point in force. The values calibrated to keep "
                 "95% of laboratory leaves are stricter and are recorded here."),
    }
    destination = settings.metadata_dir / f"ood_{model}.json"
    destination.write_text(json.dumps(calibrated, indent=2), encoding="utf-8")
    print(f"Wrote {destination}")
    print(f"  in force   : msp_min={args.msp_min}  entropy_max={args.entropy_max}")
    print(f"  calibrated : msp_min={calibrated_values['msp_min']:.4f}  "
          f"entropy_max={calibrated_values['entropy_max']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
