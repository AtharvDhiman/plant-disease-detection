"""Assemble an averaged ensemble from two trained runs and record it like a model.

The ensemble (``ensemble_cbam`` in the model zoo) averages the members'
temperature-scaled probabilities. This script

1. loads the two members' checkpoints and their fitted temperatures,
2. saves the combined weights as one checkpoint, so the torch registry, the
   evaluation code and the ONNX exporter can treat it as a single model,
3. evaluates it on the lab test split exactly as ``run_experiments.py`` evaluates
   a trained model (metrics, calibration, confusion matrix, latency), and
4. adds a ledger record (protocol ``<first member's protocol>_ensemble``), so the
   documentation, the export step and the API see it like any other run.

Nothing is trained. Run it only when no training job is running: a training run
rewrites the whole ledger when it finishes and would drop this record.

    python scripts/build_ensemble.py --members <efficientnet-id> <resnet50-id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

import app.core.runtime  # noqa: F401,E402  # isort:skip

import torch  # noqa: E402
from config import TrainConfig  # noqa: E402
from evaluate import collect_predictions, evaluate_checkpoint  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tracker import ExperimentTracker  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.ml.architectures import MODEL_ZOO, AveragedEnsemble, build_model, count_parameters  # noqa: E402
from app.ml.datasets import load_split  # noqa: E402
from app.ml.transforms import build_eval_transform  # noqa: E402

ENSEMBLE = "ensemble_cbam"


def member_temperature(record: dict) -> float:
    return float(((record.get("test") or {}).get("calibration") or {}).get("temperature") or 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--members", nargs=2, required=True,
                        help="Ledger ids, in the model zoo's member order "
                             "(EfficientNet-B0 + CBAM first: it is the explained member).")
    parser.add_argument("--tag", default=None, help="Protocol tag in the id (default: derived).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    tracker = ExperimentTracker(settings.results_dir)
    records = [tracker.get(i) for i in args.members]
    missing = [i for i, r in zip(args.members, records, strict=True) if r is None]
    if missing:
        raise SystemExit(f"Not in the ledger: {missing}")
    expected = list(MODEL_ZOO[ENSEMBLE].kwargs["members"])
    if [r["model_name"] for r in records] != expected:
        raise SystemExit(f"Members must be {expected} in that order; got "
                         f"{[r['model_name'] for r in records]}")
    first = records[0]
    if first["config"]["image_size"] != records[1]["config"]["image_size"]:
        raise SystemExit("Members were trained at different image sizes.")

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    members, class_names = [], None
    for record in records:
        path = Path(record["checkpoint_path"])
        path = path if path.is_absolute() else PROJECT_ROOT / path
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if class_names is not None and payload["class_names"] != class_names:
            raise SystemExit("Members were trained on different class lists.")
        class_names = payload["class_names"]
        member = build_model(record["model_name"], len(class_names), pretrained=False)
        member.load_state_dict(payload["model_state_dict"])
        members.append(member)
    temperatures = [member_temperature(r) for r in records]
    ensemble = AveragedEnsemble(members, temperatures).eval()

    protocol = f"{first['protocol']}_ensemble"
    tag = args.tag or (first["experiment_id"].split("__")[1] + "ens")
    config = TrainConfig.from_dict({**first["config"], "model_name": ENSEMBLE,
                                    "init_checkpoint": None, "num_workers": 0})
    experiment_id = ExperimentTracker.make_experiment_id(
        ENSEMBLE, tag, {**config.to_dict(), "members": args.members})
    config.experiment_id = experiment_id

    checkpoint = settings.checkpoints_dir / f"{experiment_id}.pt"
    torch.save({"model_state_dict": ensemble.state_dict(), "class_names": class_names,
                "config": config.to_dict(), "epoch": 0, "members": args.members,
                "member_temperatures": temperatures}, checkpoint)
    print(f"Saved {checkpoint.name}  (temperatures {[round(t, 4) for t in temperatures]})")

    split_config = json.loads(settings.splits_path.read_text(encoding="utf-8"))
    test_result = evaluate_checkpoint(checkpoint, split_config, config.test_split, device,
                                      calibration_split=config.val_split, make_plots=True,
                                      plot_prefix=experiment_id)

    # Validation accuracy, for the ledger's best_val_accuracy column.
    ensemble.to(device)
    val_ds = load_split(split_config, config.val_split, build_eval_transform(config.image_size))
    _, val_prob, val_labels = collect_predictions(
        ensemble, DataLoader(val_ds, batch_size=32, shuffle=False), device, amp=False)
    val_accuracy = float((val_prob.argmax(axis=1) == val_labels).mean())

    spec = MODEL_ZOO[ENSEMBLE]
    total, _ = count_parameters(ensemble)
    record = {
        "experiment_id": experiment_id,
        "model_name": ENSEMBLE,
        "label": spec.label,
        "group": spec.group,
        "proposed": False,
        "notes": ("Not trained: the average of two fine-tuned networks' calibrated "
                  f"probabilities ({args.members[0]} and {args.members[1]}). "
                  "Training time is the members' combined."),
        "members": args.members,
        "member_temperatures": temperatures,
        "protocol": protocol,
        "dataset_version": split_config.get("created_at"),
        "config": config.to_dict(),
        "hardware": first.get("hardware"),
        "history": {},
        "best_val_accuracy": val_accuracy,
        "best_val_loss": None,
        "best_epoch": None,
        "epochs_run": 0,
        "training_time_s": sum(r.get("training_time_s") or 0 for r in records),
        "total_parameters": total,
        "trainable_parameters": 0,
        "model_size_mb": test_result["model_size_mb"],
        "stopped_early": False,
        "checkpoint_path": str(checkpoint),
        "test": test_result,
        "plots": test_result.get("plots", {}),
    }
    tracker.log(record)
    m = test_result["metrics"]
    print(f"{experiment_id}: lab test accuracy {m['accuracy'] * 100:.2f}%  macro F1 "
          f"{m['f1_macro'] * 100:.2f}%  val {val_accuracy * 100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
