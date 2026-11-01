"""Record a training run that was killed after its best checkpoint was saved.

``trainer.train_model`` writes the best-so-far checkpoint to disk every time
validation improves, and evaluates *that* file after the last epoch. So when a
run dies late - here, a machine restart during epoch 10 of 10 - the checkpoint
on disk is already the model the run would have been tested on, unless the
unfinished epoch would have improved on it.

This script does what the end of a normal run does, for such a checkpoint:

* reads the epoch history back from ``experiments.log`` (4 decimal places, not
  the 6 a normal run stores - the in-memory history died with the process);
* scores the checkpoint on the held-out test split with the same evaluation
  code, including temperature fitting on the validation split;
* appends a ledger record, flagged ``interrupted`` with the epochs actually
  run against the number planned, so the shortfall is visible everywhere the
  record is shown rather than only here.

It refuses to run for an experiment that is already in the ledger, and it does
not invent anything: every number is either read from the checkpoint, the log
or the evaluation.

    python scripts/recover_interrupted_run.py --suite field_long \
        --experiment-id efficientnet_b0_cbam__fieldlong__98176ce8
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
from config import TrainConfig, probe_hardware  # noqa: E402
from evaluate import evaluate_checkpoint  # noqa: E402
from metrics import plot_training_curves  # noqa: E402
from run_experiments import SUITES  # noqa: E402
from tracker import ExperimentTracker  # noqa: E402
from trainer import TrainResult, save_history  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.ml.architectures import MODEL_ZOO, build_model, count_parameters, model_size_mb  # noqa: E402
from app.ml.datasets import load_split_config  # noqa: E402


def history_from_log(log_path: Path, experiment_id: str, model_name: str) -> dict:
    """Epoch rows logged after this experiment's start line."""
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [],
               "val_acc": [], "lr": [], "epoch_time_s": []}
    started = False
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        message = str(entry.get("message", ""))
        if entry.get("experiment_id") == experiment_id and "training" in message:
            started = True
            history = {key: [] for key in history}      # a restart begins again
            continue
        if not started:
            continue
        if message.startswith("[") and "training" in message:
            break                                        # the next experiment began
        if message.startswith("epoch ") and entry.get("model") == model_name:
            history["epoch"].append(int(message.split()[1].split("/")[0]))
            history["train_loss"].append(entry["train_loss"])
            history["train_acc"].append(entry["train_acc"])
            history["val_loss"].append(entry["val_loss"])
            history["val_acc"].append(entry["val_acc"])
            history["lr"].append(float(entry["lr"]))
            history["epoch_time_s"].append(entry["sec"])
    return history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=list(SUITES))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    tracker = ExperimentTracker(settings.results_dir)
    if tracker.has(args.experiment_id):
        raise SystemExit(f"{args.experiment_id} is already in the ledger - nothing to recover.")

    checkpoint = settings.checkpoints_dir / f"{args.experiment_id}.pt"
    if not checkpoint.is_file():
        raise SystemExit(f"No checkpoint at {checkpoint}.")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = TrainConfig.from_dict(payload["config"])
    if config.experiment_id != args.experiment_id:
        raise SystemExit("Checkpoint belongs to a different experiment id.")

    history = history_from_log(settings.logs_dir / "experiments.log",
                               args.experiment_id, config.model_name)
    if not history["epoch"]:
        raise SystemExit("No epoch rows found in experiments.log for this experiment.")
    if payload["epoch"] not in history["epoch"]:
        raise SystemExit(f"Checkpoint is from epoch {payload['epoch']} but the log has "
                         f"epochs {history['epoch']}.")

    model = build_model(config.model_name, len(payload["class_names"]), pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    total, trainable = count_parameters(model)
    result = TrainResult(
        config=config, history=history,
        best_val_accuracy=payload["val_accuracy"], best_val_loss=payload["val_loss"],
        best_epoch=payload["epoch"], epochs_run=len(history["epoch"]),
        training_time_s=sum(history["epoch_time_s"]), checkpoint_path=str(checkpoint),
        total_parameters=total, trainable_parameters=trainable,
        model_size_mb=model_size_mb(model), stopped_early=False,
    )
    print(f"Recovered: best epoch {result.best_epoch}, {result.epochs_run} of "
          f"{config.epochs} planned epochs run, val acc {result.best_val_accuracy:.4f}")

    spec = MODEL_ZOO[config.model_name]
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    split_config = load_split_config(settings.splits_path)
    save_history(result, settings.results_dir / f"history_{args.experiment_id}.json")
    plot_training_curves(history, settings.plots_dir / f"curves_{args.experiment_id}.png",
                         f"{spec.label} - training curves (interrupted at epoch "
                         f"{result.epochs_run}/{config.epochs})")
    test_result = evaluate_checkpoint(
        checkpoint, split_config, config.test_split, device,
        calibration_split=config.val_split, make_plots=True, plot_prefix=args.experiment_id,
    )

    note = (f"Interrupted by a machine restart during epoch {result.epochs_run + 1} of "
            f"{config.epochs}; the best checkpoint (epoch {result.best_epoch}) was evaluated, "
            "as a finished run would have done. History reconstructed from the log.")
    record = {
        "experiment_id": args.experiment_id,
        "model_name": config.model_name,
        "label": spec.label,
        "group": spec.group,
        "proposed": spec.proposed,
        "notes": f"{spec.notes} {note}".strip(),
        "protocol": args.suite,
        "interrupted": True,
        "epochs_planned": config.epochs,
        "dataset_version": split_config.get("created_at"),
        "config": config.to_dict(),
        "hardware": probe_hardware(settings.metadata_dir / "amp_probe.json").as_dict(),
        "history": history,
        "best_val_accuracy": result.best_val_accuracy,
        "best_val_loss": result.best_val_loss,
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "training_time_s": result.training_time_s,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "model_size_mb": result.model_size_mb,
        "stopped_early": False,
        "checkpoint_path": str(checkpoint),
        "test": test_result,
        "plots": {
            "training_curves": str(settings.plots_dir / f"curves_{args.experiment_id}.png"),
            **test_result.get("plots", {}),
        },
    }
    tracker.log(record)
    metrics = test_result["metrics"]
    print(f"Test accuracy {metrics['accuracy']:.4f}  macro F1 {metrics['f1_macro']:.4f}")
    print(f"Ledger: {tracker.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
