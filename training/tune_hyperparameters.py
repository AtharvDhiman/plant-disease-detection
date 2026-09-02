"""Small, honest hyper-parameter search on the proposed architecture.

Scope, and why it is small
--------------------------
A full sweep is not affordable on a 4 GB GPU that takes ~10 minutes per
configuration. Rather than pretend otherwise, this searches the parameters that
actually matter for this problem, on a reduced epoch budget, and reports the
result as what it is: a coarse search that identifies a good region, not an
optimum.

Searched by default (on ``cnn_cbam``):

* **learning rate** - the single most influential setting for AdamW
* **weight decay**  - regularisation strength
* **dropout**       - the other regularisation knob, and the one most likely to
  interact with attention, which already suppresses uninformative channels

Every trial shares the benchmark protocol's data subset, seed and schedule, so
the trials differ *only* in the searched values. Results are appended to the
same experiment ledger under the ``tuning`` group and never mixed into the
benchmark table.

    python training/tune_hyperparameters.py
    python training/tune_hyperparameters.py --model efficientnet_b0_cbam --epochs 6
    python training/tune_hyperparameters.py --strategy random --trials 8
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import app.core.runtime  # noqa: F401  # isort:skip

import torch  # noqa: E402
from config import (  # noqa: E402
    BENCHMARK_PROTOCOL,
    TrainConfig,
    auto_batch_size,
    probe_hardware,
    recommended_workers,
)
from tracker import ExperimentTracker  # noqa: E402


def _now() -> str:
    """UTC timestamp for the ledger record."""
    return datetime.now(timezone.utc).isoformat()
from trainer import train_model  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.ml.architectures import MODEL_ZOO  # noqa: E402
from app.ml.datasets import load_split_config  # noqa: E402

log = get_logger(__name__)

SEARCH_SPACE = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "dropout": [0.2, 0.4],
}


def grid(space: dict) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, values, strict=False)) for values in itertools.product(*space.values())]


def sample(space: dict, trials: int, seed: int) -> list[dict]:
    """Random search - reaches a good region in far fewer trials than a grid.

    Bergstra & Bengio (2012): when only a few parameters actually matter, random
    search covers the important axes better than a grid of the same size, because
    a grid wastes trials repeating identical values along the unimportant axes.
    """
    rng = random.Random(seed)
    keys = list(space)
    seen, out = set(), []
    while len(out) < trials and len(seen) < len(grid(space)):
        candidate = {key: rng.choice(space[key]) for key in keys}
        signature = tuple(sorted(candidate.items()))
        if signature in seen:
            continue
        seen.add(signature)
        out.append(candidate)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="cnn_cbam", choices=list(MODEL_ZOO))
    parser.add_argument("--strategy", default="random", choices=["grid", "random"])
    parser.add_argument("--trials", type=int, default=6,
                        help="Number of trials for the random strategy.")
    parser.add_argument("--epochs", type=int, default=8,
                        help="Reduced budget - this ranks configurations, it does not "
                             "produce a final model.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    configure_logging(settings.log_level, settings.logs_dir / "tuning.log")
    settings.ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware = probe_hardware(settings.metadata_dir / "amp_probe.json")
    split_config = load_split_config(settings.splits_path)
    tracker = ExperimentTracker(settings.results_dir)

    candidates = (grid(SEARCH_SPACE) if args.strategy == "grid"
                  else sample(SEARCH_SPACE, args.trials, split_config.get("seed", 42)))
    log.info("Hyper-parameter search",
             fields={"model": args.model, "strategy": args.strategy,
                     "trials": len(candidates), "epochs": args.epochs})

    results = []
    for index, params in enumerate(candidates, start=1):
        config = TrainConfig(
            model_name=args.model,
            **{**BENCHMARK_PROTOCOL, "epochs": args.epochs},
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            batch_size=auto_batch_size(args.model, BENCHMARK_PROTOCOL["image_size"],
                                       hardware, settings.batch_size),
            num_workers=recommended_workers(hardware),
            amp=hardware.supports_amp,
            seed=split_config.get("seed", 42),
            group="tuning",
            notes=f"Hyper-parameter trial {index}/{len(candidates)}",
        )
        config.experiment_id = ExperimentTracker.make_experiment_id(
            args.model, "tune", {**config.to_dict(), "dropout": params["dropout"]}
        )

        if tracker.has(config.experiment_id) and not args.force:
            log.info(f"[{index}/{len(candidates)}] skipping (already completed)",
                     fields=params)
            record = tracker.get(config.experiment_id)
            results.append({"params": params, "val_accuracy": record["best_val_accuracy"],
                            "experiment_id": config.experiment_id,
                            "training_time_s": record["training_time_s"]})
            continue

        log.info(f"[{index}/{len(candidates)}] trial", fields=params)
        try:
            # Dropout is a constructor argument, not a TrainConfig field, so it is
            # threaded through as a build override.
            outcome = _train_with_dropout(config, split_config, device, params["dropout"])
        except Exception as exc:  # noqa: BLE001 - one bad trial must not stop the search
            log.error(f"Trial {index} failed: {exc}")
            continue

        results.append({
            "params": params,
            "experiment_id": config.experiment_id,
            "val_accuracy": outcome.best_val_accuracy,
            "val_loss": outcome.best_val_loss,
            "best_epoch": outcome.best_epoch,
            "epochs_run": outcome.epochs_run,
            "training_time_s": outcome.training_time_s,
        })

        # Write the trial to the ledger the moment it finishes, and flush.
        #
        # The skip check at the top of this loop reads the ledger, but nothing
        # used to write to it until all six trials had completed - so the check
        # could never fire and the search was only resumable in appearance. When
        # the process was killed during trial 4, three finished trials and
        # 46 minutes of GPU time were lost, and the restart began again at trial
        # 1. On a machine that kills long jobs, a search that cannot resume is a
        # search that may never finish.
        tracker.log({
            "experiment_id": config.experiment_id,
            "model_name": args.model,
            "label": f"{MODEL_ZOO[args.model].label} (tuning trial {index})",
            "group": "tuning",
            "protocol": "tuning",
            "proposed": False,
            "notes": config.notes,
            "config": {**config.to_dict(), "dropout": params["dropout"]},
            "hyperparameters": params,
            "best_val_accuracy": outcome.best_val_accuracy,
            "best_val_loss": outcome.best_val_loss,
            "best_epoch": outcome.best_epoch,
            "epochs_run": outcome.epochs_run,
            "training_time_s": outcome.training_time_s,
            "logged_at": _now(),
        })
        tracker.flush()

        log.info(f"[{index}/{len(candidates)}] done",
                 fields={"val_acc": round(outcome.best_val_accuracy, 4), **params})

    if not results:
        raise SystemExit("Every trial failed; see artifacts/logs/tuning.log.")

    results.sort(key=lambda r: -(r["val_accuracy"] or 0))
    best = results[0]
    spread = (results[0]["val_accuracy"] or 0) - (results[-1]["val_accuracy"] or 0)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "strategy": args.strategy,
        "epochs_per_trial": args.epochs,
        "search_space": SEARCH_SPACE,
        "selection_metric": "best validation accuracy",
        "trials": results,
        "best": best,
        "spread_val_accuracy": spread,
        "caveat": (
            f"Trials ran for {args.epochs} epochs on the benchmark subset - enough to "
            "rank configurations, not to produce a final model. Selection is on "
            "validation accuracy; the test split is not touched here. With one run "
            "per configuration, a gap smaller than the run-to-run noise floor "
            "reported in the ablation summary should not be read as a real "
            "difference."
        ),
    }
    destination = settings.results_dir / "tuning_results.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 84)
    print(f"HYPER-PARAMETER SEARCH - {args.model} ({args.strategy}, {len(results)} trials)")
    print("=" * 84)
    print(f"{'LR':>10}{'WEIGHT DECAY':>15}{'DROPOUT':>10}{'VAL ACC':>10}{'EPOCH':>8}{'TIME s':>9}")
    print("-" * 84)
    for entry in results:
        p = entry["params"]
        print(f"{p['learning_rate']:>10.0e}{p['weight_decay']:>15.0e}{p['dropout']:>10.2f}"
              f"{entry['val_accuracy']:>10.4f}{entry.get('best_epoch', 0):>8}"
              f"{entry.get('training_time_s', 0):>9.0f}")
    print("=" * 84)
    print(f"Best: lr={best['params']['learning_rate']:.0e}  "
          f"wd={best['params']['weight_decay']:.0e}  "
          f"dropout={best['params']['dropout']}  "
          f"-> validation accuracy {best['val_accuracy']:.4f}")
    print(f"Spread across trials: {spread * 100:.2f} accuracy points")
    print(f"\n{payload['caveat']}\n")
    print(f"Wrote {destination}\n")
    return 0


def _train_with_dropout(config: TrainConfig, split_config: dict, device, dropout: float):
    """Train with a non-default dropout by patching the model factory.

    ``TrainConfig`` deliberately holds only optimisation settings; dropout is an
    architecture argument. Rather than widen the config for one search, the
    builder is temporarily wrapped so the trial stays a one-line change.
    """
    import trainer as trainer_module

    from app.ml.architectures import build_model as original_build

    def build_with_dropout(name, num_classes, pretrained=True, **overrides):
        return original_build(name, num_classes, pretrained=pretrained,
                              dropout=dropout, **overrides)

    trainer_module.build_model = build_with_dropout
    try:
        return train_model(config, split_config, settings.checkpoints_dir / "tuning",
                           tensorboard_dir=None, device=device)
    finally:
        trainer_module.build_model = original_build


if __name__ == "__main__":
    sys.exit(main())
