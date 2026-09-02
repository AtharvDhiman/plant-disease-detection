"""Orchestrate the deep-learning experiments: benchmark, ablation, placement, production.

    python training/run_experiments.py --suite benchmark
    python training/run_experiments.py --suite ablation
    python training/run_experiments.py --suite placement
    python training/run_experiments.py --suite production --models cnn_cbam efficientnet_b0_cbam
    python training/run_experiments.py --suite all

Every run is trained under the protocol declared in ``config.py``, evaluated on
the held-out test split and appended to ``artifacts/results/experiments.json``.
Re-running an already-completed configuration is skipped unless ``--force`` is
passed, so the suite is resumable after an interruption.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import app.core.runtime  # noqa: F401  # isort:skip

import torch  # noqa: E402
from config import (  # noqa: E402
    BENCHMARK_PROTOCOL,
    PRODUCTION_PROTOCOL,
    make_configs,
    probe_hardware,
    save_configs,
)
from evaluate import evaluate_checkpoint  # noqa: E402
from metrics import plot_training_curves  # noqa: E402
from tracker import ExperimentTracker  # noqa: E402
from trainer import save_history, train_model  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.ml.architectures import MODEL_ZOO  # noqa: E402
from app.ml.datasets import load_split_config  # noqa: E402

log = get_logger(__name__)

SUITES: dict[str, dict] = {
    "benchmark": {
        "models": [n for n, s in MODEL_ZOO.items() if s.group == "benchmark"],
        "protocol": BENCHMARK_PROTOCOL,
        "tag": "bench",
    },
    "ablation": {
        "models": [n for n, s in MODEL_ZOO.items() if s.group == "ablation"],
        "protocol": BENCHMARK_PROTOCOL,
        "tag": "abl",
    },
    "placement": {
        "models": [n for n, s in MODEL_ZOO.items() if s.group == "placement"],
        "protocol": BENCHMARK_PROTOCOL,
        "tag": "place",
    },
    "production": {
        # Retraining only the CBAM models would decide the production question
        # before asking it: whichever won would be a CBAM model by construction.
        # The benchmark gave CBAM no advantage that clears the measured noise
        # floor, so the full-data suite has to include credible non-CBAM
        # candidates for the selection criterion to mean anything.
        #
        #   cnn_cbam              the proposed architecture, retrained at full scale
        #   efficientnet_b0_cbam  the proposed transfer model
        #   efficientnet_b0       the same backbone without CBAM - the control that
        #                         makes "CBAM was selected" a finding rather than
        #                         an artefact of the candidate list
        #   place_cbam_last2      best custom-CNN variant in the benchmark, and the
        #                         only attention result that cleared the noise floor
        "models": [
            "cnn_cbam",
            "efficientnet_b0_cbam",
            "efficientnet_b0",
            "place_cbam_last2",
        ],
        "protocol": PRODUCTION_PROTOCOL,
        "tag": "prod",
    },
}


def run_suite(
    suite_name: str,
    models: list[str] | None,
    force: bool,
    device: torch.device,
    split_config: dict,
    tracker: ExperimentTracker,
    overrides: dict,
    worker_override: int | None = None,
    seed_override: int | None = None,
) -> list[dict]:
    """Train + evaluate every model in one suite; returns the new records."""
    suite = SUITES[suite_name]
    names = models or suite["models"]
    hardware = probe_hardware(settings.metadata_dir / "amp_probe.json")

    configs = make_configs(names, suite["protocol"], hardware,
                           base_batch_size=settings.batch_size,
                           worker_override=worker_override, **overrides)

    # Imbalance handling comes from the measured dataset, not from a guess.
    imbalance = split_config.get("imbalance", {})
    for config in configs:
        if imbalance.get("strategy") == "class_weights":
            config.class_weight_scheme = imbalance.get("class_weight_scheme")
        elif imbalance.get("strategy") == "balanced_sampler":
            config.use_balanced_sampler = True
        elif imbalance.get("strategy") == "focal_loss":
            config.use_focal_loss = True
        config.seed = seed_override if seed_override is not None else split_config.get("seed", config.seed)
        config.group = MODEL_ZOO[config.model_name].group
        config.notes = MODEL_ZOO[config.model_name].notes
        config.experiment_id = ExperimentTracker.make_experiment_id(
            config.model_name, suite["tag"], config.to_dict()
        )

    save_configs(configs, settings.results_dir / f"configs_{suite_name}.json")
    log.info(f"Suite '{suite_name}'", fields={"models": len(configs), "device": str(device)})

    records: list[dict] = []
    for index, config in enumerate(configs, start=1):
        spec = MODEL_ZOO[config.model_name]
        if tracker.has(config.experiment_id) and not force:
            log.info(f"[{index}/{len(configs)}] skipping (already completed)",
                     fields={"experiment_id": config.experiment_id})
            records.append(tracker.get(config.experiment_id))
            continue

        log.info(
            f"[{index}/{len(configs)}] training {spec.label}",
            fields={"experiment_id": config.experiment_id, "batch": config.batch_size,
                    "img": config.image_size, "epochs": config.epochs,
                    "split": config.train_split},
        )
        try:
            result = train_model(
                config, split_config, settings.checkpoints_dir,
                tensorboard_dir=settings.logs_dir / "tensorboard", device=device,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log.warning("CUDA OOM - retrying at half batch size",
                        fields={"model": config.model_name, "batch": config.batch_size})
            config.batch_size = max(4, config.batch_size // 2)
            config.experiment_id = ExperimentTracker.make_experiment_id(
                config.model_name, suite["tag"], config.to_dict()
            )
            result = train_model(
                config, split_config, settings.checkpoints_dir,
                tensorboard_dir=settings.logs_dir / "tensorboard", device=device,
            )
        except Exception:  # noqa: BLE001 - one bad model must not kill the suite
            log.error(f"Training failed for {config.model_name}\n{traceback.format_exc()}")
            continue

        save_history(result, settings.results_dir / f"history_{config.experiment_id}.json")
        plot_training_curves(
            result.history,
            settings.plots_dir / f"curves_{config.experiment_id}.png",
            f"{spec.label} - training curves",
        )

        test_result = evaluate_checkpoint(
            Path(result.checkpoint_path), split_config, config.test_split, device,
            calibration_split=config.val_split, make_plots=True,
            plot_prefix=config.experiment_id,
        )

        record = {
            "experiment_id": config.experiment_id,
            "model_name": config.model_name,
            "label": spec.label,
            "group": spec.group,
            "proposed": spec.proposed,
            "notes": spec.notes,
            "protocol": suite_name,
            "dataset_version": split_config.get("created_at"),
            "config": config.to_dict(),
            "hardware": hardware.as_dict(),
            "history": result.history,
            "best_val_accuracy": result.best_val_accuracy,
            "best_val_loss": result.best_val_loss,
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "training_time_s": result.training_time_s,
            "total_parameters": result.total_parameters,
            "trainable_parameters": result.trainable_parameters,
            "model_size_mb": result.model_size_mb,
            "stopped_early": result.stopped_early,
            "checkpoint_path": result.checkpoint_path,
            "test": test_result,
            "plots": {
                "training_curves": str(settings.plots_dir / f"curves_{config.experiment_id}.png"),
                **test_result.get("plots", {}),
            },
        }
        tracker.log(record)
        records.append(record)

        m = test_result["metrics"]
        log.info(
            f"[{index}/{len(configs)}] done",
            fields={"model": config.model_name, "test_acc": round(m["accuracy"], 4),
                    "macro_f1": round(m["f1_macro"], 4),
                    "ms/img": round(test_result["latency"]["single_image_ms"] or 0, 2),
                    "train_s": round(result.training_time_s)},
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=["benchmark"], nargs="+",
                        choices=[*SUITES, "all", "research"],
                        help="Which experiment suite(s) to run. 'research' = "
                             "benchmark + ablation + placement; 'all' adds production.")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Restrict the suite to these model names.")
    parser.add_argument("--force", action="store_true",
                        help="Retrain configurations that are already in the ledger.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the split config seed. Used to train seed replicates of the "
                             "same architecture, which measure how much of an accuracy gap between "
                             "two models is initialisation noise rather than architecture.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override the auto-estimated DataLoader worker count.")
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--test-split", default=None)
    args = parser.parse_args()

    configure_logging(settings.log_level, settings.logs_dir / "experiments.log")
    settings.ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware = probe_hardware(settings.metadata_dir / "amp_probe.json")
    log.info("Hardware", fields=hardware.as_dict())

    if not settings.splits_path.exists():
        raise FileNotFoundError("data/splits.json missing - run training/prepare_splits.py first.")
    split_config = load_split_config(settings.splits_path)

    overrides = {k: v for k, v in {
        "epochs": args.epochs,
        "image_size": args.image_size,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "test_split": args.test_split,
    }.items() if v is not None}
    if args.batch_size is not None:
        settings.batch_size = args.batch_size

    tracker = ExperimentTracker(settings.results_dir)
    requested: list[str] = []
    for name in args.suite:
        if name == "all":
            requested.extend(SUITES)
        elif name == "research":
            requested.extend(["benchmark", "ablation", "placement"])
        else:
            requested.append(name)
    # Preserve order while removing duplicates.
    suites = list(dict.fromkeys(requested))

    all_records: list[dict] = []
    for suite_name in suites:
        all_records.extend(
            run_suite(suite_name, args.models, args.force, device, split_config, tracker,
                      overrides, worker_override=args.num_workers,
                      seed_override=args.seed)
        )

    print("\n" + "=" * 108)
    print(f"{'MODEL':<34}{'GROUP':<11}{'ACC':>8}{'MACRO F1':>10}{'TOP-3':>8}"
          f"{'PARAMS':>11}{'MB':>8}{'ms/img':>9}{'TRAIN s':>9}")
    print("=" * 108)
    for record in sorted(all_records, key=lambda r: -(r.get("test", {}).get("metrics", {}).get("accuracy") or 0)):
        m = record.get("test", {}).get("metrics", {})
        lat = record.get("test", {}).get("latency", {})
        marker = "*" if record.get("proposed") else " "
        print(f"{marker}{record['label']:<33}{record['group']:<11}"
              f"{m.get('accuracy', 0):>8.4f}{m.get('f1_macro', 0):>10.4f}"
              f"{m.get('top3_accuracy', 0):>8.4f}{record['total_parameters']:>11,}"
              f"{record['model_size_mb']:>8.1f}{lat.get('single_image_ms') or 0:>9.2f}"
              f"{record['training_time_s']:>9.0f}")
    print("=" * 108)
    print("* = proposed architecture")
    print(f"Ledger: {tracker.json_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
