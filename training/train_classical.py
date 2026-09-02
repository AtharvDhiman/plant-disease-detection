"""Classical machine-learning baselines on hand-crafted features.

Extracts the descriptors defined in ``app.ml.features`` for the benchmark
subset, then trains and evaluates Logistic Regression, a linear SVM, an RBF SVM,
Random Forest and XGBoost on *exactly* the splits the deep models use, so the
two families are directly comparable.

    python training/train_classical.py
    python training/train_classical.py --models logistic_regression random_forest
    python training/train_classical.py --train-split train --test-split test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import app.core.runtime  # noqa: F401  # isort:skip

import numpy as np  # noqa: E402
from metrics import compute_metrics, plot_confusion_matrix, plot_model_comparison  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC, LinearSVC  # noqa: E402
from tracker import ExperimentTracker  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.ml.datasets import load_split_config, read_manifest  # noqa: E402
from app.ml.features import extract_from_path, feature_group_slices  # noqa: E402

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Feature caching
# --------------------------------------------------------------------------- #


def _worker(path: str):
    """Top-level so it can be pickled for Windows' spawn-based process pool."""
    return extract_from_path(path)


def build_feature_matrix(
    split_config: dict, split_name: str, workers: int, cache_dir: Path
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract (and cache) features for one split.

    Extraction costs ~16 ms per image; caching to ``.npz`` means the five model
    families are trained on identical features without paying that cost five
    times.
    """
    cache_path = cache_dir / f"features_{split_name}.npz"
    manifest = Path(split_config["splits"][split_name]["manifest"])
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if int(cached["n_source_rows"]) == len(read_manifest(manifest)):
            log.info("Loaded cached features",
                     fields={"split": split_name, "shape": list(cached["X"].shape)})
            return cached["X"], cached["y"], {"cached": True, "extraction_time_s": 0.0}

    entries = read_manifest(manifest)
    paths = [e.path for e in entries]
    labels = np.array([e.label for e in entries], dtype=np.int64)

    log.info("Extracting features", fields={"split": split_name, "images": len(paths),
                                            "workers": workers})
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        vectors = list(pool.map(_worker, paths, chunksize=32))
    elapsed = time.perf_counter() - started

    keep = [i for i, v in enumerate(vectors) if v is not None]
    if len(keep) < len(vectors):
        log.warning("Unreadable images skipped during feature extraction",
                    fields={"split": split_name, "skipped": len(vectors) - len(keep)})

    X = np.stack([vectors[i] for i in keep]).astype(np.float32)
    y = labels[keep]

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, n_source_rows=len(entries))
    log.info("Extracted features",
             fields={"split": split_name, "shape": list(X.shape),
                     "seconds": round(elapsed, 1),
                     "ms_per_image": round(elapsed * 1000 / max(len(paths), 1), 2)})
    return X, y, {"cached": False, "extraction_time_s": elapsed,
                  "ms_per_image": elapsed * 1000 / max(len(paths), 1)}


# --------------------------------------------------------------------------- #
# Model definitions
# --------------------------------------------------------------------------- #


def build_classical_models(n_classes: int, seed: int, n_jobs: int,
                           xgb_device: str = "auto") -> dict[str, dict]:
    """Model factories with the rationale for each choice recorded alongside."""
    return {
        "logistic_regression": {
            "label": "Logistic Regression (hand-crafted features)",
            "needs_scaling": True,
            "factory": lambda: LogisticRegression(
                max_iter=1500, C=1.0, multi_class="multinomial", n_jobs=n_jobs,
                random_state=seed,
            ),
            "notes": "Linear probe - shows how separable the classes are in feature space.",
        },
        "linear_svm": {
            "label": "Linear SVM (hand-crafted features)",
            "needs_scaling": True,
            # LinearSVC is one-vs-rest, so with 38 classes each calibration fold
            # costs 38 binary fits. Running the folds in parallel is the
            # difference between ~7 and ~20 minutes on this machine.
            "factory": lambda: CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=4000, dual="auto", random_state=seed),
                method="sigmoid", cv=3, n_jobs=n_jobs,
            ),
            "notes": "Max-margin linear classifier; Platt-scaled so top-k metrics are defined.",
        },
        "rbf_svm": {
            "label": "RBF SVM (hand-crafted features)",
            "needs_scaling": True,
            "factory": lambda: SVC(
                C=8.0, gamma="scale", kernel="rbf", probability=True,
                cache_size=512, random_state=seed,
            ),
            "notes": "Non-linear kernel; O(n^2) training, so only run on the benchmark subset.",
        },
        "random_forest": {
            "label": "Random Forest (hand-crafted features)",
            "needs_scaling": False,
            "factory": lambda: RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=1,
                max_features="sqrt", n_jobs=n_jobs, random_state=seed,
            ),
            "notes": "Bagged trees - strong tabular baseline, gives feature importances.",
        },
        "xgboost": {
            "label": "XGBoost (hand-crafted features)",
            "needs_scaling": False,
            "factory": lambda: _make_xgboost(n_classes, seed, n_jobs, xgb_device),
            "notes": "Gradient-boosted trees, histogram split finding.",
        },
    }


def _make_xgboost(n_classes: int, seed: int, n_jobs: int, device: str = "auto"):
    """XGBoost classifier.

    ``device`` defaults to the GPU when one is present, but can be forced to
    ``cpu`` so this script can run alongside a deep-learning training job
    without competing for the (small) VRAM budget.
    """
    import torch
    from xgboost import XGBClassifier

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return XGBClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.12,
        subsample=0.85,
        colsample_bytree=0.7,
        objective="multi:softprob",
        num_class=n_classes,
        tree_method="hist",
        device=device,
        n_jobs=n_jobs,
        random_state=seed,
        eval_metric="mlogloss",
    )


# --------------------------------------------------------------------------- #
# Training + evaluation
# --------------------------------------------------------------------------- #


def evaluate_classical(
    name: str,
    spec: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    make_plots: bool,
) -> dict:
    """Fit one classical model and produce the same metric bundle as the CNNs."""
    steps = []
    if spec["needs_scaling"]:
        # Colour histograms live in [0,1] while GLCM contrast can reach hundreds;
        # without standardisation the margin-based models are dominated by scale.
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", spec["factory"]()))
    pipeline = Pipeline(steps)

    started = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - started

    started = time.perf_counter()
    probabilities = pipeline.predict_proba(X_test)
    inference_time = time.perf_counter() - started
    predictions = probabilities.argmax(axis=1)

    # Some estimators only see the classes present in the training split; expand
    # to the full class space so the metric shapes match the deep models.
    if probabilities.shape[1] != len(class_names):
        full = np.zeros((len(probabilities), len(class_names)), dtype=np.float64)
        full[:, pipeline.named_steps["model"].classes_] = probabilities
        probabilities = full
        predictions = probabilities.argmax(axis=1)

    metrics = compute_metrics(y_test, predictions, probabilities, class_names)

    plots = {}
    if make_plots:
        cm_path = settings.confusion_dir / f"classical_{name}.png"
        plot_confusion_matrix(y_test, predictions, class_names, cm_path,
                              normalize=True, title=f"{spec['label']} - test")
        plots["confusion_matrix"] = str(cm_path)

    model_obj = pipeline.named_steps["model"]
    importances = None
    if hasattr(model_obj, "feature_importances_"):
        groups = feature_group_slices()
        raw = np.asarray(model_obj.feature_importances_, dtype=np.float64)
        importances = {g: float(raw[s].sum()) for g, s in groups.items()}

    return {
        "model_name": name,
        "label": spec["label"],
        "notes": spec["notes"],
        "family": "classical",
        "metrics": metrics,
        "training_time_s": train_time,
        "inference_total_s": inference_time,
        "inference_ms_per_image": inference_time * 1000 / max(len(X_test), 1),
        "feature_group_importance": importances,
        "plots": plots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--train-split", default="train_bench")
    parser.add_argument("--test-split", default="test_bench")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-rbf", action="store_true",
                        help="Skip the RBF SVM, which is the slowest model here.")
    parser.add_argument("--xgb-device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Force XGBoost onto the CPU when the GPU is busy training.")
    parser.add_argument("--force", action="store_true",
                        help="Refit models that are already in the experiment ledger.")
    args = parser.parse_args()

    configure_logging(settings.log_level, settings.logs_dir / "classical.log")
    settings.ensure_dirs()

    split_config = load_split_config(settings.splits_path)
    class_names = split_config["class_names"]
    seed = split_config.get("seed", settings.seed)
    cache_dir = settings.artifacts_dir / "features"

    X_train, y_train, train_info = build_feature_matrix(
        split_config, args.train_split, args.workers, cache_dir
    )
    X_test, y_test, test_info = build_feature_matrix(
        split_config, args.test_split, args.workers, cache_dir
    )
    log.info("Feature matrices ready",
             fields={"train": list(X_train.shape), "test": list(X_test.shape)})

    zoo = build_classical_models(len(class_names), seed, n_jobs=args.workers,
                                 xgb_device=args.xgb_device)
    names = args.models or list(zoo)
    if args.skip_rbf:
        names = [n for n in names if n != "rbf_svm"]

    tracker = ExperimentTracker(settings.results_dir)
    results = []
    for index, name in enumerate(names, start=1):
        if name not in zoo:
            log.warning("Unknown classical model", fields={"name": name})
            continue

        experiment_id = f"classical__{name}__{args.train_split}"
        # Fitting an RBF SVM or a calibrated linear SVM costs tens of minutes on
        # this hardware; like the deep suite, a rerun should resume rather than
        # repeat work that is already in the ledger.
        if tracker.has(experiment_id) and not args.force:
            record = tracker.get(experiment_id)
            log.info(f"[{index}/{len(names)}] skipping (already in the ledger)",
                     fields={"model": name,
                             "acc": round(record["test"]["metrics"]["accuracy"], 4)})
            results.append({
                "model_name": name,
                "label": record["label"],
                "notes": record.get("notes", ""),
                "family": "classical",
                "metrics": record["test"]["metrics"],
                "training_time_s": record.get("training_time_s"),
                "inference_ms_per_image": record["test"]["latency"]["single_image_ms"],
                "feature_group_importance": record.get("feature_group_importance"),
                "plots": record["test"].get("plots", {}),
                "resumed": True,
            })
            continue

        log.info(f"[{index}/{len(names)}] fitting {zoo[name]['label']}")
        try:
            result = evaluate_classical(
                name, zoo[name], X_train, y_train, X_test, y_test,
                class_names, not args.no_plots,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            log.error(f"{name} failed: {exc}")
            continue

        results.append(result)
        m = result["metrics"]
        log.info(f"[{index}/{len(names)}] done",
                 fields={"model": name, "acc": round(m["accuracy"], 4),
                         "macro_f1": round(m["f1_macro"], 4),
                         "fit_s": round(result["training_time_s"], 1)})

        tracker.log({
            "experiment_id": experiment_id,
            "model_name": name,
            "label": result["label"],
            "group": "classical",
            "proposed": False,
            "notes": result["notes"],
            "protocol": "classical",
            "dataset_version": split_config.get("created_at"),
            "config": {"train_split": args.train_split, "test_split": args.test_split,
                       "feature_dim": int(X_train.shape[1]), "image_size": 96},
            "training_time_s": result["training_time_s"],
            "total_parameters": None,
            "model_size_mb": None,
            "best_val_accuracy": None,
            "test": {
                "metrics": result["metrics"],
                "latency": {"single_image_ms": result["inference_ms_per_image"]},
                "plots": result["plots"],
            },
            "feature_group_importance": result["feature_group_importance"],
        })

    payload = {
        "train_split": args.train_split,
        "test_split": args.test_split,
        "feature_dim": int(X_train.shape[1]),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_extraction": {"train": train_info, "test": test_info},
        "results": results,
    }
    out = settings.results_dir / "classical_results.json"
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    if results and not args.no_plots:
        rows = [{"label": r["label"], "accuracy": r["metrics"]["accuracy"],
                 "f1_macro": r["metrics"]["f1_macro"], "proposed": False} for r in results]
        plot_model_comparison(rows, "accuracy", settings.plots_dir / "classical_accuracy.png",
                              "Classical ML baselines - test accuracy")
        plot_model_comparison(rows, "f1_macro", settings.plots_dir / "classical_f1.png",
                              "Classical ML baselines - macro F1")

    print("\n" + "=" * 88)
    print(f"{'CLASSICAL MODEL':<44}{'ACC':>9}{'MACRO F1':>11}{'TOP-3':>9}{'FIT s':>9}")
    print("=" * 88)
    for r in sorted(results, key=lambda r: -r["metrics"]["accuracy"]):
        m = r["metrics"]
        print(f"{r['label']:<44}{m['accuracy']:>9.4f}{m['f1_macro']:>11.4f}"
              f"{m['top3_accuracy']:>9.4f}{r['training_time_s']:>9.1f}")
    print("=" * 88)
    print(f"Saved: {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
