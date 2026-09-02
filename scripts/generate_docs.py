"""Generate the results documentation from the experiment ledger.

``docs/results.md``, ``docs/model_comparison.md`` and ``docs/ablation_study.md``
are *generated*, never hand-edited, so they can never disagree with the
experiments that produced them. The same generator fills the results section of
``README.md`` between the RESULTS markers.

Run after every training sweep:

    python scripts/generate_docs.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from app.core.config import settings  # noqa: E402

GENERATED_BANNER = (
    "<!-- GENERATED FILE - do not edit by hand.\n"
    "     Produced by scripts/generate_docs.py from artifacts/results/experiments.json.\n"
    "     Re-run that script after any training sweep. -->\n"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def get(record: dict, path: str, default=None):
    node = record
    for key in path.split("."):
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            return default
    return node


def pct(value, digits=2) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def dec(value, digits=4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def num(value) -> str:
    return "—" if value is None else f"{value:,}"


def params(value) -> str:
    if not value:
        return "—"
    return f"{value / 1e6:.2f}M" if value >= 1e6 else f"{value / 1e3:.1f}K"


def canonical_seed(ledger: list[dict]) -> int | None:
    """The seed the study was run at; every other seed is a replicate."""
    seeds = [(r.get("config") or {}).get("seed") for r in ledger]
    seeds = [s for s in seeds if s is not None]
    return min(seeds) if seeds else None


def rows_from_ledger(ledger: list[dict], groups: tuple[str, ...]) -> list[dict]:
    # Seed replicates exist to measure run-to-run variation, not to compete in
    # the comparison. Leaving them in printed the same model three times in the
    # benchmark table, which reads as three different architectures that happen
    # to share a name.
    baseline_seed = canonical_seed(ledger)

    out = []
    for record in ledger:
        if record.get("group") not in groups:
            continue
        if get(record, "test.metrics.accuracy") is None:
            continue
        seed = (record.get("config") or {}).get("seed")
        if baseline_seed is not None and seed is not None and seed != baseline_seed:
            continue
        out.append({
            "label": record.get("label"),
            "model_name": record.get("model_name"),
            "group": record.get("group"),
            "protocol": record.get("protocol"),
            "proposed": bool(record.get("proposed")),
            "notes": record.get("notes", ""),
            "accuracy": get(record, "test.metrics.accuracy"),
            "precision": get(record, "test.metrics.precision_macro"),
            "recall": get(record, "test.metrics.recall_macro"),
            "f1_macro": get(record, "test.metrics.f1_macro"),
            "f1_weighted": get(record, "test.metrics.f1_weighted"),
            "top3": get(record, "test.metrics.top3_accuracy"),
            "top5": get(record, "test.metrics.top5_accuracy"),
            "roc_auc": get(record, "test.metrics.roc_auc_macro_ovr"),
            "kappa": get(record, "test.metrics.cohen_kappa"),
            "ece": get(record, "test.calibration.ece"),
            "ece_scaled": get(record, "test.calibration.after_temperature.ece"),
            "temperature": get(record, "test.calibration.temperature"),
            "parameters": record.get("total_parameters"),
            "size_mb": record.get("model_size_mb"),
            "latency": get(record, "test.latency.single_image_ms"),
            "train_s": record.get("training_time_s"),
            "epochs": record.get("epochs_run"),
            "best_epoch": record.get("best_epoch"),
            "val_acc": record.get("best_val_accuracy"),
            "image_size": get(record, "config.image_size"),
            "test_split": get(record, "test.split"),
            "test_n": get(record, "test.metrics.num_samples"),
        })
    return sorted(out, key=lambda r: -(r["accuracy"] or 0))


def marker(row: dict) -> str:
    return "**★**" if row["proposed"] else ""


# --------------------------------------------------------------------------- #
# Section builders
# --------------------------------------------------------------------------- #


def megabytes(value) -> str:
    return "—" if value is None else f"{value:.1f} MB"


def benchmark_table(rows: list[dict]) -> str:
    lines = [
        "| | Model | Protocol | Accuracy | Precision | Recall | Macro F1 | Top-3 | Params | Size | ms/img | ECE |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {marker(row)} | {row['label']} | {row['protocol']} | {pct(row['accuracy'])} | "
            f"{dec(row['precision'])} | {dec(row['recall'])} | {dec(row['f1_macro'])} | "
            f"{pct(row['top3'])} | {params(row['parameters'])} | {megabytes(row['size_mb'])} | "
            f"{dec(row['latency'], 2)} | {dec(row['ece'])} |"
        )
    return "\n".join(lines)


def ablation_table(summary: dict) -> str:
    if not summary or not summary.get("available"):
        return "_The ablation suite has not been run yet._"
    lines = [
        "| Arm | Configuration | Accuracy | Macro F1 | Macro recall | Δ Macro F1 | Δ Accuracy | Params | ms/img |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in summary["arms"]:
        delta = arm["delta_vs_control"]
        f1_delta = "—" if arm["arm"] == "A" else f"{delta['f1_macro'] * 100:+.2f} pt"
        acc_delta = "—" if arm["arm"] == "A" else f"{delta['accuracy'] * 100:+.2f} pt"
        lines.append(
            f"| **{arm['arm']}** | {arm['label']} | {pct(arm['accuracy'])} | "
            f"{dec(arm['f1_macro'])} | {dec(arm['recall_macro'])} | {f1_delta} | {acc_delta} | "
            f"{num(arm['parameters'])} | {dec(arm['inference_ms'], 2)} |"
        )
    return "\n".join(lines)


def headline_block(rows: list[dict], manifest: dict | None, dataset: dict | None) -> str:
    if not rows:
        return "_No experiments have been logged yet. Run the training pipeline._"

    best_acc = max(rows, key=lambda r: r["accuracy"] or 0)
    best_f1 = max(rows, key=lambda r: r["f1_macro"] or 0)
    with_latency = [r for r in rows if r["latency"]]
    fastest = min(with_latency, key=lambda r: r["latency"]) if with_latency else None
    with_size = [r for r in rows if r["size_mb"]]
    smallest = min(with_size, key=lambda r: r["size_mb"]) if with_size else None
    proposed = [r for r in rows if r["proposed"]]

    # The protocol column is not decoration. A benchmark row is scored on the
    # 4,560-image subset split and a production row on the full 17,572-image
    # split, so "best accuracy" across both is a comparison between different
    # measurements. Naming the protocol on every row is what stops the table
    # reading as a single ranking.
    lines = [
        "| Result | Model | Protocol | Value |",
        "|---|---|---|---|",
        f"| Best accuracy | {best_acc['label']} | {best_acc['protocol']} | "
        f"**{pct(best_acc['accuracy'])}** |",
        f"| Best macro F1 | {best_f1['label']} | {best_f1['protocol']} | "
        f"**{dec(best_f1['f1_macro'])}** |",
    ]
    if fastest:
        lines.append(f"| Fastest inference | {fastest['label']} | {fastest['protocol']} | "
                     f"{dec(fastest['latency'], 2)} ms/image |")
    if smallest:
        lines.append(f"| Smallest model | {smallest['label']} | {smallest['protocol']} | "
                     f"{smallest['size_mb']:.1f} MB |")
    if proposed:
        top = max(proposed, key=lambda r: r["f1_macro"] or 0)
        lines.append(f"| Best proposed (CBAM) model | {top['label']} | {top['protocol']} | "
                     f"{pct(top['accuracy'])} accuracy, {dec(top['f1_macro'])} macro F1 |")
    if manifest:
        lines.append(f"| Serving in the application | {manifest.get('label')} | "
                     f"{manifest.get('protocol', '—')} | selected by "
                     f"`{get(manifest, 'selection.criterion')}` |")
    if dataset:
        lines.append(f"| Evaluated on | held-out test split | — | "
                     f"{num(dataset.get('official_splits', {}).get('valid'))} images, "
                     f"{dataset.get('num_classes')} classes |")
    lines.append("")
    lines.append("Rows with different protocols are **not** comparable: `benchmark` "
                 "models were trained on a fixed subset and scored on a 4,560-image "
                 "split, `production` models on the full data and scored on 17,572 "
                 "images.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #


def build_results_doc(ledger, manifest, dataset, ablation, ood, classical) -> str:
    all_rows = rows_from_ledger(ledger, ("benchmark", "classical", "ablation", "placement"))
    bench_rows = rows_from_ledger(ledger, ("benchmark", "classical"))
    production_rows = [r for r in all_rows if r["protocol"] == "production"]

    parts = [
        GENERATED_BANNER,
        "# Results\n",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from "
        f"`artifacts/results/experiments.json` ({len(all_rows)} completed experiments)._\n",
        "## Headline\n",
        headline_block(all_rows, manifest, dataset),
        "\n★ marks the proposed CBAM architectures.\n",
        "## Benchmark\n",
        "Every model in the `benchmark` protocol was trained on an identical "
        "fixed-budget subset (300 train / 60 val / 120 test images per class, "
        "160 px, 12 epochs, identical optimiser and seed). Only the architecture "
        "differs. Rows marked `production` were trained on the full training "
        "split at 224 px and are **not** directly comparable with the subset rows.\n",
        benchmark_table(bench_rows),
        "",
    ]

    if production_rows:
        parts += [
            "\n## Full-data production runs\n",
            benchmark_table(production_rows),
            "",
        ]

    if ablation and ablation.get("available"):
        verdict = ablation.get("verdict") or {}
        parts += [
            "\n## Attention ablation\n",
            "**Research question:** does integrating channel and spatial attention "
            "through CBAM improve plant disease classification accuracy, robustness "
            "and interpretability compared with conventional CNN architectures?\n",
            ablation_table(ablation),
            "",
            f"\n**Answer from the data:** {verdict.get('statement', '—')}\n",
            f"\n> {verdict.get('caveat', '')}\n",
        ]

    calibrated = [r for r in all_rows if r["ece"] is not None]
    if calibrated:
        parts += ["\n## Confidence calibration\n",
                  "Temperature scaling fits one scalar on the validation split. It "
                  "cannot change which class wins, so accuracy is unchanged by "
                  "construction; only the confidence values move.\n",
                  "| Model | ECE (raw) | ECE (scaled) | Temperature | Improvement |",
                  "|---|---:|---:|---:|---:|"]
        for row in sorted(calibrated, key=lambda r: r["ece"]):
            if row["ece_scaled"] is None:
                continue
            improvement = (row["ece"] - row["ece_scaled"]) / row["ece"] * 100 if row["ece"] else 0
            parts.append(
                f"| {row['label']} | {dec(row['ece'])} | {dec(row['ece_scaled'])} | "
                f"{dec(row['temperature'], 3)} | {improvement:+.0f}% |"
            )
        parts.append("")

    if ood:
        pipeline = get(ood, "metrics.pipeline") or {}
        parts += [
            "\n## Out-of-distribution handling\n",
            f"Calibrated on {ood.get('in_distribution', {}).get('count', '—')} held-out test "
            f"images against {get(ood, 'ood_source.count', '—')} "
            f"{get(ood, 'ood_source.kind', 'OOD')} samples, at a "
            f"{pct(get(ood, 'selected.target_tpr'), 0)} target true-positive rate.\n",
            "| Score | AUROC | Threshold | OOD caught |",
            "|---|---:|---:|---:|",
        ]
        for name, key in (("Max softmax probability", "msp"),
                          ("Normalised entropy", "entropy"),
                          ("Free energy", "energy")):
            auroc = get(ood, f"metrics.auroc.{name}")
            threshold = get(ood, f"selected.{'msp_min' if key == 'msp' else key + '_max'}")
            caught = get(ood, f"selected.per_score.{key}.ood_detection_rate")
            parts.append(f"| {name} | {dec(auroc)} | {dec(threshold)} | {pct(caught, 1)} |")

        if pipeline:
            parts += [
                "",
                "\n### Whole-pipeline rejection\n",
                "The image-quality gate runs *before* the model, so the end-to-end "
                "rate is what a user actually experiences.\n",
                "| Stage | OOD rejected |",
                "|---|---:|",
                f"| Image-quality gate alone | {pct(pipeline.get('quality_gate_rejects_ood'), 1)} |",
                f"| OOD score alone | {pct(pipeline.get('ood_score_rejects_ood'), 1)} |",
                f"| **Combined pipeline** | **{pct(pipeline.get('pipeline_rejects_ood'), 1)}** |",
                f"| Cost: genuine leaves rejected | {pct(pipeline.get('quality_gate_rejects_real_leaves'), 1)} |",
                "",
            ]
            per_family = pipeline.get("per_family") or {}
            if per_family:
                parts += ["\n| OOD family | Quality gate | OOD score | Combined |",
                          "|---|---:|---:|---:|"]
                for family, rates in per_family.items():
                    parts.append(
                        f"| {family} | {pct(rates['quality_gate_rejected'], 1)} | "
                        f"{pct(rates['ood_score_rejected'], 1)} | "
                        f"{pct(rates['pipeline_rejected'], 1)} |"
                    )
                parts.append("")
        if get(ood, "ood_source.caveat"):
            parts.append(f"\n> **Caveat.** {get(ood, 'ood_source.caveat')}\n")

    if classical:
        parts += [
            "\n## Classical baselines\n",
            f"Trained on {classical.get('feature_dim')} hand-crafted features "
            f"(colour histograms and moments, LBP, GLCM, HOG) extracted from "
            f"{num(classical.get('train_samples'))} images.\n",
        ]
        # Naming what was not run matters as much as reporting what was. A
        # reader comparing against the standard four-baseline set would
        # otherwise have to guess whether a missing model was forgotten, failed,
        # or deliberately skipped.
        trained = {row["model_name"] for row in rows_from_ledger(ledger, ("classical",))}
        expected = {
            "logistic_regression": "Logistic Regression",
            "linear_svm": "Linear SVM",
            "random_forest": "Random Forest",
            "xgboost": "XGBoost",
            "rbf_svm": "RBF SVM",
        }
        missing = {key: label for key, label in expected.items() if key not in trained}
        if missing:
            parts.append(
                "\n**Not trained:** "
                + ", ".join(sorted(missing.values()))
                + ". "
            )
            if "rbf_svm" in missing:
                parts.append(
                    "The RBF SVM is skipped by default: `probability=True` fits "
                    "five internal cross-validation folds of an O(n^2) solver over "
                    f"{num(classical.get('train_samples'))} samples, which costs "
                    "hours on this hardware for a baseline the linear SVM already "
                    "covers. Run it explicitly with "
                    "`python training/train_classical.py --models rbf_svm` if the "
                    "comparison is wanted.\n"
                )

    parts += [
        "\n## Reproducing these numbers\n",
        "Every stage is idempotent and resumable, so the whole sequence runs with "
        "one command:\n\n"
        "```bash\n"
        "python scripts/run_pipeline.py\n"
        "```\n\n"
        "or step by step:\n\n"
        "```bash\n"
        "python scripts/probe_hardware.py\n"
        "python scripts/download_dataset.py\n"
        "#   add --path /path/to/dataset to use a copy you already have;\n"
        "#   this is the only stage that can need Kaggle credentials\n"
        "python training/dataset_inspect.py\n"
        "python training/prepare_splits.py\n"
        "python scripts/build_disease_info.py\n"
        "python training/train_classical.py\n"
        "python training/run_experiments.py --suite research\n"
        "python training/run_experiments.py --suite production\n"
        "python training/analyse_results.py\n"
        "python training/export_model.py --criterion f1_macro --torchscript --onnx\n"
        "python training/calibrate_ood.py\n"
        "python scripts/benchmark_serving.py\n"
        "python scripts/generate_docs.py\n"
        "```\n\n"
        "On a memory-constrained machine, wrap the training sweep in the supervisor: "
        "it relaunches the suite if the process is killed and skips whatever is "
        "already in the ledger.\n\n"
        "```bash\n"
        "python scripts/supervise_training.py --suite research\n"
        "```\n",
    ]
    return "\n".join(parts)


def build_comparison_doc(ledger, manifest) -> str:
    rows = rows_from_ledger(ledger, ("benchmark", "classical"))
    parts = [
        GENERATED_BANNER,
        "# Model comparison\n",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}._\n",
        "## Full metric table\n",
        benchmark_table(rows),
        "",
        "\n## Extended metrics\n",
        "| Model | Weighted F1 | Top-5 | ROC-AUC (OvR) | Cohen's κ | Val accuracy | Best epoch | Train time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        train_min = f"{row['train_s'] / 60:.1f} min" if row["train_s"] else "—"
        parts.append(
            f"| {row['label']} | {dec(row['f1_weighted'])} | {pct(row['top5'])} | "
            f"{dec(row['roc_auc'])} | {dec(row['kappa'])} | {pct(row['val_acc'])} | "
            f"{row['best_epoch'] or '—'} | {train_min} |"
        )

    if manifest:
        parts += [
            "\n\n## Production selection\n",
            f"**Serving:** {manifest.get('label')} (`{manifest.get('model_name')}`)\n",
            f"**Criterion:** `{get(manifest, 'selection.criterion')}` — "
            f"{get(manifest, 'selection.criterion_description')}\n",
            f"\n> {get(manifest, 'selection.note', '')}\n",
        ]
        best_by = manifest.get("best_by") or {}
        if best_by:
            parts += ["\n| Criterion | Winner |", "|---|---|"]
            for criterion, name in best_by.items():
                parts.append(f"| `{criterion}` | {name} |")
            parts.append("")

    parts += [
        "\n## Charts\n",
        "| | |",
        "|---|---|",
        "| ![Accuracy](../artifacts/plots/compare_accuracy.png) | ![Macro F1](../artifacts/plots/compare_f1_macro.png) |",
        "| ![Precision](../artifacts/plots/compare_precision.png) | ![Recall](../artifacts/plots/compare_recall.png) |",
        "| ![Size vs accuracy](../artifacts/plots/tradeoff_size_accuracy.png) | ![Latency vs accuracy](../artifacts/plots/tradeoff_latency_accuracy.png) |",
        "",
    ]
    return "\n".join(parts)


def build_ablation_doc(ablation, ledger) -> str:
    placement = rows_from_ledger(ledger, ("placement",))
    parts = [
        GENERATED_BANNER,
        "# Ablation study\n",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}._\n",
        "## Research question\n",
        "> Does integrating channel and spatial attention through CBAM improve "
        "plant disease classification accuracy, robustness and interpretability "
        "compared with conventional CNN architectures?\n",
        "## Design\n",
        "Five arms share **one** backbone, **one** training budget, **one** data "
        "subset and **one** seed. Only the attention block differs, which is what "
        "makes this an ablation rather than five unrelated experiments.\n",
        "| Arm | Attention block | Answers |",
        "|---|---|---|",
        "| A | none | Control |",
        "| B | CBAM channel branch | Does *what to look at* help on its own? |",
        "| C | CBAM spatial branch | Does *where to look* help on its own? |",
        "| D | Squeeze-and-Excitation | Is CBAM's extra max-pooling path worth it over SE? |",
        "| E | Full CBAM | Do the two branches compose? |",
        "\n## Results\n",
        ablation_table(ablation),
        "",
    ]

    noise = (ablation or {}).get("noise_floor") or {}
    if noise.get("available"):
        parts += [
            "\n## Measured noise floor\n",
            "Before reading any delta it is worth knowing how large a difference "
            "this setup produces by chance. Two separate floors are measured, and "
            "they are not interchangeable.\n",
        ]

        nondeterminism = noise.get("nondeterminism")
        if nondeterminism:
            parts += [
                "\n### Floor 1 - repeating the same configuration\n",
                f"{len(nondeterminism['observations'])} configuration(s) here were "
                "trained more than once by construction - same architecture, same "
                "hyperparameters, same seed, differing only in which suite ran "
                "them. The spread within each group is how far apart repeated runs "
                "of the same thing land.\n",
                "| Repeated configuration | Runs | Accuracy | Macro F1 | Macro F1 spread |",
                "|---|---:|---|---|---:|",
            ]
            for observation in nondeterminism["observations"]:
                # `pair` names the extremes; `group` names every run in the set,
                # which can be more than two.
                members = observation.get("group") or observation["pair"]
                names = ", ".join(f"`{name}`" for name in members)
                accuracies = " to ".join(pct(v) for v in observation["accuracy"])
                f1s = " to ".join(dec(v) for v in observation["f1_macro"])
                parts.append(
                    f"| {names} | {observation.get('runs', len(members))} | "
                    f"{accuracies} | {f1s} | "
                    f"{observation['f1_macro_gap'] * 100:.2f} pt |"
                )
            differing = sorted({
                entry
                for observation in nondeterminism["observations"]
                for entry in observation.get("differs_in", [])
            })
            parts += [
                "",
                "\nWhat differs between repeated runs of one configuration:\n",
            ]
            parts += [f"* {entry}" for entry in differing]

            gaps = [o["f1_macro_gap"] * 100 for o in nondeterminism["observations"]]
            if len(gaps) > 1 and max(gaps) > 0:
                parts += [
                    f"\nThe two pairs disagreed by {min(gaps):.2f} pt and "
                    f"{max(gaps):.2f} pt respectively - close to an order of "
                    f"magnitude apart. The larger is used, because a single "
                    f"close-agreeing pair would badly understate how far apart two "
                    f"runs of one configuration can land.\n",
                ]

            parts += [
                "\nThis floor is still a **lower bound**, not a significance "
                "threshold. It holds the initialisation fixed, whereas the arms being "
                "compared do not: CBAM and the plain CNN have different parameter "
                "shapes, so they consume the random stream differently and start from "
                "different weights even at the same seed. A gap between two "
                "architectures therefore carries initialisation variance that this "
                "measurement excludes by construction.\n",
            ]

        seed = noise.get("seed")
        if seed:
            parts += [
                f"\n### Floor 2 - seed variation{' (the threshold actually used)' if noise.get('threshold_basis') == 'seed' else ' (measured, but narrower than Floor 1)'}\n",
                f"`{seed['model_name']}` retrained at {len(seed['f1_macro'])} seeds. "
                "Everything that differs between two architectures now varies - "
                "weight initialisation, augmentation draws, batch composition - "
                "except the architecture itself.\n",
                "| Seed | Accuracy | Macro F1 |",
                "|---|---|---:|",
            ]
            for index, seed_value in enumerate(seed["seeds"]):
                accuracy = seed["accuracy"][index] if index < len(seed["accuracy"]) else None
                parts.append(
                    f"| {seed_value} | {pct(accuracy)} | {dec(seed['f1_macro'][index])} |"
                )
            parts += [
                "",
                f"\nMean macro F1 {dec(seed['mean_f1_macro'])}, "
                f"SD {seed['sd_f1_macro'] * 100:.2f} pt, "
                f"range {seed['spread_f1_macro'] * 100:.2f} pt.\n",
            ]
        else:
            parts += [
                "\n### Floor 2 - seed variation (not yet measured)\n",
                "Seed replicates have not been trained, so the wider floor is "
                "unknown and the fixed-seed bound above is being used in its place. "
                "Every conclusion below is correspondingly weaker than it would be "
                "with replicates. To measure it:\n",
                "```bash",
                "python training/run_experiments.py --models cnn_baseline --seed 43",
                "python training/run_experiments.py --models cnn_baseline --seed 44",
                "```",
            ]

        parts += [
            f"\n**{noise['interpretation']}**\n",
        ]

    if ablation and ablation.get("verdict"):
        verdict = ablation["verdict"]
        parts += [
            "\n## Verdict\n",
            f"{verdict['statement']}\n",
            "\n| Measure | Value |",
            "|---|---:|",
            f"| CBAM Δ macro F1 vs control | {verdict['cbam_delta_f1_macro'] * 100:+.2f} pt |",
            f"| CBAM Δ accuracy vs control | {verdict['cbam_delta_accuracy'] * 100:+.2f} pt |",
            f"| CBAM inference overhead | {verdict['cbam_inference_overhead_pct']:+.1f}% |",
            f"| CBAM parameter overhead | {verdict['cbam_parameter_overhead_pct']:+.2f}% |",
            f"| Best-performing arm | {verdict['best_arm']} (macro F1 {dec(verdict['best_arm_f1'])}) |",
            f"| Is CBAM the best arm? | {'yes' if verdict['cbam_is_best'] else 'no'} |",
            f"\n> **Caveat.** {verdict['caveat']}\n",
            "\n![Ablation](../artifacts/plots/ablation.png)\n",
        ]

    pairs = (ablation or {}).get("transfer_pairs") or {}
    if pairs.get("available"):
        parts += [
            "\n## A second test: CBAM on pretrained backbones\n",
            "The five-arm ablation answers the research question for one from-scratch "
            "CNN. The benchmark independently contains **matched pairs** - the same "
            "ImageNet backbone with and without a CBAM block on its final feature map, "
            "trained under the identical protocol. Those pairs test the same question "
            "in a different regime, and are worth reading precisely because a "
            "pretrained trunk has already learned to suppress irrelevant features, so "
            "there may be less for attention to add.\n",
            "| Backbone | Macro F1 without | Macro F1 with | Delta | Params | Latency |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for entry in pairs["pairs"]:
            marker = " *" if entry.get("beyond_noise") is False else ""
            parts.append(
                f"| {entry['backbone']} | {dec(entry['without_cbam']['f1_macro'])} | "
                f"{dec(entry['with_cbam']['f1_macro'])} | "
                f"{entry['delta_f1_macro'] * 100:+.2f} pt{marker} | "
                f"{entry['parameter_overhead_pct']:+.1f}% | "
                f"{entry['inference_overhead_pct']:+.1f}% |"
            )
        parts += ["", f"\n**Reading:** {pairs['verdict']}\n"]
        # Only explain the marker if something is actually marked; the noise
        # floor is unknown until the ablation suite has run.
        if any(e.get("beyond_noise") is False for e in pairs["pairs"]):
            parts.append(
                f"\n`*` marks a difference smaller than the measured "
                f"{pairs['significance_threshold_f1'] * 100:.2f}-point run-to-run "
                "variation, which a single run each cannot separate from noise.\n"
            )
        parts.append(f"\n> {pairs['caveat']}\n")

    place = (ablation or {}).get("placement") or {}
    if placement:
        parts += [
            "\n## CBAM placement\n",
            "The five arms above vary *which* attention mechanism is used, applying "
            "it at every stage. This varies the other axis: the same CBAM block, "
            "inserted at a different depth. It is a separate question, and on this "
            "dataset it produced the larger effect.\n",
        ]
        if place.get("available"):
            parts += [
                "| Placement | Stages | Accuracy | Macro F1 | Δ vs control | Beyond noise | Params | ms/img |",
                "|---|---:|---:|---:|---:|:---:|---:|---:|",
            ]
            for entry in sorted(place["variants"],
                                key=lambda e: e["stages_with_attention"]):
                beyond = entry.get("beyond_noise")
                mark = "yes" if beyond else "no" if beyond is False else "-"
                parts.append(
                    f"| {entry['label']} | {entry['stages_with_attention']} | "
                    f"{pct(entry['accuracy'])} | {dec(entry['f1_macro'])} | "
                    f"{entry['delta_f1_macro'] * 100:+.2f} pt | {mark} | "
                    f"{num(entry['parameters'])} | {dec(entry['inference_ms'], 2)} |"
                )
            parts += [
                "",
                f"\nControl (`{place['control']}`, no attention): "
                f"{dec(place['control_f1_macro'])} macro F1.\n",
                f"\n**{place['statement']}**\n",
            ]
            replication = place.get("production_replication")
            if replication and replication.get("available"):
                parts += [
                    "\n#### Does it survive full-data training?\n",
                    f"{replication['statement']}\n",
                    "| Placement | Macro F1 | Accuracy | Params |",
                    "|---|---:|---:|---:|",
                    f"| Every stage (`cnn_cbam`) | "
                    f"{dec(replication['all_stages']['f1_macro'])} | "
                    f"{pct(replication['all_stages']['accuracy'])} | "
                    f"{num(replication['all_stages']['parameters'])} |",
                    f"| Last two stages (`place_cbam_last2`) | "
                    f"{dec(replication['last_two_stages']['f1_macro'])} | "
                    f"{pct(replication['last_two_stages']['accuracy'])} | "
                    f"{num(replication['last_two_stages']['parameters'])} |",
                    "",
                ]
            parts += [
                f"\n> {place['caveat']}\n",
            ]
        else:
            parts += [
                "| Placement | Accuracy | Macro F1 | Params | ms/img |",
                "|---|---:|---:|---:|---:|",
            ]
            for row in placement:
                parts.append(
                    f"| {row['label']} | {pct(row['accuracy'])} | {dec(row['f1_macro'])} | "
                    f"{num(row['parameters'])} | {dec(row['latency'], 2)} |"
                )
            parts.append("")
            if place.get("reason"):
                parts.append(f"\n_{place['reason']}_\n")

    return "\n".join(parts)


def update_readme(readme_path: Path, block: str) -> bool:
    """Replace the content between the RESULTS markers in the README."""
    if not readme_path.exists():
        return False
    text = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)", re.DOTALL
    )
    if not pattern.search(text):
        return False
    replacement = f"<!-- RESULTS:START -->\n{block}\n<!-- RESULTS:END -->"
    readme_path.write_text(pattern.sub(lambda _: replacement, text), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-readme", action="store_true")
    args = parser.parse_args()

    ledger = load(settings.results_dir / "experiments.json", []) or []
    if not ledger:
        raise SystemExit("No experiments logged. Run training/run_experiments.py first.")

    manifest = load(settings.exported_dir / "production.json")
    dataset = load(settings.dataset_report_path)
    dataset_summary = None
    if dataset:
        dataset_summary = {
            "num_classes": dataset["classes"]["count"],
            "official_splits": {k: v["num_images"] for k, v in dataset["splits"].items()},
        }
    ablation = load(settings.results_dir / "ablation_summary.json")
    ood = load(settings.metadata_dir / "ood_thresholds.json")
    classical = load(settings.results_dir / "classical_results.json")

    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name, content in (
        ("results.md", build_results_doc(ledger, manifest, dataset_summary, ablation, ood, classical)),
        ("model_comparison.md", build_comparison_doc(ledger, manifest)),
        ("ablation_study.md", build_ablation_doc(ablation, ledger)),
    ):
        (docs_dir / name).write_text(content, encoding="utf-8")
        written.append(name)

    if not args.skip_readme:
        rows = rows_from_ledger(ledger, ("benchmark", "classical", "ablation", "placement"))
        # Benchmark and production rows are NOT comparable and must not share a
        # table. The benchmark protocol scores 300 images per class at 160px on
        # the 4,560-image subset test split; production scores the full 63,265
        # training images at 224px on the 17,572-image test split. Sorting them
        # together put a production model above a benchmark one on a difference
        # that is entirely an artefact of the evaluation set.
        all_rows = rows_from_ledger(ledger, ("benchmark", "classical"))
        bench_rows = [r for r in all_rows if r["protocol"] != "production"]
        prod_rows = [r for r in all_rows if r["protocol"] == "production"]

        sections = [
            headline_block(rows, manifest, dataset_summary),
            "",
            "### Benchmark (identical protocol across all architectures)",
            "",
            "Every row below was trained on the same fixed budget and scored on the "
            "same held-out subset, so the ranking is meaningful. Absolute numbers "
            "are lower than a full-data run would give, by design.",
            "",
            benchmark_table(bench_rows),
        ]
        if prod_rows:
            sections += [
                "",
                "### Full-data production runs",
                "",
                "Trained on the complete training split at 224px and scored on the "
                "full test split. **These numbers cannot be compared with the "
                "benchmark table above** - a different training budget and a "
                "different, larger test set. They are what the deployed system "
                "actually achieves.",
                "",
                benchmark_table(prod_rows),
            ]
        sections += [
            "",
            "### Attention ablation",
            "",
            ablation_table(ablation),
        ]
        block = "\n".join(sections)
        if update_readme(PROJECT_ROOT / "README.md", block):
            written.append("README.md (results section)")

    print("Generated:")
    for name in written:
        print(f"  docs/{name}" if name.endswith(".md") and "README" not in name else f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
