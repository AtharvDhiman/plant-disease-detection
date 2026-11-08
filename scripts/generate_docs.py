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


def is_field(protocol) -> bool:
    """Field-augmentation rounds (field, field_long) are reported separately."""
    return str(protocol).startswith("field")


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
                 "split; `production`, `field` and `field_long` models were trained on "
                 "the full data and scored on 17,572 images.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Sections drawn from the other result files
# --------------------------------------------------------------------------- #
#
# Each takes the parsed JSON (or None) and returns markdown lines, or [] when
# the file has not been produced - a missing measurement is left out rather
# than shown as a placeholder.


def production_experiment(ledger: list[dict], manifest: dict | None) -> dict | None:
    """The ledger entry of the model the application serves."""
    if not manifest:
        return None
    wanted = manifest.get("experiment_id")
    return next((e for e in ledger if e.get("experiment_id") == wanted), None)


def display_names(dataset_report: dict | None) -> dict:
    details = get(dataset_report or {}, "classes.details") or {}
    if isinstance(details, dict):
        return {raw: info.get("display", raw) for raw, info in details.items()}
    return {info["raw"]: info.get("display", info["raw"]) for info in details}


def dataset_section(report: dict | None) -> list[str]:
    if not report:
        return []
    splits = report.get("splits", {})
    classes = report.get("classes", {})
    lines = ["\n## Dataset\n",
             f"Source: Kaggle `{get(report, 'dataset.kaggle_slug', '—')}` "
             f"(PlantVillage, augmented). {classes.get('count')} classes across "
             f"{classes.get('plant_count')} plant species.\n",
             "| Split | Images | Classes | Smallest class | Largest class | Imbalance |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, split in splits.items():
        counts = list((split.get("class_counts") or {}).values())
        low, high = (min(counts), max(counts)) if counts else (None, None)
        ratio = f"{high / low:.2f}×" if low else "—"
        lines.append(f"| {name} | {num(split.get('num_images'))} | {split.get('num_classes')} | "
                     f"{num(low)} | {num(high)} | {ratio} |")
    # The report stores these either as name lists or as plain counts.
    healthy, diseased = (len(v) if isinstance(v, list) else v
                         for v in (classes.get("healthy_classes"), classes.get("diseased_classes")))
    lines += ["",
              f"- **Healthy classes:** {healthy} · **disease classes:** {diseased}",
              # Semicolons: one plant is itself called "Pepper, bell".
              f"- **Plants:** {'; '.join(classes.get('plants', []))}"]
    dup = report.get("duplicates") or {}
    if dup:
        leak = get(dup, "cross_split_leakage.colliding_hashes")
        lines.append(
            f"- **Duplicate / leakage check** ({dup.get('method')}, "
            f"{dup.get('sampled_per_class')} images sampled per class): "
            f"{get(dup, 'train.duplicate_images', '—')} duplicates in train, "
            f"{get(dup, 'valid.duplicate_images', '—')} in valid, "
            f"{leak if leak is not None else '—'} train/valid collisions.")
    lines.append(
        "\nThe dataset's own `valid` folder is used as the held-out **test** split; "
        "validation for early stopping and calibration is carved out of `train` "
        "(see `data/splits.json`), so no test image influences training or model "
        "selection.\n")
    return lines


def production_metrics_section(experiment: dict | None) -> list[str]:
    metrics = get(experiment or {}, "test.metrics")
    if not metrics:
        return []
    calib = get(experiment, "test.calibration") or {}
    conf = get(experiment, "test.confidence_stats") or {}
    rows = [
        ("Test images", num(metrics.get("num_samples"))),
        ("Accuracy (top-1)", pct(metrics.get("accuracy"))),
        ("Top-3 accuracy", pct(metrics.get("top3_accuracy"))),
        ("Top-5 accuracy", pct(metrics.get("top5_accuracy"))),
        ("Balanced accuracy", pct(metrics.get("balanced_accuracy"))),
        ("Macro precision", dec(metrics.get("precision_macro"))),
        ("Macro recall", dec(metrics.get("recall_macro"))),
        ("Macro F1", dec(metrics.get("f1_macro"))),
        ("Weighted F1", dec(metrics.get("f1_weighted"))),
        ("Cohen's kappa", dec(metrics.get("cohen_kappa"))),
        ("Matthews correlation", dec(metrics.get("matthews_corrcoef"))),
        ("ROC AUC (macro, one-vs-rest)", dec(metrics.get("roc_auc_macro_ovr"), 5)),
        ("Expected calibration error (raw)", dec(calib.get("ece"))),
        ("Brier score", dec(calib.get("brier"))),
        ("Mean confidence when right / wrong",
         f"{pct(conf.get('mean_when_correct'), 1)} / {pct(conf.get('mean_when_wrong'), 1)}"),
        ("Inference, batch 1", f"{dec(get(experiment, 'test.latency.by_batch_size.batch_1.ms_per_image'), 2)} ms "
                               f"({get(experiment, 'test.latency.device', '—')})"),
    ]
    errors = None
    if metrics.get("num_samples") and metrics.get("accuracy") is not None:
        errors = round(metrics["num_samples"] * (1 - metrics["accuracy"]))
    lines = [f"\n## The deployed model: {experiment.get('label')}\n",
             f"Every metric for the model the application serves "
             f"(`{experiment.get('experiment_id')}`), on the full held-out test split.\n",
             "| Metric | Value |", "|---|---:|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    if errors is not None:
        lines.append(f"\n{num(errors)} of {num(metrics['num_samples'])} test images are "
                     "misclassified.\n")
    return lines


def per_class_section(experiment: dict | None, names: dict) -> list[str]:
    per_class = get(experiment or {}, "test.metrics.per_class")
    if not per_class:
        return []
    ordered = sorted(per_class.items(), key=lambda kv: kv[1]["f1"])
    lines = ["\n## Per-class accuracy (deployed model)\n",
             "Sorted hardest first. Recall is the per-class accuracy: the share of that "
             "disease's test images the model gets right.\n",
             "| # | Class | Recall (accuracy) | Precision | F1 | Test images |",
             "|---:|---|---:|---:|---:|---:|"]
    for index, (raw, m) in enumerate(ordered, 1):
        lines.append(f"| {index} | {names.get(raw, raw)} | {pct(m['recall'])} | "
                     f"{pct(m['precision'])} | {dec(m['f1'])} | {num(m['support'])} |")
    perfect = sum(1 for _, m in ordered if m["recall"] >= 1.0)
    lines.append(f"\n{perfect} of {len(ordered)} classes are recognised with 100% recall; "
                 f"the hardest is **{names.get(ordered[0][0], ordered[0][0])}** at "
                 f"{pct(ordered[0][1]['recall'])}.\n")
    return lines


DOMAIN_LABELS = {
    "original": "Original lab photo",
    "background": "Leaf on a soil background",
    "angle": "Photographed off-axis",
    "distance": "Leaf small in the frame (far away)",
    "lighting": "Harsh directional sunlight",
    "compression": "Heavy JPEG compression",
    "combined": "All of the above (closest to a field photo)",
}


def domain_shift_section(shift: dict | None) -> list[str]:
    conditions = (shift or {}).get("conditions")
    if not conditions:
        return []
    base = (conditions.get("original") or {}).get("accuracy")
    lines = ["\n## Accuracy on field-like photos\n",
             "Every number above comes from laboratory photographs. This measures how "
             "the full pipeline (quality gate, dust check, model, OOD check) behaves "
             f"when the same leaves are made to look like field photos - "
             f"{shift.get('images_per_condition')} disease images per condition, "
             "produced by `scripts/domain_shift_probe.py`.\n",
             "| Condition | Correct class | vs lab | Not marked reliable |",
             "|---|---:|---:|---:|"]
    for key, entry in conditions.items():
        acc = entry.get("accuracy")
        delta = "—" if key == "original" or acc is None or base is None \
            else f"{(acc - base) * 100:+.1f} pt"
        lines.append(f"| {DOMAIN_LABELS.get(key, key)} | {pct(acc, 1)} "
                     f"({entry.get('correct')}/{entry.get('total')}) | {delta} | "
                     f"{pct(entry.get('refused_rate'), 1)} |")
    lines += ["",
              "*Correct class* counts the top prediction when the app gave one, including "
              "a provisional or low-confidence answer. A refusal (poor quality, not a "
              "plant, heavy dust) produces no class and counts as not correct - which is "
              "why even the unmodified photos score below the model's lab test accuracy: "
              "the pipeline's own checks refuse a few percent of clean laboratory photos. "
              "*Not marked reliable* is any outcome other than a plain `ok`.",
              ""]
    single = {k: v.get("accuracy") for k, v in conditions.items()
              if k not in ("original", "combined") and v.get("accuracy") is not None}
    if single:
        hardest = min(single, key=single.get)
        lines += [f"The hardest single condition is **{DOMAIN_LABELS.get(hardest, hardest).lower()}** "
                  f"({pct(single[hardest], 1)}). When the leaf is small in the frame the app "
                  "advises moving closer.", ""]
    if shift.get("caveat"):
        lines.append(f"> {shift['caveat']}\n")
    return lines


def dust_section(dust: dict | None) -> list[str]:
    evaluation = (dust or {}).get("evaluation")
    if not evaluation:
        return []
    labels = {
        "clean": "Clean leaf (lab photo)",
        "clean_field": "Clean leaf, field-like conditions",
        "dust_light": "Light dust",
        "dust_moderate": "Moderate dust",
        "dust_heavy": "Heavy dust",
        "dust_light_field": "Light dust, field-like",
        "dust_moderate_field": "Moderate dust, field-like",
        "dust_heavy_field": "Heavy dust, field-like",
    }
    possible = get(evaluation, "threshold.rates") or {}
    heavy = get(evaluation, "heavy_threshold.rates") or {}
    lines = ["\n## Dust, soil and sand detection\n",
             "PlantVillage has no \"dusty leaf\" class, so a dust film used to be "
             "classified as a disease (often powdery mildew) or could hide one. A "
             "small detector in `backend/app/ml/dust.py` flags it before diagnosis. "
             f"Evaluated on {num(evaluation.get('eval_rows'))} held-out images "
             f"(AUC **{dec(evaluation.get('auc'), 3)}**; "
             f"{dec(get(evaluation, 'leave_crops_out.auc'), 3)} on crops withheld from "
             f"training: {', '.join(get(evaluation, 'leave_crops_out.crops', []))}).\n",
             "| Image | n | Flagged \"possible dust\" (provisional) | Flagged \"heavy dust\" (withheld) |",
             "|---|---:|---:|---:|"]
    for key in labels:
        if key in possible:
            lines.append(f"| {labels[key]} | {possible[key]['n']} | "
                         f"{pct(possible[key]['flagged'], 1)} | "
                         f"{pct(heavy.get(key, {}).get('flagged'), 1)} |")
    powdery = evaluation.get("powdery_mildew_false_alarm") or {}
    lines += ["",
              f"Rows marked *clean* are false alarms. Genuine powdery-mildew leaves "
              f"(n={powdery.get('n')}) trip the possible-dust warning "
              f"{pct(powdery.get('at_threshold'), 1)} of the time and the heavy-dust "
              f"refusal {pct(powdery.get('at_heavy'), 1)}, because a pale fungal "
              "coating and dust look alike; that is why the warning says so instead "
              "of refusing.",
              ""]
    if dust.get("caveat"):
        lines.append(f"> {dust['caveat']}\n")
    return lines


def tuning_section(tuning: dict | None) -> list[str]:
    trials = (tuning or {}).get("trials")
    if not trials:
        return []
    keys = list(trials[0]["params"])
    lines = ["\n## Hyperparameter search\n",
             f"{tuning.get('strategy', '').capitalize()} search over "
             f"`{tuning.get('model')}`, {len(trials)} trials of "
             f"{tuning.get('epochs_per_trial')} epochs, ranked by "
             f"{tuning.get('selection_metric')}.\n",
             "| Rank | " + " | ".join(keys) + " | Val accuracy | Val loss | Minutes |",
             "|---:|" + "---:|" * len(keys) + "---:|---:|---:|"]
    ranked = sorted(trials, key=lambda t: t.get("val_accuracy") or 0, reverse=True)
    for rank, trial in enumerate(ranked, 1):
        values = " | ".join(str(trial["params"][k]) for k in keys)
        lines.append(f"| {rank} | {values} | {pct(trial.get('val_accuracy'))} | "
                     f"{dec(trial.get('val_loss'))} | "
                     f"{(trial.get('training_time_s') or 0) / 60:.0f} |")
    best = ranked[0]
    lines.append(f"\nBest: {', '.join(f'{k}={v}' for k, v in best['params'].items())} "
                 f"at {pct(best.get('val_accuracy'))} validation accuracy.")
    # State the learning-rate finding only if the data actually shows it.
    if "learning_rate" in keys:
        top_lr = best["params"]["learning_rate"]
        same = [t["val_accuracy"] for t in trials if t["params"]["learning_rate"] == top_lr]
        other = [t["val_accuracy"] for t in trials if t["params"]["learning_rate"] != top_lr]
        if other and min(same) > max(other):
            lines.append(f"The learning rate dominates: every trial at {top_lr} beats "
                         "every trial at any other rate within this short budget.")
    lines.append("")
    return lines


def serving_section(serving: dict | None) -> list[str]:
    per_request = (serving or {}).get("per_request")
    if not per_request:
        return []
    stages = {
        "quality_ms": "Image-quality checks",
        "preprocess_ms": "Preprocessing",
        "inference_ms": "Model inference",
        "explain_ms": "Explanation heat maps",
        "total_ms": "**Total request**",
    }
    measured = (serving.get("generated_at") or "")[:10] or "—"
    lines = ["\n## Serving latency\n",
             f"Measured through the running API with `scripts/benchmark_serving.py` on "
             f"{measured}: {get(serving, 'model.label')} on {get(serving, 'model.device')}, "
             f"{serving.get('iterations')} requests, with the dust/soil check included in "
             "the quality stage. Re-run the script after changing the model or the gate.\n",
             "| Stage | Median | p95 |", "|---|---:|---:|"]
    for key, label in stages.items():
        if key in per_request:
            lines.append(f"| {label} | {dec(per_request[key]['median_ms'], 1)} ms | "
                         f"{dec(per_request[key]['p95_ms'], 1)} ms |")
    lines.append(f"\nThroughput: {dec(serving.get('throughput_requests_per_second'), 2)} "
                 "requests/s with explanations on. The free Render deployment runs the "
                 "same model through ONNX Runtime on a shared CPU, so expect slower "
                 "responses there.\n")
    return lines


def field_training_section(ledger: list[dict], manifest: dict | None, extra: dict) -> list[str]:
    """The field-augmentation experiment: lab scores, field-like scores, the caveats."""
    runs = [r for r in ledger
            if r.get("model_name") in ("efficientnet_b0", "efficientnet_b0_cbam", "resnet50_cbam")
            and (r.get("protocol") == "production" or is_field(r.get("protocol")))
            and get(r, "test.metrics.accuracy") is not None]
    if not any(is_field(r.get("protocol")) for r in runs):
        return []
    order = {"production": 0, "field": 1, "field_long": 2, "field_long_resnet50": 3,
             "field_real": 4, "field_real_resnet50": 5}
    runs.sort(key=lambda r: (order.get(r["protocol"], 9), r["model_name"]))
    serving = (manifest or {}).get("experiment_id")

    lines = ["\n## Field-condition training\n",
             "Every score above comes from PlantVillage's laboratory photographs, and "
             "the first measurement through the full pipeline showed the gap: the "
             "model trained on lab photos fell to about 70% when the same leaves were "
             "given a soil backdrop, harsh sun and heavy JPEG together, and to under "
             "10% when the leaf was small in the frame. Two training rounds added "
             "**field-condition augmentation** "
             "(`backend/app/ml/field_augment.py`: leaf cut-outs on soil, mulch, "
             "foliage and stone backdrops, small-in-frame placement, directional "
             "light, blur, noise, JPEG - the leaf's hue is never altered).\n",
             "| Model | Protocol | Epochs run | Augmentation | Test accuracy | Macro F1 | "
             "Test errors | Serving |",
             "|---|---|---:|---|---:|---:|---:|:---:|"]
    for r in runs:
        m = get(r, "test.metrics")
        epochs = f"{r.get('epochs_run')}/{get(r, 'config.epochs')}"
        errors = round(m["num_samples"] * (1 - m["accuracy"]))
        flag = " †" if r.get("interrupted") else ""
        lines.append(f"| {r['label']} | {r['protocol']} | {epochs}{flag} | "
                     f"{get(r, 'config.augmentation')} | {pct(m['accuracy'])} | "
                     f"{dec(m['f1_macro'])} | {num(errors)} | "
                     f"{'✔' if r.get('experiment_id') == serving else ''} |")
    lines += ["",
              "`production`: standard augmentation, last 3 blocks fine-tuned, 6 epochs. "
              "`field`: field augmentation on 70% of images, otherwise the same schedule. "
              "`field_long`: field augmentation on 50% of images, **the whole network "
              "fine-tuned**, 10 epochs. All scored on the same 17,572-image test split.",
              ""]
    if any(r.get("interrupted") for r in runs):
        lines += ["† The `field_long` run was interrupted by a machine restart during "
                  "epoch 10 of 10. The trainer saves the best-so-far checkpoint whenever "
                  "validation improves and tests that file at the end, so the epoch-9 "
                  "checkpoint on disk is the model a finished run would have been tested "
                  "on (unless the unfinished epoch improved on it, at a learning rate near "
                  "zero). `scripts/recover_interrupted_run.py` evaluated it exactly as a "
                  "normal run is, and the ledger record is flagged `interrupted`. Its "
                  "history is reconstructed from the training log to four decimals.",
                  ""]

    comparison = (extra or {}).get("field_comparison") or {}
    results = comparison.get("results") or {}
    if results:
        conditions = list(DOMAIN_LABELS)
        header = " | ".join(DOMAIN_LABELS[c] for c in conditions)
        lines += ["### Behaviour on field-like photos\n",
                  f"The same {comparison.get('disease_classes')} disease classes x "
                  f"{comparison.get('per_class')} held-out leaves, put through each field "
                  "condition once and scored by every model identically - through the full "
                  "serving pipeline (quality gate, dust check, classifier, OOD check) with "
                  "one set of OOD thresholds for all, so a difference is the network. "
                  "Cells show the share classified correctly, with the number of "
                  "confident wrong answers (status `ok` but wrong) in brackets.\n",
                  f"| Model | Lab test | {header} |",
                  "|---|---:|" + "---:|" * len(conditions)]

        def sort_key(item):
            alias = item[0]
            model, _, protocol = alias.partition("@")
            return (order.get(protocol, 9), model)

        for alias, result in sorted(results.items(), key=sort_key):
            cells = []
            for condition in conditions:
                s = result["conditions"][condition]
                cells.append(f"{s['correct'] / s['n'] * 100:.1f}% ({s['ok_wrong']})")
            model, _, protocol = alias.partition("@")
            marker = " **←**" if result.get("experiment_id") == serving else ""
            lines.append(f"| {result['label']}{marker} | "
                         f"{pct(result['lab_test_accuracy'])} | " + " | ".join(cells) + " |")
        n = next(iter(results.values()))["conditions"]["original"]["n"]
        lines += ["",
                  f"n = {n} images per condition. A refusal produces no class and counts "
                  "as not correct. The row marked **←** is the model the application serves.",
                  ""]

    forced_reason = get(manifest or {}, "selection.forced_reason")
    if forced_reason:
        lines += ["### Why this model serves\n", f"{forced_reason}\n",
                  "See [Accuracy on real photographs](#accuracy-on-real-photographs).\n"]
    elif get(manifest or {}, "protocol") and is_field(manifest.get("protocol")):
        lines += ["### Why this model serves\n",
                  "The serving model is chosen by macro F1 on the 17,572-image test split "
                  "among all full-data runs (`training/export_model.py`), and the field "
                  "comparison above was checked against two conditions before switching: "
                  "it had to beat the previous model on the field-like photos, and stay "
                  "within 0.5 points of its lab accuracy. It does both.\n"]
    lines += ["### Reading these results honestly\n",
              "* **Not independent of the training.** The augmentation imitates the same "
              "kinds of condition the field-like test applies (through different "
              "generators), so the field-like scores flatter the field-trained models. "
              "Real field photographs are the independent test - see "
              "[Accuracy on real photographs](#accuracy-on-real-photographs).",
              "* **The gain cannot be attributed to CBAM.** Round one trained plain "
              "EfficientNet-B0 and EfficientNet-B0 + CBAM identically and they tied "
              "(98.32% vs 98.25%). Round two trained only the CBAM model, with a longer "
              "schedule, the whole network unfrozen and milder augmentation changed "
              "together, so it can't separate the contribution of any one of those "
              "changes either - and no plain-EfficientNet control was run on that "
              "schedule. The earlier `production` models used a shorter, shallower "
              "schedule (6 epochs, last 3 blocks fine-tuned), so some of the gain may "
              "simply be that the original models were not trained as far as they could be.",
              "* **Round one cost lab accuracy** (about 0.7 points); the milder, longer "
              "round two more than recovered it.",
              ""]

    points = ((extra or {}).get("ood_points") or {}).get("points")
    # Superseded once the threshold is chosen on real photographs: this table
    # compares two settings for an earlier model and would describe a threshold
    # that is no longer in force.
    if (extra or {}).get("real_op"):
        points = None
    if points and "calibrated_95tpr" in points and "in_force" in points:
        lines += ["### Out-of-distribution operating point\n",
                  "The OOD thresholds calibrated to keep 95% of *laboratory* leaves are "
                  f"strict (top-class probability at least {points['calibrated_95tpr']['msp_min']:.2f}); "
                  "a field photograph is less confident than a lab one, so that setting "
                  "refuses photographs the model gets right. The application is served "
                  f"with the permissive setting it has used since commit 26882f4 "
                  f"(at least {points['in_force']['msp_min']:.2f}), recorded explicitly by "
                  "`scripts/set_ood_operating_point.py`. Measured through the full "
                  "pipeline on the same photos (`scripts/ood_operating_points.py`):\n",
                  "| Condition | Calibrated: right | wrong | not identified | "
                  "In force: right | wrong | not identified |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for condition in DOMAIN_LABELS:
            cells = []
            for name in ("calibrated_95tpr", "in_force"):
                t = points[name]["conditions"][condition]
                n = t["n"]
                cells += [pct(t.get("ok_right", 0) / n, 0), pct(t.get("ok_wrong", 0) / n, 0),
                          pct(t.get("not_identified", 0) / n, 0)]
            lines.append(f"| {DOMAIN_LABELS[condition]} | " + " | ".join(cells) + " |")
        lines += ["",
                  "*Right* and *wrong* are answers shown with status `ok`; *not identified* "
                  "is flagged out-of-distribution (the closest match is still shown, with a "
                  "warning). The strict setting removes most confident wrong answers but "
                  "refuses many correct photos; the permissive one shows more correct "
                  "diagnoses at the price of a few confident wrong ones. Choose with the "
                  "table, not by default.",
                  ""]
    return lines


def real_photo_section(extra: dict, manifest: dict | None) -> list[str]:
    """Accuracy on real field and web photographs (PlantDoc test split)."""
    real = (extra or {}).get("real_photos") or {}
    results = real.get("results") or {}
    if not results:
        return []
    prep = (extra or {}).get("plantdoc_prep") or {}
    serving = (manifest or {}).get("experiment_id")
    lines = ["\n## Accuracy on real photographs\n",
             "Every other table uses PlantVillage laboratory photographs, or lab photographs "
             "altered to look like field shots. This one uses **real photographs taken in "
             "fields or collected from the web**: the held-out test split of PlantDoc "
             f"(Singh et al., CODS-COMAD 2020; CC-BY-4.0), {real.get('n')} photos. Its classes "
             "map onto the model's own, so nothing about the model's outputs changes.\n"]
    if prep:
        lines.append(
            f"Before any training, {prep.get('train_photos_removed_as_test_near_duplicates', 0)} "
            "PlantDoc training photos were removed because a perceptual hash matched a test "
            f"photo ({prep.get('of_which_labelled_differently', 0)} of them carried a different "
            "label), so no test photo was seen in training.\n")
    lines += ["| Model | Trained on | Real photos: top-1 (95% CI) | Top-3 | Shown right | "
              "Shown **wrong** | Lab test accuracy |",
              "|---|---|---:|---:|---:|---:|---:|"]
    order = {"production": 0, "field": 1, "field_long": 2, "field_long_resnet50": 3,
             "field_real": 4, "field_real_resnet50": 5, "field_real_all": 6,
             "field_real_all_resnet50": 7, "field_real_plus": 8, "field_real_plus_resnet50": 9}
    trained_on = {"field_real": "lab + 1 real-photo dataset, 6 epochs",
                  "field_real_resnet50": "lab + 1 real-photo dataset, 6 epochs",
                  "field_real_all": "lab + 3 real-photo datasets, 6 epochs",
                  "field_real_all_resnet50": "lab + 3 real-photo datasets, 6 epochs",
                  "field_real_plus": "lab + 5 real-photo datasets, 10 epochs",
                  "field_real_plus_resnet50": "lab + 5 real-photo datasets, 10 epochs"}
    for experiment_id, r in sorted(results.items(), key=lambda kv: order.get(kv[1]["protocol"], 99)):
        trained = trained_on.get(r["protocol"], "lab photos only")
        low, high = r["top1_ci95"]
        marker = " **←**" if experiment_id == serving else ""
        lines.append(f"| {r['label']}{marker} | {trained} | {pct(r['top1'], 1)} "
                     f"({pct(low, 0)}–{pct(high, 0)}) | {pct(r['top3'], 1)} | "
                     f"{pct(r['pipeline'].get('ok_right', 0), 1)} | "
                     f"{pct(r['pipeline'].get('ok_wrong', 0), 1)} | "
                     f"{pct(r['lab_test_accuracy'])} |")
    lines += ["",
              "*Top-1/top-3*: the network alone, on every photo. *Shown right / wrong*: what the "
              "full application displays with status `ok` (the rest are flagged as uncertain, "
              "dusty or not identified, or refused), with every model judged at the one "
              "\"not sure\" threshold now in force - so these columns compare networks, not the "
              "settings each was originally served with. The row marked **←** is the model "
              "the application serves.",
              "",
              "**What this means.** Real photographs are much harder than laboratory ones: "
              "several leaves, real backgrounds, different lighting, cultivars and disease "
              "stages. A model's PlantVillage accuracy (~99.7%) is not its accuracy on a "
              "photo from a phone or a web search, and this table is the closer estimate. "
              "PlantDoc's labels are themselves known to contain some noise, and its test "
              "split is small - read differences smaller than the intervals as ties.",
              ""]

    sources = get(extra or {}, "splits.external_sources") or {}
    extra_prep = (extra or {}).get("extra_prep") or {}
    if sources:
        lines += ["### Real-photo training data\n",
                  "Fine-tuning used a stratified PlantVillage subset (so no class is forgotten) "
                  "plus these real-photo sources. Only classes that map unambiguously onto the "
                  "model's 38 were used; photos were resized to 512 px, de-duplicated, and any "
                  "photo matching a PlantDoc test photo was removed.\n",
                  "| Source | Photos used | Licence | Citation |", "|---|---:|---|---|"]
        used = {"plantdoc": sum(get(extra, "plantdoc_prep.splits", {}).get(k, 0)
                                for k in ("plantdoc_train", "plantdoc_val", "plantdoc_test"))}
        for name, info in get(extra_prep, "sources", {}).items():
            used[name] = info.get("photos_used")
        for name, info in sources.items():
            lines.append(f"| [{name}]({info.get('url')}) | {num(used.get(name))} | "
                         f"{info.get('licence')} | {info.get('citation')} |")
        lines += ["",
                  "PlantWild's licence (CC BY-NC-ND 4.0) is non-commercial: a model fine-tuned "
                  "on it should be used for research and education, which is this project's "
                  "purpose. No photos from any source are committed to this repository.",
                  ""]

    op = (extra or {}).get("real_op") or {}
    if op.get("chosen"):
        chosen = op["chosen"]
        at_015 = next((r for r in op.get("sweep", []) if abs(r["msp_min"] - 0.15) < 1e-6), None)
        lines += ["### When the app says \"not sure\"\n",
                  "Below a minimum top-class probability the app does not give a plain "
                  "diagnosis; it says it is unsure and shows the closest matches. That threshold "
                  f"was chosen on {num(op.get('photos'))} real validation photographs "
                  "(`scripts/choose_real_operating_point.py`; never the PlantDoc test split) as "
                  f"the most permissive one at which answers shown as a diagnosis are right at "
                  f"least {pct(op.get('target_precision'), 0)} of the time.\n",
                  "| Threshold | Shown as diagnosis | Of those, right | Confidently wrong (of all photos) |",
                  "|---:|---:|---:|---:|"]
        rows = [r for r in op.get("sweep", []) if round(r["msp_min"] * 100) % 10 == 0 or r is chosen
                or abs(r["msp_min"] - chosen["msp_min"]) < 1e-6]
        for r in rows:
            mark = " **(in force)**" if abs(r["msp_min"] - chosen["msp_min"]) < 1e-6 else ""
            lines.append(f"| {r['msp_min']:.2f}{mark} | {pct(r['coverage'], 1)} | "
                         f"{pct(r['precision'], 1)} | {pct(r['shown_wrong'], 1)} |")
        if at_015:
            lines += ["",
                      f"At the previous setting (0.15) this model would show every real photo as "
                      f"a diagnosis, {pct(at_015['shown_wrong'], 1)} of them confidently wrong; at "
                      f"{chosen['msp_min']:.2f} that falls to {pct(chosen['shown_wrong'], 1)}, at "
                      f"the cost of saying \"not sure\" for {pct(1 - chosen['coverage'], 1)} of "
                      "photos - for which the right answer is usually among the closest matches "
                      "shown."]
        lines.append("")
    return lines


def readme_extras(deployed: dict | None, names: dict, extra: dict) -> list[str]:
    """Short versions of the extra sections, linking to the full tables."""
    lines: list[str] = []
    metrics = get(deployed or {}, "test.metrics")
    if metrics:
        lines += ["", f"### The deployed model ({deployed.get('label')}) at a glance", "",
                  "| Accuracy | Top-3 | Macro F1 | Balanced acc. | Cohen's κ | ROC AUC | Test images |",
                  "|---:|---:|---:|---:|---:|---:|---:|",
                  f"| **{pct(metrics.get('accuracy'))}** | {pct(metrics.get('top3_accuracy'))} | "
                  f"{dec(metrics.get('f1_macro'))} | {pct(metrics.get('balanced_accuracy'))} | "
                  f"{dec(metrics.get('cohen_kappa'))} | {dec(metrics.get('roc_auc_macro_ovr'), 5)} | "
                  f"{num(metrics.get('num_samples'))} |"]
        per_class = metrics.get("per_class") or {}
        if per_class:
            hardest = sorted(per_class.items(), key=lambda kv: kv[1]["f1"])[:5]
            lines += ["", "Hardest classes (per-class accuracy = recall):", "",
                      "| Class | Accuracy | F1 |", "|---|---:|---:|"]
            lines += [f"| {names.get(raw, raw)} | {pct(m['recall'])} | {dec(m['f1'])} |"
                      for raw, m in hardest]
            lines.append("\nAll 38 classes: [`docs/results.md` → Per-class accuracy]"
                         "(docs/results.md#per-class-accuracy-deployed-model).")
    real = get(extra, "real_photos.results") or {}
    if real:
        lines += ["", "### On real photographs", "",
                  f"Held-out real field and web photos (PlantDoc test split, "
                  f"{get(extra, 'real_photos.n')} photos, never seen in training):", "",
                  "| Model | Right (top-1) | Right answer in top 3 | Confidently wrong |",
                  "|---|---:|---:|---:|"]
        serving = (deployed or {}).get("experiment_id")
        keep = [(k, r) for k, r in real.items()
                if r["protocol"] in ("production", "field_long") or k == serving
                or r["protocol"].startswith("field_real_plus")]
        for experiment_id, r in keep:
            what = ("**live: trained on real photos**" if experiment_id == serving
                    else "trained on lab photos only" if "real" not in r["protocol"]
                    else "trained on real photos")
            lines.append(f"| {r['label']} ({what}) | {pct(r['top1'], 1)} | {pct(r['top3'], 1)} | "
                         f"{pct(r['pipeline'].get('ok_wrong', 0), 1)} |")
        lines.append("\nWhen it is not sure, the site says so and shows the closest matches instead "
                     "of a diagnosis - [details](docs/results.md#accuracy-on-real-photographs).")
    conditions = get(extra, "domain_shift.conditions")
    if conditions:
        lines += ["", "### Beyond the lab test set", "",
                  "| Condition | Correct class |", "|---|---:|"]
        lines += [f"| {DOMAIN_LABELS.get(k, k)} | {pct(v.get('accuracy'), 1)} |"
                  for k, v in conditions.items()]
        lines.append("\nLeaves imitated as field photos, through the full live pipeline - "
                     "[details](docs/results.md#accuracy-on-field-like-photos).")
    rates = get(extra, "dust.evaluation.threshold.rates")
    if rates:
        lines += ["", f"**Dust detection:** flags {pct(get(rates, 'dust_heavy.flagged'), 0)} of "
                  f"heavily and {pct(get(rates, 'dust_moderate.flagged'), 0)} of moderately "
                  f"dusted leaves, with false alarms on {pct(get(rates, 'clean.flagged'), 1)} of "
                  "clean lab photos and "
                  f"{pct(get(rates, 'clean_field.flagged'), 1)} of clean field-like photos "
                  "([details](docs/results.md#dust-soil-and-sand-detection))."]
    return lines


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #


def build_results_doc(ledger, manifest, dataset, ablation, ood, classical,
                      extra: dict | None = None) -> str:
    extra = extra or {}
    all_rows = rows_from_ledger(ledger, ("benchmark", "classical", "ablation", "placement"))
    # Full-data runs (production and field-augmented) have their own section
    # below; ranking them inside the benchmark table put full-data runs among
    # subset runs as if comparable.
    bench_rows = [r for r in rows_from_ledger(ledger, ("benchmark", "classical"))
                  if r["protocol"] != "production" and not is_field(r["protocol"])]
    production_rows = [r for r in all_rows
                       if r["protocol"] == "production" or is_field(r["protocol"])]
    deployed = production_experiment(ledger, manifest)
    names = display_names(extra.get("dataset_report"))

    parts = [
        GENERATED_BANNER,
        "# Results\n",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from "
        f"`artifacts/results/experiments.json` ({len(all_rows)} completed experiments)._\n",
        "## Headline\n",
        headline_block(all_rows, manifest, dataset),
        "\n★ marks the proposed CBAM architectures.\n",
        *dataset_section(extra.get("dataset_report")),
        *production_metrics_section(deployed),
        *per_class_section(deployed, names),
        "## Benchmark\n",
        "Every model in the `benchmark` protocol was trained on an identical "
        "fixed-budget subset (300 train / 60 val / 120 test images per class, "
        "160 px, 12 epochs, identical optimiser and seed). Only the architecture "
        "differs. Full-data runs (`production` and `field`) are in their own "
        "section below and are **not** directly comparable with these subset rows.\n",
        benchmark_table(bench_rows),
        "",
    ]

    if production_rows:
        parts += [
            "\n## Full-data runs\n",
            "Trained on the complete training split at 224 px and scored on the full "
            "17,572-image test set, so these rows are comparable with each other. The "
            "`production` protocol uses standard augmentation; `field` and `field_long` "
            "add field-condition augmentation (see "
            "[Field-condition training](#field-condition-training)).\n",
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

    parts += field_training_section(ledger, manifest, extra)
    parts += real_photo_section(extra, manifest)
    parts += domain_shift_section(extra.get("domain_shift"))
    parts += dust_section(extra.get("dust"))
    parts += tuning_section(extra.get("tuning"))
    parts += serving_section(extra.get("serving"))

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
    placement = [r for r in rows_from_ledger(ledger, ("placement",))
                 if r["protocol"] != "production" and not is_field(r["protocol"])]
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
    extra = {
        "dataset_report": dataset,
        "domain_shift": load(settings.results_dir / "domain_shift.json"),
        "dust": load(settings.metadata_dir / "dust_detector.json"),
        "tuning": load(settings.results_dir / "tuning_results.json"),
        "serving": load(settings.results_dir / "serving_benchmark.json"),
        "field_comparison": load(settings.results_dir / "field_comparison.json"),
        "ood_points": load(settings.results_dir / "ood_operating_points.json"),
        "real_photos": load(settings.results_dir / "real_photo_eval.json"),
        "plantdoc_prep": load(settings.results_dir / "plantdoc_preparation.json"),
        "extra_prep": load(settings.results_dir / "realworld_extra_preparation.json"),
        "real_op": load(settings.results_dir / "real_operating_point.json"),
        "splits": load(settings.splits_path),
    }

    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name, content in (
        ("results.md", build_results_doc(ledger, manifest, dataset_summary, ablation, ood,
                                         classical, extra)),
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
        full_data = lambda r: r["protocol"] == "production" or is_field(r["protocol"])  # noqa: E731
        bench_rows = [r for r in all_rows if not full_data(r)]
        prod_rows = [r for r in all_rows if full_data(r)]

        sections = [
            headline_block(rows, manifest, dataset_summary),
            *readme_extras(production_experiment(ledger, manifest),
                           display_names(dataset), extra),
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
                "### Full-data runs",
                "",
                "Trained on the complete training split at 224px and scored on the "
                "full test split. **These numbers cannot be compared with the "
                "benchmark table above** - a different training budget and a "
                "different, larger test set. They are what the deployed system "
                "actually achieves. `field` and `field_long` add field-condition "
                "augmentation; see [Field-condition training]"
                "(docs/results.md#field-condition-training).",
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
