"""Generate the analysis notebooks.

The notebooks are written by a script rather than committed as hand-edited JSON
for one reason: a notebook that re-implements the pipeline will drift from it.
These notebooks **import the project's own modules** and display their real
output, so they cannot disagree with what the training scripts do. They are a
narrated view of the system, not a second copy of it.

    python scripts/build_notebooks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

BOOTSTRAP = """\
# Make the project's own modules importable, and apply the OpenMP fix that must
# run before torch is imported.
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

import app.core.runtime  # noqa: F401  (sets OMP env before numpy/torch)

import json
import numpy as np
import matplotlib.pyplot as plt

from app.core.config import settings

print("Project root:", PROJECT_ROOT)
"""


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# --------------------------------------------------------------------------- #
# 01 - dataset analysis
# --------------------------------------------------------------------------- #

NB_DATASET = notebook([
    markdown("""\
# 01 · Dataset analysis

Explores the downloaded dataset using the project's own inspection code, so
everything shown here is exactly what the training pipeline sees.

**Prerequisites**

```
python scripts/download_dataset.py
python training/dataset_inspect.py
python training/prepare_splits.py
```
"""),
    code(BOOTSTRAP),
    markdown("## The dataset report\n\nWritten by `training/dataset_inspect.py`. Nothing about the archive layout is assumed."),
    code("""\
report = json.loads(settings.dataset_report_path.read_text(encoding="utf-8"))

print("Root      :", report["dataset"]["root"])
print("Classes   :", report["classes"]["count"])
print("Plants    :", report["classes"]["plant_count"])
print("Healthy   :", len(report["classes"]["healthy_classes"]))
print("Diseased  :", len(report["classes"]["diseased_classes"]))
for name, split in report["splits"].items():
    print(f"  {name:<8} {split['num_images']:>7,} images / {split['num_classes']:>3} classes")
"""),
    markdown("## Class distribution and imbalance\n\nThe imbalance ratio decides which - if any - correction is applied."),
    code("""\
counts = report["splits"]["train"]["class_counts"]
imbalance = report["splits"]["train"]["imbalance"]

print(f"min {imbalance['min_count']} ({imbalance['min_class']})")
print(f"max {imbalance['max_count']} ({imbalance['max_class']})")
print(f"ratio {imbalance['imbalance_ratio']:.3f}   coefficient of variation {imbalance['coefficient_of_variation']:.4f}")

names = sorted(counts, key=counts.get)
fig, ax = plt.subplots(figsize=(9, 10))
ax.barh([n.replace("___", " / ").replace("_", " ") for n in names],
        [counts[n] for n in names], color="#2f7a4d")
ax.set_xlabel("Training images")
ax.set_title("Images per class")
ax.grid(axis="x", alpha=0.25, linestyle="--")
plt.tight_layout()
plt.show()
"""),
    markdown("""\
## Data-leakage check

The dataset ships pre-augmented, so the obvious question is whether augmented
copies of the same original photograph appear on both sides of the official
train/valid boundary. Perceptual hashes of a stratified sample from each split
are compared for collisions.

This is a *sampled* estimate, and pHash would not catch a rotated or flipped
augmentation of the same original - both limitations are real.
"""),
    code("""\
duplicates = report["duplicates"]
print("Method:", duplicates["method"])
print(f"Sampled per class: {duplicates['sampled_per_class']}")
print(f"Train duplicate groups: {duplicates['train']['duplicate_groups']}")

leak = duplicates.get("cross_split_leakage")
if leak:
    print(f"Train<->valid colliding hashes: {leak['colliding_hashes']}")
    print(f"Leak rate in sample: {leak['leak_rate_in_sample'] * 100:.3f}%")
"""),
    markdown("## The split protocol and the imbalance decision"),
    code("""\
splits = json.loads(settings.splits_path.read_text(encoding="utf-8"))

print(splits["protocol"]["description"], "\\n")
for name, info in splits["splits"].items():
    print(f"  {name:<12} {info['count']:>7,}")

print("\\nImbalance strategy:", splits["imbalance"]["strategy"])
print("Reason:", splits["imbalance"]["reason"])
"""),
    markdown("## Sample images\n\nOne example per class, straight from the training manifest."),
    code("""\
from PIL import Image
from app.ml.datasets import read_manifest

rows = read_manifest(settings.data_dir / "manifests" / "train.csv")
first_per_class = {}
for row in rows:
    first_per_class.setdefault(row.label, row)

labels = sorted(first_per_class)[:16]
fig, axes = plt.subplots(4, 4, figsize=(10, 10.5))
for ax, label in zip(axes.ravel(), labels):
    with Image.open(first_per_class[label].path) as img:
        ax.imshow(img.convert("RGB"))
    ax.set_title(first_per_class[label].class_name.replace("___", "\\n").replace("_", " "), fontsize=7)
    ax.axis("off")
plt.tight_layout()
plt.show()
"""),
    markdown("## Image-quality metrics on real leaves\n\nThe same checks the API runs before every prediction."),
    code("""\
from app.ml.quality import analyse_image

for row in rows[:5]:
    with Image.open(row.path) as img:
        report_q = analyse_image(img)
    print(f"{row.class_name[:34]:<34} score {report_q.score:.3f}  "
          f"sharpness {report_q.metrics['blur_variance']:>7.0f}  "
          f"coherence {report_q.metrics['spatial_coherence']:.3f}  "
          f"plant-like {report_q.metrics['plant_like_fraction']:.2f}")
"""),
])


# --------------------------------------------------------------------------- #
# 02 - classical ML
# --------------------------------------------------------------------------- #

NB_CLASSICAL = notebook([
    markdown("""\
# 02 · Classical machine-learning baselines

Hand-crafted features and classical classifiers, on exactly the splits the deep
models use. These baselines turn "why use a CNN?" into an empirical question
rather than received wisdom.

**Prerequisite:** `python training/train_classical.py`
"""),
    code(BOOTSTRAP),
    markdown("""\
## The feature vector

| Group | Dimensions | What it captures |
|---|---|---|
| RGB + HSV histograms | 192 | Colour distribution; chlorosis shifts green toward yellow |
| Colour moments | 18 | Mean, standard deviation, skew per channel |
| Local Binary Patterns | 26 | Micro-texture of lesions and mildew |
| GLCM (Haralick) | 48 | Coarse co-occurrence texture |
| HOG | 900 | Edge and gradient structure of lesion borders and veins |
"""),
    code("""\
from PIL import Image
from app.ml.datasets import read_manifest
from app.ml.features import extract_features, feature_group_slices

rows = read_manifest(settings.data_dir / "manifests" / "train_bench.csv")
with Image.open(rows[0].path) as img:
    vector = extract_features(img)

print("Feature vector:", vector.shape)
for group, span in feature_group_slices().items():
    print(f"  {group:<8} dims {span.start:>4} - {span.stop:<4} ({span.stop - span.start})")
"""),
    markdown("## How separable are the classes in feature space?\n\nA 2-D PCA projection of the colour features gives a quick visual answer."),
    code("""\
from sklearn.decomposition import PCA

cache = settings.artifacts_dir / "features" / "features_train_bench.npz"
if cache.exists():
    data = np.load(cache)
    X, y = data["X"], data["y"]
    print("Cached feature matrix:", X.shape)

    projected = PCA(n_components=2, random_state=0).fit_transform(X[:3000])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    scatter = ax.scatter(projected[:, 0], projected[:, 1], c=y[:3000], cmap="tab20", s=6, alpha=0.7)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_title("PCA of hand-crafted features (first 3,000 training images)")
    plt.tight_layout(); plt.show()
else:
    print("Run `python training/train_classical.py` first to build the feature cache.")
"""),
    markdown("## Results\n\nAll figures below come from the run, not from this notebook."),
    code("""\
path = settings.results_dir / "classical_results.json"
if path.exists():
    results = json.loads(path.read_text(encoding="utf-8"))
    print(f"Train {results['train_samples']:,} / test {results['test_samples']:,} images, "
          f"{results['feature_dim']} features\\n")
    print(f"{'MODEL':<44}{'ACC':>9}{'MACRO F1':>11}{'TOP-3':>9}{'FIT s':>9}")
    print("-" * 82)
    for entry in sorted(results["results"], key=lambda r: -r["metrics"]["accuracy"]):
        m = entry["metrics"]
        print(f"{entry['label']:<44}{m['accuracy']:>9.4f}{m['f1_macro']:>11.4f}"
              f"{m['top3_accuracy']:>9.4f}{entry['training_time_s']:>9.1f}")
else:
    print("Run `python training/train_classical.py` first.")
"""),
    markdown("## Which feature group carries the signal?\n\nTree models expose feature importances, which can be summed per group."),
    code("""\
if path.exists():
    for entry in results["results"]:
        importance = entry.get("feature_group_importance")
        if not importance:
            continue
        total = sum(importance.values()) or 1
        print(entry["label"])
        for group, value in sorted(importance.items(), key=lambda kv: -kv[1]):
            print(f"   {group:<8} {value / total * 100:5.1f}%")
        print()
"""),
    markdown("""\
## Reading these numbers

A classical baseline that reaches a high score does **not** make deep learning
pointless - it establishes the floor. If hand-crafted colour histograms reach
90%, a CNN reaching 96% bought six points, not ninety-six. Without the baseline
there is no way to separate "the model is good" from "the dataset is easy".
"""),
])


# --------------------------------------------------------------------------- #
# 03 - CNN baseline
# --------------------------------------------------------------------------- #

NB_CNN = notebook([
    markdown("""\
# 03 · CNN baseline

Builds the from-scratch CNN, inspects it layer by layer, and looks at the
training curves the real run produced.
"""),
    code(BOOTSTRAP),
    markdown("## Building the model"),
    code("""\
import torch
from app.ml.architectures import build_model, count_parameters, model_size_mb

split_config = json.loads(settings.splits_path.read_text(encoding="utf-8"))
num_classes = split_config["num_classes"]

model = build_model("cnn_baseline", num_classes, pretrained=False)
total, trainable = count_parameters(model)
print(f"Classes    : {num_classes}")
print(f"Parameters : {total:,} ({trainable:,} trainable)")
print(f"Size       : {model_size_mb(model):.2f} MB")
print(model)
"""),
    markdown("## How the spatial dimensions shrink\n\nEach stage halves the resolution and doubles the channels."),
    code("""\
x = torch.randn(1, 3, 224, 224)
print(f"{'stage':<8}{'output shape':<24}{'elements':>12}")
print("-" * 46)
print(f"{'input':<8}{str(tuple(x.shape)):<24}{x.numel():>12,}")
with torch.no_grad():
    for index, block in enumerate(model.features):
        x = block(x)
        print(f"{index:<8}{str(tuple(x.shape)):<24}{x.numel():>12,}")
"""),
    markdown("## Augmentation, visualised\n\nThe same transform the trainer uses. Every variant must still be a plausible photograph of a leaf."),
    code("""\
from PIL import Image
from app.ml.datasets import read_manifest
from app.ml.transforms import build_train_transform, denormalize

rows = read_manifest(settings.data_dir / "manifests" / "train.csv")
with Image.open(rows[0].path) as handle:
    original = handle.convert("RGB")

transform = build_train_transform(224, strength="medium")
fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
axes[0, 0].imshow(original); axes[0, 0].set_title("original"); axes[0, 0].axis("off")
for ax in axes.ravel()[1:]:
    tensor = transform(original).unsqueeze(0)
    ax.imshow(denormalize(tensor)[0].permute(1, 2, 0).numpy())
    ax.set_title("augmented", fontsize=9); ax.axis("off")
plt.tight_layout(); plt.show()
"""),
    markdown("## Training curves from the real run"),
    code("""\
ledger = json.loads((settings.results_dir / "experiments.json").read_text(encoding="utf-8"))
record = next((r for r in ledger if r["model_name"] == "cnn_baseline"), None)

if record:
    history = record["history"]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax_loss.plot(history["epoch"], history["train_loss"], label="train", color="#2f7a4d", lw=2)
    ax_loss.plot(history["epoch"], history["val_loss"], label="validation", color="#c1553b", lw=2)
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss"); ax_loss.legend(frameon=False)
    ax_acc.plot(history["epoch"], history["train_acc"], label="train", color="#2f7a4d", lw=2)
    ax_acc.plot(history["epoch"], history["val_acc"], label="validation", color="#c1553b", lw=2)
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy"); ax_acc.legend(frameon=False)
    fig.suptitle(f"{record['label']} - best validation accuracy {record['best_val_accuracy']:.4f}")
    plt.tight_layout(); plt.show()

    metrics = record["test"]["metrics"]
    print(f"Test accuracy {metrics['accuracy']:.4f}   macro F1 {metrics['f1_macro']:.4f}   "
          f"top-3 {metrics['top3_accuracy']:.4f}")
else:
    print("Run `python training/run_experiments.py --suite benchmark` first.")
"""),
])


# --------------------------------------------------------------------------- #
# 04 - attention
# --------------------------------------------------------------------------- #

NB_ATTENTION = notebook([
    markdown("""\
# 04 · Attention mechanisms

Dissects SE, channel attention, spatial attention and CBAM, then looks at the
ablation the training pipeline produced.
"""),
    code(BOOTSTRAP),
    markdown("""\
## The four blocks

| Block | Gate shape | Question it answers |
|---|---|---|
| SE | `(B, C, 1, 1)` | *What* matters — from average pooling alone |
| CBAM channel | `(B, C, 1, 1)` | *What* matters — from average **and** max pooling |
| CBAM spatial | `(B, 1, H, W)` | *Where* it matters |
| CBAM | both, in sequence | Both, channel first |
"""),
    code("""\
import torch
from app.ml.attention import CBAM, ChannelAttention, SEBlock, SpatialAttention, build_attention

x = torch.randn(2, 64, 28, 28)
print(f"{'block':<10}{'output':<22}{'params':>8}")
print("-" * 40)
for name in ("none", "se", "channel", "spatial", "cbam"):
    block = build_attention(name, 64)
    out = block(x)
    params = sum(p.numel() for p in block.parameters())
    print(f"{name:<10}{str(tuple(out.shape)):<22}{params:>8,}")
"""),
    markdown("## CBAM is exactly `x * channel_gate * spatial_gate`\n\nVerified numerically rather than asserted."),
    code("""\
cbam = CBAM(64)
out = cbam(x)
expected = x * cbam.last_channel_attention * cbam.last_spatial_attention
print("channel gate:", tuple(cbam.last_channel_attention.shape))
print("spatial gate:", tuple(cbam.last_spatial_attention.shape))
print("out == x * channel * spatial :", torch.allclose(out, expected, atol=1e-5))
"""),
    markdown("## The learned spatial gate on a real leaf\n\nRead straight out of the forward pass of the trained model."),
    code("""\
from PIL import Image
from app.ml.datasets import read_manifest
from app.ml.explain import cbam_spatial_map, colorize, overlay_heatmap
from app.ml.transforms import build_eval_transform, resized_rgb
from evaluate import load_checkpoint

manifest_path = settings.exported_dir / "production.json"
cbam_checkpoint = None
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = [manifest] + manifest.get("candidates", [])
    cbam_checkpoint = next((c for c in candidates if "cbam" in c["model_name"]), None)

if cbam_checkpoint:
    path = PROJECT_ROOT / cbam_checkpoint["checkpoint"]
    model, config, class_names = load_checkpoint(path, torch.device("cpu"))
    rows = read_manifest(settings.data_dir / "manifests" / "test_bench.csv")

    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for column, row in enumerate(rows[::700][:4]):
        with Image.open(row.path) as handle:
            image = handle.convert("RGB")
        tensor = build_eval_transform(config.image_size)(image).unsqueeze(0)
        result = cbam_spatial_map(model, tensor)
        base = np.asarray(resized_rgb(image, config.image_size))

        axes[0, column].imshow(base)
        axes[0, column].set_title(row.class_name.replace("___", "\\n").replace("_", " "), fontsize=7)
        axes[1, column].imshow(overlay_heatmap(base, result.heatmap))
        axes[1, column].set_title("CBAM spatial gate", fontsize=8)
        for ax in (axes[0, column], axes[1, column]):
            ax.axis("off")
    plt.tight_layout(); plt.show()
else:
    print("No CBAM model exported yet. Run the training suite and export_model.py.")
"""),
    markdown("""\
## Ablation results

Five arms, one backbone, one budget, one seed. Only the attention block differs.
"""),
    code("""\
path = settings.results_dir / "ablation_summary.json"
if path.exists():
    summary = json.loads(path.read_text(encoding="utf-8"))
    print(f"{'ARM':<5}{'CONFIGURATION':<34}{'ACC':>9}{'MACRO F1':>11}{'Δ F1':>10}{'PARAMS':>11}")
    print("-" * 80)
    for arm in summary["arms"]:
        delta = arm["delta_vs_control"]["f1_macro"]
        delta_text = "—" if arm["arm"] == "A" else f"{delta * 100:+.2f}pt"
        print(f"{arm['arm']:<5}{arm['label'][:33]:<34}{arm['accuracy']:>9.4f}"
              f"{arm['f1_macro']:>11.4f}{delta_text:>10}{arm['parameters']:>11,}")

    verdict = summary.get("verdict")
    if verdict:
        print("\\nVERDICT")
        print(" ", verdict["statement"])
        print("\\nCAVEAT")
        print(" ", verdict["caveat"])
else:
    print("Run `python training/run_experiments.py --suite ablation` then `analyse_results.py`.")
"""),
])


# --------------------------------------------------------------------------- #
# 05 - evaluation
# --------------------------------------------------------------------------- #

NB_EVAL = notebook([
    markdown("""\
# 05 · Model evaluation

Benchmark comparison, per-class analysis, calibration and out-of-distribution
behaviour - all read from the experiment ledger.
"""),
    code(BOOTSTRAP),
    markdown("## The benchmark table"),
    code("""\
import pandas as pd

ledger = json.loads((settings.results_dir / "experiments.json").read_text(encoding="utf-8"))

def flatten(record):
    test = record.get("test") or {}
    metrics = test.get("metrics") or {}
    return {
        "model": record.get("label"),
        "group": record.get("group"),
        "protocol": record.get("protocol"),
        "proposed": record.get("proposed"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("f1_macro"),
        "top3": metrics.get("top3_accuracy"),
        "params_M": (record.get("total_parameters") or 0) / 1e6,
        "size_MB": record.get("model_size_mb"),
        "ms_per_image": (test.get("latency") or {}).get("single_image_ms"),
        "ece": (test.get("calibration") or {}).get("ece"),
    }

frame = pd.DataFrame([flatten(r) for r in ledger if (r.get("test") or {}).get("metrics")])
frame.sort_values("accuracy", ascending=False).round(4)
"""),
    markdown("## Accuracy against cost\n\nThe interesting question is not which model wins, but what the win costs."),
    code("""\
fig, (ax_size, ax_speed) = plt.subplots(1, 2, figsize=(13, 5))
for ax, key, label in ((ax_size, "size_MB", "Model size (MB)"),
                       (ax_speed, "ms_per_image", "Inference (ms/image)")):
    subset = frame.dropna(subset=[key, "accuracy"])
    colors = ["#c1553b" if p else "#2f7a4d" for p in subset["proposed"]]
    ax.scatter(subset[key], subset["accuracy"], s=110, c=colors, edgecolor="white", linewidth=1.4)
    for _, row in subset.iterrows():
        ax.annotate(row["model"], (row[key], row["accuracy"]), fontsize=7,
                    xytext=(6, 5), textcoords="offset points")
    ax.set_xlabel(label); ax.set_ylabel("Test accuracy")
    ax.grid(alpha=0.25, linestyle="--")
fig.suptitle("Accuracy vs cost — red marks the proposed CBAM architectures")
plt.tight_layout(); plt.show()
"""),
    markdown("## Per-class performance\n\nThe classes the model finds hardest are where more data would help most."),
    code("""\
best = max((r for r in ledger if (r.get("test") or {}).get("metrics")),
           key=lambda r: r["test"]["metrics"]["accuracy"])
per_class = best["test"]["metrics"]["per_class"]

rows = sorted(per_class.items(), key=lambda kv: kv[1]["f1"])[:12]
names = [k.replace("___", " / ").replace("_", " ") for k, _ in rows]
y = np.arange(len(names))

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(y + 0.25, [v["precision"] for _, v in rows], 0.25, label="Precision", color="#2f7a4d")
ax.barh(y,        [v["recall"]    for _, v in rows], 0.25, label="Recall",    color="#7fb069")
ax.barh(y - 0.25, [v["f1"]        for _, v in rows], 0.25, label="F1",        color="#c1553b")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
ax.set_xlim(0, 1.02); ax.legend(frameon=False, ncol=3, loc="lower right")
ax.set_title(f"12 hardest classes — {best['label']}")
ax.grid(axis="x", alpha=0.25, linestyle="--")
plt.tight_layout(); plt.show()
"""),
    markdown("""\
## Confidence calibration

Temperature scaling divides the logits by one scalar fitted on the validation
split. It cannot change which class wins, so accuracy is unchanged by
construction — only the confidence values move.
"""),
    code("""\
calibration = best["test"].get("calibration")
if calibration:
    print(f"ECE  raw {calibration['ece']:.4f}   scaled {calibration['after_temperature']['ece']:.4f}")
    print(f"MCE  raw {calibration['mce']:.4f}   temperature T = {calibration['temperature']:.4f}")
    print(f"Mean confidence {calibration['mean_confidence']:.4f} vs accuracy {calibration['accuracy']:.4f}")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for bins, label, color in ((calibration["bins"], "raw", "#c1553b"),
                               (calibration["after_temperature"]["bins"], "temperature-scaled", "#2f7a4d")):
        points = [(b["confidence"], b["accuracy"]) for b in bins if b["count"] > 0]
        ax.plot([p[0] for p in points], [p[1] for p in points], "o-", label=label, color=color, lw=2)
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", label="perfect calibration")
    ax.set_xlabel("Predicted confidence"); ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Reliability — {best['label']}"); ax.legend(frameon=False)
    ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout(); plt.show()
"""),
    markdown("""\
## Out-of-distribution behaviour

The stand-alone scores are weak — that is a known property of softmax-based OOD
detection, not a bug here. It is why the image-quality gate runs first, and why
the number that matters is the whole-pipeline rate.
"""),
    code("""\
path = settings.metadata_dir / "ood_thresholds.json"
if path.exists():
    ood = json.loads(path.read_text(encoding="utf-8"))
    print("AUROC by score:")
    for name, value in ood["metrics"]["auroc"].items():
        print(f"  {name:<28}{value:.4f}")

    pipeline = ood["metrics"].get("pipeline")
    if pipeline:
        print("\\nWhole-pipeline rejection:")
        print(f"  quality gate alone      {pipeline['quality_gate_rejects_ood'] * 100:5.1f}%")
        print(f"  OOD score alone         {pipeline['ood_score_rejects_ood'] * 100:5.1f}%")
        print(f"  combined                {pipeline['pipeline_rejects_ood'] * 100:5.1f}%")
        print(f"  cost (real leaves lost) {pipeline['quality_gate_rejects_real_leaves'] * 100:5.1f}%")
    print("\\nCaveat:", ood["ood_source"].get("caveat", ""))
else:
    print("Run `python training/calibrate_ood.py` first.")
"""),
    markdown("## Explainability on real predictions"),
    code("""\
import torch
from PIL import Image
from app.ml.datasets import read_manifest
from app.ml.explain import grad_cam, grad_cam_plus_plus, overlay_heatmap
from app.ml.transforms import build_eval_transform, resized_rgb
from evaluate import load_checkpoint

manifest_path = settings.exported_dir / "production.json"
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model, config, class_names = load_checkpoint(
        PROJECT_ROOT / manifest["checkpoint"], torch.device("cpu"))
    rows = read_manifest(settings.data_dir / "manifests" / "test_bench.csv")

    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for column, row in enumerate(rows[::900][:4]):
        with Image.open(row.path) as handle:
            image = handle.convert("RGB")
        tensor = build_eval_transform(config.image_size)(image).unsqueeze(0)
        base = np.asarray(resized_rgb(image, config.image_size))

        with torch.no_grad():
            probs = torch.softmax(model(tensor)[0], dim=0)
        top = int(probs.argmax())

        axes[0, column].imshow(base)
        axes[0, column].set_title(f"{class_names[top].split('___')[-1][:22]}\\n{probs[top]:.1%}", fontsize=8)
        axes[1, column].imshow(overlay_heatmap(base, grad_cam(model, tensor, model.feature_layer).heatmap))
        axes[1, column].set_title("Grad-CAM", fontsize=8)
        axes[2, column].imshow(overlay_heatmap(base, grad_cam_plus_plus(model, tensor, model.feature_layer).heatmap))
        axes[2, column].set_title("Grad-CAM++", fontsize=8)
        for r in range(3):
            axes[r, column].axis("off")
    plt.tight_layout(); plt.show()
else:
    print("Run `python training/export_model.py` first.")
"""),
    markdown("""\
> **Reading heat maps honestly.** These show where the network pooled evidence
> for its decision, at the resolution of its final feature map. They are **not**
> a disease segmentation and do not measure lesion extent. Their real diagnostic
> value is negative evidence: if the bright region sits on the background rather
> than the leaf, the model has latched onto a spurious correlation and the
> prediction should be distrusted however confident it is.
"""),
])


NOTEBOOKS = {
    "01_dataset_analysis.ipynb": NB_DATASET,
    "02_classical_ml.ipynb": NB_CLASSICAL,
    "03_cnn_baseline.ipynb": NB_CNN,
    "04_attention_models.ipynb": NB_ATTENTION,
    "05_model_evaluation.ipynb": NB_EVAL,
}


def main() -> int:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in NOTEBOOKS.items():
        (NOTEBOOK_DIR / name).write_text(json.dumps(content, indent=1), encoding="utf-8")
        print(f"  notebooks/{name}")
    print(f"\nWrote {len(NOTEBOOKS)} notebooks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
