"""Evaluation metrics, calibration measures and the plotting helpers.

Everything reported in the README, the docs and the research dashboard is
computed here from real model outputs. No number in this project is written by
hand.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)

PALETTE = {
    "primary": "#2f7a4d",
    "accent": "#c1553b",
    "muted": "#94a3b8",
    "grid": "#cbd5e1",
}


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> dict:
    """Full metric bundle for one model on one split.

    ``probabilities`` must be softmax outputs of shape ``(N, K)``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    n_classes = len(class_names)
    present = np.unique(y_true)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_classes), zero_division=0
    )

    # ROC-AUC needs at least two classes present and is only defined over the
    # classes that actually appear in the split.
    roc_auc_macro = None
    roc_auc_weighted = None
    if len(present) > 1:
        try:
            probs_present = probabilities[:, present]
            probs_present = probs_present / probs_present.sum(axis=1, keepdims=True)
            remap = {c: i for i, c in enumerate(present)}
            y_remapped = np.array([remap[v] for v in y_true])
            roc_auc_macro = float(
                roc_auc_score(y_remapped, probs_present, multi_class="ovr", average="macro")
            )
            roc_auc_weighted = float(
                roc_auc_score(y_remapped, probs_present, multi_class="ovr", average="weighted")
            )
        except ValueError:
            pass

    def _topk(k: int) -> float | None:
        if k > n_classes:
            return None
        return float(top_k_accuracy_score(y_true, probabilities, k=k, labels=range(n_classes)))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_weighted": float(f1_w),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc_macro_ovr": roc_auc_macro,
        "roc_auc_weighted_ovr": roc_auc_weighted,
        "top1_accuracy": _topk(1),
        "top3_accuracy": _topk(3),
        "top5_accuracy": _topk(5),
        "num_samples": int(len(y_true)),
        "num_classes": n_classes,
        "per_class": {
            class_names[i]: {
                "precision": float(per_class_p[i]),
                "recall": float(per_class_r[i]),
                "f1": float(per_class_f1[i]),
                "support": int(support[i]),
            }
            for i in range(n_classes)
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=range(n_classes), target_names=class_names,
            zero_division=0, output_dict=True,
        ),
    }


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> tuple[float, float, list[dict]]:
    """Return ``(ECE, MCE, per-bin stats)``.

    ECE is the support-weighted mean gap between a bin's mean confidence and its
    empirical accuracy; MCE is the worst such gap. A well-calibrated model that
    says "92%" is right 92% of the time.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    mce = 0.0
    bins: list[dict] = []
    total = len(confidences)
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (confidences > lo) & (confidences <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"lower": float(lo), "upper": float(hi), "count": 0,
                         "confidence": None, "accuracy": None, "gap": None})
            continue
        avg_conf = float(confidences[mask].mean())
        acc = float(correct[mask].mean())
        gap = abs(avg_conf - acc)
        ece += (count / total) * gap
        mce = max(mce, gap)
        bins.append({"lower": float(lo), "upper": float(hi), "count": count,
                     "confidence": avg_conf, "accuracy": acc, "gap": gap})
    return float(ece), float(mce), bins


def brier_score(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    """Multi-class Brier score (lower is better)."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(len(y_true)), np.asarray(y_true)] = 1.0
    return float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1)))


def predictive_entropy(probabilities: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Shannon entropy of each prediction, optionally scaled to ``[0, 1]``."""
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
    if normalize:
        entropy = entropy / np.log(probabilities.shape[1])
    return entropy


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def _style(ax) -> None:
    ax.grid(alpha=0.25, linestyle="--", color=PALETTE["grid"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    destination: Path,
    normalize: bool = True,
    title: str = "Confusion matrix",
) -> np.ndarray:
    """Render (and return) the confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    display = matrix.astype(np.float64)
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, out=np.zeros_like(display), where=row_sums > 0)

    size = max(8.0, len(class_names) * 0.34)
    fig, ax = plt.subplots(figsize=(size, size * 0.92))
    im = ax.imshow(display, cmap="YlGn", vmin=0, vmax=1 if normalize else display.max())
    fig.colorbar(im, ax=ax, fraction=0.043, pad=0.02,
                 label="Fraction of true class" if normalize else "Count")

    pretty = [c.replace("___", " / ").replace("_", " ") for c in class_names]
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(pretty, rotation=90, fontsize=6)
    ax.set_yticklabels(pretty, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return matrix


def plot_training_curves(history: dict, destination: Path, title: str) -> None:
    """Loss and accuracy curves for train and validation."""
    epochs = history.get("epoch", list(range(1, len(history.get("train_loss", [])) + 1)))
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.4))

    ax_loss.plot(epochs, history["train_loss"], label="train", color=PALETTE["primary"], lw=2)
    ax_loss.plot(epochs, history["val_loss"], label="validation", color=PALETTE["accent"], lw=2)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend(frameon=False)
    _style(ax_loss)

    ax_acc.plot(epochs, history["train_acc"], label="train", color=PALETTE["primary"], lw=2)
    ax_acc.plot(epochs, history["val_acc"], label="validation", color=PALETTE["accent"], lw=2)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend(frameon=False)
    _style(ax_acc)

    if history.get("stage_boundary"):
        for ax in (ax_loss, ax_acc):
            ax.axvline(history["stage_boundary"], color=PALETTE["muted"], ls=":", lw=1.5)
            ax.text(history["stage_boundary"], ax.get_ylim()[1], " fine-tune",
                    fontsize=8, va="top", color=PALETTE["muted"])

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def plot_reliability_diagram(bins: list[dict], ece: float, destination: Path, title: str) -> None:
    """Reliability (calibration) diagram with the perfect-calibration diagonal."""
    centres, accuracies, confidences, counts = [], [], [], []
    for b in bins:
        centres.append((b["lower"] + b["upper"]) / 2)
        accuracies.append(b["accuracy"] if b["accuracy"] is not None else 0.0)
        confidences.append(b["confidence"] if b["confidence"] is not None else 0.0)
        counts.append(b["count"])

    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    width = (centres[1] - centres[0]) * 0.9 if len(centres) > 1 else 0.06
    ax.bar(centres, accuracies, width=width, color=PALETTE["primary"],
           edgecolor="white", label="Accuracy")
    ax.plot([0, 1], [0, 1], ls="--", color=PALETTE["accent"], label="Perfect calibration")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style(ax)

    ax_hist.bar(centres, counts, width=width, color=PALETTE["muted"], edgecolor="white")
    ax_hist.set_xlabel("Predicted confidence")
    ax_hist.set_ylabel("Samples")
    _style(ax_hist)

    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def plot_model_comparison(
    rows: list[dict], metric: str, destination: Path, title: str, higher_is_better: bool = True
) -> None:
    """Horizontal bar chart comparing one metric across models."""
    valid = [r for r in rows if r.get(metric) is not None]
    valid.sort(key=lambda r: r[metric], reverse=not higher_is_better)
    labels = [r["label"] for r in valid]
    values = [r[metric] for r in valid]
    proposed = [r.get("proposed", False) for r in valid]

    fig, ax = plt.subplots(figsize=(9.5, max(4.0, len(labels) * 0.45)))
    colors = [PALETTE["accent"] if p else PALETTE["primary"] for p in proposed]
    bars = ax.barh(labels, values, color=colors, edgecolor="#1f3d2b", linewidth=0.5)
    span = (max(values) - min(values)) or 1.0
    for bar, value in zip(bars, values, strict=False):
        ax.text(value + span * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}", va="center", fontsize=8)
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.set_xlim(min(values) - span * 0.12, max(values) + span * 0.16)
    _style(ax)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def plot_scatter_tradeoff(
    rows: list[dict], x_key: str, y_key: str, destination: Path,
    title: str, x_label: str, y_label: str, log_x: bool = False,
) -> None:
    """Scatter plot of an accuracy/cost trade-off (size vs accuracy, latency vs accuracy)."""
    valid = [r for r in rows if r.get(x_key) is not None and r.get(y_key) is not None]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for row in valid:
        color = PALETTE["accent"] if row.get("proposed") else PALETTE["primary"]
        ax.scatter(row[x_key], row[y_key], s=110, color=color,
                   edgecolor="white", linewidth=1.4, zorder=3)
        ax.annotate(row["label"], (row[x_key], row[y_key]), fontsize=7.5,
                    xytext=(6, 5), textcoords="offset points")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    _style(ax)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def plot_per_class_performance(
    per_class: dict[str, dict], destination: Path, title: str, top_n: int | None = None
) -> None:
    """Grouped bars of per-class precision/recall/F1, sorted by F1."""
    items = sorted(per_class.items(), key=lambda kv: kv[1]["f1"])
    if top_n:
        items = items[:top_n]
    names = [k.replace("___", " / ").replace("_", " ") for k, _ in items]
    precision = [v["precision"] for _, v in items]
    recall = [v["recall"] for _, v in items]
    f1 = [v["f1"] for _, v in items]

    y = np.arange(len(names))
    height = 0.26
    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.36)))
    ax.barh(y + height, precision, height, label="Precision", color="#2f7a4d")
    ax.barh(y, recall, height, label="Recall", color="#7fb069")
    ax.barh(y - height, f1, height, label="F1", color="#c1553b")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Score")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    _style(ax)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def plot_ablation(rows: list[dict], destination: Path) -> None:
    """Grouped bars for the attention ablation: accuracy, macro-F1, macro-recall."""
    labels = [r["label"] for r in rows]
    metrics = [("accuracy", "Accuracy", "#2f7a4d"),
               ("f1_macro", "Macro F1", "#7fb069"),
               ("recall_macro", "Macro recall", "#c1553b")]
    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.7), 5.2))
    for offset, (key, name, color) in zip((-width, 0, width), metrics, strict=False):
        values = [r.get(key) or 0.0 for r in rows]
        bars = ax.bar(x + offset, values, width, label=name, color=color)
        for bar, value in zip(bars, values, strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}",
                    ha="center", fontsize=7.2, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace(": ", ":\n") for name in labels], fontsize=8)
    lo = min(min(r.get(k) or 1.0 for k, _, _ in metrics) for r in rows)
    ax.set_ylim(max(0, lo - 0.08), 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Attention ablation study (identical CNN backbone and training budget)")
    ax.legend(frameon=False, ncol=3)
    _style(ax)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def plot_confidence_distribution(
    confidences: np.ndarray, correct: np.ndarray, destination: Path, title: str
) -> None:
    """Overlaid confidence histograms for correct and incorrect predictions."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bins = np.linspace(0, 1, 41)
    ax.hist(confidences[correct.astype(bool)], bins=bins, alpha=0.78,
            label="Correct", color=PALETTE["primary"])
    ax.hist(confidences[~correct.astype(bool)], bins=bins, alpha=0.78,
            label="Incorrect", color=PALETTE["accent"])
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Predictions")
    ax.set_title(title)
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140)
    plt.close(fig)
