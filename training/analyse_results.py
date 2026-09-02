"""Turn the experiment ledger into comparison charts and the ablation summary.

Reads ``artifacts/results/experiments.json`` and writes:

* ``artifacts/plots/compare_*.png``        - accuracy, F1, precision/recall, trade-offs
* ``artifacts/plots/ablation.png``         - the attention ablation chart
* ``artifacts/results/ablation_summary.json`` - the quantified answer to the
  research question, including deltas against the no-attention control
* ``artifacts/results/summary.json``       - headline numbers the README embeds

Every number is read from the ledger. Nothing here computes or invents a metric.

    python training/analyse_results.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import app.core.runtime  # noqa: F401  # isort:skip

from metrics import plot_ablation, plot_model_comparison, plot_scatter_tradeoff  # noqa: E402
from tracker import ExperimentTracker  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)


def _get(record: dict, path: str, default=None):
    node = record
    for key in path.split("."):
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            return default
    return node


def to_row(record: dict) -> dict:
    """Flatten one ledger record into a chart-ready row."""
    return {
        "experiment_id": record.get("experiment_id"),
        "model_name": record.get("model_name"),
        "label": record.get("label"),
        "group": record.get("group"),
        "protocol": record.get("protocol"),
        "proposed": bool(record.get("proposed")),
        "accuracy": _get(record, "test.metrics.accuracy"),
        "precision_macro": _get(record, "test.metrics.precision_macro"),
        "recall_macro": _get(record, "test.metrics.recall_macro"),
        "f1_macro": _get(record, "test.metrics.f1_macro"),
        "f1_weighted": _get(record, "test.metrics.f1_weighted"),
        "top3_accuracy": _get(record, "test.metrics.top3_accuracy"),
        "top5_accuracy": _get(record, "test.metrics.top5_accuracy"),
        "roc_auc": _get(record, "test.metrics.roc_auc_macro_ovr"),
        "ece": _get(record, "test.calibration.ece"),
        "parameters": record.get("total_parameters"),
        "parameters_millions": (record.get("total_parameters") or 0) / 1e6,
        "model_size_mb": record.get("model_size_mb"),
        "inference_ms": _get(record, "test.latency.single_image_ms"),
        "training_time_s": record.get("training_time_s"),
        "training_time_min": (record.get("training_time_s") or 0) / 60,
        "epochs_run": record.get("epochs_run"),
        "val_accuracy": record.get("best_val_accuracy"),
        "seed": (record.get("config") or {}).get("seed"),
        "num_workers": (record.get("config") or {}).get("num_workers"),
    }


def fixed_budget(rows: list[dict]) -> list[dict]:
    """Drop production runs, keeping only the controlled fixed-budget ones.

    A model can appear twice in the ledger: once under the fixed-budget research
    protocol (300 images per class, 160px, scored on the 4,560-image subset
    split) and once under production (full data, 224px, scored on 17,572
    images). Those numbers are not comparable, and a plain name->record lookup
    lets the production row silently shadow the research one.

    The symptom was the measured noise floor jumping from 0.47 to 2.14 points,
    because cnn_cbam's research and production runs were being read as two runs
    of one configuration. That inflated threshold then swallowed every real
    finding, the placement result included.

    Note the filter is "not production" rather than "protocol == benchmark":
    the benchmark, ablation and placement suites carry different protocol tags
    but share one training budget, schedule and evaluation split, which is
    exactly what makes them comparable with each other.
    """
    return [r for r in rows if r.get("protocol") != "production"]


def estimate_noise_floor(all_rows: list[dict], replicates: list[dict] | None = None) -> dict:
    """Measure how much macro F1 moves when nothing about the architecture changes.

    There are two distinct floors here, and conflating them would overstate every
    conclusion in the ablation.

    The **repeat floor** comes from configurations trained twice at the same seed:
    ``abl_cnn_none`` repeats ``cnn_baseline`` and ``abl_cnn_se`` repeats
    ``cnn_se``. Same architecture, same schedule, same initialisation. What does
    differ is cuDNN autotuning, non-deterministic GPU reduction order, and - in
    these particular runs - the DataLoader worker count, which matters because
    each worker is given its own derived seed and therefore its own augmentation
    stream. The pair is thus not a pure determinism check: it also captures
    data-order variance, which is a real part of what separates any two runs.
    The per-observation ``differs_in`` field records exactly which of these
    applied, so the claim can be checked rather than taken on trust.

    Two observations of this floor disagreed by almost an order of magnitude
    (0.04 pt and 0.35 pt), which is itself the argument for reporting the maximum
    rather than the mean: a single close-agreeing pair would badly understate how
    far apart two runs of the same thing can land.

    The **seed floor** comes from training one architecture at several seeds.
    This is the threshold that matters, because two different architectures never
    share an initialisation: CNN and CNN+CBAM have different parameter shapes, so
    they consume the random stream differently and start from different weights
    even when the seed is identical. A gap between them therefore contains
    initialisation and data-order variance in addition to any real architectural
    effect, and only the seed floor measures that.

    Reporting the non-determinism floor as if it licensed architectural claims
    would make noise look like findings, so both are reported and the ablation
    is judged against the wider one.
    """
    # Groups of entries that are the same network trained more than once. The
    # CBAM group has three members, not two: abl_cnn_cbam leaves attention_stages
    # at its default of "all", which is exactly what place_cbam_all specifies, so
    # all three are the identical architecture under different names.
    groups = [
        ["cnn_baseline", "abl_cnn_none"],
        ["cnn_se", "abl_cnn_se"],
        ["cnn_cbam", "abl_cnn_cbam", "place_cbam_all"],
    ]
    by_name = {r["model_name"]: r for r in fixed_budget(all_rows)}

    observations = []
    for group in groups:
        members = [by_name[name] for name in group if name in by_name]
        if len(members) < 2:
            continue
        # With more than two runs the spread, not a single difference, is the
        # quantity of interest: it is what a third run would have to beat.
        left = min(members, key=lambda r: r["f1_macro"] or 0)
        right = max(members, key=lambda r: r["f1_macro"] or 0)
        # Say what actually varied between the two runs. Claiming "only cuDNN
        # scheduling" when the worker count also changed would misdescribe the
        # measurement and overstate how tight the floor is.
        differs_in = ["cuDNN algorithm selection", "GPU reduction order"]
        # Compare across every member, not just the two extremes: a three-run
        # group can vary in worker count between members that are not the min
        # and max, and claiming otherwise would understate what varied.
        worker_counts = sorted({m.get("num_workers") for m in members
                                if m.get("num_workers") is not None})
        if len(worker_counts) > 1:
            differs_in.append(
                f"DataLoader worker count ({' vs '.join(str(w) for w in worker_counts)}), "
                f"and so the per-worker augmentation seeds and batch composition"
            )
        observations.append({
            "pair": [left["model_name"], right["model_name"]],
            "group": [m["model_name"] for m in members],
            "runs": len(members),
            "accuracy": [left["accuracy"], right["accuracy"]],
            "f1_macro": [left["f1_macro"], right["f1_macro"]],
            "accuracy_gap": abs((left["accuracy"] or 0) - (right["accuracy"] or 0)),
            "f1_macro_gap": abs((left["f1_macro"] or 0) - (right["f1_macro"] or 0)),
            "differs_in": differs_in,
        })

    result: dict = {"available": bool(observations)}

    if observations:
        f1_gaps = [o["f1_macro_gap"] for o in observations]
        result["nondeterminism"] = {
            "method": (
                "Identical architectures and hyperparameters trained once in the "
                "benchmark suite and once in the ablation suite, at the same seed. "
                "See differs_in on each observation for what varied; where the "
                "DataLoader worker count differs the pair also captures augmentation "
                "and batch-order variance, not only GPU non-determinism."
            ),
            "observations": observations,
            "max_accuracy_gap": max(o["accuracy_gap"] for o in observations),
            "max_f1_macro_gap": max(f1_gaps),
            "mean_f1_macro_gap": sum(f1_gaps) / len(f1_gaps),
        }

    # Seed replicates: the same architecture trained from several initialisations.
    seed_floor = None
    if replicates and len(replicates) >= 2:
        f1s = [r["f1_macro"] for r in replicates if r.get("f1_macro") is not None]
        accs = [r["accuracy"] for r in replicates if r.get("accuracy") is not None]
        if len(f1s) >= 2:
            mean_f1 = sum(f1s) / len(f1s)
            spread = max(f1s) - min(f1s)
            sd = (sum((x - mean_f1) ** 2 for x in f1s) / (len(f1s) - 1)) ** 0.5
            seed_floor = spread
            result["seed"] = {
                "method": (
                    f"The same architecture ({replicates[0]['model_name']}) trained "
                    f"{len(f1s)} times at different seeds. Everything varies that "
                    f"varies between two different architectures - weight "
                    f"initialisation, augmentation draws, batch composition - except "
                    f"the architecture itself."
                ),
                "model_name": replicates[0]["model_name"],
                "seeds": [r.get("seed") for r in replicates],
                "f1_macro": f1s,
                "accuracy": accs,
                "mean_f1_macro": mean_f1,
                "sd_f1_macro": sd,
                "spread_f1_macro": spread,
            }

    # The ablation is judged against the widest floor actually measured. Neither
    # floor dominates the other: the seed replicates vary the initialisation but
    # only for one architecture, while the repeated configurations span several
    # architectures and, in this study, a change of DataLoader worker count.
    # Both are lower bounds on different components of the same noise, so taking
    # the larger is the only choice that cannot license a claim the other
    # measurement contradicts. Promoting the seed floor merely because it is the
    # more principled *design* would, when it happens to come out narrower, lower
    # the bar on evidence already in hand.
    repeat_floor = result["nondeterminism"]["max_f1_macro_gap"] if observations else None
    candidates = [f for f in (seed_floor, repeat_floor) if f is not None]

    if seed_floor is not None and candidates:
        widest = max(candidates)
        result["threshold_f1_macro"] = widest
        result["threshold_basis"] = "seed" if widest == seed_floor else "nondeterminism"
        seeds_run = len(result["seed"]["f1_macro"])
        sd = result["seed"]["sd_f1_macro"]
        if repeat_floor is not None and repeat_floor > seed_floor:
            result["interpretation"] = (
                f"Retraining {result['seed']['model_name']} at {seeds_run} seeds moved "
                f"macro F1 across {seed_floor * 100:.2f} points (SD {sd * 100:.2f}). "
                f"Repeating identical configurations moved it further still, by up to "
                f"{repeat_floor * 100:.2f} points, so {repeat_floor * 100:.2f} points is "
                f"used as the threshold: it is the largest variation observed between "
                f"runs that should have been equivalent. A gap between attention arms "
                f"smaller than this is not evidence of an architectural effect."
            )
        else:
            result["interpretation"] = (
                f"Retraining {result['seed']['model_name']} at {seeds_run} seeds moved "
                f"macro F1 across a range of {seed_floor * 100:.2f} points "
                f"(SD {sd * 100:.2f}), wider than the "
                f"{(repeat_floor or 0) * 100:.2f} points seen when repeating an "
                f"identical configuration. Two different architectures never share an "
                f"initialisation, so a gap between attention arms smaller than "
                f"{seed_floor * 100:.2f} points is not evidence of an architectural "
                f"effect."
            )
    elif observations:
        nd = result["nondeterminism"]["max_f1_macro_gap"]
        result["threshold_f1_macro"] = nd
        result["threshold_basis"] = "nondeterminism"
        spread = ""
        if len(observations) > 1:
            gaps = sorted(o["f1_macro_gap"] * 100 for o in observations)
            spread = (
                f" Across {len(observations)} such pairs the gap ranged from "
                f"{gaps[0]:.2f} to {gaps[-1]:.2f} points, so the largest is used."
            )
        result["interpretation"] = (
            f"Retraining the same configuration moved macro F1 by up to "
            f"{nd * 100:.2f} points.{spread} This remains a LOWER BOUND on the noise "
            f"relevant to the ablation: these repeats share an initialisation, "
            f"whereas two different architectures have different parameter shapes "
            f"and so start from different weights. Treat {nd * 100:.2f} points as "
            f"the smallest gap that could be meaningful, not as a significance "
            f"threshold. Seed replicates (training/run_experiments.py --models "
            f"cnn_baseline --seed N) would measure the threshold properly."
        )
    else:
        result["reason"] = "Needs both the benchmark and ablation suites to have run."

    return result


def transfer_cbam_pairs(all_rows: list[dict], noise_floor: float | None) -> dict:
    """Test the research question a second time, on pretrained backbones.

    The ablation answers "does CBAM help?" for one from-scratch CNN. The
    benchmark happens to contain three *matched pairs* - the same ImageNet
    backbone with and without a CBAM block on its final feature map, trained
    under the identical protocol. Those are three further independent tests of
    the same question, and they are worth reading precisely because they probe a
    different regime: a pretrained trunk has already learned to suppress
    irrelevant features, so there may be less for attention to add.
    """
    pairs = [
        ("efficientnet_b0", "efficientnet_b0_cbam", "EfficientNet-B0"),
        ("resnet50", "resnet50_cbam", "ResNet50"),
        ("mobilenet_v3_large", "mobilenet_v3_large_cbam", "MobileNetV3-Large"),
    ]
    by_name = {r["model_name"]: r for r in fixed_budget(all_rows)}

    entries = []
    for plain_name, cbam_name, label in pairs:
        plain, cbam = by_name.get(plain_name), by_name.get(cbam_name)
        if not plain or not cbam:
            continue
        delta_f1 = (cbam["f1_macro"] or 0) - (plain["f1_macro"] or 0)
        delta_acc = (cbam["accuracy"] or 0) - (plain["accuracy"] or 0)
        entries.append({
            "backbone": label,
            "without_cbam": {"model_name": plain_name, "accuracy": plain["accuracy"],
                             "f1_macro": plain["f1_macro"],
                             "parameters": plain["parameters"],
                             "inference_ms": plain["inference_ms"]},
            "with_cbam": {"model_name": cbam_name, "accuracy": cbam["accuracy"],
                          "f1_macro": cbam["f1_macro"],
                          "parameters": cbam["parameters"],
                          "inference_ms": cbam["inference_ms"]},
            "delta_accuracy": delta_acc,
            "delta_f1_macro": delta_f1,
            "parameter_overhead_pct": (
                100 * ((cbam["parameters"] or 0) - (plain["parameters"] or 0))
                / (plain["parameters"] or 1)
            ),
            "inference_overhead_pct": (
                100 * ((cbam["inference_ms"] or 0) - (plain["inference_ms"] or 0))
                / (plain["inference_ms"] or 1)
            ),
            "beyond_noise": (abs(delta_f1) > noise_floor) if noise_floor else None,
        })

    if not entries:
        return {"available": False,
                "reason": "Needs both the plain and CBAM variant of at least one backbone."}

    helped = [e for e in entries if e["delta_f1_macro"] > 0]
    threshold = noise_floor if noise_floor else 0.005
    decisive = [e for e in entries if abs(e["delta_f1_macro"]) > threshold]

    if not decisive:
        verdict = (
            f"Across {len(entries)} matched backbone pair(s), every CBAM/no-CBAM "
            f"difference is smaller than the {threshold * 100:.2f}-point run-to-run "
            "variation measured on this setup. On pretrained backbones at this "
            "scale, the evidence neither supports nor refutes a CBAM benefit."
        )
    elif len(helped) > len(entries) / 2:
        verdict = (
            f"CBAM improved macro F1 on {len(helped)} of {len(entries)} pretrained "
            "backbones."
        )
    else:
        verdict = (
            f"CBAM improved macro F1 on only {len(helped)} of {len(entries)} pretrained "
            "backbones. On a trunk already pretrained on 1.2M ImageNet images, the "
            "features reaching the attention block are largely free of the "
            "irrelevant activations CBAM exists to suppress, so there is less for "
            "it to do than in the from-scratch case."
        )

    return {
        "available": True,
        "pairs": entries,
        "improved_count": len(helped),
        "total_pairs": len(entries),
        "significance_threshold_f1": threshold,
        "verdict": verdict,
        "caveat": (
            "One run per configuration. These pairs are a secondary line of "
            "evidence alongside the controlled five-arm ablation, not a "
            "replacement for it - the ablation holds the backbone fixed, whereas "
            "each pair here changes only where CBAM is inserted in an already "
            "pretrained network."
        ),
    }


def placement_analysis(all_rows: list[dict], noise_floor: float | None) -> dict:
    """Ask where in the network CBAM earns its keep, not merely whether it does.

    The five-arm ablation varies *which* attention mechanism is used while
    applying it everywhere. This varies the opposite axis: the same CBAM block,
    inserted at a different depth. It is a distinct question, and on this dataset
    it turned out to be the one with the larger effect - which is only visible if
    the placements are compared against the no-attention control and against each
    other rather than listed in a table.
    """
    by_name = {r["model_name"]: r for r in fixed_budget(all_rows)}
    control = by_name.get("abl_cnn_none") or by_name.get("cnn_baseline")
    if control is None:
        return {"available": False,
                "reason": "The no-attention control has not been trained."}

    variants = [
        ("place_cbam_last", "Last stage only", 1),
        ("place_cbam_last2", "Last two stages", 2),
        ("place_cbam_all", "Every stage", 4),
    ]
    entries = []
    for name, label, stages in variants:
        row = by_name.get(name)
        if row is None:
            continue
        delta = (row["f1_macro"] or 0) - (control["f1_macro"] or 0)
        entries.append({
            "model_name": name,
            "label": label,
            "stages_with_attention": stages,
            "accuracy": row["accuracy"],
            "f1_macro": row["f1_macro"],
            "parameters": row["parameters"],
            "inference_ms": row["inference_ms"],
            "delta_f1_macro": delta,
            "beyond_noise": (abs(delta) > noise_floor) if noise_floor else None,
        })

    if not entries:
        return {"available": False,
                "reason": "No CBAM placement variants have been trained."}

    best = max(entries, key=lambda e: e["f1_macro"] or 0)
    worst = min(entries, key=lambda e: e["f1_macro"] or 0)
    spread = (best["f1_macro"] or 0) - (worst["f1_macro"] or 0)

    statement = (
        f"Attention on the {best['label'].lower()} scored highest "
        f"({best['f1_macro']:.4f} macro F1, {best['delta_f1_macro'] * 100:+.2f} pt "
        f"against the identical network without attention)."
    )
    if len(entries) > 1:
        statement += (
            f" The spread between the best and worst placement is "
            f"{spread * 100:.2f} pt"
        )
        if noise_floor:
            statement += (
                f", {'larger' if spread > noise_floor else 'smaller'} than the "
                f"{noise_floor * 100:.2f}-pt run-to-run floor"
            )
            if spread > noise_floor and worst["delta_f1_macro"] < 0:
                statement += (
                    f". The worst placement is {abs(worst['delta_f1_macro']) * 100:.2f} pt "
                    f"*below* the no-attention control, so the same block can help or "
                    f"hurt depending only on where it is inserted"
                )
        statement += "."

    # The fixed-budget study answers the question on a small subset. The
    # production suite happens to contain the same contrast at full scale -
    # cnn_cbam puts CBAM on every stage, place_cbam_last2 on the last two, and
    # they have the same parameter count - which is an independent replication
    # rather than a restatement. A finding that survives a 5.5x change in
    # training data is worth far more than one measured once.
    production = {r["model_name"]: r for r in all_rows if r.get("protocol") == "production"}
    replication = None
    all_stages = production.get("cnn_cbam")
    last_two = production.get("place_cbam_last2")
    if all_stages and last_two:
        delta = (last_two["f1_macro"] or 0) - (all_stages["f1_macro"] or 0)
        replication = {
            "available": True,
            "all_stages": {"model_name": "cnn_cbam", "f1_macro": all_stages["f1_macro"],
                           "accuracy": all_stages["accuracy"],
                           "parameters": all_stages["parameters"]},
            "last_two_stages": {"model_name": "place_cbam_last2",
                                "f1_macro": last_two["f1_macro"],
                                "accuracy": last_two["accuracy"],
                                "parameters": last_two["parameters"]},
            "delta_f1_macro": delta,
            "agrees_with_fixed_budget": delta > 0,
            "statement": (
                f"At full data scale the same contrast holds: CBAM on the last two "
                f"stages reaches {last_two['f1_macro']:.4f} macro F1 against "
                f"{all_stages['f1_macro']:.4f} for CBAM on every stage, a "
                f"{delta * 100:+.2f} pt difference at an identical parameter count. "
                f"The fixed-budget study and the full-data runs therefore agree on "
                f"direction, which a single measurement could not establish."
                if delta > 0 else
                f"At full data scale the contrast reverses: CBAM on the last two "
                f"stages reaches {last_two['f1_macro']:.4f} macro F1 against "
                f"{all_stages['f1_macro']:.4f} for CBAM on every stage "
                f"({delta * 100:+.2f} pt). The fixed-budget result does not survive "
                f"the change of scale and should not be generalised."
            ),
        }

    return {
        "available": True,
        "control": control["model_name"],
        "control_f1_macro": control["f1_macro"],
        "variants": entries,
        "best": best["model_name"],
        "best_label": best["label"],
        "spread_f1_macro": spread,
        "spread_beyond_noise": (spread > noise_floor) if noise_floor else None,
        "production_replication": replication,
        "statement": statement,
        "caveat": (
            "One run per placement. The spread is compared against the measured "
            "run-to-run floor, which is a lower bound rather than a significance "
            "test, so this identifies where to look next rather than settling the "
            "question."
        ),
    }


def build_ablation_summary(rows: list[dict], all_rows: list[dict] | None = None,
                           replicates: list[dict] | None = None) -> dict:
    """Quantify what each attention variant adds over the no-attention control."""
    by_name = {r["model_name"]: r for r in fixed_budget(rows)}
    control = by_name.get("abl_cnn_none")
    if control is None:
        return {"available": False,
                "reason": "The no-attention control (abl_cnn_none) has not been trained."}

    arms = [
        ("abl_cnn_none", "A", "CNN (control)", "No attention."),
        ("abl_cnn_channel", "B", "CNN + Channel attention",
         "CBAM's channel branch only - learns *what* is important."),
        ("abl_cnn_spatial", "C", "CNN + Spatial attention",
         "CBAM's spatial branch only - learns *where* it is important."),
        ("abl_cnn_se", "D", "CNN + SE",
         "Squeeze-and-Excitation: channel attention from average pooling alone."),
        ("abl_cnn_cbam", "E", "CNN + CBAM",
         "Both branches, channel then spatial (the proposed configuration)."),
    ]

    entries = []
    for name, letter, label, description in arms:
        row = by_name.get(name)
        if row is None:
            continue
        entries.append({
            "arm": letter,
            "model_name": name,
            "label": label,
            "description": description,
            "accuracy": row["accuracy"],
            "f1_macro": row["f1_macro"],
            "recall_macro": row["recall_macro"],
            "precision_macro": row["precision_macro"],
            "top3_accuracy": row["top3_accuracy"],
            "ece": row["ece"],
            "parameters": row["parameters"],
            "inference_ms": row["inference_ms"],
            "training_time_s": row["training_time_s"],
            "delta_vs_control": {
                "accuracy": (row["accuracy"] or 0) - (control["accuracy"] or 0),
                "f1_macro": (row["f1_macro"] or 0) - (control["f1_macro"] or 0),
                "recall_macro": (row["recall_macro"] or 0) - (control["recall_macro"] or 0),
                "parameter_overhead": (row["parameters"] or 0) - (control["parameters"] or 0),
                "parameter_overhead_pct": (
                    100 * ((row["parameters"] or 0) - (control["parameters"] or 0))
                    / (control["parameters"] or 1)
                ),
                "inference_overhead_ms": (row["inference_ms"] or 0) - (control["inference_ms"] or 0),
                "inference_overhead_pct": (
                    100 * ((row["inference_ms"] or 0) - (control["inference_ms"] or 0))
                    / (control["inference_ms"] or 1)
                ),
            },
        })

    cbam = next((e for e in entries if e["model_name"] == "abl_cnn_cbam"), None)
    best = max(entries, key=lambda e: e["f1_macro"] or 0)

    noise = estimate_noise_floor(all_rows or rows, replicates)
    # Prefer the measured noise floor over the 0.5-point rule of thumb when the
    # duplicate-configuration pairs are available.
    threshold = (noise["threshold_f1_macro"]
                 if noise.get("available") and noise.get("threshold_f1_macro")
                 else 0.005)

    verdict = None
    if cbam is not None:
        delta_f1 = cbam["delta_vs_control"]["f1_macro"]
        delta_acc = cbam["delta_vs_control"]["accuracy"]
        overhead = cbam["delta_vs_control"]["inference_overhead_pct"]
        # A single training run cannot separate a difference smaller than the
        # measured run-to-run variation from noise, so the wording is graded by
        # effect size rather than claiming significance the experiment cannot
        # support.
        # Clearing the floor by a hair is not the same as clearing it decisively.
        # The floor itself is estimated from a handful of repeats, so a delta only
        # slightly above it is within the uncertainty of the threshold, let alone
        # of the measurement. Requiring 2x before using unqualified language keeps
        # the strength of the claim proportional to the evidence.
        if delta_f1 > 2 * threshold:
            direction = (
                f"CBAM improved macro F1 by {delta_f1 * 100:.2f} points and accuracy by "
                f"{delta_acc * 100:.2f} points over the identical CNN without attention. "
                f"That is more than twice the {threshold * 100:.2f}-point run-to-run "
                f"variation measured here, so the direction of the effect is "
                f"well supported even though a single run per arm cannot quantify it."
            )
        elif delta_f1 > threshold:
            direction = (
                f"CBAM improved macro F1 by {delta_f1 * 100:.2f} points and accuracy by "
                f"{delta_acc * 100:.2f} points over the identical CNN without attention, "
                f"but only narrowly clears the {threshold * 100:.2f}-point run-to-run "
                f"variation measured here ({delta_f1 / threshold:.1f}x). The floor is "
                f"itself estimated from a few repeats, so this is suggestive rather "
                f"than established; seed replicates would be needed to confirm it."
            )
        elif delta_f1 > 0.0:
            direction = (
                f"CBAM improved macro F1 by {delta_f1 * 100:.2f} points, which is within "
                f"the {threshold * 100:.2f}-point run-to-run variation measured on this "
                "setup. The improvement is directionally positive but not established."
            )
        else:
            direction = (
                f"CBAM did **not** improve on the plain CNN in this experiment: macro F1 "
                f"changed by {delta_f1 * 100:.2f} points and accuracy by "
                f"{delta_acc * 100:.2f} points."
            )
        verdict = {
            "cbam_delta_f1_macro": delta_f1,
            "cbam_delta_accuracy": delta_acc,
            "cbam_inference_overhead_pct": overhead,
            "cbam_parameter_overhead_pct": cbam["delta_vs_control"]["parameter_overhead_pct"],
            "best_arm": best["label"],
            "best_arm_f1": best["f1_macro"],
            "cbam_is_best": best["model_name"] == "abl_cnn_cbam",
            "statement": direction,
            "significance_threshold_f1": threshold,
            "margin_over_threshold": (delta_f1 / threshold) if threshold else None,
            "evidence_strength": (
                "supported" if delta_f1 > 2 * threshold
                else "suggestive" if delta_f1 > threshold
                else "within noise" if delta_f1 > 0
                else "no improvement"
            ),
            "caveat": (
                "All five arms share one architecture, one training budget, one data "
                "subset and one seed; only the attention block differs. A single run "
                "per arm cannot establish statistical significance. "
                + (noise["interpretation"] if noise.get("available")
                   else "Differences below roughly half a point should be read as "
                        "inconclusive.")
            ),
        }

    return {"available": True, "control": control["model_name"], "arms": entries,
            "verdict": verdict, "noise_floor": noise}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    configure_logging(settings.log_level, settings.logs_dir / "analyse.log")
    settings.ensure_dirs()

    tracker = ExperimentTracker(settings.results_dir)
    if not tracker.records:
        raise SystemExit("No experiments logged yet. Run training/run_experiments.py first.")

    all_runs = [to_row(r) for r in tracker.records if _get(r, "test.metrics.accuracy") is not None]

    # Seed replicates repeat one architecture at extra seeds purely to measure
    # run-to-run variance. They must not enter the comparison tables, or the same
    # model would appear several times in every chart; they feed the noise floor.
    canonical_seed = min((r["seed"] for r in all_runs if r["seed"] is not None), default=None)
    replicates = [r for r in all_runs
                  if r["seed"] is not None and r["seed"] != canonical_seed]
    rows = [r for r in all_runs if r not in replicates]

    replicate_groups: dict[str, list[dict]] = {}
    for run in replicates:
        replicate_groups.setdefault(run["model_name"], []).append(run)
    # A replicate set is only usable alongside its canonical run.
    seed_sets = []
    for model_name, runs in replicate_groups.items():
        primary = next((r for r in rows if r["model_name"] == model_name), None)
        if primary is not None:
            seed_sets.append([primary, *runs])
    seed_replicates = max(seed_sets, key=len) if seed_sets else None
    if seed_replicates:
        log.info("Seed replicates found",
                 fields={"model": seed_replicates[0]["model_name"],
                         "runs": len(seed_replicates),
                         "seeds": [r["seed"] for r in seed_replicates]})

    benchmark = [r for r in rows if r["group"] in {"benchmark", "classical"}]
    ablation = [r for r in rows if r["group"] == "ablation"]
    placement = [r for r in rows if r["group"] == "placement"]
    production = [r for r in rows if r["protocol"] == "production"]

    log.info("Loaded ledger", fields={"total": len(rows), "benchmark": len(benchmark),
                                      "ablation": len(ablation), "placement": len(placement),
                                      "production": len(production)})

    if not args.no_plots and benchmark:
        plot_model_comparison(benchmark, "accuracy", settings.plots_dir / "compare_accuracy.png",
                              "Test accuracy by model")
        plot_model_comparison(benchmark, "f1_macro", settings.plots_dir / "compare_f1_macro.png",
                              "Macro F1 by model")
        plot_model_comparison(benchmark, "precision_macro",
                              settings.plots_dir / "compare_precision.png",
                              "Macro precision by model")
        plot_model_comparison(benchmark, "recall_macro", settings.plots_dir / "compare_recall.png",
                              "Macro recall by model")
        plot_model_comparison(benchmark, "top3_accuracy",
                              settings.plots_dir / "compare_top3.png", "Top-3 accuracy by model")
        plot_model_comparison(benchmark, "inference_ms",
                              settings.plots_dir / "compare_latency.png",
                              "Single-image inference latency (lower is better)",
                              higher_is_better=False)
        plot_model_comparison(benchmark, "ece", settings.plots_dir / "compare_ece.png",
                              "Expected calibration error (lower is better)",
                              higher_is_better=False)
        plot_scatter_tradeoff(benchmark, "model_size_mb", "accuracy",
                              settings.plots_dir / "tradeoff_size_accuracy.png",
                              "Model size vs accuracy", "Model size (MB)", "Test accuracy",
                              log_x=True)
        plot_scatter_tradeoff(benchmark, "inference_ms", "accuracy",
                              settings.plots_dir / "tradeoff_latency_accuracy.png",
                              "Inference latency vs accuracy", "ms / image", "Test accuracy")
        plot_scatter_tradeoff(benchmark, "parameters_millions", "f1_macro",
                              settings.plots_dir / "tradeoff_params_f1.png",
                              "Parameter count vs macro F1", "Parameters (millions)", "Macro F1",
                              log_x=True)
        log.info("Wrote comparison charts")

    ablation_summary = build_ablation_summary(ablation, rows, seed_replicates)

    # The matched CBAM/no-CBAM backbone pairs live in the benchmark, so this
    # evidence exists whether or not the ablation suite has been run.
    noise = estimate_noise_floor(rows, seed_replicates)
    pair_analysis = transfer_cbam_pairs(
        rows, noise.get("threshold_f1_macro") if noise.get("available") else None
    )
    ablation_summary.setdefault("transfer_pairs", pair_analysis)
    placement_summary = placement_analysis(
        rows, noise.get("threshold_f1_macro") if noise.get("available") else None
    )
    ablation_summary.setdefault("placement", placement_summary)

    if ablation_summary.get("available") or pair_analysis.get("available"):
        settings.results_dir.joinpath("ablation_summary.json").write_text(
            json.dumps(ablation_summary, indent=2), encoding="utf-8"
        )
        if not args.no_plots:
            plot_ablation(
                [{"label": e["label"], "accuracy": e["accuracy"], "f1_macro": e["f1_macro"],
                  "recall_macro": e["recall_macro"]} for e in ablation_summary["arms"]],
                settings.plots_dir / "ablation.png",
            )
        log.info("Wrote ablation summary")

    if placement and not args.no_plots:
        plot_model_comparison(placement, "f1_macro",
                              settings.plots_dir / "cbam_placement.png",
                              "CBAM placement: macro F1 by number of attended stages")

    best_accuracy = max(rows, key=lambda r: r["accuracy"] or 0)
    best_f1 = max(rows, key=lambda r: r["f1_macro"] or 0)
    fastest = min((r for r in rows if r["inference_ms"]), key=lambda r: r["inference_ms"])
    smallest = min((r for r in rows if r["model_size_mb"]), key=lambda r: r["model_size_mb"])
    proposed = [r for r in rows if r["proposed"]]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_count": len(rows),
        "groups": {
            "benchmark": len(benchmark), "ablation": len(ablation),
            "placement": len(placement), "production": len(production),
        },
        "best_accuracy": {"model": best_accuracy["label"], "value": best_accuracy["accuracy"],
                          "model_name": best_accuracy["model_name"]},
        "best_f1_macro": {"model": best_f1["label"], "value": best_f1["f1_macro"],
                          "model_name": best_f1["model_name"]},
        "fastest": {"model": fastest["label"], "ms_per_image": fastest["inference_ms"],
                    "accuracy": fastest["accuracy"]},
        "smallest": {"model": smallest["label"], "size_mb": smallest["model_size_mb"],
                     "accuracy": smallest["accuracy"]},
        "proposed_models": [
            {"model_name": r["model_name"], "label": r["label"], "accuracy": r["accuracy"],
             "f1_macro": r["f1_macro"], "protocol": r["protocol"]}
            for r in sorted(proposed, key=lambda r: -(r["f1_macro"] or 0))
        ],
        "rows": rows,
        "ablation": ablation_summary,
        "noise_floor": noise,
        "transfer_cbam_pairs": pair_analysis,
        "placement_analysis": placement_summary,
    }
    settings.results_dir.joinpath("summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 104)
    print(f"{'MODEL':<38}{'GROUP':<11}{'ACC':>8}{'MACRO F1':>10}{'TOP-3':>8}"
          f"{'MB':>8}{'ms/img':>9}{'ECE':>8}")
    print("=" * 104)
    def cell(value, width, digits):
        # A missing metric is not zero. Classical models have no on-disk size or
        # calibration record, and printing 0.0000 would read as a real measurement.
        return f"{'n/a':>{width}}" if value is None else f"{value:>{width}.{digits}f}"

    for row in sorted(rows, key=lambda r: -(r["accuracy"] or 0)):
        marker = "*" if row["proposed"] else " "
        print(f"{marker}{row['label'][:36]:<37}{row['group']:<11}"
              f"{cell(row['accuracy'], 8, 4)}{cell(row['f1_macro'], 10, 4)}"
              f"{cell(row['top3_accuracy'], 8, 4)}{cell(row['model_size_mb'], 8, 1)}"
              f"{cell(row['inference_ms'], 9, 2)}{cell(row['ece'], 8, 4)}")
    print("=" * 104)
    print("* = proposed architecture")

    if ablation_summary.get("verdict"):
        verdict = ablation_summary["verdict"]
        print("\nABLATION VERDICT")
        print("-" * 104)
        print(f"  {verdict['statement']}")
        print(f"  Inference overhead: {verdict['cbam_inference_overhead_pct']:+.1f}%   "
              f"Parameter overhead: {verdict['cbam_parameter_overhead_pct']:+.2f}%")
        print(f"  Best arm: {verdict['best_arm']} (macro F1 {verdict['best_arm_f1']:.4f})")
        noise = ablation_summary.get("noise_floor") or {}
        if noise.get("available"):
            repeats = len((noise.get("nondeterminism") or {}).get("observations", []))
            basis = ("seed replicates" if noise.get("threshold_basis") == "seed"
                     else f"{repeats} repeated configuration(s)")
            print(f"  Measured run-to-run noise floor: "
                  f"{noise['threshold_f1_macro'] * 100:.2f} points of macro F1 "
                  f"(from {basis})")
        print(f"  {verdict['caveat']}")

    pairs = pair_analysis or {}
    if pairs.get("available"):
        print("\nCBAM ON PRETRAINED BACKBONES (matched pairs from the benchmark)")
        print("-" * 104)
        print(f"  {'BACKBONE':<22}{'F1 without':>12}{'F1 with':>10}{'delta':>11}"
              f"{'params':>10}{'latency':>10}")
        for entry in pairs["pairs"]:
            print(f"  {entry['backbone']:<22}"
                  f"{entry['without_cbam']['f1_macro']:>12.4f}"
                  f"{entry['with_cbam']['f1_macro']:>10.4f}"
                  f"{entry['delta_f1_macro'] * 100:>+9.2f}pt"
                  f"{entry['parameter_overhead_pct']:>+9.1f}%"
                  f"{entry['inference_overhead_pct']:>+9.1f}%")
        print(f"\n  {pairs['verdict']}")

    print(f"\nWrote {settings.results_dir / 'summary.json'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
