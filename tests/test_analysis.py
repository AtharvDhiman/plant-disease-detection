"""Tests for the results-analysis layer.

These functions produce the headline scientific claims of the project - the
ablation verdict, the significance threshold and the transfer-pair comparison -
so a silent error here would be published as a finding. The tests below pin the
behaviour that keeps those claims honest: that a threshold measured with the
initialisation held fixed is labelled a lower bound rather than a significance
test, that seed replicates widen the threshold when they exist, and that a
difference smaller than the threshold is never reported as an improvement.
"""
from __future__ import annotations

# conftest and analyse_results both import app.core.runtime before matplotlib,
# so the OpenMP guard is already in place by the time this module is imported.
import pytest
from analyse_results import (
    build_ablation_summary,
    estimate_noise_floor,
    placement_analysis,
    to_row,
    transfer_cbam_pairs,
)


def row(model_name: str, f1: float, *, accuracy: float | None = None,
        group: str = "ablation", seed: int = 42, parameters: int = 1_000_000,
        inference_ms: float = 2.0, label: str | None = None,
        num_workers: int = 4) -> dict:
    """Build an analysis row through the production flattener.

    Going through ``to_row`` rather than hand-writing a dict means these tests
    keep the real schema: if a new field is added to the row, the tests get it
    too instead of failing on a KeyError that has nothing to do with the
    behaviour under test.
    """
    record = {
        "experiment_id": f"{model_name}__test__{seed}",
        "model_name": model_name,
        "label": label or model_name,
        "group": group,
        "protocol": "benchmark",
        "proposed": False,
        "total_parameters": parameters,
        "model_size_mb": parameters * 4 / 1e6,
        "training_time_s": 600.0,
        "epochs_run": 12,
        "best_val_accuracy": f1,
        "config": {"seed": seed, "num_workers": num_workers},
        "test": {
            "metrics": {
                "accuracy": accuracy if accuracy is not None else f1,
                "precision_macro": f1,
                "recall_macro": f1,
                "f1_macro": f1,
                "f1_weighted": f1,
                "top3_accuracy": min(1.0, f1 + 0.03),
                "top5_accuracy": min(1.0, f1 + 0.04),
                "roc_auc_macro_ovr": min(1.0, f1 + 0.02),
            },
            "calibration": {"ece": 0.06},
            "latency": {"single_image_ms": inference_ms},
        },
    }
    return to_row(record)


# --------------------------------------------------------------------------- #
# Noise floor
# --------------------------------------------------------------------------- #

def test_noise_floor_without_duplicates_is_unavailable():
    result = estimate_noise_floor([row("cnn_baseline", 0.95)])
    assert result["available"] is False
    assert "reason" in result


def test_fixed_seed_duplicates_are_labelled_a_lower_bound():
    """A same-seed repeat must not be presented as a significance threshold.

    Two runs of an identical configuration share their initialisation. Two
    *different* architectures do not - different parameter shapes draw different
    starting weights from the same seed - so the same-seed gap understates the
    noise in every comparison the ablation actually makes.
    """
    result = estimate_noise_floor([
        row("cnn_baseline", 0.9578, group="benchmark"),
        row("abl_cnn_none", 0.9582),
    ])
    assert result["available"] is True
    assert result["threshold_basis"] == "nondeterminism"
    assert result["threshold_f1_macro"] == abs(0.9582 - 0.9578)
    # The wording is the safeguard: it has to say this is a floor, not a test.
    assert "LOWER BOUND" in result["interpretation"]
    assert "not as a significance threshold" in result["interpretation"]


def test_seed_replicates_take_over_as_the_threshold():
    """When replicates exist they supersede the same-seed floor, and widen it."""
    rows = [
        row("cnn_baseline", 0.9578, group="benchmark"),
        row("abl_cnn_none", 0.9582),
    ]
    replicates = [
        row("cnn_baseline", 0.9578, group="benchmark", seed=42),
        row("cnn_baseline", 0.9491, group="benchmark", seed=43),
        row("cnn_baseline", 0.9623, group="benchmark", seed=44),
    ]
    result = estimate_noise_floor(rows, replicates)

    assert result["threshold_basis"] == "seed"
    assert result["threshold_f1_macro"] == 0.9623 - 0.9491
    # The seed floor must be the wider of the two, or it would license claims
    # that the narrower fixed-seed floor does not support.
    assert result["threshold_f1_macro"] > result["nondeterminism"]["max_f1_macro_gap"]
    assert result["seed"]["seeds"] == [42, 43, 44]
    assert result["seed"]["sd_f1_macro"] > 0
    assert "LOWER BOUND" not in result["interpretation"]


def test_single_replicate_is_not_enough_for_a_seed_floor():
    """One extra run gives no spread; the floor must fall back, not divide by zero."""
    result = estimate_noise_floor(
        [row("cnn_baseline", 0.9578, group="benchmark"), row("abl_cnn_none", 0.9582)],
        [row("cnn_baseline", 0.9578, group="benchmark", seed=42)],
    )
    assert result["threshold_basis"] == "nondeterminism"
    assert "seed" not in result


# --------------------------------------------------------------------------- #
# Ablation verdict
# --------------------------------------------------------------------------- #

def test_gap_below_the_threshold_is_not_called_an_improvement():
    """The central honesty guarantee: sub-noise gaps are reported as inconclusive."""
    rows = [
        row("abl_cnn_none", 0.9580),
        row("abl_cnn_cbam", 0.9584, parameters=1_030_000, inference_ms=2.4),
    ]
    # cnn_baseline/abl_cnn_none are the duplicate pair: a 0.10pt gap sets the
    # floor, above the 0.04pt that CBAM gains.
    all_rows = rows + [row("cnn_baseline", 0.9570, group="benchmark")]
    summary = build_ablation_summary(rows, all_rows)

    assert summary["available"] is True
    cbam = next(a for a in summary["arms"] if a["model_name"] == "abl_cnn_cbam")
    # +0.04pt sits just under the measured floor, so it cannot be claimed.
    assert 0 < cbam["delta_vs_control"]["f1_macro"] < summary["verdict"]["significance_threshold_f1"]

    statement = summary["verdict"]["statement"]
    assert "within" in statement and "run-to-run variation" in statement
    assert "not established" in statement
    # A gain this small must never be stated as a bare improvement.
    assert not statement.startswith("CBAM improved macro F1 by 0.04 points and accuracy")


def test_gap_above_the_threshold_is_reported_as_a_real_gain():
    """The mirror case: a genuine effect must not be hedged into nothing."""
    rows = [
        row("abl_cnn_none", 0.9400),
        row("abl_cnn_cbam", 0.9600, parameters=1_030_000, inference_ms=2.4),
    ]
    all_rows = rows + [row("cnn_baseline", 0.9398, group="benchmark")]
    summary = build_ablation_summary(rows, all_rows)

    statement = summary["verdict"]["statement"]
    assert statement.startswith("CBAM improved macro F1 by 2.00 points")
    assert "not established" not in statement
    assert summary["verdict"]["cbam_is_best"] is True


def test_cbam_regression_is_stated_plainly():
    """If the proposed architecture loses, the summary has to say so."""
    rows = [
        row("abl_cnn_none", 0.9600),
        row("abl_cnn_cbam", 0.9400, parameters=1_030_000, inference_ms=2.4),
    ]
    all_rows = rows + [row("cnn_baseline", 0.9598, group="benchmark")]
    summary = build_ablation_summary(rows, all_rows)

    statement = summary["verdict"]["statement"]
    assert "did **not** improve" in statement
    assert summary["verdict"]["cbam_is_best"] is False


def test_ablation_needs_its_control():
    summary = build_ablation_summary([row("abl_cnn_cbam", 0.96)])
    assert summary["available"] is False
    assert "abl_cnn_none" in summary["reason"]


# --------------------------------------------------------------------------- #
# Transfer pairs
# --------------------------------------------------------------------------- #

def test_transfer_pairs_report_negative_deltas_honestly():
    """A backbone where CBAM hurts must be reported as hurting."""
    rows = [
        row("efficientnet_b0", 0.9700, group="benchmark", parameters=5_300_000),
        row("efficientnet_b0_cbam", 0.9681, group="benchmark", parameters=5_570_000),
        row("resnet50", 0.9912, group="benchmark", parameters=23_500_000),
        row("resnet50_cbam", 0.9914, group="benchmark", parameters=24_000_000),
    ]
    result = transfer_cbam_pairs(rows, 0.0005)
    assert result["available"] is True
    assert len(result["pairs"]) == 2

    effnet = next(p for p in result["pairs"]
                  if p["without_cbam"]["model_name"] == "efficientnet_b0")
    assert effnet["delta_f1_macro"] < 0
    # -0.20pt is far outside a 0.05pt floor, so this is a real regression and
    # must not be softened into "within noise".
    assert effnet["beyond_noise"] is True


def test_transfer_pairs_absent_without_matched_runs():
    result = transfer_cbam_pairs([row("resnet50", 0.99, group="benchmark")], 0.0005)
    assert result["available"] is False


def test_floor_takes_the_widest_pair_not_the_average():
    """One close-agreeing pair must not be allowed to understate the floor.

    The real runs produced gaps of 0.04 pt and 0.35 pt for the two duplicate
    pairs. Averaging them, or reporting only the first, would license claims the
    second pair plainly contradicts.
    """
    rows = [
        row("cnn_baseline", 0.9578, group="benchmark"),
        row("abl_cnn_none", 0.9582),
        row("cnn_se", 0.9598, group="benchmark"),
        row("abl_cnn_se", 0.9634),
    ]
    result = estimate_noise_floor(rows)

    gaps = [o["f1_macro_gap"] for o in result["nondeterminism"]["observations"]]
    assert len(gaps) == 2
    assert result["threshold_f1_macro"] == max(gaps)
    assert result["threshold_f1_macro"] > result["nondeterminism"]["mean_f1_macro_gap"]
    # The spread between pairs is worth stating, not hiding behind one number.
    assert "ranged from" in result["interpretation"]


def test_differing_worker_count_is_disclosed():
    """The report must not claim only GPU non-determinism when more varied.

    Each DataLoader worker gets its own derived seed, so changing the worker
    count changes the augmentation stream. Describing such a pair as a pure
    determinism check would misdescribe what was measured.
    """
    rows = [
        row("cnn_baseline", 0.9578, group="benchmark", num_workers=4),
        row("abl_cnn_none", 0.9582, num_workers=2),
    ]
    observation = estimate_noise_floor(rows)["nondeterminism"]["observations"][0]

    disclosure = " ".join(observation["differs_in"])
    # Counts are sorted so the wording is stable regardless of which member
    # happened to score highest.
    assert "worker count (2 vs 4)" in disclosure
    assert "augmentation" in disclosure


def test_matched_worker_count_claims_only_non_determinism():
    """When the worker count matches, the extra caveat must not be invented."""
    rows = [
        row("cnn_baseline", 0.9578, group="benchmark", num_workers=4),
        row("abl_cnn_none", 0.9582, num_workers=4),
    ]
    observation = estimate_noise_floor(rows)["nondeterminism"]["observations"][0]

    assert not any("worker count" in entry for entry in observation["differs_in"])


def test_transfer_deltas_inside_the_floor_are_called_inconclusive():
    """A -0.20pt delta against a 0.35pt floor is noise, not a regression."""
    rows = [
        row("efficientnet_b0", 0.9700, group="benchmark", parameters=5_300_000),
        row("efficientnet_b0_cbam", 0.9681, group="benchmark", parameters=5_570_000),
    ]
    result = transfer_cbam_pairs(rows, 0.0035)

    pair = result["pairs"][0]
    assert pair["delta_f1_macro"] < 0
    assert pair["beyond_noise"] is False


def test_narrow_margin_over_the_floor_is_called_suggestive():
    """Clearing the floor by 1.3x must not be worded like a solid result.

    The floor is itself estimated from a handful of repeats, so a delta only
    slightly above it is inside the uncertainty of the threshold. The wording
    has to stay proportional to how much evidence there actually is.
    """
    rows = [
        row("abl_cnn_none", 0.9582),
        row("abl_cnn_cbam", 0.9628, parameters=1_030_000, inference_ms=7.4),
    ]
    # A duplicate pair 0.35pt apart: the CBAM gain of 0.46pt is only 1.3x that.
    all_rows = rows + [
        row("cnn_se", 0.9598, group="benchmark"),
        row("abl_cnn_se", 0.9634),
    ]
    summary = build_ablation_summary(rows, all_rows)
    verdict = summary["verdict"]

    assert verdict["evidence_strength"] == "suggestive"
    assert 1.0 < verdict["margin_over_threshold"] < 2.0
    assert "only narrowly clears" in verdict["statement"]
    assert "suggestive rather than established" in verdict["statement"]


def test_wide_margin_over_the_floor_is_called_supported():
    rows = [
        row("abl_cnn_none", 0.9400),
        row("abl_cnn_cbam", 0.9600, parameters=1_030_000, inference_ms=2.4),
    ]
    all_rows = rows + [row("cnn_baseline", 0.9398, group="benchmark")]
    verdict = build_ablation_summary(rows, all_rows)["verdict"]

    assert verdict["evidence_strength"] == "supported"
    assert verdict["margin_over_threshold"] > 2.0
    assert "more than twice" in verdict["statement"]


# --------------------------------------------------------------------------- #
# CBAM placement
# --------------------------------------------------------------------------- #

def test_placement_reports_a_harmful_placement_as_harmful():
    """The same block below the control must be stated, not averaged away."""
    rows = [
        row("abl_cnn_none", 0.9582),
        row("place_cbam_last", 0.9569, group="placement"),
        row("place_cbam_last2", 0.9676, group="placement"),
    ]
    result = placement_analysis(rows, 0.0035)

    assert result["available"] is True
    assert result["best"] == "place_cbam_last2"
    worst = next(v for v in result["variants"] if v["model_name"] == "place_cbam_last")
    assert worst["delta_f1_macro"] < 0
    assert "*below* the no-attention control" in result["statement"]
    # A 1.07pt spread against a 0.35pt floor is a real effect.
    assert result["spread_beyond_noise"] is True


def test_placement_spread_inside_the_floor_is_not_overclaimed():
    rows = [
        row("abl_cnn_none", 0.9582),
        row("place_cbam_last", 0.9600, group="placement"),
        row("place_cbam_last2", 0.9610, group="placement"),
    ]
    result = placement_analysis(rows, 0.0035)

    assert result["spread_beyond_noise"] is False
    assert "smaller than" in result["statement"]
    assert "*below* the no-attention control" not in result["statement"]


def test_placement_needs_a_control():
    result = placement_analysis([row("place_cbam_last2", 0.9676, group="placement")], 0.0035)
    assert result["available"] is False


def test_identical_architectures_are_one_group_not_two_pairs():
    """cnn_cbam, abl_cnn_cbam and place_cbam_all are the same network.

    abl_cnn_cbam leaves attention_stages at its default of "all", which is what
    place_cbam_all specifies. Treating them as separate pairs would count the
    same evidence twice and report a narrower spread than the three runs show.
    """
    rows = [
        row("cnn_cbam", 0.9605, group="benchmark"),
        row("abl_cnn_cbam", 0.9628),
        row("place_cbam_all", 0.9660, group="placement"),
    ]
    result = estimate_noise_floor(rows)
    observations = result["nondeterminism"]["observations"]

    assert len(observations) == 1
    assert observations[0]["runs"] == 3
    # The spread must span the full group, not just the first two members.
    assert observations[0]["f1_macro_gap"] == pytest.approx(0.9660 - 0.9605)


def test_narrow_seed_floor_does_not_lower_the_threshold():
    """A seed study that comes out tight must not weaken evidence already held.

    This is the real case from the study: three seeds of the control spanned
    0.22 pt, while three runs of one identical CBAM configuration spanned
    0.47 pt. Promoting the seed floor because it is the more principled design
    would have halved the bar, licensing claims the repeat runs contradict.
    """
    rows = [
        row("cnn_cbam", 0.9605, group="benchmark"),
        row("abl_cnn_cbam", 0.9628),
        row("place_cbam_all", 0.9652, group="placement"),
    ]
    replicates = [
        row("cnn_baseline", 0.9578, group="benchmark", seed=42),
        row("cnn_baseline", 0.9556, group="benchmark", seed=43),
        row("cnn_baseline", 0.9570, group="benchmark", seed=44),
    ]
    result = estimate_noise_floor(rows, replicates)

    seed_spread = result["seed"]["spread_f1_macro"]
    repeat_spread = result["nondeterminism"]["max_f1_macro_gap"]
    assert seed_spread < repeat_spread

    assert result["threshold_f1_macro"] == pytest.approx(repeat_spread)
    assert result["threshold_basis"] == "nondeterminism"
    # Both measurements have to be visible, not just the winning one.
    assert "at 3 seeds" in result["interpretation"]
    assert "moved it further still" in result["interpretation"]


def test_wide_seed_floor_does_raise_the_threshold():
    """The mirror case: when seeds vary more, they set the bar."""
    rows = [
        row("cnn_baseline", 0.9578, group="benchmark"),
        row("abl_cnn_none", 0.9582),
    ]
    replicates = [
        row("cnn_baseline", 0.9578, group="benchmark", seed=42),
        row("cnn_baseline", 0.9450, group="benchmark", seed=43),
        row("cnn_baseline", 0.9620, group="benchmark", seed=44),
    ]
    result = estimate_noise_floor(rows, replicates)

    assert result["threshold_basis"] == "seed"
    assert result["threshold_f1_macro"] == pytest.approx(0.9620 - 0.9450)
    assert "wider than" in result["interpretation"]


def test_production_runs_never_enter_the_noise_floor():
    """A production run is a different experiment, not a repeat of a research one.

    cnn_cbam exists under both protocols: 0.9605 on the fixed-budget subset and
    0.9842 on full data. Treating those as two runs of one configuration
    reported a 2.14-point noise floor instead of 0.47, and that inflated
    threshold then swallowed the placement finding entirely.
    """
    rows = [
        row("cnn_cbam", 0.9605, group="benchmark"),
        row("abl_cnn_cbam", 0.9628),
        row("place_cbam_all", 0.9652, group="placement"),
    ]
    rows[0]["protocol"] = "benchmark"
    rows[1]["protocol"] = "ablation"
    rows[2]["protocol"] = "placement"

    production = row("cnn_cbam", 0.9842, group="benchmark")
    production["protocol"] = "production"

    clean = estimate_noise_floor(rows)
    contaminated = estimate_noise_floor([*rows, production])

    assert contaminated["threshold_f1_macro"] == clean["threshold_f1_macro"]
    assert clean["threshold_f1_macro"] == pytest.approx(0.9652 - 0.9605)


def test_benchmark_ablation_and_placement_share_one_budget():
    """The filter must not drop ablation or placement rows.

    They carry different protocol tags but the same training budget, schedule
    and evaluation split. Keeping only `protocol == "benchmark"` would empty
    the duplicate groups and silently disable the noise floor.
    """
    from analyse_results import fixed_budget

    rows = []
    for name, protocol in (("cnn_baseline", "benchmark"), ("abl_cnn_none", "ablation"),
                           ("place_cbam_all", "placement"), ("cnn_cbam", "production")):
        entry = row(name, 0.96)
        entry["protocol"] = protocol
        rows.append(entry)

    kept = {r["protocol"] for r in fixed_budget(rows)}
    assert kept == {"benchmark", "ablation", "placement"}


def test_production_replication_reports_agreement():
    """The full-data runs are an independent check, not a restatement.

    cnn_cbam and place_cbam_last2 have the same parameter count and differ only
    in where CBAM sits, so their production runs test the placement finding at
    5.5x the training data.
    """
    rows = [
        row("abl_cnn_none", 0.9582),
        row("place_cbam_last2", 0.9676, group="placement"),
    ]
    for entry in (row("cnn_cbam", 0.9842, group="benchmark"),
                  row("place_cbam_last2", 0.9880, group="placement")):
        entry["protocol"] = "production"
        rows.append(entry)

    result = placement_analysis(rows, 0.0047)
    replication = result["production_replication"]

    assert replication["available"] is True
    assert replication["agrees_with_fixed_budget"] is True
    assert replication["delta_f1_macro"] == pytest.approx(0.9880 - 0.9842)
    assert "same contrast holds" in replication["statement"]
    # The production rows must not have leaked into the fixed-budget variants.
    assert all(v["model_name"] != "cnn_cbam" for v in result["variants"])


def test_production_replication_reports_disagreement_honestly():
    """If full data reverses the finding, the report must say it does not hold."""
    rows = [row("abl_cnn_none", 0.9582), row("place_cbam_last2", 0.9676, group="placement")]
    for entry in (row("cnn_cbam", 0.9880, group="benchmark"),
                  row("place_cbam_last2", 0.9842, group="placement")):
        entry["protocol"] = "production"
        rows.append(entry)

    replication = placement_analysis(rows, 0.0047)["production_replication"]
    assert replication["agrees_with_fixed_budget"] is False
    assert "does not survive" in replication["statement"]
    assert "should not be generalised" in replication["statement"]
