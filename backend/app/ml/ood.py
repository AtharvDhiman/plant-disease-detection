"""Out-of-distribution detection: deciding when *not* to answer.

The quality checks in :mod:`app.ml.quality` catch *photographic* problems - a
blurred, dark or tiny image. They cannot catch a perfectly sharp, well-lit
photograph of something that simply is not a plant leaf. That is this module's
job, and it works from the classifier's own output distribution.

Scores implemented
------------------
``msp``      Maximum softmax probability. Simple and surprisingly strong, but
             neural networks are famously over-confident on unfamiliar inputs.
``entropy``  Normalised Shannon entropy of the class distribution. High entropy
             means the model spread its belief across many classes.
``energy``   ``-logsumexp(logits)`` (Liu et al., 2020). Unlike softmax it keeps
             the logits' absolute magnitude, which drops for inputs unlike
             anything in training, so it separates near-OOD better than MSP.

The thresholds are not guessed: ``training/calibrate_ood.py`` measures each
score on the held-out test split and on out-of-distribution proxies, then picks
operating points at a target true-positive rate. The resulting JSON is loaded
here. Without that file the module falls back to the configured defaults and
says so in its output, rather than pretending to be calibrated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def msp_score(probabilities: np.ndarray) -> np.ndarray:
    """Maximum softmax probability; higher means more in-distribution."""
    return probabilities.max(axis=-1)


def normalised_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Shannon entropy scaled to ``[0, 1]``; higher means more uncertain."""
    num_classes = probabilities.shape[-1]
    if num_classes <= 1:
        return np.zeros(probabilities.shape[:-1], dtype=np.float32)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=-1)
    return entropy / np.log(num_classes)


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Free-energy OOD score; *lower* (more negative) means in-distribution.

    Returned with the paper's sign convention ``E(x) = -T * logsumexp(logits/T)``.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        temperature = 1.0
    scaled = logits / temperature
    shifted = np.where(np.isfinite(scaled.max(axis=-1)), scaled.max(axis=-1), 0.0)
    logsumexp = shifted + np.log(np.maximum(np.exp(scaled - shifted[..., None]).sum(axis=-1), 1e-12))
    return -temperature * logsumexp


def margin_score(probabilities: np.ndarray) -> np.ndarray:
    """Gap between the top-1 and top-2 probabilities."""
    top2 = np.sort(probabilities, axis=-1)[..., -2:]
    return top2[..., -1] - top2[..., -2]


# --------------------------------------------------------------------------- #
# Calibrated decision
# --------------------------------------------------------------------------- #


@dataclass
class OODThresholds:
    """Operating points for the OOD decision."""

    msp_min: float
    entropy_max: float
    energy_max: float
    method: str = "msp+entropy"
    calibrated: bool = False
    target_tpr: float | None = None
    source: str | None = None
    metrics: dict | None = None
    model_name: str | None = None

    @classmethod
    def from_file(cls, path: Path, expected_model: str | None = None) -> OODThresholds | None:
        """Load calibrated thresholds, refusing ones fitted to a different model.

        Thresholds are only meaningful for the network they were measured on:
        the softmax floor and the energy limit both depend on that model's logit
        scale. Applying one model's thresholds to another rejects correct,
        confident predictions - which is exactly what happened during
        development when the production model changed and the generic
        ``ood_thresholds.json`` was left behind. Returning ``None`` here makes
        the caller fall back to the configured defaults, which are conservative
        but at least not actively wrong.
        """
        if not Path(path).exists():
            return None
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        chosen = data.get("selected", {})
        if not chosen:
            return None

        calibrated_for = data.get("model")
        if expected_model and calibrated_for and calibrated_for != expected_model:
            return None

        return cls(
            msp_min=float(chosen["msp_min"]),
            entropy_max=float(chosen["entropy_max"]),
            energy_max=float(chosen["energy_max"]),
            method=chosen.get("method", "msp+entropy"),
            calibrated=True,
            target_tpr=chosen.get("target_tpr"),
            source=str(path),
            metrics=data.get("metrics"),
            model_name=calibrated_for,
        )

    @classmethod
    def defaults(cls, msp_min: float, entropy_max: float) -> OODThresholds:
        return cls(
            msp_min=msp_min,
            entropy_max=entropy_max,
            # Energy is unbounded, so an uncalibrated default cannot gate on it.
            energy_max=float("inf"),
            method="msp+entropy",
            calibrated=False,
        )


@dataclass
class OODVerdict:
    """Whether one prediction should be trusted, and why."""

    is_ood: bool
    reasons: list[str]
    scores: dict
    thresholds: dict

    def to_dict(self) -> dict:
        return {
            "is_out_of_distribution": self.is_ood,
            "reasons": self.reasons,
            "scores": {k: round(float(v), 5) for k, v in self.scores.items()},
            "thresholds": self.thresholds,
        }


def assess(logits: np.ndarray, thresholds: OODThresholds) -> OODVerdict:
    """Judge a single prediction's logits against the calibrated thresholds."""
    logits = np.asarray(logits, dtype=np.float64).reshape(1, -1)
    probabilities = softmax(logits)

    msp = float(msp_score(probabilities)[0])
    entropy = float(normalised_entropy(probabilities)[0])
    energy = float(energy_score(logits)[0])
    margin = float(margin_score(probabilities)[0])

    # Only the criteria named by the calibrated method may veto. Applying all
    # three unconditionally was wrong twice over.
    #
    # First, it contradicted the calibration's own recorded contract: the
    # thresholds file states that serving enforces MSP and entropy, and that the
    # energy limit is kept for reference because it is the least stable of the
    # three across retraining - it depends on the absolute logit scale, which
    # moves whenever the model does.
    #
    # Second, and worse, each threshold is independently set to keep 95% of
    # in-distribution inputs. Combining them with OR does not preserve that
    # rate: three criteria each rejecting their own 5% tail reject up to 15%
    # between them. The docstring on select_thresholds promises "about 5% of
    # genuine leaf photographs will be asked for a retake", and enforcing the
    # union quietly broke that promise. A correct 99.95%-confidence prediction
    # on an image from the dataset's own validation split was being refused,
    # because it happened to sit in the energy tail while passing both criteria
    # the calibration actually selected.
    active = {name.strip() for name in (thresholds.method or "msp+entropy").split("+")}

    reasons: list[str] = []
    if "msp" in active and msp < thresholds.msp_min:
        reasons.append(
            # Say "uncalibrated" explicitly. The confidence shown to the user is
            # temperature-scaled, so quoting this raw figure as "confidence"
            # made the page appear to contradict itself: 94.7% in the headline,
            # 55.9% in the refusal. Both are correct on their own scale, and the
            # thresholds were measured on this one.
            f"Uncalibrated top-class probability {msp * 100:.1f}% is below the "
            f"{thresholds.msp_min * 100:.1f}% operating point measured on genuine "
            f"leaves. (The headline confidence is temperature-scaled and so reads "
            f"higher; the reliability decision uses the uncalibrated scale the "
            f"thresholds were fitted on.)"
        )
    if "entropy" in active and entropy > thresholds.entropy_max:
        reasons.append(
            f"The prediction is spread across many classes "
            f"(normalised entropy {entropy:.2f} > {thresholds.entropy_max:.2f})."
        )
    if ("energy" in active and np.isfinite(thresholds.energy_max)
            and energy > thresholds.energy_max):
        reasons.append(
            f"Free-energy score {energy:.2f} exceeds the calibrated limit "
            f"{thresholds.energy_max:.2f}, indicating an input unlike the training data."
        )

    return OODVerdict(
        is_ood=bool(reasons),
        reasons=reasons,
        scores={"msp": msp, "entropy": entropy, "energy": energy, "margin": margin},
        thresholds={
            "msp_min": thresholds.msp_min,
            "entropy_max": thresholds.entropy_max,
            "energy_max": None if not np.isfinite(thresholds.energy_max) else thresholds.energy_max,
            "calibrated": thresholds.calibrated,
            "method": thresholds.method,
            "enforced": sorted(active),
        },
    )


# --------------------------------------------------------------------------- #
# Threshold selection (used by training/calibrate_ood.py)
# --------------------------------------------------------------------------- #


def select_thresholds(
    in_dist_logits: np.ndarray,
    ood_logits: np.ndarray,
    target_tpr: float = 0.95,
) -> dict:
    """Choose operating points that keep ``target_tpr`` of in-distribution inputs.

    Fixing the true-positive rate (rather than maximising accuracy on a proxy OOD
    set) means the user-visible behaviour is predictable: at a 0.95 target, about
    5% of genuine leaf photographs will be asked for a retake.
    """
    in_probs = softmax(in_dist_logits)
    ood_probs = softmax(ood_logits)

    in_msp, ood_msp = msp_score(in_probs), msp_score(ood_probs)
    in_ent, ood_ent = normalised_entropy(in_probs), normalised_entropy(ood_probs)
    in_en, ood_en = energy_score(in_dist_logits), energy_score(ood_logits)

    msp_min = float(np.quantile(in_msp, 1 - target_tpr))
    entropy_max = float(np.quantile(in_ent, target_tpr))
    energy_max = float(np.quantile(in_en, target_tpr))

    def _rates(in_scores, ood_scores, threshold, higher_is_in_dist):
        if higher_is_in_dist:
            tpr = float((in_scores >= threshold).mean())
            fpr = float((ood_scores >= threshold).mean())
        else:
            tpr = float((in_scores <= threshold).mean())
            fpr = float((ood_scores <= threshold).mean())
        return {"tpr": tpr, "fpr": fpr, "ood_detection_rate": 1 - fpr}

    return {
        "target_tpr": target_tpr,
        "msp_min": msp_min,
        "entropy_max": entropy_max,
        "energy_max": energy_max,
        "per_score": {
            "msp": _rates(in_msp, ood_msp, msp_min, True),
            "entropy": _rates(in_ent, ood_ent, entropy_max, False),
            "energy": _rates(in_en, ood_en, energy_max, False),
        },
        "distributions": {
            "in_distribution": {
                "msp_mean": float(in_msp.mean()), "entropy_mean": float(in_ent.mean()),
                "energy_mean": float(in_en.mean()),
            },
            "ood": {
                "msp_mean": float(ood_msp.mean()), "entropy_mean": float(ood_ent.mean()),
                "energy_mean": float(ood_en.mean()),
            },
        },
    }


def auroc(in_scores: np.ndarray, ood_scores: np.ndarray, higher_is_in_dist: bool = True) -> float:
    """AUROC for separating in-distribution from OOD, via the rank-sum identity."""
    in_scores = np.asarray(in_scores, dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    if not higher_is_in_dist:
        in_scores, ood_scores = -in_scores, -ood_scores

    combined = np.concatenate([in_scores, ood_scores])
    order = combined.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)
    # Average ranks within ties so exact duplicates do not bias the statistic.
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    tie_sums = np.zeros(len(counts))
    np.add.at(tie_sums, inverse, ranks)
    ranks = (tie_sums / counts)[inverse]

    n_in, n_ood = len(in_scores), len(ood_scores)
    rank_sum = ranks[:n_in].sum()
    return float((rank_sum - n_in * (n_in + 1) / 2) / (n_in * n_ood))
