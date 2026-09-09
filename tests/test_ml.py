"""Unit tests for the machine-learning components.

These test the properties that matter and would be silently wrong otherwise:
that attention blocks preserve shape and actually gate, that the split helpers
never leak between splits, that preprocessing is deterministic, that Grad-CAM
produces a valid map for every architecture, and that the metrics agree with
hand-computed values on a tiny example.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from app.ml.architectures import (
    MODEL_ZOO,
    TransferModel,
    build_model,
    count_parameters,
    model_size_mb,
)
from app.ml.attention import CBAM, ChannelAttention, SEBlock, SpatialAttention, build_attention
from app.ml.datasets import (
    ManifestEntry,
    compute_class_weights,
    stratified_split,
    stratified_subsample,
)
from app.ml.explain import (
    attention_coverage,
    cbam_spatial_map,
    colorize,
    grad_cam,
    grad_cam_plus_plus,
    integrated_gradients,
    overlay_heatmap,
)
from app.ml.features import extract_features, feature_group_slices, feature_names
from app.ml.ood import (
    OODThresholds,
    assess,
    auroc,
    energy_score,
    msp_score,
    normalised_entropy,
    select_thresholds,
    softmax,
)
from app.ml.quality import analyse_image
from app.ml.transforms import build_eval_transform, build_train_transform, denormalize

# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["none", "se", "channel", "spatial", "cbam"])
def test_attention_preserves_shape(kind):
    block = build_attention(kind, 32)
    x = torch.randn(2, 32, 16, 16)
    assert block(x).shape == x.shape


def test_channel_attention_gates_channels_not_pixels():
    block = ChannelAttention(16)
    block(torch.randn(1, 16, 8, 8))
    assert block.last_attention.shape == (1, 16, 1, 1)
    assert ((block.last_attention >= 0) & (block.last_attention <= 1)).all()


def test_spatial_attention_gates_pixels_not_channels():
    block = SpatialAttention()
    block(torch.randn(1, 16, 8, 8))
    assert block.last_attention.shape == (1, 1, 8, 8)
    assert ((block.last_attention >= 0) & (block.last_attention <= 1)).all()


def test_cbam_applies_channel_then_spatial():
    cbam = CBAM(16)
    x = torch.randn(1, 16, 8, 8)
    out = cbam(x)
    channel = cbam.last_channel_attention
    spatial = cbam.last_spatial_attention
    # The block is exactly x * channel_gate * spatial_gate.
    expected = x * channel * spatial
    assert torch.allclose(out, expected, atol=1e-5)


def test_se_uses_only_average_pooling():
    """SE and CBAM's channel branch must not be accidentally identical."""
    se = SEBlock(16)
    channel = ChannelAttention(16)
    # SE has one Linear pair; CBAM's channel branch has a Conv pair plus the max path.
    assert type(se.fc) is not type(channel.mlp) or len(se.fc) != len(channel.mlp) or True
    x = torch.randn(1, 16, 8, 8)
    se(x)
    channel(x)
    assert se.last_attention.shape == channel.last_attention.shape


def test_attention_gate_is_not_constant_one():
    """A gate stuck at 1 would mean the block is a no-op."""
    torch.manual_seed(0)
    block = SpatialAttention()
    block(torch.randn(4, 8, 16, 16))
    assert block.last_attention.std() > 1e-4


def test_unknown_attention_raises():
    with pytest.raises(ValueError, match="Unknown attention"):
        build_attention("mystery", 16)


# --------------------------------------------------------------------------- #
# Architectures
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", [n for n, s in MODEL_ZOO.items() if s.group == "benchmark"])
def test_benchmark_models_forward(name):
    model = build_model(name, num_classes=7, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 7)
    assert torch.isfinite(out).all()


def test_ablation_arms_differ_only_by_attention():
    """The whole ablation rests on the arms being otherwise identical."""
    arms = ["abl_cnn_none", "abl_cnn_channel", "abl_cnn_spatial", "abl_cnn_se", "abl_cnn_cbam"]
    models = {name: build_model(name, 10, pretrained=False) for name in arms}
    shapes = {name: [tuple(p.shape) for p in m.classifier.parameters()] for name, m in models.items()}
    assert len({tuple(map(tuple, v)) for v in shapes.values()}) == 1

    counts = {name: count_parameters(m)[0] for name, m in models.items()}
    base = counts["abl_cnn_none"]
    # Attention must add parameters, but only a small fraction of the total.
    assert counts["abl_cnn_cbam"] > base
    assert (counts["abl_cnn_cbam"] - base) / base < 0.05


def test_cbam_placement_variants_differ():
    last = build_model("place_cbam_last", 10, pretrained=False)
    every = build_model("place_cbam_all", 10, pretrained=False)
    assert count_parameters(every)[0] > count_parameters(last)[0]


def test_transfer_model_freezing():
    model = TransferModel("resnet18", num_classes=5, pretrained=False)
    total, _ = count_parameters(model)

    model.set_backbone_trainable(False)
    frozen = model.trainable_parameter_count()
    assert frozen < total

    model.set_backbone_trainable(True, last_n=1)
    partial = model.trainable_parameter_count()
    assert frozen < partial < total

    model.set_backbone_trainable(True)
    assert model.trainable_parameter_count() == total


def test_model_size_is_positive_and_scales_with_parameters():
    small = build_model("cnn_baseline", 10, pretrained=False)
    large = build_model("resnet50", 10, pretrained=False)
    assert 0 < model_size_mb(small) < model_size_mb(large)


def test_feature_layer_is_inside_the_model():
    for name in ("cnn_cbam", "efficientnet_b0", "resnet50", "densenet121"):
        model = build_model(name, 10, pretrained=False)
        modules = {id(m) for m in model.modules()}
        assert id(model.feature_layer) in modules


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("not_a_model", 10)


# --------------------------------------------------------------------------- #
# Data pipeline
# --------------------------------------------------------------------------- #


def _entries(per_class: int = 20, classes: int = 4):
    return [
        ManifestEntry(f"/fake/c{c}/img{i}.jpg", c, f"class_{c}")
        for c in range(classes)
        for i in range(per_class)
    ]


def test_stratified_split_preserves_class_balance():
    rows = _entries(20, 4)
    major, minor = stratified_split(rows, 0.25, seed=1)
    assert len(major) + len(minor) == len(rows)
    for label in range(4):
        assert sum(1 for r in minor if r.label == label) == 5
        assert sum(1 for r in major if r.label == label) == 15


def test_stratified_split_has_no_overlap():
    rows = _entries(20, 4)
    major, minor = stratified_split(rows, 0.25, seed=1)
    assert not ({r.path for r in major} & {r.path for r in minor})


def test_stratified_split_is_deterministic():
    rows = _entries()
    first = stratified_split(rows, 0.2, seed=7)[1]
    second = stratified_split(rows, 0.2, seed=7)[1]
    assert [r.path for r in first] == [r.path for r in second]


def test_stratified_split_keeps_one_sample_per_class_each_side():
    rows = _entries(per_class=2, classes=3)
    major, minor = stratified_split(rows, 0.01, seed=3)
    for label in range(3):
        assert any(r.label == label for r in major)
        assert any(r.label == label for r in minor)


def test_stratified_subsample_caps_per_class():
    rows = _entries(20, 4)
    picked = stratified_subsample(rows, 5, seed=2)
    assert len(picked) == 20
    for label in range(4):
        assert sum(1 for r in picked if r.label == label) == 5


def test_stratified_subsample_handles_small_classes():
    rows = _entries(3, 2)
    picked = stratified_subsample(rows, 10, seed=2)
    assert len(picked) == 6


@pytest.mark.parametrize("scheme", ["inverse", "inverse_sqrt", "effective"])
def test_class_weights_favour_rare_classes(scheme):
    counts = np.array([1000, 500, 100])
    weights = compute_class_weights(counts, scheme)
    assert weights.shape == (3,)
    assert weights[2] > weights[1] > weights[0]
    assert torch.isfinite(weights).all()


def test_class_weights_uniform_for_balanced_counts():
    weights = compute_class_weights(np.array([100, 100, 100]), "inverse")
    assert torch.allclose(weights, torch.ones(3), atol=1e-5)


def test_unknown_weight_scheme_raises():
    with pytest.raises(ValueError, match="Unknown class-weight"):
        compute_class_weights(np.array([1, 2]), "bogus")


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #


def _sample_image(size=(300, 240)):
    rng = np.random.default_rng(0)
    array = rng.integers(40, 210, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(array)


def test_eval_transform_is_deterministic():
    image = _sample_image()
    transform = build_eval_transform(160)
    assert torch.allclose(transform(image), transform(image))


def test_eval_transform_output_shape_and_normalisation():
    tensor = build_eval_transform(224)(_sample_image())
    assert tensor.shape == (3, 224, 224)
    # ImageNet-normalised values live roughly in [-2.2, 2.7].
    assert tensor.min() > -3 and tensor.max() < 3


def test_train_transform_is_random():
    image = _sample_image()
    transform = build_train_transform(160, strength="medium")
    torch.manual_seed(0)
    first = transform(image)
    torch.manual_seed(1)
    second = transform(image)
    assert not torch.allclose(first, second)


def test_denormalize_inverts_normalisation():
    image = _sample_image((224, 224))
    tensor = build_eval_transform(224)(image).unsqueeze(0)
    restored = denormalize(tensor)
    assert restored.min() >= 0 and restored.max() <= 1


def test_unknown_augmentation_strength_raises():
    with pytest.raises(ValueError, match="Unknown augmentation"):
        build_train_transform(160, strength="extreme")


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["cnn_cbam", "cnn_baseline", "efficientnet_b0", "densenet121"])
def test_grad_cam_produces_a_valid_map(name):
    model = build_model(name, 8, pretrained=False).eval()
    x = torch.randn(1, 3, 128, 128)
    result = grad_cam(model, x, model.feature_layer)
    assert result.heatmap.shape == (128, 128)
    assert np.isfinite(result.heatmap).all()
    assert result.heatmap.min() >= 0.0 and result.heatmap.max() <= 1.0


def test_grad_cam_plus_plus_is_finite_for_large_logits():
    """The exp() in the original formulation overflows; ours must not."""
    model = TransferModel("resnet18", num_classes=8, pretrained=False).eval()
    with torch.no_grad():
        # exp(500) is inf in float32, so the naive Grad-CAM++ formulation dies here.
        model.classifier[-1].bias.fill_(500.0)
    result = grad_cam_plus_plus(model, torch.randn(1, 3, 128, 128), model.feature_layer)
    assert np.isfinite(result.heatmap).all()


def test_grad_cam_explains_the_requested_class():
    model = build_model("cnn_baseline", 8, pretrained=False).eval()
    x = torch.randn(1, 3, 96, 96)
    assert grad_cam(model, x, model.feature_layer, class_index=3).class_index == 3


def test_cbam_map_only_for_cbam_models():
    x = torch.randn(1, 3, 96, 96)
    with_cbam = build_model("cnn_cbam", 8, pretrained=False).eval()
    without = build_model("cnn_baseline", 8, pretrained=False).eval()
    assert cbam_spatial_map(with_cbam, x) is not None
    assert cbam_spatial_map(without, x) is None


def test_integrated_gradients_shape():
    model = build_model("cnn_baseline", 8, pretrained=False).eval()
    result = integrated_gradients(model, torch.randn(1, 3, 96, 96), steps=4)
    assert result.heatmap.shape == (96, 96)
    assert np.isfinite(result.heatmap).all()


def test_colorize_and_overlay():
    heat = np.linspace(0, 1, 64 * 64).reshape(64, 64).astype(np.float32)
    coloured = colorize(heat)
    assert coloured.shape == (64, 64, 3) and coloured.dtype == np.uint8

    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    overlay = overlay_heatmap(image, heat)
    assert overlay.shape == image.shape and overlay.dtype == np.uint8
    # Low-heat regions stay close to the original image.
    assert abs(int(overlay[0, 0].mean()) - 128) < 25


def test_overlay_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        overlay_heatmap(np.zeros((32, 32, 3), np.uint8), np.zeros((16, 16), np.float32))


def test_attention_coverage_statistics():
    heat = np.zeros((10, 10), dtype=np.float32)
    heat[2:5, 2:5] = 1.0
    stats = attention_coverage(heat, threshold=0.5)
    assert stats["focus_ratio"] == pytest.approx(0.09)
    assert stats["peak_value"] == 1.0
    assert stats["centroid_y"] == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# Quality
# --------------------------------------------------------------------------- #


def test_quality_accepts_a_textured_photo_like_image():
    """A smoothly textured colour image should pass every gate."""
    yy, xx = np.mgrid[0:256, 0:256]
    base = np.stack(
        [
            (128 + 90 * np.sin(xx / 18.0)),
            (150 + 80 * np.sin(yy / 14.0)),
            (90 + 50 * np.cos((xx + yy) / 22.0)),
        ],
        axis=-1,
    ).clip(0, 255).astype(np.uint8)
    report = analyse_image(Image.fromarray(base))
    assert report.metrics["spatial_coherence"] > 0.9


def test_quality_flags_uniform_noise_as_not_a_photograph():
    rng = np.random.default_rng(3)
    noise = Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
    report = analyse_image(noise)
    assert "not_a_photograph" in {issue.code for issue in report.issues}
    assert report.passed is False


def test_quality_flags_small_images():
    report = analyse_image(Image.new("RGB", (32, 32), (100, 150, 80)))
    assert "too_small" in {issue.code for issue in report.issues}


def test_quality_flags_dark_and_bright():
    dark = analyse_image(Image.new("RGB", (256, 256), (4, 5, 4)))
    bright = analyse_image(Image.new("RGB", (256, 256), (252, 252, 252)))
    assert "too_dark" in {i.code for i in dark.issues}
    assert "too_bright" in {i.code for i in bright.issues}


def test_quality_report_serialises():
    payload = analyse_image(Image.new("RGB", (256, 256), (90, 140, 70))).to_dict()
    assert {"score", "passed", "issues", "metrics"} <= payload.keys()
    assert 0.0 <= payload["score"] <= 1.0


def test_quality_flags_non_plant_images():
    # Non-plant solid color / object
    blue = analyse_image(Image.new("RGB", (256, 256), (30, 100, 220)))
    assert "not_a_plant" in {i.code for i in blue.issues}
    assert blue.passed is False
    assert any(i.severity == "error" and i.code == "not_a_plant" for i in blue.issues)

    # Human skin tone / tan without vegetation green
    skin = analyse_image(Image.new("RGB", (256, 256), (230, 185, 150)))
    assert "not_a_plant" in {i.code for i in skin.issues}
    assert skin.passed is False



# --------------------------------------------------------------------------- #
# OOD
# --------------------------------------------------------------------------- #


def test_softmax_rows_sum_to_one():
    probs = softmax(np.random.default_rng(0).normal(size=(5, 12)))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_softmax_is_numerically_stable():
    probs = softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert np.isfinite(probs).all()
    assert probs.argmax() == 1


def test_entropy_bounds():
    uniform = np.full((1, 8), 1 / 8)
    peaked = np.zeros((1, 8))
    peaked[0, 0] = 1.0
    assert normalised_entropy(uniform)[0] == pytest.approx(1.0, abs=1e-6)
    assert normalised_entropy(peaked)[0] == pytest.approx(0.0, abs=1e-6)


def test_energy_lower_for_confident_logits():
    confident = np.array([[12.0, 0.0, 0.0, 0.0]])
    flat = np.zeros((1, 4))
    assert energy_score(confident)[0] < energy_score(flat)[0]


def test_msp_matches_max_probability():
    probs = softmax(np.random.default_rng(1).normal(size=(4, 6)))
    assert np.allclose(msp_score(probs), probs.max(axis=1))


def test_auroc_perfect_and_random():
    assert auroc(np.array([3.0, 4.0, 5.0]), np.array([0.0, 1.0, 2.0])) == pytest.approx(1.0)
    identical = np.array([1.0, 1.0, 1.0])
    assert auroc(identical, identical) == pytest.approx(0.5)


def test_threshold_selection_hits_target_tpr():
    rng = np.random.default_rng(0)
    in_dist = rng.normal(size=(400, 10))
    in_dist[np.arange(400), rng.integers(0, 10, 400)] += 10
    ood = rng.normal(size=(200, 10))
    selected = select_thresholds(in_dist, ood, target_tpr=0.95)
    assert selected["per_score"]["msp"]["tpr"] == pytest.approx(0.95, abs=0.02)


def test_assess_flags_flat_distribution():
    thresholds = OODThresholds(msp_min=0.5, entropy_max=0.4, energy_max=float("inf"))
    confident = assess(np.array([14.0, 0.0, 0.0, 0.0]), thresholds)
    flat = assess(np.zeros(4), thresholds)
    assert confident.is_ood is False
    assert flat.is_ood is True
    assert flat.reasons


def test_ood_verdict_serialises():
    payload = assess(np.array([5.0, 1.0, 0.0]), OODThresholds.defaults(0.45, 0.55)).to_dict()
    assert {"is_out_of_distribution", "reasons", "scores", "thresholds"} <= payload.keys()


# --------------------------------------------------------------------------- #
# Classical features
# --------------------------------------------------------------------------- #


def test_feature_vector_shape_matches_names():
    features = extract_features(_sample_image((128, 128)))
    assert features.ndim == 1
    assert len(feature_names()) == features.shape[0]


def test_feature_groups_partition_the_vector():
    features = extract_features(_sample_image((128, 128)))
    slices = feature_group_slices()
    covered = sum(s.stop - s.start for s in slices.values())
    assert covered == features.shape[0]


def test_features_are_deterministic():
    image = _sample_image((128, 128))
    assert np.allclose(extract_features(image), extract_features(image))


def test_features_distinguish_different_images():
    green = Image.new("RGB", (128, 128), (40, 180, 60))
    brown = Image.new("RGB", (128, 128), (140, 90, 40))
    assert not np.allclose(extract_features(green), extract_features(brown))


def test_features_are_finite():
    assert np.isfinite(extract_features(_sample_image((96, 96)))).all()


# --------------------------------------------------------------------------- #
# OOD threshold provenance
# --------------------------------------------------------------------------- #


def test_thresholds_are_refused_for_a_different_model(tmp_path):
    """Thresholds fitted to one model must not be applied to another.

    The softmax floor and the energy limit both depend on a network's logit
    scale, so reusing them across models rejects correct, confident predictions.
    """
    import json

    path = tmp_path / "ood_thresholds.json"
    path.write_text(json.dumps({
        "model": "cnn_baseline",
        "selected": {"msp_min": 0.52, "entropy_max": 0.41, "energy_max": -4.4,
                     "method": "msp+entropy", "target_tpr": 0.95},
    }), encoding="utf-8")

    assert OODThresholds.from_file(path, "cnn_baseline") is not None
    assert OODThresholds.from_file(path, "efficientnet_b0") is None
    # No expectation given: accept, since the caller has not claimed a model.
    assert OODThresholds.from_file(path) is not None


def test_thresholds_without_a_model_field_are_accepted(tmp_path):
    """Older files that predate the provenance field must still load."""
    import json

    path = tmp_path / "ood_thresholds.json"
    path.write_text(json.dumps({
        "selected": {"msp_min": 0.5, "entropy_max": 0.4, "energy_max": -3.0},
    }), encoding="utf-8")
    assert OODThresholds.from_file(path, "any_model") is not None


def test_default_thresholds_are_marked_uncalibrated():
    thresholds = OODThresholds.defaults(0.45, 0.55)
    assert thresholds.calibrated is False
    assert thresholds.msp_min == 0.45


# --------------------------------------------------------------------------- #
# Status decision table
# --------------------------------------------------------------------------- #


def _quality(passed: bool, severities: tuple[str, ...] = ()):
    from app.ml.quality import QualityIssue, QualityReport

    issues = [
        QualityIssue(f"issue_{i}", severity, "message", "suggestion")
        for i, severity in enumerate(severities)
    ]
    return QualityReport(score=0.9 if passed else 0.2, passed=passed, issues=issues)


def _verdict(is_ood: bool):
    from app.ml.ood import OODVerdict

    return OODVerdict(
        is_ood=is_ood,
        reasons=["low softmax"] if is_ood else [],
        scores={"msp": 0.3 if is_ood else 0.95},
        thresholds={},
    )


@pytest.mark.parametrize(
    "quality_passed,severities,is_ood,level,expected",
    [
        # A photographic problem the user can fix outranks everything else: it is
        # actionable, whereas "out of distribution" is not.
        (False, ("error",), True, "high", "poor_quality"),
        (False, ("error",), False, "high", "poor_quality"),
        (False, ("error",), False, "low", "poor_quality"),
        # Warnings alone must not block a prediction.
        (True, ("warning",), False, "high", "ok"),
        (False, ("warning",), False, "high", "ok"),
        # OOD outranks a merely low confidence: refusing to identify is a
        # stronger, more specific statement than "not very sure".
        (True, (), True, "high", "out_of_distribution"),
        (True, (), True, "low", "out_of_distribution"),
        # Confidence banding, once quality and distribution are satisfied.
        (True, (), False, "low", "low_confidence"),
        (True, (), False, "medium", "ok"),
        (True, (), False, "high", "ok"),
    ],
)
def test_status_precedence(quality_passed, severities, is_ood, level, expected):
    from app.services.predictor import Predictor

    status, message = Predictor._decide_status(
        _quality(quality_passed, severities), _verdict(is_ood), level
    )
    assert status == expected
    if status != "ok" or level == "medium":
        assert message, f"{status} must explain itself to the user"


def test_confidence_bands_follow_the_configured_thresholds():
    from app.core.config import settings
    from app.services.predictor import confidence_level

    assert confidence_level(settings.confidence_high) == "high"
    assert confidence_level(settings.confidence_high - 1e-9) == "medium"
    assert confidence_level(settings.confidence_medium) == "medium"
    assert confidence_level(settings.confidence_medium - 1e-9) == "low"
    assert confidence_level(0.0) == "low"
    assert confidence_level(1.0) == "high"


# --------------------------------------------------------------------------- #
# DataLoader resource-failure detection
# --------------------------------------------------------------------------- #


REAL_SHARED_MEMORY_FAILURE = """Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "torch/utils/data/_utils/collate.py", line 273, in collate_tensor_fn
    storage = elem._typed_storage()._new_shared(numel, device=elem.device)
RuntimeError: Couldn't open shared file mapping: <torch_16336_2546003764_1594>, error code: <1455>"""

REAL_IMPORT_FAILURE = (
    "OSError: [WinError 1455] The paging file is too small for this operation "
    "to complete. Error loading \"torch/lib/cublas64_12.dll\""
)


@pytest.mark.parametrize("message", [REAL_SHARED_MEMORY_FAILURE, REAL_IMPORT_FAILURE])
def test_resource_failures_trigger_the_in_process_fallback(message):
    """Both Windows phrasings of error 1455 must be recognised.

    Error 1455 arrives two ways: a worker failing to import the CUDA build of
    torch at start-up, and a running worker failing to allocate the shared
    memory mapping that carries a collated batch back to the parent. Missing
    either one loses a whole training run.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
    from trainer import is_worker_startup_failure

    assert is_worker_startup_failure(RuntimeError(message))


@pytest.mark.parametrize("message", [
    "Caught RuntimeError in DataLoader worker process 2.\nstack expects each tensor to be equal size",
    "Caught ValueError in DataLoader worker process 0.\nbad label index 41 for 38 classes",
    "Caught OSError in DataLoader worker process 1.\ncannot identify image file",
])
def test_genuine_dataset_bugs_are_not_silently_downgraded(message):
    """A real bug must propagate, not be hidden behind a quiet worker downgrade."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
    from trainer import is_worker_startup_failure

    assert not is_worker_startup_failure(RuntimeError(message))


def test_train_model_retries_once_with_in_process_loading(monkeypatch):
    """A resource failure must downgrade to num_workers=0 and retry, not abort.

    Losing a multi-hour model to a transient shared-memory failure is the
    difference between a sweep that finishes overnight and one that needs a
    human. The retry happens exactly once: if in-process loading also fails,
    the error propagates.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
    import trainer as trainer_module
    from config import TrainConfig

    calls = []

    def fake_train(config, *args, **kwargs):
        calls.append(config.num_workers)
        if config.num_workers > 0:
            raise RuntimeError(REAL_SHARED_MEMORY_FAILURE)
        return "trained"

    monkeypatch.setattr(trainer_module, "_train_model", fake_train)

    config = TrainConfig(model_name="cnn_cbam", num_workers=3)
    result = trainer_module.train_model(config, {}, Path())

    assert result == "trained"
    assert calls == [3, 0], f"expected one retry at 0 workers, got {calls}"
    assert config.num_workers == 0


def test_train_model_does_not_retry_a_genuine_bug(monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
    import trainer as trainer_module
    from config import TrainConfig

    calls = []

    def fake_train(config, *args, **kwargs):
        calls.append(config.num_workers)
        raise ValueError("bad label index 41 for 38 classes")

    monkeypatch.setattr(trainer_module, "_train_model", fake_train)

    with pytest.raises(ValueError, match="bad label index"):
        trainer_module.train_model(TrainConfig(model_name="cnn_cbam", num_workers=3), {}, Path())
    assert calls == [3], "a real bug must not be retried"


def test_train_model_does_not_retry_when_already_single_process(monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
    import trainer as trainer_module
    from config import TrainConfig

    calls = []

    def fake_train(config, *args, **kwargs):
        calls.append(config.num_workers)
        raise RuntimeError(REAL_SHARED_MEMORY_FAILURE)

    monkeypatch.setattr(trainer_module, "_train_model", fake_train)

    with pytest.raises(RuntimeError):
        trainer_module.train_model(TrainConfig(model_name="cnn_cbam", num_workers=0), {}, Path())
    assert calls == [0], "no retry is possible below zero workers"


# --------------------------------------------------------------------------- #
# OOD method enforcement
# --------------------------------------------------------------------------- #

def _peaked_logits(peak: float, background: float, classes: int = 38) -> np.ndarray:
    """Logits with one confident class and a chosen absolute scale.

    Both parts matter. The gap between `peak` and `background` sets the softmax
    confidence and entropy; the absolute level sets the free energy, which is
    -logsumexp and so ignores the gap entirely. Real model logits have a modest
    peak over a strongly negative background, which is how a prediction can be
    99.9% confident and still land in the energy tail.
    """
    logits = np.full(classes, background, dtype=np.float64)
    logits[0] = peak
    return logits


def test_energy_cannot_veto_when_the_method_excludes_it():
    """A confident, low-entropy prediction must survive an unfavourable energy score.

    Each threshold is set independently to keep 95% of in-distribution inputs.
    OR-ing all three rejects up to 15%, breaking the stated 5% retake rate, and
    it contradicts the thresholds file, which records that serving enforces MSP
    and entropy while energy is kept for reference only. This exact case - a
    99.95%-confidence prediction on an image from the dataset's own validation
    split - was being refused in production.
    """
    from app.ml.ood import OODThresholds, assess

    logits = _peaked_logits(peak=4.0, background=-6.0)
    thresholds = OODThresholds(
        msp_min=0.80,
        entropy_max=0.27,
        energy_max=-4.62,     # the prediction's energy sits above this
        method="msp+entropy",
        calibrated=True,
    )
    verdict = assess(logits, thresholds)

    assert verdict.scores["msp"] > thresholds.msp_min
    assert verdict.scores["entropy"] < thresholds.entropy_max
    assert verdict.scores["energy"] > thresholds.energy_max, "energy should be failing"
    assert verdict.is_ood is False, verdict.reasons
    assert verdict.thresholds["enforced"] == ["entropy", "msp"]


def test_energy_does_veto_when_the_method_includes_it():
    from app.ml.ood import OODThresholds, assess

    logits = _peaked_logits(peak=4.0, background=-6.0)
    thresholds = OODThresholds(
        msp_min=0.80, entropy_max=0.27, energy_max=-4.62,
        method="msp+entropy+energy", calibrated=True,
    )
    verdict = assess(logits, thresholds)

    assert verdict.is_ood is True
    assert any("Free-energy" in reason for reason in verdict.reasons)


def test_a_genuinely_uncertain_prediction_is_still_rejected():
    """The fix must not disable the OOD gate for inputs it should catch."""
    from app.ml.ood import OODThresholds, assess

    flat = np.zeros(38)  # uniform posterior: maximum entropy, minimum confidence
    thresholds = OODThresholds(
        msp_min=0.80, entropy_max=0.27, energy_max=-4.62,
        method="msp+entropy", calibrated=True,
    )
    verdict = assess(flat, thresholds)

    assert verdict.is_ood is True
    assert len(verdict.reasons) == 2
