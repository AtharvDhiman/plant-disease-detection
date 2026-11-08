"""The averaged ensemble: its output, its checkpoint round trip, and its ONNX export."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import app.core.runtime  # noqa: F401,E402  # isort:skip
from app.ml.architectures import MODEL_ZOO, AveragedEnsemble, build_model  # noqa: E402


def test_softmax_of_the_output_is_the_mean_of_the_members_calibrated_probabilities():
    torch.manual_seed(0)
    members = [build_model("cnn_baseline", 5).eval(), build_model("cnn_se", 5).eval()]
    ensemble = AveragedEnsemble(members, [0.5, 2.0]).eval()
    x = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        expected = (torch.softmax(members[0](x) / 0.5, 1) + torch.softmax(members[1](x) / 2.0, 1)) / 2
        produced = torch.softmax(ensemble(x), 1)
    assert torch.allclose(produced, expected, atol=1e-6)
    assert ensemble.feature_layer is members[0].feature_layer
    assert ensemble.explain_member is members[0]


def test_temperature_count_must_match_the_members():
    with pytest.raises(ValueError):
        AveragedEnsemble([build_model("cnn_baseline", 5)], [1.0, 2.0])


def test_zoo_entry_builds_its_members_and_the_temperatures_travel_with_the_weights():
    spec = MODEL_ZOO["ensemble_cbam"]
    assert spec.group == "ensemble"           # never swept by a benchmark suite
    model = build_model("ensemble_cbam", 7, pretrained=False)
    assert [type(m).__name__ for m in model.members] == ["TransferModel", "TransferModel"]
    model.temperatures.copy_(torch.tensor([0.9, 1.3]))
    rebuilt = build_model("ensemble_cbam", 7, pretrained=False)
    rebuilt.load_state_dict(model.state_dict())
    assert rebuilt.temperatures.tolist() == pytest.approx([0.9, 1.3])


def test_onnx_export_matches_torch_and_its_heat_map_is_the_first_members(tmp_path):
    """The graph's average matches torch, and its CAM output is exactly the first
    member's CAM head - the head export_serving_onnx.verify_cam proves equal to
    autograd Grad-CAM on the *trained* network at export time. (That proof cannot
    run here: with untrained weights the maps are ~1e-14, i.e. numerically zero,
    and normalising them is noise for a single network just as for an ensemble.)"""
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort
    from export_serving_onnx import WithCamHead, export, verify_numerics

    torch.manual_seed(0)
    members = [build_model("efficientnet_b0_cbam", 4, pretrained=False).eval(),
               build_model("mobilenet_v3_large", 4, pretrained=False).eval()]
    for member in members:
        for parameter in member.parameters():
            parameter.requires_grad_(False)
    ensemble = AveragedEnsemble(members, [1.1, 0.8]).eval()
    path = tmp_path / "ensemble.onnx"

    assert export(ensemble, 96, path, with_cam=True)
    assert verify_numerics(ensemble, path, 96, trials=2) < 1e-3

    x = torch.randn(1, 3, 96, 96, generator=torch.Generator().manual_seed(3))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    logits, cam = session.run(None, {"image": x.numpy()})
    with torch.no_grad():
        _, member_cam = WithCamHead(members[0]).eval()(x)
    member_cam = member_cam.numpy()
    assert cam.shape == member_cam.shape and cam.shape[:2] == (1, 4)
    assert np.allclose(cam, member_cam, rtol=1e-3, atol=1e-3 * float(np.abs(member_cam).max()))
    assert np.allclose(np.exp(logits).sum(axis=1), 1.0, atol=1e-5)   # log-probabilities
