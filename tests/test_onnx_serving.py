"""Guards on the ONNX serving backend.

The backend exists so the app fits a 512 MB host: the PyTorch path needs ~657 MB
resident and is OOM-killed there, the ONNX path needs ~145 MB. Swapping the
runtime under a trained model has one genuinely dangerous failure mode, and it
is not a crash - it is silently changing what the model predicts. A crash is
obvious. A preprocessing pipeline that drifts by half a pixel is not, and it
would degrade accuracy in a way no deployment check would catch.

So the load-bearing test here is :func:`test_preprocessing_matches_torchvision`.
The rest are cheaper structural checks that the slim image stays slim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

BUNDLE = PROJECT_ROOT / "models" / "exported" / "serving.json"
GRAPH = PROJECT_ROOT / "models" / "exported" / "serving.onnx"

# Shapes chosen to exercise the two things a naive reimplementation gets wrong:
# non-square inputs (the short side is what gets resized) and an odd size
# difference at the centre crop (which rounds rather than floors).
SHAPES = [(256, 256), (640, 480), (480, 640), (1000, 300), (300, 1000), (225, 224)]


def _image(width: int, height: int) -> Image.Image:
    rng = np.random.RandomState((width * height) % 9973)
    return Image.fromarray(rng.randint(0, 256, (height, width, 3), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# The one that matters
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("width,height", SHAPES)
def test_preprocessing_matches_torchvision(width, height):
    """app.ml.preprocess must be bit-identical to the torchvision eval transform.

    Both backends feed their model from ``eval_preprocess``, so any drift here
    changes predictions on *both* paths, not just the ONNX one.
    """
    torch = pytest.importorskip("torch")

    from app.ml.preprocess import eval_preprocess
    from app.ml.transforms import build_eval_transform

    image = _image(width, height)
    reference = build_eval_transform(224)(image).unsqueeze(0).numpy()
    produced = eval_preprocess(image, 224)

    assert produced.shape == reference.shape
    assert produced.dtype == np.float32
    # Exactly zero, not "close": torchvision delegates to Pillow for PIL inputs
    # and preprocess.py makes the same Pillow calls, so any difference at all
    # means the pipelines have genuinely diverged.
    assert np.abs(produced - reference).max() == 0.0


def test_eval_crop_matches_resized_rgb():
    """Heat maps are drawn over this crop; a mismatch misaligns every overlay."""
    pytest.importorskip("torch")

    from app.ml.preprocess import eval_crop
    from app.ml.transforms import resized_rgb

    for width, height in SHAPES:
        image = _image(width, height)
        assert np.array_equal(np.asarray(eval_crop(image, 224)),
                              np.asarray(resized_rgb(image, 224)))


def test_softmax_matches_torch_with_temperature():
    """Reported confidence is temperature-scaled; the numpy version must agree."""
    torch = pytest.importorskip("torch")

    from app.ml.preprocess import softmax

    rng = np.random.RandomState(0)
    logits = rng.randn(1, 38).astype(np.float32) * 8.0
    for temperature in (1.0, 0.5072429180145264, 2.5):
        expected = torch.softmax(torch.from_numpy(logits) / temperature, dim=1).numpy()
        assert np.abs(softmax(logits, temperature) - expected).max() < 1e-6


# --------------------------------------------------------------------------- #
# The exported bundle, when one has been built
# --------------------------------------------------------------------------- #


needs_bundle = pytest.mark.skipif(
    not (BUNDLE.exists() and GRAPH.exists()),
    reason="no ONNX bundle; run scripts/export_serving_onnx.py",
)


@needs_bundle
def test_bundle_is_self_contained():
    """The container reads only these two files, so everything must be in them."""
    manifest = json.loads(BUNDLE.read_text(encoding="utf-8"))
    for key in ("model_name", "class_names", "image_size", "temperature",
                "ood_thresholds", "graph", "explain_methods"):
        assert key in manifest, f"serving.json is missing {key!r}"
    assert len(manifest["class_names"]) == len(set(manifest["class_names"]))
    assert manifest["ood_thresholds"].get("energy_max") is not None, (
        "energy_max is required to rebuild OODThresholds; without it the served "
        "OOD decision silently falls back to uncalibrated defaults"
    )


@needs_bundle
def test_graph_matches_the_manifest():
    """A manifest that disagrees with its graph 500s on the first request."""
    ort = pytest.importorskip("onnxruntime")

    manifest = json.loads(BUNDLE.read_text(encoding="utf-8"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(GRAPH), options, providers=["CPUExecutionProvider"])

    (image_input,) = session.get_inputs()
    assert image_input.name == "image"
    assert image_input.shape[1:] == [3, manifest["image_size"], manifest["image_size"]]

    outputs = {o.name: o for o in session.get_outputs()}
    assert "logits" in outputs
    assert outputs["logits"].shape[-1] == len(manifest["class_names"])
    if manifest.get("cam_exact"):
        assert "cam" in outputs, (
            "manifest claims cam_exact but the graph has no CAM output; "
            "grad_cam would fall back to occlusion at runtime"
        )


@needs_bundle
def test_onnx_predictions_match_pytorch():
    """The whole swap is only safe if the two runtimes agree on the logits."""
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")

    manifest = json.loads(BUNDLE.read_text(encoding="utf-8"))
    checkpoint = Path(manifest["source_checkpoint"])
    if not checkpoint.exists():
        pytest.skip("source checkpoint not present on this machine")

    from app.ml.architectures import build_model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(manifest["model_name"], len(payload["class_names"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(GRAPH), options, providers=["CPUExecutionProvider"])

    size = manifest["image_size"]
    generator = torch.Generator().manual_seed(7)
    example = torch.randn(1, 3, size, size, generator=generator)
    with torch.no_grad():
        expected = model(example).numpy()
    produced = session.run(None, {"image": example.numpy()})[0]

    assert np.abs(expected - produced).max() < 1e-3
    assert int(expected.argmax()) == int(produced.argmax())


# --------------------------------------------------------------------------- #
# Keeping the slim image slim
# --------------------------------------------------------------------------- #


def test_serving_requirements_exclude_the_training_stack():
    """These packages are the footprint. Re-adding one re-creates the OOM."""
    text = (PROJECT_ROOT / "requirements-serve.txt").read_text(encoding="utf-8")
    installed = {line.split("[")[0].split(">")[0].split("=")[0].strip().lower()
                 for line in text.splitlines()
                 if line.strip() and not line.lstrip().startswith("#")}

    for banned in ("torch", "torchvision", "scikit-image", "scikit-learn",
                   "pandas", "matplotlib", "seaborn", "xgboost", "tensorboard",
                   "kagglehub"):
        assert banned not in installed, (
            f"{banned} is back in requirements-serve.txt; the serving image no "
            "longer fits a 512 MB host"
        )
    for required in ("onnxruntime", "numpy", "pillow", "fastapi", "uvicorn"):
        assert required in installed, f"requirements-serve.txt is missing {required}"


def test_render_image_never_installs_torch():
    """The Render image must not acquire torch by any route."""
    text = (PROJECT_ROOT / "docker" / "render.Dockerfile").read_text(encoding="utf-8")
    body = "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#"))
    assert "requirements-serve.txt" in body
    assert "requirements.txt" not in body.replace("requirements-serve.txt", "")
    assert "download.pytorch.org" not in body
    assert "PDD_SERVING_BACKEND=onnx" in body


def test_render_image_ships_the_bundle_not_a_checkpoint():
    """A .pt in the image would need torch to read it, defeating the point."""
    text = (PROJECT_ROOT / "docker" / "render.Dockerfile").read_text(encoding="utf-8")
    body = "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#"))
    assert "serving.onnx" in body and "serving.json" in body
    assert ".pt" not in body


def test_serving_bundle_is_not_gitignored():
    """Render builds from the repo; an ignored graph means an image with no model."""
    text = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "!models/exported/serving.onnx" in text, (
        "serving.onnx is excluded by models/exported/**/*.onnx and needs an "
        "explicit negation, or Render will build an image with no model in it"
    )
