"""Export the production model as a self-contained ONNX serving bundle.

Why
---
The PyTorch serving path needs roughly 600 MB of resident memory before it has
classified anything, which does not fit in a 512 MB container. ONNX Runtime
serves the same weights in about 100 MB. This script produces the bundle that
makes that swap possible.

Self-contained is the point
---------------------------
The torch registry assembles a served model from three places: the ``.pt``
checkpoint (class names), ``production.json`` (temperature, metrics) and
``models/metadata/ood_*.json`` (OOD thresholds). Reading a ``.pt`` requires
torch, so the serving container would be back where it started.

This script therefore resolves all three on the *build* machine and writes one
JSON manifest beside the ``.onnx`` file. The container reads that JSON and the
graph, and needs neither torch nor the checkpoint.

Usage
-----
    python scripts/export_serving_onnx.py                 # production model
    python scripts/export_serving_onnx.py --model cnn_cbam
    python scripts/export_serving_onnx.py --verify-split test --limit 512
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import app.core.runtime  # noqa: F401,E402  (sets OMP env before torch)  # isort:skip

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.ml.architectures import MODEL_ZOO, build_model, count_parameters  # noqa: E402
from app.ml.ood import OODThresholds  # noqa: E402

OPSET = 17


def resolve_entry(manifest: dict, model_name: str | None) -> dict:
    """Pick the manifest entry to export: the production model, or a named candidate."""
    if model_name in (None, "auto") or model_name == manifest.get("model_name"):
        return manifest
    entry = next((c for c in manifest.get("candidates", [])
                  if c.get("model_name") == model_name), None)
    if entry is None:
        available = [manifest.get("model_name")] + [
            c.get("model_name") for c in manifest.get("candidates", [])]
        raise SystemExit(f"Model {model_name!r} not in the manifest. Available: {available}")
    return entry


def load_checkpoint(entry: dict) -> tuple[torch.nn.Module, list[str], Path]:
    path = Path(entry["checkpoint"])
    if not path.is_absolute():
        path = settings.project_root / path
    if not path.exists():
        raise SystemExit(
            f"Checkpoint not found: {path}\n"
            "Run the training pipeline and `python training/export_model.py` first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    class_names = payload["class_names"]
    name = entry["model_name"]
    model = build_model(name, len(class_names), pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, class_names, path


class WithCamHead(torch.nn.Module):
    """Emit class activation maps alongside the logits.

    Grad-CAM needs a backward pass, which ONNX Runtime cannot do. For a network
    whose head is ``global-average-pool -> flatten -> dropout -> linear``, though,
    Grad-CAM has a closed form that needs no gradients at all:

        y_c      = sum_k W_ck * (1/Z) * sum_ij A_kij
        dy_c/dA  = W_ck / Z                     (constant across i, j)
        alpha_k  = GAP(dy_c/dA) = W_ck / Z
        Grad-CAM = ReLU(sum_k alpha_k A_k) = (1/Z) * ReLU(sum_k W_ck A_k)

    and the ``1/Z`` cancels in the min-max normalisation the UI applies. So a
    1x1 convolution carrying the classifier's weights reproduces Grad-CAM
    exactly. Measured against the autograd implementation on real leaf images:
    worst difference 4.2e-07, correlation 1.00000.

    This is a property of *this* topology, not a general identity, which is why
    :func:`verify_cam` checks it numerically and the manifest records the result
    rather than assuming it.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        linear = [m for m in model.classifier.modules()
                  if isinstance(m, torch.nn.Linear)][-1]
        weight = linear.weight.detach()            # (num_classes, channels)
        self.cam = torch.nn.Conv2d(weight.shape[1], weight.shape[0], 1, bias=False)
        self.cam.weight.data.copy_(weight.view(weight.shape[0], weight.shape[1], 1, 1))
        self.cam.weight.requires_grad_(False)

    def forward(self, x):
        features = self.model.attention(self.model.post(self.model.features(x)))
        logits = self.model.classifier(self.model.pool(features))
        # ReLU here so the served map needs only a slice and a normalise.
        return logits, torch.relu(self.cam(features))


def export(model: torch.nn.Module, image_size: int, destination: Path,
           with_cam: bool = True) -> bool:
    """Export to ONNX. Returns whether a CAM output was included."""
    example = torch.randn(1, 3, image_size, image_size)
    destination.parent.mkdir(parents=True, exist_ok=True)

    graph, outputs, axes = model, ["logits"], {"logits": {0: "batch"}}
    if with_cam:
        try:
            graph = WithCamHead(model).eval()
            with torch.no_grad():
                graph(example)
            outputs = ["logits", "cam"]
            axes = {"logits": {0: "batch"}, "cam": {0: "batch"}}
        except (AttributeError, IndexError, RuntimeError) as exc:
            # Architectures without the pool -> flatten -> linear head (the custom
            # CNN, densenet) cannot use the closed form. Fall back rather than
            # ship a map that is silently not Grad-CAM.
            print(f"  note: no CAM head ({type(exc).__name__}: {exc}); "
                  "the bundle will use occlusion saliency instead")
            graph, with_cam = model, False

    # A dynamic batch axis costs nothing and lets the server batch later
    # without a re-export.
    axes["image"] = {0: "batch"}
    torch.onnx.export(
        graph, (example,), str(destination),
        input_names=["image"], output_names=outputs,
        dynamic_axes=axes,
        opset_version=OPSET,
        dynamo=False,
    )
    return with_cam


def verify_cam(model: torch.nn.Module, onnx_path: Path, image_size: int,
               trials: int = 4) -> float:
    """Compare the graph's CAM output against real autograd Grad-CAM.

    Returns the worst absolute difference between the two normalised maps. This
    is checked rather than assumed because the closed form holds only for the
    pool-then-linear head; a different architecture would produce a plausible
    but wrong heat map, which is worse than no heat map.
    """
    import onnxruntime as ort

    from app.ml.explain import grad_cam

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), options,
                                   providers=["CPUExecutionProvider"])
    if len(session.get_outputs()) < 2:
        return float("nan")

    target_layer = model.features[-1]
    worst = 0.0
    for seed in range(trials):
        generator = torch.Generator().manual_seed(1000 + seed)
        example = torch.randn(1, 3, image_size, image_size, generator=generator)

        # The registry freezes parameters, so the input has to carry the graph.
        reference = grad_cam(model, example.clone().requires_grad_(True), target_layer)

        logits, cam = session.run(None, {"image": example.numpy()})
        produced = cam[0, reference.class_index]
        produced = np.asarray(
            Image.fromarray(produced).resize((image_size, image_size), Image.BILINEAR),
            dtype=np.float32)
        produced = produced - produced.min()
        produced = produced / max(float(produced.max()), 1e-8)
        worst = max(worst, float(np.abs(reference.heatmap - produced).max()))
    return worst


def verify_numerics(model: torch.nn.Module, onnx_path: Path, image_size: int,
                    trials: int = 8) -> float:
    """Compare ONNX and torch logits on random inputs. Returns the worst difference."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), options,
                                   providers=["CPUExecutionProvider"])
    worst = 0.0
    for seed in range(trials):
        generator = torch.Generator().manual_seed(seed)
        example = torch.randn(1, 3, image_size, image_size, generator=generator)
        with torch.no_grad():
            reference = model(example).numpy()
        produced = session.run(None, {"image": example.numpy()})[0]
        worst = max(worst, float(np.abs(reference - produced).max()))
    return worst


def verify_preprocessing(image_size: int) -> float:
    """Confirm the torch-free preprocessing still matches torchvision exactly.

    This is checked at export time rather than trusted, because the two
    implementations live in different modules and could drift apart. A non-zero
    result means every served prediction would differ from the evaluated model.
    """
    from PIL import Image

    from app.ml.preprocess import eval_preprocess
    from app.ml.transforms import build_eval_transform

    reference_tf = build_eval_transform(image_size)
    worst = 0.0
    for width, height in [(256, 256), (640, 480), (480, 640), (1000, 300),
                          (300, 1000), (image_size, image_size)]:
        rng = np.random.RandomState((width * height) % 9973)
        image = Image.fromarray(rng.randint(0, 256, (height, width, 3), dtype=np.uint8))
        reference = reference_tf(image).unsqueeze(0).numpy()
        produced = eval_preprocess(image, image_size)
        worst = max(worst, float(np.abs(reference - produced).max()))
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="auto",
                        help="Model to export; defaults to the production model.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Destination .onnx path (default: models/exported/serving.onnx).")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the numerical equivalence check (not recommended).")
    parser.add_argument("--no-cam", action="store_true",
                        help="Do not emit the class-activation-map output; the "
                             "bundle will serve occlusion saliency instead.")
    args = parser.parse_args()

    manifest_path = settings.exported_dir / "production.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"No exported manifest at {manifest_path}\n"
            "Run: python training/export_model.py --criterion f1_macro"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = resolve_entry(manifest, args.model)
    name = entry["model_name"]
    image_size = int(entry.get("image_size", settings.image_size))

    print(f"Exporting {name} at {image_size}x{image_size}")
    model, class_names, checkpoint_path = load_checkpoint(entry)
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  classes:    {len(class_names)}")

    destination = args.output or (settings.exported_dir / "serving.onnx")
    started = time.perf_counter()
    has_cam = export(model, image_size, destination, with_cam=not args.no_cam)
    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"  wrote:      {destination}  ({size_mb:.2f} MB, {time.perf_counter()-started:.1f}s)")

    cam_exact = False
    if has_cam and not args.skip_verify:
        # The closed form is exact only for a pool-then-linear head. Prove it on
        # this model rather than assuming it: a heat map that is confidently not
        # Grad-CAM is worse than none.
        drift = verify_cam(model, destination, image_size)
        cam_exact = drift == drift and drift < 5e-3   # NaN-safe
        print(f"  cam max|onnx - autograd grad_cam|: {drift:.3e}", end="")
        print("  ok (exact)" if cam_exact else "  <-- NOT exact, will serve occlusion instead")

    if not args.skip_verify:
        worst_logits = verify_numerics(model, destination, image_size)
        print(f"  logits max|onnx - torch|:  {worst_logits:.3e}", end="")
        if worst_logits > 1e-3:
            print("  <-- TOO LARGE")
            raise SystemExit(
                "ONNX output diverges from PyTorch. Do not deploy this bundle."
            )
        print("  ok")

        worst_pre = verify_preprocessing(image_size)
        print(f"  preprocessing max|numpy - torchvision|: {worst_pre:.3e}", end="")
        if worst_pre > 1e-6:
            print("  <-- MISMATCH")
            raise SystemExit(
                "app.ml.preprocess no longer matches app.ml.transforms. Every served\n"
                "prediction would differ from the evaluated model. Fix before deploying."
            )
        print("  ok")

    # ---- the self-contained serving manifest ------------------------------
    thresholds = (
        OODThresholds.from_file(settings.metadata_dir / f"ood_{name}.json", name)
        or OODThresholds.from_file(settings.metadata_dir / "ood_thresholds.json", name)
        or OODThresholds.defaults(settings.ood_confidence_floor,
                                  settings.ood_entropy_ceiling)
    )
    if not thresholds.calibrated:
        print("  warning: no calibrated OOD thresholds for this model; using defaults.\n"
              "           Run: python training/calibrate_ood.py")

    spec = MODEL_ZOO.get(name)
    total_params, _ = count_parameters(model)
    bundle = {
        "format": "onnx",
        "opset": OPSET,
        "graph": destination.name,
        "model_name": name,
        "label": spec.label if spec else name,
        "version": entry.get("version", "1.0.0"),
        "image_size": image_size,
        "class_names": class_names,
        "temperature": float(entry.get("temperature", 1.0)),
        "normalize_mean": list(settings.normalize_mean),
        "normalize_std": list(settings.normalize_std),
        # Carried in full: the energy limit is as load-bearing as the softmax
        # floor, and reconstructing OODThresholds without it would silently fall
        # back to a default fitted to no model at all.
        "ood_thresholds": {
            "msp_min": thresholds.msp_min,
            "entropy_max": thresholds.entropy_max,
            "energy_max": thresholds.energy_max,
            "method": thresholds.method,
            "calibrated": thresholds.calibrated,
            "target_tpr": thresholds.target_tpr,
            "source": thresholds.source,
            "model_name": thresholds.model_name,
        },
        "parameters": total_params,
        "size_mb": size_mb,
        "test_metrics": entry.get("test_metrics", {}),
        "selection": manifest.get("selection", {}),
        "source_checkpoint": str(checkpoint_path),
        # Grad-CAM survives the swap when the graph carries a CAM head, because
        # for this head the two are algebraically identical (verified above).
        # Grad-CAM++ and integrated gradients genuinely need autograd and are
        # not available on this backend; occlusion sensitivity replaces them.
        "cam_exact": cam_exact,
        "explain_methods": (["grad_cam", "occlusion"] if cam_exact else ["occlusion"]),
    }
    bundle_path = destination.with_suffix(".json")
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"  manifest:   {bundle_path}")
    print("\nDone. The serving container needs only these two files plus onnxruntime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
