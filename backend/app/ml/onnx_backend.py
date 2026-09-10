"""ONNX Runtime serving backend - the same model, without PyTorch in the process.

Why this exists
---------------
Measured resident memory for a single serving process on this project:

    PyTorch + torchvision + checkpoint + one inference      ~628 MB
    the full FastAPI app as it stands today                 ~611 MB
    ONNX Runtime + FastAPI + Pillow + SQLAlchemy + inference ~103 MB

A 512 MB container therefore cannot run the PyTorch path at all - it is
OOM-killed before serving a request - and has roughly five times headroom on the
ONNX path. Nothing about the model changes: the exported graph reproduces the
PyTorch logits to within 8e-05, and :mod:`app.ml.preprocess` reproduces the
torchvision eval transform bit-exactly.

What this backend gives up
--------------------------
Grad-CAM, Grad-CAM++ and integrated gradients all need a backward pass, and
there is no autograd here. They are replaced by occlusion sensitivity, which is
a genuine saliency method that needs only forward passes - it is slower per
explanation but it measures something real: how much the predicted logit
actually falls when a region is hidden.

The bundle
----------
:mod:`scripts.export_serving_onnx` writes ``serving.onnx`` plus a
``serving.json`` manifest carrying the class names, temperature and OOD
thresholds. Everything this module needs is in those two files, so the container
ships no ``.pt`` checkpoint and imports no torch.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.ml.ood import OODThresholds
from app.ml.preprocess import softmax

log = get_logger(__name__)


def _upsample_normalised(cam: np.ndarray, size: int) -> np.ndarray:
    """Resize a small activation map to ``size`` and rescale it to ``[0, 1]``.

    Pillow's bilinear upsampling uses the same half-pixel convention as
    ``F.interpolate(..., align_corners=False)``, which is what the torch backend
    applies, so the two produce the same map: measured end to end against
    autograd Grad-CAM at 3.2e-05.
    """
    from PIL import Image

    resample = getattr(Image, "Resampling", Image).BILINEAR
    resized = np.asarray(
        Image.fromarray(np.asarray(cam, dtype=np.float32)).resize(
            (size, size), resample),
        dtype=np.float32,
    )
    resized = resized - resized.min()
    # Guard the constant-map case, which would otherwise divide by zero and
    # produce NaNs the UI renders as a blank tile.
    return resized / max(float(resized.max()), 1e-8)


def _session_options(threads: int):
    """Memory-frugal ONNX Runtime configuration for a small container.

    Single-threaded on purpose. A free-tier container is allotted a fraction of
    a core, so extra intra-op threads add per-thread arenas and contention
    without buying throughput.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # Full graph optimisation happens once at load and shrinks the resident
    # graph; the extra second of startup is worth it on a memory-bound host.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Let the arena hand pages back after the warm-up burst rather than holding
    # the high-water mark for the life of the process.
    options.enable_cpu_mem_arena = False
    return options


@dataclass
class OnnxModel:
    """A loaded ONNX graph plus the metadata the API reports.

    Mirrors the shape of :class:`app.ml.registry.LoadedModel` so the response
    payloads are identical on either backend.
    """

    name: str
    label: str
    session: object
    class_names: list[str]
    image_size: int
    temperature: float
    ood_thresholds: OODThresholds
    graph_path: str
    parameters: int
    size_mb: float
    load_time_ms: float
    metrics: dict = field(default_factory=dict)
    version: str = "1.0.0"
    selection: dict = field(default_factory=dict)
    # ONNX Runtime sessions are internally thread-safe for concurrent Run calls,
    # but occlusion mapping issues a burst of them and we would rather bound
    # peak memory than overlap two bursts on a 512 MB host.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    # Explainability here is forward-pass only; the gradient methods are not
    # available without autograd.
    has_cbam: bool = False
    device: str = "cpu"

    # Whether the graph carries a class-activation-map output that is provably
    # identical to autograd Grad-CAM for this architecture. Set by the exporter,
    # which checks it numerically rather than assuming it.
    cam_exact: bool = False
    explain_methods: tuple[str, ...] = ("occlusion",)

    # ------------------------------------------------------------- inference
    def logits(self, batch: np.ndarray) -> np.ndarray:
        """Run the graph. ``batch`` is ``(N, 3, S, S)`` float32."""
        return self.session.run(None, {"image": batch.astype(np.float32, copy=False)})[0]

    def logits_and_cam(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Run the graph, returning the class activation maps when present."""
        outputs = self.session.run(None, {"image": batch.astype(np.float32, copy=False)})
        return outputs[0], (outputs[1] if len(outputs) > 1 else None)

    def probabilities(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(raw_logits, temperature-scaled probabilities)``.

        The OOD scores are defined on the *raw* logits while the reported
        confidence must be calibrated, so both are returned rather than
        collapsing them here.
        """
        raw = self.logits(batch)
        return raw, softmax(raw, self.temperature)

    # -------------------------------------------------------- explainability
    def grad_cam(self, batch: np.ndarray,
                 class_index: int | None = None) -> tuple[np.ndarray, int] | None:
        """Grad-CAM, computed from the graph's CAM head with no backward pass.

        Returns ``None`` when the bundle has no CAM output, so callers can fall
        back to :meth:`occlusion_map` rather than silently serving a different
        method under the Grad-CAM label.

        The map is upsampled to the input resolution and min-max normalised, to
        match what :func:`app.ml.explain.grad_cam` returns on the torch backend.
        """
        raw, cam = self.logits_and_cam(batch)
        if cam is None:
            return None
        index = int(raw[0].argmax()) if class_index is None else int(class_index)
        return _upsample_normalised(cam[0, index], batch.shape[-1]), index

    def occlusion_map(self, batch: np.ndarray, class_index: int | None = None,
                      grid: int = 8, baseline: float = 0.0) -> tuple[np.ndarray, int]:
        """Gradient-free saliency: how far the class logit falls when a patch is hidden.

        Slides a ``grid x grid`` mask over the input and records the drop in the
        target logit. Unlike Grad-CAM this needs no backward pass, so it works on
        a graph-only runtime. It costs ``grid**2 + 1`` forward passes - about 1.3 s
        at 8x8 on one CPU core - which is why the API only runs it on request.

        Returns the ``(grid, grid)`` map and the class index it explains.
        """
        base = self.logits(batch)[0]
        index = int(base.argmax()) if class_index is None else int(class_index)
        reference = float(base[index])

        size = batch.shape[-1]
        # ceil, so the final patch covers the remainder instead of leaving a
        # strip of the image never occluded.
        step = -(-size // grid)

        occluded = np.repeat(batch, grid * grid, axis=0)
        for cell in range(grid * grid):
            row, col = divmod(cell, grid)
            occluded[cell, :,
                     row * step:min((row + 1) * step, size),
                     col * step:min((col + 1) * step, size)] = baseline

        # One batched call rather than grid**2 separate ones: ONNX Runtime
        # amortises its per-run overhead and this is ~3x faster in practice.
        scores = self.logits(occluded)[:, index]
        return (reference - scores).reshape(grid, grid).astype(np.float32), index

    # ------------------------------------------------------------- reporting
    def describe(self) -> dict:
        """Serialisable summary, matching the torch backend's ``describe()``."""
        return {
            "name": self.name,
            "label": self.label,
            "version": self.version,
            "num_classes": len(self.class_names),
            "image_size": self.image_size,
            "device": self.device,
            "backend": "onnx",
            "parameters": self.parameters,
            "size_mb": round(self.size_mb, 3),
            "load_time_ms": round(self.load_time_ms, 2),
            "temperature": round(self.temperature, 4),
            "has_cbam": self.has_cbam,
            "checkpoint": Path(self.graph_path).name,
            "test_metrics": self.metrics,
            "selection": self.selection,
            "explain_methods": list(self.explain_methods),
            "ood_thresholds": {
                "msp_min": self.ood_thresholds.msp_min,
                "entropy_max": self.ood_thresholds.entropy_max,
                "calibrated": self.ood_thresholds.calibrated,
                "method": self.ood_thresholds.method,
            },
        }


class OnnxRegistry:
    """Loads and caches the ONNX serving bundle. Interface mirrors ModelRegistry."""

    def __init__(self) -> None:
        self._model: OnnxModel | None = None
        self._lock = threading.Lock()

    @staticmethod
    def bundle_path() -> Path:
        """Location of the serving manifest written by ``export_serving_onnx.py``."""
        configured = getattr(settings, "onnx_bundle", None)
        if configured:
            path = Path(configured)
            return path if path.is_absolute() else settings.project_root / path
        return settings.exported_dir / "serving.json"

    def production_manifest(self) -> dict | None:
        path = self.bundle_path()
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def resolve_device() -> str:
        """ONNX Runtime here is CPU-only; the GPU providers are not installed."""
        return "cpu"

    def load(self, name: str | None = None) -> OnnxModel:
        with self._lock:
            if self._model is not None:
                return self._model

            manifest_path = self.bundle_path()
            manifest = self.production_manifest()
            if manifest is None:
                raise FileNotFoundError(
                    f"No ONNX serving bundle at {manifest_path}. Build one with:\n"
                    "    python scripts/export_serving_onnx.py"
                )
            if name not in (None, "auto") and name != manifest.get("model_name"):
                raise KeyError(
                    f"The ONNX backend serves only the exported bundle "
                    f"({manifest.get('model_name')!r}); {name!r} was requested. "
                    "Re-export with --model to switch."
                )

            graph_path = manifest_path.parent / manifest.get("graph", "serving.onnx")
            if not graph_path.exists():
                raise FileNotFoundError(f"ONNX graph not found: {graph_path}")

            started = time.perf_counter()
            import onnxruntime as ort

            session = ort.InferenceSession(
                str(graph_path),
                _session_options(int(getattr(settings, "onnx_threads", 1) or 1)),
                providers=["CPUExecutionProvider"],
            )

            raw = manifest.get("ood_thresholds") or {}
            if raw:
                thresholds = OODThresholds(
                    msp_min=float(raw.get("msp_min", settings.ood_confidence_floor)),
                    entropy_max=float(raw.get("entropy_max", settings.ood_entropy_ceiling)),
                    energy_max=float(raw["energy_max"]),
                    method=str(raw.get("method", "msp+entropy")),
                    calibrated=bool(raw.get("calibrated", False)),
                    target_tpr=raw.get("target_tpr"),
                    source=raw.get("source"),
                    model_name=raw.get("model_name"),
                )
            else:
                thresholds = OODThresholds.defaults(settings.ood_confidence_floor,
                                                    settings.ood_entropy_ceiling)
            if not thresholds.calibrated:
                log.warning(
                    "Serving with uncalibrated OOD thresholds",
                    fields={"model": manifest["model_name"],
                            "hint": "python training/calibrate_ood.py, then re-export"},
                )

            model = OnnxModel(
                name=manifest["model_name"],
                label=manifest.get("label", manifest["model_name"]),
                session=session,
                class_names=list(manifest["class_names"]),
                image_size=int(manifest.get("image_size", settings.image_size)),
                temperature=float(manifest.get("temperature", 1.0)),
                ood_thresholds=thresholds,
                graph_path=str(graph_path),
                parameters=int(manifest.get("parameters", 0)),
                size_mb=float(manifest.get("size_mb", graph_path.stat().st_size / (1024 * 1024))),
                load_time_ms=(time.perf_counter() - started) * 1000,
                metrics=manifest.get("test_metrics", {}),
                version=manifest.get("version", "1.0.0"),
                selection=manifest.get("selection", {}),
                cam_exact=bool(manifest.get("cam_exact", False)),
                explain_methods=tuple(manifest.get("explain_methods", ("occlusion",))),
            )
            # Trust the graph over the manifest: a bundle whose manifest claims a
            # CAM head the graph does not have would 500 on every explain request.
            if model.cam_exact and len(session.get_outputs()) < 2:
                log.warning("Manifest claims a CAM head but the graph has none; "
                            "falling back to occlusion saliency",
                            fields={"graph": str(graph_path)})
                model.cam_exact = False
                model.explain_methods = ("occlusion",)
            self._model = model
            log.info("Loaded ONNX model",
                     fields={"model": model.name, "classes": len(model.class_names),
                             "load_ms": round(model.load_time_ms, 1),
                             "image_size": model.image_size,
                             "graph_mb": round(model.size_mb, 2)})
            return model

    def warm_up(self, name: str | None = None) -> OnnxModel:
        """Load and run one inference so the first real request is not the slow one."""
        model = self.load(name)
        blank = np.zeros((1, 3, model.image_size, model.image_size), dtype=np.float32)
        with model.lock:
            model.logits(blank)
        return model

    def available_names(self) -> list[str]:
        manifest = self.production_manifest()
        return [manifest["model_name"]] if manifest else []

    def loaded_names(self) -> list[str]:
        return [self._model.name] if self._model else []


registry = OnnxRegistry()
