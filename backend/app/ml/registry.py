"""Model loading and caching for serving.

The application loads its weights **once**, at startup, and every request reuses
the same in-memory model. Training and inference are entirely separate: nothing
in this module can train, and the FastAPI process never touches the dataset.

Which checkpoint gets served is decided by ``models/exported/production.json``,
written by ``training/export_model.py`` from the measured benchmark results, so
promoting a new model is a matter of re-running the export rather than editing
code.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from app.core.config import settings
from app.core.logging import get_logger
from app.ml.architectures import MODEL_ZOO, build_model, count_parameters, model_size_mb
from app.ml.ood import OODThresholds

log = get_logger(__name__)


@dataclass
class LoadedModel:
    """A ready-to-serve model plus the metadata the API reports.

    ``lock`` guards every use of :attr:`model`. FastAPI runs synchronous
    endpoints in a threadpool, so two ``/api/predict`` requests genuinely do
    execute concurrently against this one shared module - and inference here is
    not read-only. Grad-CAM registers hooks, toggles ``requires_grad``,
    back-propagates and zeroes gradients, while the attention blocks cache their
    last gate on the module itself. Without serialisation, concurrent requests
    would corrupt each other's heat maps and gradients.

    Serialising costs nothing here: a single GPU processes one batch at a time
    regardless, so the lock removes a correctness hazard without reducing
    throughput.
    """

    name: str
    label: str
    model: torch.nn.Module
    class_names: list[str]
    image_size: int
    device: torch.device
    temperature: float
    ood_thresholds: OODThresholds
    checkpoint_path: str
    parameters: int
    size_mb: float
    load_time_ms: float
    metrics: dict = field(default_factory=dict)
    has_cbam: bool = False
    version: str = "1.0.0"
    selection: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def describe(self) -> dict:
        """Serialisable summary used by ``GET /api/models`` and the UI."""
        return {
            "name": self.name,
            "label": self.label,
            "version": self.version,
            "num_classes": len(self.class_names),
            "image_size": self.image_size,
            "device": str(self.device),
            "parameters": self.parameters,
            "size_mb": round(self.size_mb, 3),
            "load_time_ms": round(self.load_time_ms, 2),
            "temperature": round(self.temperature, 4),
            "has_cbam": self.has_cbam,
            "checkpoint": Path(self.checkpoint_path).name,
            "test_metrics": self.metrics,
            "selection": self.selection,
            "ood_thresholds": {
                "msp_min": self.ood_thresholds.msp_min,
                "entropy_max": self.ood_thresholds.entropy_max,
                "calibrated": self.ood_thresholds.calibrated,
                "method": self.ood_thresholds.method,
            },
        }


class ModelRegistry:
    """Process-wide cache of loaded models, keyed by name."""

    def __init__(self) -> None:
        self._models: dict[str, LoadedModel] = {}
        self._lock = threading.Lock()
        self._production_name: str | None = None

    # ------------------------------------------------------------- device
    @staticmethod
    def resolve_device() -> torch.device:
        """Honour ``PDD_DEVICE``; fall back to CUDA when available."""
        requested = settings.device.lower()
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("PDD_DEVICE=cuda but no CUDA device is available.")
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------- manifest
    @staticmethod
    def production_manifest() -> dict | None:
        """Read ``models/exported/production.json`` if the export step has run."""
        path = settings.exported_dir / "production.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- load
    def load(self, name: str | None = None) -> LoadedModel:
        """Load (or return the cached) model. Thread-safe and idempotent."""
        manifest = self.production_manifest()
        if name in (None, "auto"):
            if manifest is None:
                raise FileNotFoundError(
                    "No production model registered. Run the training pipeline and then "
                    "`python training/export_model.py`."
                )
            name = manifest["model_name"]

        with self._lock:
            if name in self._models:
                return self._models[name]

            entry = None
            if manifest and manifest.get("model_name") == name:
                entry = manifest
            elif manifest:
                entry = next(
                    (c for c in manifest.get("candidates", []) if c.get("model_name") == name),
                    None,
                )
            if entry is None:
                raise KeyError(
                    f"Model {name!r} is not in the exported manifest. "
                    f"Available: {self.available_names()}"
                )

            started = time.perf_counter()
            device = self.resolve_device()
            checkpoint_path = Path(entry["checkpoint"])
            if not checkpoint_path.is_absolute():
                checkpoint_path = settings.project_root / checkpoint_path
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
            class_names = payload["class_names"]
            model = build_model(name, len(class_names), pretrained=False)
            model.load_state_dict(payload["model_state_dict"])
            model.to(device).eval()
            # Inference never needs gradients except for Grad-CAM, which re-enables
            # them locally; leaving them off here saves memory on every request.
            for parameter in model.parameters():
                parameter.requires_grad_(False)

            spec = MODEL_ZOO.get(name)
            total_params, _ = count_parameters(model)
            # Prefer thresholds calibrated for *this* model. The generic file is
            # accepted only when it records the same model, because thresholds
            # fitted to a different network reject correct predictions.
            thresholds = (
                OODThresholds.from_file(settings.metadata_dir / f"ood_{name}.json", name)
                or OODThresholds.from_file(settings.metadata_dir / "ood_thresholds.json", name)
                or OODThresholds.defaults(settings.ood_confidence_floor,
                                          settings.ood_entropy_ceiling)
            )
            if not thresholds.calibrated:
                log.warning(
                    "No OOD thresholds calibrated for this model; using configured defaults",
                    fields={"model": name,
                            "hint": "Run: python training/calibrate_ood.py"},
                )

            loaded = LoadedModel(
                name=name,
                label=spec.label if spec else name,
                model=model,
                class_names=class_names,
                image_size=entry.get("image_size", settings.image_size),
                device=device,
                temperature=float(entry.get("temperature", 1.0)),
                ood_thresholds=thresholds,
                checkpoint_path=str(checkpoint_path),
                parameters=total_params,
                size_mb=model_size_mb(model),
                load_time_ms=(time.perf_counter() - started) * 1000,
                metrics=entry.get("test_metrics", {}),
                has_cbam=any("cbam" in n.lower() or "attention" in n.lower()
                             for n, _ in model.named_modules() if "spatial" in n.lower()),
                version=entry.get("version", "1.0.0"),
                selection=manifest.get("selection", {}) if manifest else {},
            )
            self._models[name] = loaded
            if manifest and manifest.get("model_name") == name:
                self._production_name = name

            log.info(
                "Loaded model",
                fields={"model": name, "device": str(device), "classes": len(class_names),
                        "params": total_params, "load_ms": round(loaded.load_time_ms, 1),
                        "image_size": loaded.image_size},
            )
            return loaded

    # --------------------------------------------------------------- query
    def available_names(self) -> list[str]:
        manifest = self.production_manifest()
        if not manifest:
            return []
        names = [manifest["model_name"]]
        names += [c["model_name"] for c in manifest.get("candidates", [])
                  if c["model_name"] != manifest["model_name"]]
        return names

    def loaded_names(self) -> list[str]:
        return sorted(self._models)

    @property
    def production_name(self) -> str | None:
        return self._production_name

    def warm_up(self, name: str | None = None) -> LoadedModel:
        """Load the model and run one dummy forward pass.

        The first CUDA forward pass pays a large one-off kernel-initialisation
        cost. Doing it at startup keeps the first real user request fast.
        """
        loaded = self.load(name)
        dummy = torch.zeros(1, 3, loaded.image_size, loaded.image_size, device=loaded.device)
        with torch.no_grad():
            loaded.model(dummy)
        if loaded.device.type == "cuda":
            torch.cuda.synchronize()
        return loaded

    def clear(self) -> None:
        """Drop cached models (used by tests)."""
        with self._lock:
            self._models.clear()
            self._production_name = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


registry = ModelRegistry()
