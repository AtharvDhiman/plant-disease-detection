"""Centralised, environment-driven configuration.

Every tunable in the project (paths, image size, thresholds, database URL) is
declared here so nothing is hard-coded in the training scripts, the API or the
inference code. Values can be overridden through a ``.env`` file at the repo
root or through real environment variables prefixed with ``PDD_``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> backend/app/core -> backend/app -> backend -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings; see ``.env.example`` for the documented knobs."""

    model_config = SettingsConfigDict(
        env_prefix="PDD_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # ---------------------------------------------------------------- paths
    project_root: Path = PROJECT_ROOT
    dataset_path: Path | None = None
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    upload_dir: Path = PROJECT_ROOT / "uploads"

    # ------------------------------------------------------------- database
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'plant_disease.db').as_posix()}"

    # ---------------------------------------------------------------- model
    # Which exported bundle the API serves. ``auto`` resolves via
    # models/exported/production.json written by the export step.
    serving_model: str = "auto"
    # Which runtime executes the model. ``torch`` is the full-featured path used
    # in development and for GPU serving. ``onnx`` runs the exported graph
    # through ONNX Runtime instead, which drops the process from ~610 MB to
    # ~103 MB resident - the difference between fitting a 512 MB container and
    # being OOM-killed at startup - at the cost of the gradient-based
    # explanations. See app/ml/onnx_backend.py.
    serving_backend: str = "torch"  # torch | onnx
    # Bundle written by scripts/export_serving_onnx.py; relative paths resolve
    # against the project root.
    onnx_bundle: str = "models/exported/serving.json"
    # One thread by default: a free-tier container gets a fraction of a core, so
    # extra intra-op threads cost memory and contention without buying speed.
    onnx_threads: int = 1
    image_size: int = 224
    # ImageNet statistics — used by every backbone we fine-tune.
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    device: str = "auto"  # auto | cuda | cpu

    # ----------------------------------------------------------- thresholds
    confidence_high: float = 0.85
    confidence_medium: float = 0.60
    # Below this the prediction is reported as "unable to confidently identify".
    ood_confidence_floor: float = 0.15
    # Normalised predictive entropy above which we flag out-of-distribution.
    ood_entropy_ceiling: float = 0.85

    # ------------------------------------------------------- image quality
    quality_min_side: int = 64
    quality_blur_threshold: float = 45.0     # variance of Laplacian
    quality_dark_threshold: float = 35.0     # mean luminance 0-255
    quality_bright_threshold: float = 225.0
    quality_min_score: float = 0.45          # below this we warn the user
    # Lag-1 pixel autocorrelation; camera photographs sit well above 0.9, while
    # synthetic noise sits near 0. 0.5 leaves a wide margin for grainy phone shots.
    quality_min_coherence: float = 0.50

    # -------------------------------------------------------------- uploads
    max_upload_size: int = 10 * 1024 * 1024  # 10 MiB
    allowed_extensions: tuple[str, ...] | str = (".jpg", ".jpeg", ".png", ".webp")
    allowed_content_types: tuple[str, ...] | str = (
        "image/jpeg",
        "image/png",
        "image/webp",
    )

    # ------------------------------------------------------------- training
    seed: int = 42
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4

    # ------------------------------------------------------------------ api
    api_title: str = "Plant Disease Detection API"
    api_version: str = "1.0.0"
    cors_origins: tuple[str, ...] | str = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    )
    log_level: str = "INFO"

    # ---------------------------------------------------------- validators
    @field_validator("cors_origins", "allowed_extensions", "allowed_content_types", mode="after")
    @classmethod
    def _split_csv(cls, value):
        """Accept ``a,b,c`` strings from .env as well as real sequences."""
        if isinstance(value, str):
            if value.strip().startswith("["):
                try:
                    return tuple(json.loads(value))
                except Exception:
                    pass
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return tuple(value)

    # ------------------------------------------------------ derived helpers
    @property
    def checkpoints_dir(self) -> Path:
        return self.models_dir / "checkpoints"

    @property
    def exported_dir(self) -> Path:
        return self.models_dir / "exported"

    @property
    def metadata_dir(self) -> Path:
        return self.models_dir / "metadata"

    @property
    def results_dir(self) -> Path:
        return self.artifacts_dir / "results"

    @property
    def plots_dir(self) -> Path:
        return self.artifacts_dir / "plots"

    @property
    def confusion_dir(self) -> Path:
        return self.artifacts_dir / "confusion_matrices"

    @property
    def explain_dir(self) -> Path:
        return self.artifacts_dir / "explainability"

    @property
    def logs_dir(self) -> Path:
        return self.artifacts_dir / "logs"

    @property
    def dataset_report_path(self) -> Path:
        return self.artifacts_dir / "dataset_report.json"

    @property
    def splits_path(self) -> Path:
        return self.data_dir / "splits.json"

    @property
    def disease_info_path(self) -> Path:
        return self.data_dir / "disease_info.json"

    def resolve_dataset_path(self) -> Path:
        """Return the dataset root, falling back to data/dataset_path.json."""
        if self.dataset_path is not None:
            return Path(self.dataset_path)
        pointer = self.data_dir / "dataset_path.json"
        if pointer.exists():
            return Path(json.loads(pointer.read_text(encoding="utf-8"))["path"])
        raise FileNotFoundError(
            "Dataset location unknown. Run `python scripts/download_dataset.py` "
            "or set PDD_DATASET_PATH."
        )

    def ensure_dirs(self) -> None:
        """Create every directory the application writes into."""
        for path in (
            self.data_dir,
            self.upload_dir,
            self.checkpoints_dir,
            self.exported_dir,
            self.metadata_dir,
            self.results_dir,
            self.plots_dir,
            self.confusion_dir,
            self.explain_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


settings = get_settings()
