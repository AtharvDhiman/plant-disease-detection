"""Backend selection: hand callers whichever model registry is configured.

Import ``registry`` from here rather than from :mod:`app.ml.registry` or
:mod:`app.ml.onnx_backend` directly. The choice is made once, at import time,
from ``PDD_SERVING_BACKEND``.

Why import time rather than per call
------------------------------------
The whole reason the ONNX backend exists is that importing torch costs roughly
530 MB of resident memory. A lazy per-call lookup would not help: something has
to import the module eventually, and on a 512 MB host "eventually" means the
process is killed mid-request instead of at startup. Branching on the import
itself is what keeps torch out of the process entirely.

Both registries expose the same surface - ``warm_up``, ``load``,
``production_manifest``, ``resolve_device``, ``available_names``,
``loaded_names`` - and both return a model whose ``describe()`` produces the
same response payload, so callers do not need to know which one they got.
"""
from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

BACKEND = (settings.serving_backend or "torch").strip().lower()

if BACKEND not in {"torch", "onnx"}:
    raise ValueError(
        f"PDD_SERVING_BACKEND must be 'torch' or 'onnx', got {settings.serving_backend!r}"
    )

if BACKEND == "onnx":
    from app.ml.onnx_backend import registry  # noqa: F401
else:
    from app.ml.registry import registry  # noqa: F401


def backend_name() -> str:
    """Which runtime is executing the model, for /api/health and the UI."""
    return BACKEND


def supports_gradient_explanations() -> bool:
    """Grad-CAM, Grad-CAM++ and integrated gradients need autograd.

    The ONNX backend offers occlusion sensitivity instead, which measures how
    far the predicted logit falls when a region is masked - a real saliency
    signal, but computed with forward passes only.
    """
    return BACKEND == "torch"
