"""The inference pipeline: image in, structured diagnosis out.

Order of operations, and why:

1. **Quality analysis** first, on the original image, so a blurred or dark
   upload is reported as such rather than being classified anyway.
2. **Preprocessing** with the exact transform used at validation time.
3. **Inference** under ``torch.no_grad``, with temperature scaling applied to
   the logits so the reported confidence is calibrated.
4. **OOD assessment** on the (uncalibrated) logits, because the energy score is
   defined on raw logits.
5. **Explainability**, optional per request because Grad-CAM needs a backward
   pass and roughly doubles the latency.

Each stage is timed separately and the timings are returned to the client and
stored, which is what the research dashboard's performance panel reports.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.logging import get_logger
from app.ml import ood
from app.ml.heatmap import CamResult, attention_coverage, overlay_heatmap
# One preprocessing implementation for both backends. It is torch-free and
# bit-identical to build_eval_transform (verified at export time, and again by
# tests), which is what lets the ONNX path serve the same predictions while
# keeping torchvision - 77 MB - out of the serving process entirely.
from app.ml.preprocess import eval_crop, eval_preprocess
from app.ml.quality import QualityReport, analyse_image, quality_band
from app.ml.serving import BACKEND, registry
from app.services.knowledge import knowledge_base

if BACKEND == "torch":
    # Imported only on the torch backend. On the ONNX backend neither torch nor
    # explain.py exists in the image, and importing them is the ~530 MB that
    # does not fit in a 512 MB container.
    import torch

    from app.ml.explain import (
        cbam_channel_profile,
        cbam_spatial_map,
        grad_cam,
        grad_cam_plus_plus,
        integrated_gradients,
    )

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclass
class ClassScore:
    """One entry of the top-k list."""

    class_name: str
    display_name: str
    plant: str
    condition: str
    probability: float
    is_healthy: bool


@dataclass
class PredictionResult:
    """Everything one analysis produced."""

    predicted_class: str
    display_name: str
    plant: str
    condition: str
    is_healthy: bool
    confidence: float
    confidence_level: str
    status: str
    status_message: str | None
    top_predictions: list[ClassScore]
    quality: QualityReport
    ood: ood.OODVerdict
    model: dict
    timings: dict
    explanations: dict = field(default_factory=dict)
    attention_stats: dict = field(default_factory=dict)
    # method -> reason, for explanations that were requested but could not be
    # produced (a transient CUDA failure under load, for example). Reporting the
    # failure beats silently returning fewer tabs than the caller asked for.
    explanation_errors: dict = field(default_factory=dict)
    channel_attention: list[float] | None = None
    disease_info: dict | None = None
    image_meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Confidence banding
# --------------------------------------------------------------------------- #


def confidence_level(confidence: float) -> str:
    """Bucket a probability into HIGH / MEDIUM / LOW using configured thresholds."""
    if confidence >= settings.confidence_high:
        return "high"
    if confidence >= settings.confidence_medium:
        return "medium"
    return "low"


CONFIDENCE_MESSAGES = {
    "high": None,
    "medium": (
        "Moderate confidence. The visual evidence is consistent with this class but "
        "not decisive - compare the alternatives below and consider a second photo "
        "from a different angle."
    ),
    "low": (
        "Low confidence prediction. Consider uploading a clearer, closer image of a "
        "single leaf, or consulting an agricultural extension officer before acting "
        "on this result."
    ),
}


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #


class Predictor:
    """Stateless service wrapping the loaded model.

    The model comes from whichever registry :mod:`app.ml.serving` selected - a
    :class:`app.ml.registry.LoadedModel` on the torch backend or an
    :class:`app.ml.onnx_backend.OnnxModel` on the ONNX one. The two expose the
    same attributes, so everything below the inference call is shared.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name

    def _loaded(self):
        return registry.load(self.model_name)

    # ---------------------------------------------------------------- main
    def predict(
        self,
        image: Image.Image,
        top_k: int = 5,
        explain: bool = True,
        explain_methods: tuple[str, ...] = ("grad_cam", "grad_cam_plus_plus", "cbam_spatial"),
        save_dir: Path | None = None,
        file_stem: str | None = None,
    ) -> PredictionResult:
        """Run the full pipeline on one PIL image."""
        loaded = self._loaded()
        total_start = time.perf_counter()

        # 1 ------------------------------------------------------- quality
        quality_start = time.perf_counter()
        quality = analyse_image(image)
        quality_ms = (time.perf_counter() - quality_start) * 1000

        # 2 -------------------------------------------------- preprocessing
        preprocess_start = time.perf_counter()
        batch = eval_preprocess(image, loaded.image_size)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000

        # 3 ------------------------------------------------------ inference
        # The model is shared across the request threadpool and inference is not
        # read-only (attention blocks cache their gate), so hold the model lock.
        inference_start = time.perf_counter()
        if BACKEND == "torch":
            tensor = torch.from_numpy(batch).to(loaded.device)
            with loaded.lock, torch.no_grad():
                logits = loaded.model(tensor)
                if loaded.device.type == "cuda":
                    torch.cuda.synchronize()
            raw_logits = logits.float().cpu().numpy()[0]
        else:
            tensor = batch
            with loaded.lock:
                raw_logits = loaded.logits(batch)[0]
        inference_ms = (time.perf_counter() - inference_start) * 1000
        # Temperature scaling only rescales the logits, so the argmax - and
        # therefore the predicted class - is unchanged; only confidence moves.
        calibrated = raw_logits / max(loaded.temperature, 1e-6)
        probabilities = ood.softmax(calibrated.reshape(1, -1))[0]

        order = np.argsort(probabilities)[::-1][: min(top_k, len(loaded.class_names))]
        top = [
            ClassScore(
                class_name=loaded.class_names[i],
                display_name=knowledge_base.display_name(loaded.class_names[i]),
                plant=knowledge_base.plant_of(loaded.class_names[i]),
                condition=knowledge_base.condition_of(loaded.class_names[i]),
                probability=float(probabilities[i]),
                is_healthy=knowledge_base.is_healthy(loaded.class_names[i]),
            )
            for i in order
        ]
        best = top[0]
        confidence = best.probability
        level = confidence_level(confidence)

        # 4 ------------------------------------------------------------ OOD
        verdict = ood.assess(raw_logits, loaded.ood_thresholds)

        status, message = self._decide_status(quality, verdict, level)
        is_rejected = status in ("not_a_plant", "poor_quality", "out_of_distribution")

        # 5 ------------------------------------------------- explainability
        explanations: dict = {}
        attention_stats: dict = {}
        explanation_errors: dict = {}
        channel_profile: list[float] | None = None
        explain_ms = 0.0
        if explain and not is_rejected:
            explain_start = time.perf_counter()
            explanations, attention_stats, channel_profile, explanation_errors = self._explain(
                loaded, tensor, image, int(order[0]), explain_methods, save_dir, file_stem
            )
            explain_ms = (time.perf_counter() - explain_start) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000

        if is_rejected:
            predicted_class = "Unknown"
            display_name = "Not Accepted"
            plant = "Unknown"
            condition = "Not a recognized plant leaf" if status != "poor_quality" else "Insufficient image quality"
            is_healthy = False
            reported_confidence = 0.0
            reported_level = "low"
            info = None
            top_predictions = []
        else:
            predicted_class = best.class_name
            display_name = best.display_name
            plant = best.plant
            condition = best.condition
            is_healthy = best.is_healthy
            reported_confidence = confidence
            reported_level = level
            info = knowledge_base.get(best.class_name)
            top_predictions = top

        return PredictionResult(
            predicted_class=predicted_class,
            display_name=display_name,
            plant=plant,
            condition=condition,
            is_healthy=is_healthy,
            confidence=reported_confidence,
            confidence_level=reported_level,
            status=status,
            status_message=message,
            top_predictions=top_predictions,
            quality=quality,
            ood=verdict,
            model={
                "name": loaded.name,
                "label": loaded.label,
                "version": loaded.version,
                "image_size": loaded.image_size,
                "device": str(loaded.device),
                "temperature": round(loaded.temperature, 4),
                "parameters": loaded.parameters,
            },
            timings={
                "quality_ms": round(quality_ms, 2),
                "preprocess_ms": round(preprocess_ms, 2),
                "inference_ms": round(inference_ms, 2),
                "explain_ms": round(explain_ms, 2),
                "total_ms": round(total_ms, 2),
            },
            explanations=explanations,
            attention_stats=attention_stats,
            explanation_errors=explanation_errors,
            channel_attention=channel_profile,
            disease_info=info,
            image_meta={
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "quality_band": quality_band(quality.score),
            },
        )

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _decide_status(
        quality: QualityReport, verdict: ood.OODVerdict, level: str
    ) -> tuple[str, str | None]:
        """Combine the quality and OOD signals into one user-facing status.

        Quality wins over OOD: a blurred photograph of a real leaf should be
        reported as a photo problem the user can fix, not as an unrecognised
        subject.
        """
        # Photographic errors outrank non-plant detection so that dark/blurry shots
        # are flagged as camera defects to fix:
        if not quality.passed and any(i.severity == "error" and i.code != "not_a_plant" for i in quality.issues):
            return "poor_quality", (
                "Image quality is insufficient for a reliable prediction. "
                + " ".join(issue.suggestion for issue in quality.issues
                           if issue.severity == "error")
            )

        not_a_plant_issue = next((i for i in quality.issues if i.code == "not_a_plant"), None)
        if not_a_plant_issue:
            return "not_a_plant", (
                "Photo rejected: No plant leaf detected. "
                "This system is trained exclusively to diagnose leaf diseases across 14 crop species. "
                + not_a_plant_issue.suggestion
            )

        if not quality.passed and any(i.severity == "error" for i in quality.issues):
            return "poor_quality", (
                "Image quality is insufficient for a reliable prediction. "
                + " ".join(issue.suggestion for issue in quality.issues
                           if issue.severity == "error")
            )
        if verdict.is_ood:
            return "out_of_distribution", (
                "Unable to identify a plant leaf in this image. It does not resemble the "
                "crop leaf photographs this model was trained on. " + " ".join(verdict.reasons)
            )
        if level == "low":
            return "low_confidence", CONFIDENCE_MESSAGES["low"]
        if level == "medium":
            return "ok", CONFIDENCE_MESSAGES["medium"]
        return "ok", None

    @staticmethod
    def _explain_onnx(loaded, batch: np.ndarray, class_index: int,
                      method: str, errors: dict) -> CamResult | None:
        """Explanations available without autograd.

        ``grad_cam`` is the real thing, not an approximation: for a head that is
        global-average-pool then linear, Grad-CAM reduces algebraically to the
        class activation map, which the exported graph emits directly. The
        exporter verifies this numerically per model and records ``cam_exact``,
        so a model whose head does not have that form falls back to occlusion
        rather than serving a map labelled Grad-CAM that is not one.

        The remaining torch methods need genuine higher-order gradients and have
        no closed form here. They are reported as unavailable rather than
        silently substituted, so a missing tab in the UI is explained.
        """
        if method == "grad_cam" and getattr(loaded, "cam_exact", False):
            produced = loaded.grad_cam(batch, class_index)
            if produced is not None:
                heatmap, index = produced
                return CamResult(heatmap, index, "grad_cam")

        if method in ("occlusion", "grad_cam"):
            # Reached for `occlusion` directly, or for `grad_cam` on a bundle
            # with no CAM head - occlusion is the honest substitute there.
            heatmap, index = loaded.occlusion_map(batch, class_index)
            # Occlusion returns a coarse grid of logit drops; the UI expects a
            # full-resolution map in [0, 1] like every other method.
            from app.ml.onnx_backend import _upsample_normalised
            resized = _upsample_normalised(heatmap, batch.shape[-1])
            if method == "grad_cam":
                errors["grad_cam"] = (
                    "Grad-CAM needs either autograd or a CAM head in the exported "
                    "graph; neither is present. Showing occlusion sensitivity instead."
                )
            return CamResult(resized, index, "occlusion")

        if method in ("grad_cam_plus_plus", "integrated_gradients", "cbam_spatial"):
            errors[method] = (
                f"{method} requires a backward pass and is not available on the "
                "ONNX serving backend. Available here: "
                f"{', '.join(getattr(loaded, 'explain_methods', ('occlusion',)))}."
            )
            return None

        errors[method] = "Unknown explanation method."
        log.warning("Unknown explanation method", fields={"method": method})
        return None

    def _explain(
        self,
        loaded,
        tensor,
        original: Image.Image,
        class_index: int,
        methods: tuple[str, ...],
        save_dir: Path | None,
        file_stem: str | None,
    ) -> tuple[dict, dict, list[float] | None, dict]:
        """Produce heat maps, overlays, their statistics, and any failures."""
        base_image = np.asarray(eval_crop(original, loaded.image_size))
        stem = file_stem or uuid.uuid4().hex
        target_dir = save_dir or settings.upload_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        outputs: dict = {}
        stats: dict = {}
        errors: dict = {}
        channel_profile: list[float] | None = None

        # Grad-CAM needs gradients w.r.t. activations; the registry disabled
        # requires_grad on the parameters, so re-enable just for this call.
        needs_grad = (BACKEND == "torch"
                      and bool({"grad_cam", "grad_cam_plus_plus", "integrated_gradients"}
                               & set(methods)))
        # Everything below mutates shared model state - hooks, requires_grad
        # flags, accumulated gradients and the attention blocks' cached gates -
        # so it must not overlap with another request's explanation pass.
        loaded.lock.acquire()
        if needs_grad:
            for parameter in loaded.model.parameters():
                parameter.requires_grad_(True)
        try:
            for method in methods:
                try:
                    if BACKEND == "torch":
                        if method == "grad_cam":
                            result = grad_cam(loaded.model, tensor, loaded.model.feature_layer,
                                              class_index)
                        elif method == "grad_cam_plus_plus":
                            result = grad_cam_plus_plus(loaded.model, tensor,
                                                        loaded.model.feature_layer, class_index)
                        elif method == "cbam_spatial":
                            result = cbam_spatial_map(loaded.model, tensor, class_index)
                            if result is None:
                                continue
                            channel_profile = cbam_channel_profile(loaded.model)
                        elif method == "integrated_gradients":
                            result = integrated_gradients(loaded.model, tensor, class_index,
                                                          steps=24)
                        else:
                            errors[method] = "Unknown explanation method."
                            log.warning("Unknown explanation method", fields={"method": method})
                            continue
                    else:
                        result = self._explain_onnx(loaded, tensor, class_index, method, errors)
                        if result is None:
                            continue
                except Exception as exc:  # noqa: BLE001 - never fail a prediction on a heat map
                    # A heat map is an enhancement; losing one must not lose the
                    # diagnosis. But the client is told, so a missing tab reads as
                    # a failure rather than as "this model has no such view".
                    errors[method] = f"{type(exc).__name__}: {exc}"[:300]
                    log.warning("Explanation failed",
                                fields={"method": method, "error": str(exc)})
                    continue

                overlay = overlay_heatmap(base_image, result.heatmap)
                filename = f"{stem}_{result.method}.png"
                Image.fromarray(overlay).save(target_dir / filename, optimize=True)
                outputs[result.method] = filename
                stats[result.method] = attention_coverage(result.heatmap)
        finally:
            if needs_grad:
                for parameter in loaded.model.parameters():
                    parameter.requires_grad_(False)
                loaded.model.zero_grad(set_to_none=True)
            loaded.lock.release()

        original_name = f"{stem}_input.png"
        Image.fromarray(base_image).save(target_dir / original_name, optimize=True)
        outputs["original"] = original_name
        return outputs, stats, channel_profile, errors


predictor = Predictor()
