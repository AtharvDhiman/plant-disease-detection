"""Explainability: Grad-CAM, Grad-CAM++, CBAM attention and Integrated Gradients.

What each method actually tells you
-----------------------------------
*Grad-CAM* weights the final convolutional feature maps by the average gradient
of the target logit with respect to each channel, so bright regions are those
whose activation, if increased, would most increase the score for the predicted
class. *Grad-CAM++* replaces the average with a positive-gradient weighting that
handles multiple lesions in one image better. *CBAM attention* is not a
saliency map at all - it is the spatial gate the network itself learned, read
straight out of the forward pass.

An important honesty caveat, surfaced in the UI as well: none of these maps is a
disease segmentation. They show where evidence for the decision was pooled from,
at the resolution of the final feature map (typically 7x7 or 14x14 upsampled to
the input). A map that covers a lesion is evidence the model is looking at the
right thing; it is not a measurement of lesion extent.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from app.ml.attention import SpatialAttention

# Rendering and statistics live in a torch-free module so the ONNX serving
# backend can use them too; re-exported here so this module's API is unchanged.
from app.ml.heatmap import (  # noqa: F401
    JET_ANCHORS,
    CamResult,
    attention_coverage,
    colorize,
    overlay_heatmap,
)

# --------------------------------------------------------------------------- #
# Hook plumbing
# --------------------------------------------------------------------------- #


class ActivationGradientHook:
    """Capture the forward activations and backward gradients of one layer."""

    def __init__(self, layer: torch.nn.Module) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            layer.register_forward_hook(self._save_activation),
            layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output.detach() if isinstance(output, torch.Tensor) else output[0].detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> ActivationGradientHook:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _normalise(cam: torch.Tensor) -> torch.Tensor:
    """Scale a heat map to ``[0, 1]`` per sample.

    The epsilon guards division by zero only. Adding it to the denominator
    unconditionally would silently compress genuinely small-range maps (an
    untrained model's CAM spans ~1e-9, which an absolute epsilon would squash to
    0.05 instead of 1.0), so a degenerate range is detected explicitly and
    returned as all-zeros.
    """
    flat = cam.flatten(1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    span = hi - lo
    degenerate = span <= 0
    scaled = torch.where(degenerate, torch.zeros_like(flat),
                         (flat - lo) / span.clamp_min(1e-30))
    return scaled.clamp(0.0, 1.0).view_as(cam)


def _relu_cam(weighted_sum: torch.Tensor) -> torch.Tensor:
    """Apply Grad-CAM's ReLU, falling back to magnitude if it erases everything.

    Grad-CAM keeps only positive evidence. When every location has a negative
    contribution the ReLU produces an all-black map that says nothing; rendering
    the magnitude instead is more informative than showing an empty tab, and the
    caller records which happened.
    """
    positive = F.relu(weighted_sum)
    if float(positive.abs().max()) > 0:
        return positive
    return weighted_sum.abs()


def _resize(cam: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(cam.unsqueeze(1), size=size, mode="bilinear", align_corners=False).squeeze(1)


# --------------------------------------------------------------------------- #
# CAM methods
# --------------------------------------------------------------------------- #


def grad_cam(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_layer: torch.nn.Module,
    class_index: int | None = None,
) -> CamResult:
    """Grad-CAM (Selvaraju et al., 2017) for a single-image batch."""
    model.eval()
    model.zero_grad(set_to_none=True)

    with ActivationGradientHook(target_layer) as hook:
        # Autocast is deliberately off: fp16 gradients here are small enough to
        # underflow to zero, producing an all-black map.
        logits = model(inputs)
        index = int(logits.argmax(dim=1).item()) if class_index is None else class_index
        logits[:, index].sum().backward()

        activations = hook.activations
        gradients = hook.gradients

    if activations is None or gradients is None:  # pragma: no cover - guarded by callers
        raise RuntimeError("Grad-CAM hooks captured nothing; is target_layer inside the model?")

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = _relu_cam((weights * activations).sum(dim=1))
    cam = _normalise(_resize(cam, inputs.shape[-2:]))
    return CamResult(cam[0].cpu().numpy().astype(np.float32), index, "grad_cam")


def grad_cam_plus_plus(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_layer: torch.nn.Module,
    class_index: int | None = None,
) -> CamResult:
    """Grad-CAM++ (Chattopadhyay et al., 2018).

    Uses second- and third-order gradient terms so that several disjoint
    evidence regions - multiple lesions on one leaf - all contribute, instead of
    the single strongest one dominating as it can in plain Grad-CAM.
    """
    model.eval()
    model.zero_grad(set_to_none=True)

    with ActivationGradientHook(target_layer) as hook:
        logits = model(inputs)
        index = int(logits.argmax(dim=1).item()) if class_index is None else class_index
        # The paper derives alpha from the gradients of exp(score), but
        # exp(score) overflows to inf for large logits. Back-propagating the raw
        # score instead differs only by the positive scalar exp(score), which
        # cancels between the alpha numerator and denominator and is removed
        # entirely by the final normalisation.
        logits[:, index].sum().backward()

        activations = hook.activations
        gradients = hook.gradients

    if activations is None or gradients is None:  # pragma: no cover
        raise RuntimeError("Grad-CAM++ hooks captured nothing.")

    grad2 = gradients.pow(2)
    grad3 = grad2 * gradients
    activation_sum = activations.sum(dim=(2, 3), keepdim=True)
    denominator = 2 * grad2 + activation_sum * grad3
    alpha = grad2 / torch.where(denominator.abs() > 1e-12, denominator,
                                torch.full_like(denominator, 1e-12))
    alpha = torch.where(gradients != 0, alpha, torch.zeros_like(alpha))

    weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
    cam = _relu_cam((weights * activations).sum(dim=1))
    cam = _normalise(_resize(cam, inputs.shape[-2:]))
    return CamResult(cam[0].cpu().numpy().astype(np.float32), index, "grad_cam_plus_plus")


@torch.no_grad()
def cbam_spatial_map(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    class_index: int | None = None,
) -> CamResult | None:
    """Read the learned CBAM spatial gate out of a forward pass.

    Returns ``None`` for models without a CBAM/spatial-attention block, which is
    what the UI uses to hide the "CBAM Attention" tab for the baselines.
    """
    blocks = [m for m in model.modules() if isinstance(m, SpatialAttention)]
    if not blocks:
        return None

    model.eval()
    logits = model(inputs)
    index = int(logits.argmax(dim=1).item()) if class_index is None else class_index

    # The deepest block has the most class-specific, least texture-driven gate.
    gate = blocks[-1].last_attention
    if gate is None:  # pragma: no cover - forward always populates it
        return None
    cam = _normalise(_resize(gate.squeeze(1), inputs.shape[-2:]))
    return CamResult(cam[0].cpu().numpy().astype(np.float32), index, "cbam_spatial")


@torch.no_grad()
def cbam_channel_profile(model: torch.nn.Module) -> list[float] | None:
    """Per-channel CBAM gate values from the last forward pass, for the UI chart."""
    from app.ml.attention import ChannelAttention

    blocks = [m for m in model.modules() if isinstance(m, ChannelAttention)]
    if not blocks or blocks[-1].last_attention is None:
        return None
    gate = blocks[-1].last_attention[0].flatten()
    return [round(float(v), 5) for v in gate.cpu().numpy()]


def integrated_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    class_index: int | None = None,
    steps: int = 32,
    baseline: torch.Tensor | None = None,
) -> CamResult:
    """Integrated Gradients (Sundararajan et al., 2017).

    Attribution is the path integral of gradients from a black-image baseline to
    the real input, which satisfies completeness (attributions sum to the score
    difference) - a guarantee Grad-CAM does not offer. Pixel-level and therefore
    noisier, so the UI presents it as a secondary view.
    """
    model.eval()
    base = torch.zeros_like(inputs) if baseline is None else baseline

    with torch.no_grad():
        index = int(model(inputs).argmax(dim=1).item()) if class_index is None else class_index

    total = torch.zeros_like(inputs)
    for step in range(1, steps + 1):
        point = (base + (step / steps) * (inputs - base)).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model(point)[:, index].sum().backward()
        if point.grad is not None:
            total = total + point.grad.detach()

    attribution = (inputs - base) * total / steps
    saliency = attribution.abs().sum(dim=1)
    # Clip the top 1% before normalising; a few extreme pixels otherwise flatten
    # the whole map to near zero.
    cap = torch.quantile(saliency.flatten(1), 0.99, dim=1).view(-1, 1, 1)
    saliency = _normalise(torch.minimum(saliency, cap))
    return CamResult(saliency[0].cpu().numpy().astype(np.float32), index, "integrated_gradients")


# --------------------------------------------------------------------------- #
# Registry
#
# Rendering (colorize / overlay_heatmap / attention_coverage) moved to
# app.ml.heatmap, which has no torch dependency, and is re-exported above.
# --------------------------------------------------------------------------- #


CAM_METHODS: dict[str, Callable] = {
    "grad_cam": grad_cam,
    "grad_cam_plus_plus": grad_cam_plus_plus,
    "integrated_gradients": integrated_gradients,
}


def available_methods(model: torch.nn.Module) -> list[str]:
    """Which explanation tabs make sense for this model."""
    methods = ["grad_cam", "grad_cam_plus_plus"]
    if any(isinstance(m, SpatialAttention) for m in model.modules()):
        methods.append("cbam_spatial")
    methods.append("integrated_gradients")
    return methods
