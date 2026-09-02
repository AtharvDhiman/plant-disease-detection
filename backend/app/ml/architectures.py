"""Model zoo for the benchmark, the ablation study and production serving.

Two families are provided:

``PlantCNN``
    A from-scratch convolutional network with a pluggable attention block after
    each stage.  Setting ``attention`` to ``none``/``se``/``channel``/``spatial``/
    ``cbam`` gives the five arms of the ablation study while holding every other
    hyper-parameter fixed, which is what makes the comparison a real ablation
    rather than five unrelated models.

``TransferModel``
    An ImageNet-pretrained torchvision backbone with the classifier replaced by
    ``[optional attention] -> GAP -> Dropout -> Linear``.  Freezing is
    controlled by :meth:`TransferModel.set_backbone_trainable` so the two-stage
    transfer-learning schedule (head first, then fine-tune) is explicit.

Every model exposes ``feature_layer`` - the last convolutional feature map -
which Grad-CAM and Grad-CAM++ hook into.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torchvision import models as tv_models

from app.ml.attention import CBAM, build_attention

# --------------------------------------------------------------------------- #
# Custom CNN
# --------------------------------------------------------------------------- #


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> attention -> MaxPool.

    Attention is applied to the activated, normalised feature map and before
    down-sampling, which is where CBAM has the most spatial detail to work with.
    """

    def __init__(self, in_channels: int, out_channels: int, attention: str = "none") -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.attention = build_attention(attention, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.attention(self.conv(x)))


class PlantCNN(nn.Module):
    """Custom CNN backbone with configurable attention placement.

    Parameters
    ----------
    num_classes:
        Number of output classes.
    attention:
        One of ``none``, ``se``, ``channel``, ``spatial``, ``cbam``.
    stage_channels:
        Width of each stage. Four stages at 224 px give a 14x14 final map, a good
        compromise between spatial resolution for the heat maps and memory on a
        4 GB GPU.
    attention_stages:
        Which stages get an attention block - ``all``, ``last``, ``last2`` or
        ``deep`` (the second half).  Used by the CBAM-placement experiment.
    """

    def __init__(
        self,
        num_classes: int,
        attention: str = "none",
        stage_channels: tuple[int, ...] = (32, 64, 128, 256),
        attention_stages: str = "all",
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.attention_kind = attention
        self.attention_stages = attention_stages

        n = len(stage_channels)
        if attention_stages == "all":
            enabled = set(range(n))
        elif attention_stages == "last":
            enabled = {n - 1}
        elif attention_stages == "last2":
            enabled = {n - 2, n - 1}
        elif attention_stages == "deep":
            enabled = set(range(n // 2, n))
        else:
            raise ValueError(f"Unknown attention_stages={attention_stages!r}")

        blocks: list[nn.Module] = []
        in_ch = 3
        for idx, out_ch in enumerate(stage_channels):
            kind = attention if idx in enabled else "none"
            blocks.append(ConvBlock(in_ch, out_ch, attention=kind))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )
        self.feature_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))

    @property
    def feature_layer(self) -> nn.Module:
        """Last convolutional stage - the Grad-CAM target."""
        return self.features[-1].conv

    def set_backbone_trainable(self, trainable: bool, last_n: int | None = None) -> None:
        """Present for API parity with :class:`TransferModel`; trains everything."""
        for param in self.features.parameters():
            param.requires_grad = trainable


# --------------------------------------------------------------------------- #
# Transfer-learning models
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BackboneSpec:
    """How to build a torchvision backbone and where its feature map lives."""

    builder: Callable[..., nn.Module]
    weights: str | None
    feature_channels: int
    trunk: Callable[[nn.Module], nn.Module]


def _resnet_trunk(model: nn.Module) -> nn.Module:
    """ResNet exposes the trunk as its children minus avgpool and fc."""
    return nn.Sequential(*list(model.children())[:-2])


BACKBONES: dict[str, BackboneSpec] = {
    "mobilenet_v3_large": BackboneSpec(
        tv_models.mobilenet_v3_large,
        "MobileNet_V3_Large_Weights.IMAGENET1K_V2",
        960,
        lambda m: m.features,
    ),
    "mobilenet_v3_small": BackboneSpec(
        tv_models.mobilenet_v3_small,
        "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
        576,
        lambda m: m.features,
    ),
    "mobilenet_v2": BackboneSpec(
        tv_models.mobilenet_v2,
        "MobileNet_V2_Weights.IMAGENET1K_V2",
        1280,
        lambda m: m.features,
    ),
    "efficientnet_b0": BackboneSpec(
        tv_models.efficientnet_b0,
        "EfficientNet_B0_Weights.IMAGENET1K_V1",
        1280,
        lambda m: m.features,
    ),
    "efficientnet_b1": BackboneSpec(
        tv_models.efficientnet_b1,
        "EfficientNet_B1_Weights.IMAGENET1K_V2",
        1280,
        lambda m: m.features,
    ),
    "resnet50": BackboneSpec(
        tv_models.resnet50,
        "ResNet50_Weights.IMAGENET1K_V2",
        2048,
        _resnet_trunk,
    ),
    "resnet18": BackboneSpec(
        tv_models.resnet18,
        "ResNet18_Weights.IMAGENET1K_V1",
        512,
        _resnet_trunk,
    ),
    "densenet121": BackboneSpec(
        tv_models.densenet121,
        "DenseNet121_Weights.IMAGENET1K_V1",
        1024,
        lambda m: m.features,
    ),
}


class TransferModel(nn.Module):
    """ImageNet backbone + optional attention + GAP + dropout + linear head."""

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        attention: str = "none",
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}; expected one of {sorted(BACKBONES)}")
        spec = BACKBONES[backbone]
        self.backbone_name = backbone
        self.attention_kind = attention

        weights = spec.weights if pretrained else None
        net = spec.builder(weights=weights)
        self.features = spec.trunk(net)
        self.feature_channels = spec.feature_channels

        # DenseNet's `features` ends with a BatchNorm but no ReLU; without one the
        # pooled descriptor mixes signed activations and the head trains poorly.
        # Not in-place: Grad-CAM registers a backward hook on `features[-1]`, and
        # modifying that hook's output tensor in place is an autograd error.
        self.post = nn.ReLU(inplace=False) if backbone.startswith("densenet") else nn.Identity()

        self.attention = build_attention(attention, self.feature_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(self.post(self.features(x)))
        return self.classifier(self.pool(x))

    @property
    def feature_layer(self) -> nn.Module:
        """Last block of the convolutional trunk - the Grad-CAM target."""
        return self.features[-1]

    def set_backbone_trainable(self, trainable: bool, last_n: int | None = None) -> None:
        """Freeze/unfreeze the trunk.

        ``last_n`` unfreezes only the final ``n`` top-level blocks, which is the
        stage-2 fine-tuning schedule: earlier layers keep their generic ImageNet
        edge/texture filters while the leaf-specific upper layers adapt.
        """
        children = list(self.features.children())
        if last_n is None:
            for param in self.features.parameters():
                param.requires_grad = trainable
            return
        for param in self.features.parameters():
            param.requires_grad = not trainable
        for block in children[-last_n:]:
            for param in block.parameters():
                param.requires_grad = trainable

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSpec:
    """Declarative description of one entry in the benchmark."""

    name: str
    family: str                 # "custom" | "transfer"
    label: str                  # human-readable name for the UI
    kwargs: dict = field(default_factory=dict)
    group: str = "benchmark"    # benchmark | ablation | placement
    proposed: bool = False
    notes: str = ""


MODEL_ZOO: dict[str, ModelSpec] = {
    # ---- benchmark: the eight architectures required by the brief -----------
    "cnn_baseline": ModelSpec(
        "cnn_baseline", "custom", "Custom CNN (baseline)",
        {"attention": "none"}, notes="Model 1 - from-scratch CNN, no attention.",
    ),
    "cnn_se": ModelSpec(
        "cnn_se", "custom", "Custom CNN + SE",
        {"attention": "se"}, notes="Model 2 - Squeeze-and-Excitation channel attention.",
    ),
    "mobilenet_v3_large": ModelSpec(
        "mobilenet_v3_large", "transfer", "MobileNetV3-Large",
        {"backbone": "mobilenet_v3_large", "attention": "none"},
        notes="Model 3 - lightweight ImageNet transfer baseline.",
    ),
    "efficientnet_b0": ModelSpec(
        "efficientnet_b0", "transfer", "EfficientNet-B0",
        {"backbone": "efficientnet_b0", "attention": "none"},
        notes="Model 4 - compound-scaled ImageNet transfer baseline.",
    ),
    "resnet50": ModelSpec(
        "resnet50", "transfer", "ResNet50",
        {"backbone": "resnet50", "attention": "none"},
        notes="Model 5 - deep residual ImageNet transfer baseline.",
    ),
    "densenet121": ModelSpec(
        "densenet121", "transfer", "DenseNet121",
        {"backbone": "densenet121", "attention": "none"},
        notes="Model 6 - densely connected ImageNet transfer baseline.",
    ),
    "cnn_cbam": ModelSpec(
        "cnn_cbam", "custom", "Custom CNN + CBAM",
        {"attention": "cbam"}, proposed=True,
        notes="Model 7 - the proposed from-scratch architecture.",
    ),
    "efficientnet_b0_cbam": ModelSpec(
        "efficientnet_b0_cbam", "transfer", "EfficientNet-B0 + CBAM",
        {"backbone": "efficientnet_b0", "attention": "cbam"}, proposed=True,
        notes="Model 8 - transfer learning combined with CBAM.",
    ),
    "resnet50_cbam": ModelSpec(
        "resnet50_cbam", "transfer", "ResNet50 + CBAM",
        {"backbone": "resnet50", "attention": "cbam"},
        notes="Model 8b - deeper transfer backbone with CBAM.",
    ),
    "mobilenet_v3_large_cbam": ModelSpec(
        "mobilenet_v3_large_cbam", "transfer", "MobileNetV3-Large + CBAM",
        {"backbone": "mobilenet_v3_large", "attention": "cbam"},
        notes="Lightweight transfer backbone with CBAM - deployment candidate.",
    ),
    # ---- ablation: identical CNN, only the attention block changes ----------
    "abl_cnn_none": ModelSpec(
        "abl_cnn_none", "custom", "Ablation A: CNN", {"attention": "none"},
        group="ablation", notes="Control arm.",
    ),
    "abl_cnn_channel": ModelSpec(
        "abl_cnn_channel", "custom", "Ablation B: CNN + Channel attention",
        {"attention": "channel"}, group="ablation",
        notes="CBAM channel branch only.",
    ),
    "abl_cnn_spatial": ModelSpec(
        "abl_cnn_spatial", "custom", "Ablation C: CNN + Spatial attention",
        {"attention": "spatial"}, group="ablation",
        notes="CBAM spatial branch only.",
    ),
    "abl_cnn_se": ModelSpec(
        "abl_cnn_se", "custom", "Ablation D: CNN + SE", {"attention": "se"},
        group="ablation", notes="Squeeze-and-Excitation reference point.",
    ),
    "abl_cnn_cbam": ModelSpec(
        "abl_cnn_cbam", "custom", "Ablation E: CNN + CBAM", {"attention": "cbam"},
        group="ablation", proposed=True, notes="Both branches (proposed).",
    ),
    # ---- CBAM placement experiment -----------------------------------------
    "place_cbam_last": ModelSpec(
        "place_cbam_last", "custom", "CBAM on last stage only",
        {"attention": "cbam", "attention_stages": "last"}, group="placement",
    ),
    "place_cbam_last2": ModelSpec(
        "place_cbam_last2", "custom", "CBAM on last two stages",
        {"attention": "cbam", "attention_stages": "last2"}, group="placement",
    ),
    "place_cbam_all": ModelSpec(
        "place_cbam_all", "custom", "CBAM on every stage",
        {"attention": "cbam", "attention_stages": "all"}, group="placement",
    ),
}


def build_model(name: str, num_classes: int, pretrained: bool = True, **overrides) -> nn.Module:
    """Instantiate a model from :data:`MODEL_ZOO` by name."""
    if name not in MODEL_ZOO:
        raise ValueError(f"Unknown model {name!r}; expected one of {sorted(MODEL_ZOO)}")
    spec = MODEL_ZOO[name]
    kwargs = {**spec.kwargs, **overrides}
    if spec.family == "custom":
        kwargs.pop("pretrained", None)
        return PlantCNN(num_classes=num_classes, **kwargs)
    return TransferModel(num_classes=num_classes, pretrained=pretrained, **kwargs)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model: nn.Module) -> float:
    """Approximate on-disk size of the fp32 state dict, in mebibytes."""
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return (params + buffers) / 1024**2


def has_cbam(model: nn.Module) -> bool:
    """True when the model contains at least one CBAM block."""
    return any(isinstance(m, CBAM) for m in model.modules())
