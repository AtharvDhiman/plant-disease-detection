"""Attention blocks used by the proposed architecture and the ablation study.

The module implements three families so the ablation in section 12 of the
project brief compares like with like:

* :class:`SEBlock`            - Squeeze-and-Excitation (channel only, one pooling path)
* :class:`ChannelAttention`   - CBAM channel branch (avg **and** max pooling, shared MLP)
* :class:`SpatialAttention`   - CBAM spatial branch
* :class:`CBAM`               - channel branch followed by spatial branch

References
----------
Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.

Every block is residual-multiplicative (``x * sigmoid(attention)``) and keeps the
input shape, so it can be dropped anywhere in a feature pipeline.  Each block
caches its last attention map in ``last_attention`` which the explainability
service reads to render the "CBAM Attention" tab in the UI.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Squeezes each channel to a single number with global average pooling, then
    learns a per-channel gate through a bottleneck MLP.  It answers *what* is
    important but uses only the average-pooled statistic.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        squeeze = F.adaptive_avg_pool2d(x, 1).view(b, c)
        gate = torch.sigmoid(self.fc(squeeze)).view(b, c, 1, 1)
        self.last_attention = gate.detach()
        return x * gate


class ChannelAttention(nn.Module):
    """CBAM channel branch.

    Two descriptors are produced for every channel - global average pooling and
    global max pooling - pushed through a *shared* bottleneck MLP and summed
    before the sigmoid.  Average pooling captures the extent of a feature over
    the leaf, max pooling captures its strongest evidence; using both is what
    distinguishes CBAM's channel branch from SE.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1))
        gate = torch.sigmoid(avg_out + max_out)
        self.last_attention = gate.detach()
        return x * gate


class SpatialAttention(nn.Module):
    """CBAM spatial branch.

    Collapses the channel axis with average and max pooling to a 2-channel map,
    then a single large-kernel convolution learns *where* the informative pixels
    are.  The resulting ``(B, 1, H, W)`` map is exactly what the UI renders as
    the CBAM attention heat-map.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("CBAM spatial attention expects kernel_size in {3, 7}")
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.amax(x, dim=1, keepdim=True)
        gate = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        self.last_attention = gate.detach()
        return x * gate


class CBAM(nn.Module):
    """Convolutional Block Attention Module - channel first, then spatial.

    The sequential ordering follows the original paper, which found it to beat
    both the parallel arrangement and the spatial-first ordering.
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_attention(self.channel_attention(x))

    @property
    def last_channel_attention(self) -> torch.Tensor | None:
        return self.channel_attention.last_attention

    @property
    def last_spatial_attention(self) -> torch.Tensor | None:
        return self.spatial_attention.last_attention


class Identity(nn.Module):
    """No-op attention, used as the control arm of the ablation study."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x


ATTENTION_REGISTRY = {
    "none": lambda channels: Identity(),
    "se": lambda channels: SEBlock(channels),
    "channel": lambda channels: ChannelAttention(channels),
    "spatial": lambda channels: SpatialAttention(),
    "cbam": lambda channels: CBAM(channels),
}


def build_attention(kind: str, channels: int) -> nn.Module:
    """Instantiate an attention block by name (see ``ATTENTION_REGISTRY``)."""
    try:
        return ATTENTION_REGISTRY[kind](channels)
    except KeyError as exc:  # pragma: no cover - guarded by config validation
        raise ValueError(
            f"Unknown attention kind {kind!r}; expected one of {sorted(ATTENTION_REGISTRY)}"
        ) from exc


def collect_spatial_attention_modules(model: nn.Module) -> list[SpatialAttention]:
    """Return every :class:`SpatialAttention` inside ``model``, in forward order."""
    return [m for m in model.modules() if isinstance(m, SpatialAttention)]
