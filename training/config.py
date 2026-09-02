"""Experiment configuration and hardware-aware defaults.

The training protocol is described declaratively here so that
``artifacts/results/experiments.json`` can record the exact configuration that
produced every number in the report, and so the same run can be reproduced from
the JSON alone.
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

# --------------------------------------------------------------------------- #
# Hardware probing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HardwareProfile:
    """What the machine can actually do, measured rather than assumed."""

    device: str
    gpu_name: str | None
    vram_gb: float
    system_ram_gb: float
    cpu_count: int
    cpu_name: str
    supports_amp: bool
    torch_version: str
    cuda_version: str | None
    # Measured fp32/fp16 forward-time ratio; >1 means half precision is faster.
    amp_speedup: float = 0.0
    # Free commit charge in GiB: the system commit limit (RAM + pagefile) minus
    # what is already committed. This, not physical RAM, is what a spawned
    # DataLoader worker consumes when it re-imports the CUDA build of torch.
    free_commit_gb: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _free_commit_gb() -> float:
    """Commit charge still available, in GiB.

    Physical RAM is the wrong number to budget workers against on Windows. A
    spawned worker re-imports the CUDA build of torch, which *reserves* roughly
    2 GB of commit charge for its DLLs while its resident set stays small - so
    the failure mode is exhausting the commit limit, not exhausting RAM. The
    commit limit is RAM plus the pagefile, and on this machine enlarging the
    pagefile raised it from 31 GB to 48 GB without adding a byte of RAM.

    Returns 0.0 when it cannot be measured, which makes the caller fall back to
    the conservative RAM-based estimate.
    """
    if platform.system() != "Windows":
        return 0.0
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "$m = Get-CimInstance Win32_PerfRawData_PerfOS_Memory; "
             "[math]::Round(($m.CommitLimit - $m.CommittedBytes) / 1GB, 2)"],
            capture_output=True, text=True, timeout=25, check=False,
        )
        return max(0.0, float(out.stdout.strip()))
    except Exception:  # noqa: BLE001
        return 0.0


def _system_ram_gb() -> float:
    """Total physical RAM in GiB (best effort, no psutil dependency)."""
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                capture_output=True, text=True, timeout=25, check=False,
            )
            return round(int(out.stdout.strip()) / 1024**3, 2)
        except Exception:  # noqa: BLE001
            return 0.0
    try:
        import os

        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 2)
    except Exception:  # noqa: BLE001
        return 0.0


def benchmark_amp_speedup(cache_path: Path | None = None) -> float:
    """Measure the real fp16-vs-fp32 speed ratio on this GPU.

    Enabling AMP is normally free speed, but only on hardware with tensor cores.
    On a tensor-core-less card (GTX 16xx, for instance) fp16 convolutions run
    through a slow emulation path and autocast makes training *several times
    slower*. Assuming AMP is a win would silently triple this project's training
    time, so the ratio is measured once and cached.

    Returns ``fp32_time / fp16_time``; values above 1 mean AMP is faster.
    """
    if not torch.cuda.is_available():
        return 0.0
    if cache_path and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("gpu") == torch.cuda.get_device_name(0):
                return float(cached["speedup"])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    import time

    import torch.nn as nn

    device = torch.device("cuda")
    conv = nn.Sequential(
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
    ).to(device)
    sample = torch.randn(16, 32, 112, 112, device=device)

    # The tensors are passed in rather than captured: they are freed immediately
    # afterwards to keep the 4 GB card clear, and a closure over a name that is
    # about to be deleted is a trap waiting for the next edit.
    def _time(block, tensor, use_amp: bool) -> float:
        with torch.no_grad():
            for _ in range(6):
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    block(tensor)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(20):
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    block(tensor)
            torch.cuda.synchronize()
        return time.perf_counter() - start

    speedup = _time(conv, sample, False) / max(_time(conv, sample, True), 1e-9)
    del conv, sample
    torch.cuda.empty_cache()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"gpu": torch.cuda.get_device_name(0), "speedup": speedup}, indent=2),
            encoding="utf-8",
        )
    return speedup


def probe_hardware(amp_cache: Path | None = None) -> HardwareProfile:
    """Detect the compute environment; used to size batches and pick models."""
    import os

    cuda = torch.cuda.is_available()
    gpu_name = None
    vram = 0.0
    supports_amp = False
    amp_speedup = 0.0
    if cuda:
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        vram = round(props.total_memory / 1024**3, 2)
        amp_speedup = benchmark_amp_speedup(amp_cache)
        # Require a real margin: a 1.05x "win" is noise and not worth the
        # GradScaler's extra complexity and occasional skipped steps.
        supports_amp = amp_speedup > 1.15

    return HardwareProfile(
        device="cuda" if cuda else "cpu",
        gpu_name=gpu_name,
        vram_gb=vram,
        system_ram_gb=_system_ram_gb(),
        free_commit_gb=_free_commit_gb(),
        cpu_count=os.cpu_count() or 1,
        cpu_name=platform.processor() or "unknown",
        supports_amp=supports_amp,
        amp_speedup=round(amp_speedup, 3),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )


# Per-model VRAM cost at 224 px with AMP, measured empirically on a 4 GB card and
# expressed as a batch-size ceiling. Used to shrink batches automatically instead
# of crashing with CUDA OOM half-way through a benchmark.
VRAM_BATCH_CEILING: dict[str, dict[float, int]] = {
    # backbone family -> {vram_gb_threshold: max_batch}
    "resnet50": {4.0: 24, 6.0: 40, 8.0: 64, 999.0: 96},
    "densenet121": {4.0: 20, 6.0: 32, 8.0: 56, 999.0: 96},
    "efficientnet_b0": {4.0: 32, 6.0: 48, 8.0: 72, 999.0: 128},
    "efficientnet_b1": {4.0: 24, 6.0: 40, 8.0: 64, 999.0: 96},
    "mobilenet_v3": {4.0: 48, 6.0: 64, 8.0: 96, 999.0: 160},
    "mobilenet_v2": {4.0: 40, 6.0: 56, 8.0: 88, 999.0: 144},
    "custom": {4.0: 64, 6.0: 96, 8.0: 128, 999.0: 192},
}


def _family_for(model_name: str) -> str:
    for family in ("resnet50", "densenet121", "efficientnet_b1", "efficientnet_b0",
                   "mobilenet_v3", "mobilenet_v2"):
        if family in model_name:
            return family
    return "custom"


def auto_batch_size(model_name: str, image_size: int, hardware: HardwareProfile, requested: int) -> int:
    """Clamp the requested batch size to what the detected GPU can hold."""
    if hardware.device == "cpu":
        return min(requested, 16)
    table = VRAM_BATCH_CEILING[_family_for(model_name)]
    ceiling = next(cap for threshold, cap in sorted(table.items()) if hardware.vram_gb <= threshold)
    # Activation memory scales roughly with the square of the input side.
    ceiling = int(ceiling * (224 / image_size) ** 2)
    # The table is calibrated for fp16 activations; full precision needs twice
    # the activation memory, so halve the ceiling when AMP is off.
    if not hardware.supports_amp:
        ceiling //= 2
    return max(4, min(requested, ceiling))


def recommended_workers(hardware: HardwareProfile, override: int | None = None) -> int:
    """How many DataLoader workers this machine can actually afford.

    On Windows the ``spawn`` start method gives every worker a fresh interpreter
    that re-imports torch, and importing the CUDA build reserves on the order of
    2 GB of *commit charge* per process for its DLLs - even though the resident
    set stays small. Exceed the system commit limit and Windows raises
    ``OSError: [WinError 1455] The paging file is too small``, which surfaces as
    a training run that hangs waiting on workers that never started.

    The budget is therefore taken against *free commit charge*, not physical RAM
    - an earlier version named commit charge in this docstring and then measured
    RAM, which capped this machine at one worker even after its pagefile was
    enlarged enough to afford six. The
    Pass ``override`` (from ``PDD_NUM_WORKERS``) to bypass the estimate on a
    machine known to have the commit headroom; the trainer still falls back to
    in-process loading if the workers turn out not to start.
    """
    if override is not None and override >= 0:
        return min(override, max(0, hardware.cpu_count - 1))
    if hardware.device == "cpu":
        return 0

    # Budget against free commit charge when it could be measured, since that is
    # the quantity a worker actually consumes. Keep 6 GB in reserve for the main
    # process, the CUDA context and everything else running.
    if hardware.free_commit_gb > 0:
        affordable = int(max(0.0, hardware.free_commit_gb - 6.0) // 2.0)
    else:
        # Fall back to the old physical-RAM estimate. It is far too pessimistic
        # on a machine with a large pagefile, but it never over-commits.
        affordable = int(max(0.0, hardware.system_ram_gb - 4.0) // 2.0)

    return max(0, min(6, hardware.cpu_count - 2, affordable))


# --------------------------------------------------------------------------- #
# Experiment configuration
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    """Every knob for one training run."""

    model_name: str
    experiment_id: str = ""
    # data
    train_split: str = "train_bench"
    val_split: str = "val_bench"
    test_split: str = "test_bench"
    image_size: int = 160
    batch_size: int = 32
    num_workers: int = 4
    augmentation: str = "light"
    # optimisation
    epochs: int = 14
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"          # cosine | plateau | none
    warmup_epochs: int = 1
    label_smoothing: float = 0.05
    grad_clip: float | None = 1.0
    # two-stage transfer learning
    two_stage: bool = True
    stage1_epochs: int = 3             # frozen backbone, head only
    stage2_lr_scale: float = 0.15      # fine-tuning LR = lr * this
    unfreeze_last_n: int | None = 3    # None unfreezes the whole trunk
    # regularisation / imbalance
    class_weight_scheme: str | None = None
    use_balanced_sampler: bool = False
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    # runtime
    amp: bool = True
    seed: int = 42
    early_stopping_patience: int = 5
    min_delta: float = 1e-4
    pretrained: bool = True
    group: str = "benchmark"
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------- #
# Protocol presets
# --------------------------------------------------------------------------- #


BENCHMARK_PROTOCOL = {
    "train_split": "train_bench",
    "val_split": "val_bench",
    "test_split": "test_bench",
    "image_size": 160,
    "epochs": 12,
    "augmentation": "light",
    "stage1_epochs": 3,
    "early_stopping_patience": 4,
}
"""Identical budget for every architecture in the benchmark and ablation.

Fixing the image size, epoch count, augmentation strength, optimiser and data
subset across all arms is what makes the resulting table a controlled
comparison. Only the architecture changes.
"""

PRODUCTION_PROTOCOL = {
    "train_split": "train",
    "val_split": "val",
    "test_split": "test",
    "image_size": 224,
    # Six epochs, not twelve. The full split has 5.5x more images per class than
    # the benchmark subset, so each epoch carries 5.5x the gradient signal: the
    # subset needed twelve epochs over 300 images per class, the full run reaches
    # the same place in far fewer over 1,660. Early stopping still cuts it short
    # when validation accuracy plateaus, and the epoch budget is recorded in the
    # ledger alongside every result.
    "epochs": 6,
    "augmentation": "medium",
    "stage1_epochs": 2,
    "early_stopping_patience": 2,
}
"""Full-data protocol used to train the models that ship in the application."""


def make_configs(
    model_names: list[str],
    protocol: dict[str, Any],
    hardware: HardwareProfile,
    base_batch_size: int = 32,
    worker_override: int | None = None,
    **overrides: Any,
) -> list[TrainConfig]:
    """Build one :class:`TrainConfig` per model under a shared protocol."""
    configs: list[TrainConfig] = []
    for name in model_names:
        params: dict[str, Any] = {"model_name": name, **protocol, **overrides}
        params["batch_size"] = auto_batch_size(
            name, params.get("image_size", 224), hardware, base_batch_size
        )
        params["amp"] = hardware.supports_amp
        params["num_workers"] = recommended_workers(hardware, worker_override)
        configs.append(TrainConfig(**params))
    return configs


def save_configs(configs: list[TrainConfig], destination: Path) -> None:
    """Persist a list of configs for reproducibility."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([c.to_dict() for c in configs], indent=2), encoding="utf-8"
    )
