"""The training engine: one function trains one model under one configuration.

Implements the modern-practice checklist from the brief: AdamW, warm-up plus
cosine (or plateau) LR scheduling, label smoothing, gradient clipping, mixed
precision, early stopping, best-checkpoint saving, TensorBoard logging,
reproducible seeding, and the two-stage transfer-learning schedule.

The trainer is deliberately architecture-agnostic - it only asks the model for
``set_backbone_trainable`` - so every arm of the benchmark and the ablation runs
through exactly the same code path.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TrainConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from app.core.logging import get_logger
from app.ml.architectures import build_model, count_parameters, model_size_mb
from app.ml.datasets import (
    ManifestImageDataset,
    build_balanced_sampler,
    compute_class_weights,
    load_split,
)
from app.ml.transforms import build_eval_transform, build_train_transform

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG the training loop touches.

    ``deterministic`` additionally disables cuDNN autotuning. It is left off by
    default because it costs roughly 15-20% throughput and the benchmark's
    conclusions are drawn from differences far larger than the residual
    non-determinism; the seed alone already makes runs closely reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:  # pragma: no cover - runs in subprocess
    """Give each DataLoader worker a deterministic, distinct seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017).

    Down-weights already-well-classified examples so training focuses on the
    hard, typically minority-class, samples.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else torch.tensor([]))
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.weight.numel() else None
        ce = F.cross_entropy(
            logits, target, weight=weight, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_criterion(config: TrainConfig, class_counts: np.ndarray, device: torch.device) -> nn.Module:
    """Assemble the loss function from the configured imbalance strategy."""
    weight = None
    if config.class_weight_scheme:
        weight = compute_class_weights(class_counts, config.class_weight_scheme).to(device)
    if config.use_focal_loss:
        return FocalLoss(config.focal_gamma, weight, config.label_smoothing).to(device)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=config.label_smoothing)


# --------------------------------------------------------------------------- #
# Schedulers
# --------------------------------------------------------------------------- #


def build_scheduler(optimizer, config: TrainConfig, steps_per_epoch: int):
    """Return ``(scheduler, step_granularity)``.

    Cosine decay with linear warm-up steps every batch; ReduceLROnPlateau steps
    once per epoch on the validation loss.
    """
    if config.scheduler == "cosine":
        total_steps = max(1, config.epochs * steps_per_epoch)
        warmup_steps = max(1, config.warmup_epochs * steps_per_epoch)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            # Floor at 1% of the peak LR so late epochs still make progress.
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), "step"

    if config.scheduler == "plateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.4, patience=2, min_lr=1e-6
            ),
            "epoch",
        )
    return None, "none"


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class TrainResult:
    """What one training run produced."""

    config: TrainConfig
    history: dict
    best_val_accuracy: float
    best_val_loss: float
    best_epoch: int
    epochs_run: int
    training_time_s: float
    checkpoint_path: str
    total_parameters: int
    trainable_parameters: int
    model_size_mb: float
    stopped_early: bool


# --------------------------------------------------------------------------- #
# Data loaders
# --------------------------------------------------------------------------- #


class WorkerStartupError(RuntimeError):
    """A DataLoader worker process could not be started."""


def is_worker_startup_failure(error: BaseException) -> bool:
    """Recognise a *resource* failure in the DataLoader worker pool.

    Each spawned worker re-imports the CUDA build of torch, which reserves
    ~1 GB of Windows commit charge for its DLLs. When the system commit limit is
    reached the worker dies with ``WinError 1455`` and the parent can block
    forever on an empty queue. Falling back to in-process loading recovers from
    that without human intervention.

    The markers below are deliberately specific to memory and process-creation
    failures. A generic "Caught SomeError in DataLoader worker" must *not* match:
    that is how a genuine bug in the dataset or the transform surfaces, and
    silently switching to single-process loading would hide it.
    """
    # The message that matters is usually in the chained cause, not the wrapper.
    parts = []
    current: BaseException | None = error
    seen = 0
    while current is not None and seen < 5:
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
        seen += 1
    text = " ".join(parts).lower()

    markers = (
        # Windows reports error 1455 ("the paging file is too small") in two
        # different phrasings: from the loader failing to import torch at
        # start-up, and from a running worker failing to create the shared
        # memory mapping that carries a collated batch back to the parent.
        "paging file is too small",
        "winerror 1455",
        "error code: <1455>",
        "couldn't open shared file mapping",
        "cannot allocate memory",
        "resource temporarily unavailable",
        "failed to create process",
        "out of memory",
        "not enough memory",
        "cannot allocate",
        "bad allocation",
        "shared memory",
    )
    return any(marker in text for marker in markers)


def build_loaders(
    config: TrainConfig, split_config: dict, device: torch.device
) -> tuple[DataLoader, DataLoader, ManifestImageDataset, ManifestImageDataset]:
    """Construct the train/validation loaders for one configuration."""
    train_tf = build_train_transform(config.image_size, strength=config.augmentation)
    eval_tf = build_eval_transform(config.image_size)

    train_ds = load_split(split_config, config.train_split, train_tf)
    val_ds = load_split(split_config, config.val_split, eval_tf)

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    sampler = None
    shuffle = True
    if config.use_balanced_sampler:
        sampler = build_balanced_sampler(train_ds.labels, len(train_ds.class_names))
        shuffle = False

    common = {
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
        "prefetch_factor": 4 if config.num_workers > 0 else None,
    }
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=shuffle, sampler=sampler,
        drop_last=len(train_ds) > config.batch_size, worker_init_fn=seed_worker,
        generator=generator, **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(config.batch_size, 32), shuffle=False, **common
    )
    return train_loader, val_loader, train_ds, val_ds


# --------------------------------------------------------------------------- #
# Epoch loops
# --------------------------------------------------------------------------- #


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    scaler=None,
    scheduler=None,
    scheduler_granularity: str = "none",
    grad_clip: float | None = None,
    amp: bool = False,
) -> tuple[float, float]:
    """One pass over ``loader``; training when ``optimizer`` is given."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    seen = 0
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                stepped = True
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if grad_clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    # A dropped scale means the step was skipped for inf/nan
                    # gradients; advancing the LR schedule then would desync it.
                    stepped = scaler.get_scale() >= scale_before
                else:
                    loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                if stepped and scheduler is not None and scheduler_granularity == "step":
                    scheduler.step()

        batch = targets.size(0)
        total_loss += loss.item() * batch
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += batch

    return total_loss / max(seen, 1), correct / max(seen, 1)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


def train_model(
    config: TrainConfig,
    split_config: dict,
    checkpoints_dir: Path,
    tensorboard_dir: Path | None = None,
    device: torch.device | None = None,
    progress_every: int = 1,
) -> TrainResult:
    """Train one model end to end, retrying without worker processes if needed."""
    try:
        return _train_model(config, split_config, checkpoints_dir, tensorboard_dir,
                            device, progress_every)
    except Exception as exc:  # noqa: BLE001 - re-raised below unless it is the known case
        if config.num_workers == 0 or not is_worker_startup_failure(exc):
            raise
        log.warning(
            "DataLoader workers hit a resource limit; retrying with in-process loading",
            fields={"model": config.model_name, "workers": config.num_workers,
                    # Log the whole message: a truncated one is useless for
                    # telling a commit-limit failure from a real dataset bug.
                    "error": str(exc)},
        )
        config.num_workers = 0
        return _train_model(config, split_config, checkpoints_dir, tensorboard_dir,
                            device, progress_every)


def _train_model(
    config: TrainConfig,
    split_config: dict,
    checkpoints_dir: Path,
    tensorboard_dir: Path | None = None,
    device: torch.device | None = None,
    progress_every: int = 1,
) -> TrainResult:
    """Train one model end to end and return its result record."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)

    class_names = split_config["class_names"]
    num_classes = len(class_names)

    train_loader, val_loader, train_ds, _ = build_loaders(config, split_config, device)
    class_counts = train_ds.class_counts()

    model = build_model(config.model_name, num_classes, pretrained=config.pretrained).to(device)
    total_params, _ = count_parameters(model)
    size_mb = model_size_mb(model)

    criterion = build_criterion(config, class_counts, device)
    # AMP on a 4 GB card is about memory as much as speed; the GradScaler keeps
    # fp16 gradients from underflowing.
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    writer = None
    if tensorboard_dir is not None:
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(tensorboard_dir / config.experiment_id))

    is_transfer = hasattr(model, "backbone_name")
    two_stage = config.two_stage and is_transfer and config.pretrained
    stage_boundary = config.stage1_epochs if two_stage else None

    if two_stage:
        # Stage 1: the randomly initialised head would otherwise send large,
        # noisy gradients through pretrained weights and destroy them.
        model.set_backbone_trainable(False)
        log.info("Stage 1: backbone frozen", fields={"model": config.model_name})

    def make_optimizer(lr: float):
        params = [p for p in model.parameters() if p.requires_grad]
        if config.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=config.weight_decay)
        if config.optimizer == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=config.weight_decay)
        if config.optimizer == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=config.weight_decay,
                                   nesterov=True)
        raise ValueError(f"Unknown optimizer {config.optimizer!r}")

    optimizer = make_optimizer(config.learning_rate)
    scheduler, granularity = build_scheduler(optimizer, config, len(train_loader))

    history: dict = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [],
                     "val_acc": [], "lr": [], "epoch_time_s": [],
                     "stage_boundary": stage_boundary}

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / f"{config.experiment_id}.pt"

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = config.early_stopping_patience
    stopped_early = False
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        if two_stage and epoch == config.stage1_epochs + 1:
            # Stage 2: unfreeze the upper trunk and drop the LR so the pretrained
            # filters are nudged rather than overwritten.
            model.set_backbone_trainable(True, last_n=config.unfreeze_last_n)
            optimizer = make_optimizer(config.learning_rate * config.stage2_lr_scale)
            remaining = config.epochs - config.stage1_epochs
            stage2 = TrainConfig.from_dict({**config.to_dict(), "epochs": remaining,
                                            "warmup_epochs": 0})
            scheduler, granularity = build_scheduler(optimizer, stage2, len(train_loader))
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info("Stage 2: fine-tuning",
                     fields={"model": config.model_name, "trainable_params": trainable,
                             "lr": config.learning_rate * config.stage2_lr_scale})

        epoch_start = time.perf_counter()
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, scheduler,
            granularity, config.grad_clip, use_amp,
        )
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device, amp=use_amp)
        epoch_time = time.perf_counter() - epoch_start

        if scheduler is not None and granularity == "epoch":
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        history["lr"].append(current_lr)
        history["epoch_time_s"].append(round(epoch_time, 2))

        if writer is not None:
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, epoch)
            writer.add_scalar("lr", current_lr, epoch)

        if epoch % progress_every == 0 or epoch == config.epochs:
            log.info(
                f"epoch {epoch}/{config.epochs}",
                fields={"model": config.model_name, "train_loss": round(train_loss, 4),
                        "train_acc": round(train_acc, 4), "val_loss": round(val_loss, 4),
                        "val_acc": round(val_acc, 4), "lr": f"{current_lr:.2e}",
                        "sec": round(epoch_time, 1)},
            )

        # Model selection on validation accuracy, tie-broken by loss.
        improved = (val_acc > best_val_acc + config.min_delta) or (
            abs(val_acc - best_val_acc) <= config.min_delta and val_loss < best_val_loss
        )
        if improved:
            best_val_acc, best_val_loss, best_epoch = val_acc, val_loss, epoch
            patience_left = config.early_stopping_patience
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.to_dict(),
                    "class_names": class_names,
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("Early stopping",
                         fields={"model": config.model_name, "epoch": epoch,
                                 "best_epoch": best_epoch, "best_val_acc": round(best_val_acc, 4)})
                stopped_early = True
                break

    training_time = time.perf_counter() - started
    if writer is not None:
        writer.close()

    if train_ds.failed_paths:
        log.warning("Unreadable images skipped during training",
                    fields={"count": len(train_ds.failed_paths),
                            "example": train_ds.failed_paths[0]})

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return TrainResult(
        config=config,
        history=history,
        best_val_accuracy=best_val_acc,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        epochs_run=len(history["epoch"]),
        training_time_s=training_time,
        checkpoint_path=str(checkpoint_path),
        total_parameters=total_params,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        model_size_mb=size_mb,
        stopped_early=stopped_early,
    )


def save_history(result: TrainResult, destination: Path) -> None:
    """Write one run's history + summary to JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "experiment_id": result.config.experiment_id,
                "model_name": result.config.model_name,
                "config": result.config.to_dict(),
                "history": result.history,
                "best_val_accuracy": result.best_val_accuracy,
                "best_val_loss": result.best_val_loss,
                "best_epoch": result.best_epoch,
                "epochs_run": result.epochs_run,
                "training_time_s": result.training_time_s,
                "total_parameters": result.total_parameters,
                "model_size_mb": result.model_size_mb,
                "stopped_early": result.stopped_early,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
