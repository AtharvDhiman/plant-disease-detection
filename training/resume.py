"""Resume an interrupted training run from its best checkpoint.

Why this exists
---------------
This machine is sometimes powered off or restarted in the middle of a
multi-hour run. ``trainer`` writes the best-so-far checkpoint whenever
validation improves, so most of the work is on disk - but a plain rerun starts
again from epoch 1. Resuming continues from the saved weights instead.

What is and is not restored
---------------------------
* **Restored exactly:** the weights, the epoch counter, the best-so-far
  validation score, the stage the run was in (frozen head / fine-tuning), the
  trainable-parameter set, and the position in the learning-rate schedule
  (it is a pure function of the step count, so it is fast-forwarded).
* **Not restored:** the optimiser's moment estimates, because checkpoints have
  never stored them. AdamW restarts with empty moments, which costs a few
  hundred steps of slightly noisier updates at the learning rate in force.
  Nor is the DataLoader's random stream: it is re-seeded from the seed and the
  resume epoch, so the resumed epochs get fresh shuffles and augmentations.
  (Before 2026-11-08 it restarted from the seed alone, and each resumed epoch
  replayed an earlier one's draws - which showed as training accuracy creeping
  to 99% while validation still improved.)
  A resumed run is therefore *very close to* an uninterrupted one, not
  identical to it, and the ledger record says so.
* **Reconstructed:** the per-epoch history before the interruption, from the
  training log (four decimal places, since that is what the log carries).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

EMPTY_HISTORY_KEYS = ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr",
                      "epoch_time_s")


def history_from_log(log_path: Path, experiment_id: str, model_name: str) -> dict:
    """Epoch rows the log holds for one experiment.

    A line announcing a *fresh* start (``... training ...``) discards anything
    collected so far, because a rerun from scratch supersedes the earlier
    attempt. A line announcing a *resume* (``... resuming ...``) does not: it
    continues the epochs already collected.
    """
    history: dict = {key: [] for key in EMPTY_HISTORY_KEYS}
    started = False
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        message = str(entry.get("message", ""))
        if entry.get("experiment_id") == experiment_id:
            if "resuming" in message:
                started = True
                continue
            if "training" in message:
                started = True
                history = {key: [] for key in EMPTY_HISTORY_KEYS}
                continue
        if not started:
            continue
        if message.startswith("[") and ("training" in message or "resuming" in message) \
                and entry.get("experiment_id") != experiment_id:
            break                                        # the next experiment began
        if message.startswith("epoch ") and entry.get("model") == model_name:
            number = int(message.split()[1].split("/")[0])
            if number in history["epoch"]:
                continue                                 # a repeated line, not a new epoch
            history["epoch"].append(number)
            history["train_loss"].append(entry["train_loss"])
            history["train_acc"].append(entry["train_acc"])
            history["val_loss"].append(entry["val_loss"])
            history["val_acc"].append(entry["val_acc"])
            history["lr"].append(float(entry["lr"]))
            history["epoch_time_s"].append(entry["sec"])
    return history


def fast_forward(scheduler, steps: int) -> None:
    """Advance a step-based scheduler to where an earlier run left it.

    ``LambdaLR`` computes the learning rate from the step count alone, so
    stepping it ``steps`` times reproduces the position exactly. PyTorch warns
    when a scheduler is stepped before its optimiser; that is expected here.
    """
    if scheduler is None or steps <= 0:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(steps):
            scheduler.step()
