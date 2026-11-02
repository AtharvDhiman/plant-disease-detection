"""Resuming an interrupted training run: history recovery and schedule position."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

import app.core.runtime  # noqa: F401,E402  # isort:skip
from config import TrainConfig  # noqa: E402
from resume import fast_forward, history_from_log  # noqa: E402
from trainer import build_scheduler  # noqa: E402

EXP = "resnet50_cbam__fieldlongr50__abc"
MODEL = "resnet50_cbam"


def _epoch(number: int, val_acc: float, total: int = 10) -> dict:
    return {"message": f"epoch {number}/{total}", "model": MODEL, "train_loss": 1.0 / number,
            "train_acc": 0.5 + number / 30, "val_loss": 0.9 / number, "val_acc": val_acc,
            "lr": "3.00e-04", "sec": 100.0 + number}


def _log(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "experiments.log"
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return path


def test_history_collects_epochs_after_the_start_line(tmp_path):
    log = _log(tmp_path, [
        {"message": "[1/1] training X", "experiment_id": EXP},
        _epoch(1, 0.80), _epoch(2, 0.90), _epoch(3, 0.95),
    ])
    history = history_from_log(log, EXP, MODEL)
    assert history["epoch"] == [1, 2, 3]
    assert history["val_acc"] == [0.80, 0.90, 0.95]
    assert history["lr"] == [3e-4, 3e-4, 3e-4]


def test_a_resume_line_continues_the_history_instead_of_resetting_it(tmp_path):
    """A fresh start supersedes earlier attempts; a resume must not discard them,
    or a second interruption would lose every epoch before the first one."""
    log = _log(tmp_path, [
        {"message": "[1/1] training X", "experiment_id": EXP},
        _epoch(1, 0.80), _epoch(2, 0.90), _epoch(3, 0.95),
        {"message": "[1/1] resuming X after epoch 3", "experiment_id": EXP},
        {"message": "Continuing from epoch 3", "model": MODEL},
        _epoch(4, 0.96), _epoch(5, 0.97),
    ])
    assert history_from_log(log, EXP, MODEL)["epoch"] == [1, 2, 3, 4, 5]


def test_a_fresh_start_discards_the_abandoned_attempt(tmp_path):
    log = _log(tmp_path, [
        {"message": "[1/1] training X", "experiment_id": EXP},
        _epoch(1, 0.10), _epoch(2, 0.20),
        {"message": "[1/1] training X", "experiment_id": EXP},
        _epoch(1, 0.80),
    ])
    history = history_from_log(log, EXP, MODEL)
    assert history["epoch"] == [1]
    assert history["val_acc"] == [0.80]


def test_a_repeated_epoch_line_is_not_counted_twice(tmp_path):
    log = _log(tmp_path, [
        {"message": "[1/1] training X", "experiment_id": EXP},
        _epoch(1, 0.80), _epoch(1, 0.80), _epoch(2, 0.90),
    ])
    assert history_from_log(log, EXP, MODEL)["epoch"] == [1, 2]


def test_the_next_experiment_ends_collection(tmp_path):
    log = _log(tmp_path, [
        {"message": "[1/2] training X", "experiment_id": EXP},
        _epoch(1, 0.80),
        {"message": "[2/2] training Y", "experiment_id": "other__id"},
        _epoch(1, 0.50),
    ])
    assert history_from_log(log, EXP, MODEL)["val_acc"] == [0.80]


def test_unrelated_experiments_and_models_are_ignored(tmp_path):
    other = _epoch(1, 0.33)
    other["model"] = "efficientnet_b0"
    log = _log(tmp_path, [
        {"message": "[1/1] training X", "experiment_id": EXP}, other, _epoch(1, 0.80),
    ])
    assert history_from_log(log, EXP, MODEL)["val_acc"] == [0.80]


@pytest.mark.parametrize("steps", [0, 1, 7, 250, 1999])
def test_fast_forward_reproduces_the_uninterrupted_learning_rate(steps):
    """The schedule is a pure function of the step count, so resuming at step N
    must land on exactly the learning rate an uninterrupted run had at step N."""
    config = TrainConfig(model_name="resnet50_cbam", epochs=8, warmup_epochs=0)

    def fresh():
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=4.5e-5)
        scheduler, granularity = build_scheduler(optimizer, config, steps_per_epoch=250)
        assert granularity == "step"
        return optimizer, scheduler

    continuous_optimizer, continuous = fresh()
    for _ in range(steps):
        continuous_optimizer.step()
        continuous.step()

    resumed_optimizer, resumed = fresh()
    fast_forward(resumed, steps)
    assert resumed.get_last_lr() == pytest.approx(continuous.get_last_lr(), rel=1e-12)
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(
        continuous_optimizer.param_groups[0]["lr"], rel=1e-12)


def test_fast_forward_tolerates_no_scheduler():
    fast_forward(None, 100)      # the plateau/none case must not raise
