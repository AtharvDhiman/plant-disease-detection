"""Real-world photograph support: sources in manifests, per-source augmentation,
fine-tuning from a checkpoint without disturbing past experiment ids, and the
PlantDoc preparation's class mapping and duplicate detection."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "scripts"))

import app.core.runtime  # noqa: F401,E402  # isort:skip
from config import TrainConfig  # noqa: E402

from app.ml.datasets import ManifestEntry, ManifestImageDataset, read_manifest, write_manifest  # noqa: E402


def test_manifest_round_trips_the_source(tmp_path):
    rows = [ManifestEntry("a.jpg", 1, "X", "plantdoc"), ManifestEntry("b.jpg", 2, "Y")]
    write_manifest(rows, tmp_path / "m.csv")
    back = read_manifest(tmp_path / "m.csv")
    assert [r.source for r in back] == ["plantdoc", "plantvillage"]


def test_manifests_written_before_the_source_column_read_as_plantvillage(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("path,label,class_name\nx.jpg,3,Z\n", encoding="utf-8")
    assert read_manifest(path)[0].source == "plantvillage"


def test_dataset_applies_the_transform_for_each_photo_source(tmp_path):
    for name in ("lab.png", "field.png"):
        Image.new("RGB", (16, 16), (90, 140, 70)).save(tmp_path / name)
    dataset = ManifestImageDataset(
        [ManifestEntry(str(tmp_path / "lab.png"), 0, "A"),
         ManifestEntry(str(tmp_path / "field.png"), 1, "B", "plantdoc")],
        transform=lambda image: "lab-transform", class_names=["A", "B"],
    )
    dataset.source_transforms = {"plantdoc": lambda image: "real-transform"}
    assert dataset[0] == ("lab-transform", 0)
    assert dataset[1] == ("real-transform", 1)


def test_unset_init_checkpoint_leaves_the_config_dict_unchanged():
    """Every experiment id is a hash of this dict; a new field that appeared in it
    would silently re-identify, and so retrain, every earlier configuration."""
    plain = TrainConfig(model_name="efficientnet_b0_cbam").to_dict()
    assert "init_checkpoint" not in plain
    tuned = TrainConfig(model_name="efficientnet_b0_cbam", init_checkpoint="x.pt").to_dict()
    assert tuned["init_checkpoint"] == "x.pt"
    assert TrainConfig.from_dict(tuned).init_checkpoint == "x.pt"


def test_recorded_full_data_runs_keep_their_experiment_ids():
    from tracker import ExperimentTracker

    ledger_path = ROOT / "artifacts" / "results" / "experiments.json"
    if not ledger_path.exists():
        pytest.skip("No ledger in this checkout")
    checked = 0
    for record in json.loads(ledger_path.read_text(encoding="utf-8")):
        config = record.get("config") or {}
        if record.get("protocol") not in ("production", "field", "field_long", "field_long_resnet50"):
            continue
        model, tag, _ = record["experiment_id"].split("__")
        rebuilt = ExperimentTracker.make_experiment_id(model, tag, TrainConfig.from_dict(config).to_dict())
        assert rebuilt == record["experiment_id"]
        checked += 1
    assert checked > 0


def test_plantdoc_mapping_targets_are_model_classes():
    from prepare_plantdoc import MAPPING

    splits = ROOT / "data" / "splits.json"
    if not splits.exists():
        pytest.skip("No split configuration in this checkout")
    classes = set(json.loads(splits.read_text(encoding="utf-8"))["class_names"])
    assert set(MAPPING.values()) <= classes
    assert len(MAPPING) == len(set(MAPPING))           # no folder mapped twice


def test_perceptual_hash_matches_a_recompressed_copy_but_not_a_different_photo(tmp_path):
    from prepare_plantdoc import hamming, perceptual_hash

    rng = np.random.default_rng(0)
    base = (np.clip(rng.normal(120, 40, (64, 64, 3)), 0, 255)).astype("uint8")
    base = np.asarray(Image.fromarray(base).resize((256, 256), Image.BILINEAR))
    Image.fromarray(base).save(tmp_path / "a.png")
    Image.fromarray(base).resize((200, 200)).save(tmp_path / "a_small.jpg", quality=60)
    Image.fromarray(np.ascontiguousarray(base[::-1])).save(tmp_path / "flipped.png")

    a, a_small, flipped = (perceptual_hash(tmp_path / n) for n in ("a.png", "a_small.jpg", "flipped.png"))
    distances = hamming(np.array([a_small, flipped], dtype=np.uint64), a)
    assert distances[0] <= 6            # same photo, resized and recompressed
    assert distances[1] > 6             # genuinely different image
    assert perceptual_hash(tmp_path / "missing.png") is None
