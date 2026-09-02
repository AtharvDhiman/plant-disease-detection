"""Recompute experiment ids after a change to the id-hashing policy.

``ExperimentTracker.make_experiment_id`` hashes the run configuration, so
changing which fields participate in that hash renames every existing
experiment. Rather than discard hours of completed training, this script
recomputes the ids and renames the artefacts that are keyed by them:
checkpoints, history files, curves, confusion matrices and per-model plots.

Run with ``--dry-run`` first to see what it would do.

    python scripts/migrate_experiment_ids.py --dry-run
    python scripts/migrate_experiment_ids.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from tracker import ExperimentTracker  # noqa: E402

from app.core.config import settings  # noqa: E402

PROTOCOL_TAGS = {
    "benchmark": "bench",
    "ablation": "abl",
    "placement": "place",
    "production": "prod",
}


def rename_artifacts(old_id: str, new_id: str, dry_run: bool) -> list[str]:
    """Rename every file whose name embeds the experiment id."""
    targets = [
        settings.checkpoints_dir / f"{old_id}.pt",
        settings.results_dir / f"history_{old_id}.json",
        settings.plots_dir / f"curves_{old_id}.png",
        settings.plots_dir / f"{old_id}_per_class.png",
        settings.plots_dir / f"{old_id}_reliability.png",
        settings.plots_dir / f"{old_id}_confidence.png",
    ]
    targets += list(settings.confusion_dir.glob(f"{old_id}_*.png"))

    moved = []
    for source in targets:
        if not source.exists():
            continue
        destination = source.with_name(source.name.replace(old_id, new_id))
        moved.append(f"{source.name} -> {destination.name}")
        if not dry_run:
            destination.unlink(missing_ok=True)
            source.rename(destination)
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tracker = ExperimentTracker(settings.results_dir)
    if not tracker.records:
        print("Ledger is empty; nothing to migrate.")
        return 0

    changed = 0
    for record in tracker.records:
        config = record.get("config") or {}
        tag = PROTOCOL_TAGS.get(record.get("protocol"), record.get("protocol", "run"))
        new_id = ExperimentTracker.make_experiment_id(record["model_name"], tag, config)
        old_id = record["experiment_id"]
        if new_id == old_id:
            continue

        changed += 1
        print(f"\n{record['label']}")
        print(f"  {old_id}\n  -> {new_id}")
        for line in rename_artifacts(old_id, new_id, args.dry_run):
            print(f"     {line}")

        if args.dry_run:
            continue

        record["experiment_id"] = new_id
        config["experiment_id"] = new_id
        for key in ("checkpoint_path",):
            if record.get(key):
                record[key] = record[key].replace(old_id, new_id)
        test = record.get("test") or {}
        if test.get("checkpoint"):
            test["checkpoint"] = test["checkpoint"].replace(old_id, new_id)
        for plot_key, plot_path in (record.get("plots") or {}).items():
            record["plots"][plot_key] = plot_path.replace(old_id, new_id)
        for plot_key, plot_path in (test.get("plots") or {}).items():
            test["plots"][plot_key] = plot_path.replace(old_id, new_id)

    if changed and not args.dry_run:
        tracker.flush()
        print(f"\nMigrated {changed} experiment(s); ledger rewritten.")
        # The export manifest embeds experiment ids and checkpoint paths, so it
        # goes stale too. Leaving it stale produces an API that starts fine and
        # then 503s on every prediction.
        manifest_path = settings.exported_dir / "production.json"
        if manifest_path.exists():
            manifest_path.unlink()
            print("Removed the stale export manifest. Re-run:")
            print("  python training/export_model.py --criterion f1_macro")
    elif changed:
        print(f"\n[dry run] {changed} experiment(s) would be migrated.")
    else:
        print("All experiment ids already match the current hashing policy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
