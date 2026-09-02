"""Add a second dataset to the project, from Kaggle or from a local folder.

Why this exists
---------------
The model is a fixed-width classifier: its final layer has exactly as many
outputs as the dataset had classes. It cannot name a disease it was not trained
on, so "make it recognise wheat" always means "retrain on data that includes
wheat". This script does the part before the retrain - acquiring the data,
checking it is usable, and reporting precisely what work the new classes create.

It deliberately stops short of retraining. Discovering that a dataset is
unusable after six hours of GPU time is the failure this avoids: everything
cheap and fallible happens first, and the expensive step is only reached once
the data has been shown to be sound.

What it reports
---------------
* the classes found, and how many images each has
* classes that collide with ones already in the project
* classes with too few images to train on
* which knowledge-base entries you will have to write, since the project
  refuses to invent agricultural facts

Usage
-----
    python scripts/add_dataset.py --kaggle <owner>/<dataset-slug>
    python scripts/add_dataset.py --path "D:/data/wheat-leaf-disease"
    python scripts/add_dataset.py --path ... --plant Wheat   # prefix bare classes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "_kagglehub_cache"
os.environ.setdefault("KAGGLEHUB_CACHE", str(CACHE_DIR))
os.environ.setdefault("TQDM_DISABLE", "1")

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from download_dataset import IMAGE_SUFFIXES, validate_dataset_root  # noqa: E402

# Below this, a class cannot support a train/val/test split that means anything.
MIN_IMAGES_PER_CLASS = 40


def kaggle_credentials_present() -> tuple[bool, str]:
    """Report whether a Kaggle token exists, without reading its contents.

    The token is never opened or logged here. Only its presence matters, and
    printing any part of a credential to a build log would be a mistake that is
    hard to undo.
    """
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True, "KAGGLE_USERNAME / KAGGLE_KEY environment variables"
    token = Path.home() / ".kaggle" / "kaggle.json"
    if token.exists():
        return True, str(token)
    return False, (
        "No Kaggle credentials found. On kaggle.com go to Settings -> API ->\n"
        "  'Create New Token', which downloads kaggle.json, then save it at:\n"
        f"    {token}\n"
        "  Or skip Kaggle entirely and use --path with a folder you already have."
    )


def fetch_from_kaggle(slug: str) -> Path:
    """Download a dataset, failing with guidance rather than a stack trace."""
    ok, detail = kaggle_credentials_present()
    if not ok:
        raise SystemExit(detail)
    print(f"Credentials: {detail}", flush=True)

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit("kagglehub is not installed: pip install kagglehub") from exc

    print(f"Downloading {slug} ...", flush=True)
    try:
        return Path(kagglehub.dataset_download(slug))
    except Exception as exc:  # noqa: BLE001 - reported with context
        raise SystemExit(
            f"Download failed: {exc}\n\n"
            "If this is an authentication error the token may be expired - "
            "create a new one from Settings -> API on kaggle.com."
        ) from exc


def find_class_root(root: Path, max_depth: int = 4) -> Path | None:
    """Locate the directory holding one folder per class.

    Datasets are packaged inconsistently: some put the class folders at the top,
    some nest them under train/, some repeat the dataset name twice. Rather than
    require a fixed layout, this looks for the directory that actually contains
    class folders.
    """
    best: tuple[int, Path] | None = None
    frontier = [root]
    for _ in range(max_depth + 1):
        if not frontier:
            break
        nxt = []
        for candidate in frontier:
            if not candidate.is_dir():
                continue
            count = sum(
                1 for child in candidate.iterdir()
                if child.is_dir() and any(
                    entry.suffix.lower() in IMAGE_SUFFIXES
                    for entry in child.iterdir() if entry.is_file()
                )
            )
            if count >= 2 and (best is None or count > best[0]):
                best = (count, candidate)
            nxt.extend(child for child in candidate.iterdir() if child.is_dir())
        frontier = nxt
    return best[1] if best else None


def count_images(class_dir: Path) -> int:
    return sum(1 for entry in class_dir.rglob("*")
               if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES)


def existing_classes() -> list[str]:
    report = PROJECT_ROOT / "artifacts" / "dataset_report.json"
    if not report.exists():
        return []
    return json.loads(report.read_text(encoding="utf-8"))["classes"]["names"]


def curated_classes() -> set[str]:
    try:
        from build_disease_info import DISEASES
    except Exception:  # noqa: BLE001 - the report is still useful without it
        return set()
    return set(DISEASES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--kaggle", metavar="OWNER/SLUG",
                        help="Kaggle dataset to download. Needs credentials.")
    source.add_argument("--path", type=Path,
                        help="A dataset directory you already have. No credentials used.")
    parser.add_argument("--plant", default=None,
                        help="Prefix bare class folders as Plant___Condition, for "
                             "datasets that name folders 'Leaf_rust' rather than "
                             "'Wheat___Leaf_rust'.")
    parser.add_argument("--min-images", type=int, default=MIN_IMAGES_PER_CLASS)
    args = parser.parse_args()

    root = fetch_from_kaggle(args.kaggle) if args.kaggle else args.path
    print(f"\nDataset root: {root}")

    ok, detail = validate_dataset_root(root)
    if not ok:
        raise SystemExit(f"Not a usable dataset: {detail}")

    class_root = find_class_root(root)
    if class_root is None:
        raise SystemExit("Could not locate a directory containing class folders.")
    print(f"Class folders: {class_root}\n")

    counts = Counter()
    for child in sorted(class_root.iterdir()):
        if child.is_dir():
            counts[child.name] = count_images(child)

    known = set(existing_classes())
    curated = curated_classes()

    def canonical(name: str) -> str:
        if args.plant and "___" not in name:
            return f"{args.plant}___{name}"
        return name

    print(f"{'CLASS':<44}{'IMAGES':>8}  STATUS")
    print("-" * 78)
    thin, collisions, to_write = [], [], []
    for name, count in counts.most_common():
        full = canonical(name)
        notes = []
        if full in known:
            notes.append("already in project")
            collisions.append(full)
        if count < args.min_images:
            notes.append(f"too few (<{args.min_images})")
            thin.append(full)
        if full not in curated and full not in known:
            to_write.append(full)
        print(f"{full[:44]:<44}{count:>8}  {', '.join(notes) or 'new'}")

    total = sum(counts.values())
    print("-" * 78)
    print(f"{len(counts)} classes, {total:,} images\n")

    print("NEXT STEPS")
    if collisions:
        print(f"  ! {len(collisions)} class(es) already exist in the project. Merging "
              f"needs a decision:\n    keep both sources, or replace the existing "
              f"images. Nothing is merged automatically.")
    if thin:
        print(f"  ! {len(thin)} class(es) have fewer than {args.min_images} images and "
              f"will train poorly:\n    {', '.join(thin[:5])}"
              f"{' ...' if len(thin) > 5 else ''}")
    if to_write:
        print(f"  * Write {len(to_write)} knowledge-base entr(ies) in "
              f"scripts/build_disease_info.py.")
        print("    The project refuses to invent agricultural facts, so each needs "
              "symptoms,\n    causes, favourable conditions, prevention and management "
              "from a real source\n    (university extension services, APS compendia, "
              "FAO guidance).")
        for name in to_write[:8]:
            print(f"      - {name}")
        if len(to_write) > 8:
            print(f"      ... and {len(to_write) - 8} more")
    print("\n  Then, to retrain on the combined data:")
    print(f'    python scripts/download_dataset.py --path "{root}"')
    print("    python training/dataset_inspect.py && python training/prepare_splits.py")
    print("    python scripts/run_pipeline.py --from knowledge")
    return 0


if __name__ == "__main__":
    sys.exit(main())
