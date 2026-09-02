"""Resolve the plant-disease dataset directory, downloading it only if needed.

The dataset is ~3 GB, so re-fetching it is expensive and requires Kaggle API
credentials. This script therefore treats the *download* as the last resort
rather than the default:

1. ``--path DIR``     - use a dataset directory you already have on disk.
2. ``data/dataset_path.json`` - reuse the location recorded by a previous run.
3. the project-local cache  - a previous download that was never recorded.
4. ``kagglehub``      - fetch from Kaggle. Only this step needs credentials.

Steps 1-3 do not import ``kagglehub``, touch the network, or read any API key,
so a machine that already has the data can run the whole pipeline offline and
without a Kaggle account. Every path is validated before it is accepted: a
directory only counts as the dataset if it actually contains class folders with
images in them, so a typo fails immediately instead of producing an empty
training run.

The default kagglehub cache lives on the system drive which, in this project's
environment, does not have room for the dataset, so KAGGLEHUB_CACHE is pointed
at a project-local directory before kagglehub is imported.

Progress is written as periodic percentage lines rather than a live tqdm bar so
the script is safe to run detached with its output redirected to a log file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "_kagglehub_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("KAGGLEHUB_CACHE", str(CACHE_DIR))
os.environ.setdefault("TQDM_DISABLE", "1")

DATASET_SLUG = "vipoooool/new-plant-diseases-dataset"
ARCHIVE = CACHE_DIR / "datasets" / "vipoooool" / "new-plant-diseases-dataset" / "2.archive"
EXPECTED_BYTES = 2.70 * 1024**3
RECORD_PATH = PROJECT_ROOT / "data" / "dataset_path.json"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# The dataset must have at least this many class folders to be plausible; the
# published version has 38. A lower bound catches a half-extracted archive.
MIN_CLASS_DIRS = 5
# How far below the given root to look for the directory holding class folders.
# The Kaggle layout needs 3; one extra level tolerates an added wrapper folder.
MAX_SEARCH_DEPTH = 4


def _looks_like_image_folder(directory: Path) -> int:
    """Count immediate subdirectories of `directory` that contain images.

    ImageFolder-style datasets are one directory per class. Rather than walking
    the whole 70k-image tree, this stops at the first image in each class
    folder, which is enough to tell a real dataset from an empty shell.
    """
    if not directory.is_dir():
        return 0
    class_dirs = 0
    for child in directory.iterdir():
        if not child.is_dir():
            continue
        for entry in child.iterdir():
            if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES:
                class_dirs += 1
                break
    return class_dirs


def validate_dataset_root(root: Path) -> tuple[bool, str]:
    """Check that `root` contains a usable image-classification dataset.

    Accepts any directory near `root` that holds enough class folders, so the
    same check works for the Kaggle layout and for a user-supplied copy that is
    flattened or re-rooted.
    """
    if not root.exists():
        return False, f"{root} does not exist"
    if not root.is_dir():
        return False, f"{root} is not a directory"

    # The Kaggle archive nests "train" three levels below the version root
    # (versions/2/<name>/<name>/train), so the search has to go deeper than the
    # obvious one or two levels. A breadth-first walk bounded by MAX_SEARCH_DEPTH
    # covers that layout and a flattened user copy without ever descending into
    # the 38 class folders themselves.
    best = 0
    frontier = [root]
    for _ in range(MAX_SEARCH_DEPTH + 1):
        if not frontier:
            break
        next_frontier = []
        for candidate in frontier:
            found = _looks_like_image_folder(candidate)
            best = max(best, found)
            if best >= MIN_CLASS_DIRS:
                return True, f"found {best} class folders in {candidate}"
            next_frontier.extend(child for child in candidate.iterdir() if child.is_dir())
        frontier = next_frontier
    return False, (
        f"{root} contains no directory with at least {MIN_CLASS_DIRS} class "
        f"folders of images (best was {best}). Point --path at the folder that "
        f"holds the per-class image directories."
    )


def record(path: Path, source: str) -> None:
    """Persist the resolved location for every downstream stage to read."""
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(
        json.dumps({"slug": DATASET_SLUG, "path": str(path), "source": source}, indent=2),
        encoding="utf-8",
    )
    print(f"Recorded dataset path in {RECORD_PATH}", flush=True)


def recorded_path() -> Path | None:
    """Return the previously recorded dataset directory, if it is still valid."""
    if not RECORD_PATH.exists():
        return None
    try:
        stored = json.loads(RECORD_PATH.read_text(encoding="utf-8")).get("path")
    except (json.JSONDecodeError, OSError):
        return None
    return Path(stored) if stored else None


def cached_path() -> Path | None:
    """Find a completed download in the project-local kagglehub cache."""
    versions = CACHE_DIR / "datasets" / "vipoooool" / "new-plant-diseases-dataset" / "versions"
    if not versions.is_dir():
        return None
    numbered = sorted(
        (p for p in versions.iterdir() if p.is_dir()),
        key=lambda p: int(p.name) if p.name.isdigit() else -1,
        reverse=True,
    )
    return numbered[0] if numbered else None


def _watch(stop: threading.Event) -> None:
    """Emit one progress line per 30 s so a detached run stays observable."""
    while not stop.wait(30):
        size = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0
        print(
            f"[progress] {size / 1024**2:,.0f} MiB "
            f"({100 * size / EXPECTED_BYTES:.1f}% of expected archive)",
            flush=True,
        )


def download() -> Path:
    """Fetch the dataset from Kaggle. This is the only step needing credentials."""
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "kagglehub is not installed and no local dataset was found.\n"
            "Either install it (pip install kagglehub) and configure Kaggle API\n"
            "credentials, or point the pipeline at a dataset you already have:\n"
            "    python scripts/download_dataset.py --path <dataset-directory>"
        ) from exc

    print(f"KAGGLEHUB_CACHE = {os.environ['KAGGLEHUB_CACHE']}", flush=True)
    print(f"Downloading {DATASET_SLUG} ...", flush=True)
    start = time.time()

    stop = threading.Event()
    watcher = threading.Thread(target=_watch, args=(stop,), daemon=True)
    watcher.start()
    try:
        path = kagglehub.dataset_download(DATASET_SLUG)
    except Exception as exc:  # noqa: BLE001 - surfaced with actionable guidance
        raise SystemExit(
            f"Kaggle download failed: {exc}\n\n"
            "This step is the only one that needs Kaggle API credentials. If you\n"
            "already have the dataset on disk, skip it entirely:\n"
            "    python scripts/download_dataset.py --path <dataset-directory>"
        ) from exc
    finally:
        stop.set()

    print(f"Elapsed: {time.time() - start:,.0f}s", flush=True)
    return Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=None,
                        help="Use this dataset directory instead of downloading. "
                             "No Kaggle credentials or network access are used.")
    parser.add_argument("--force-download", action="store_true",
                        help="Ignore any local copy and re-fetch from Kaggle.")
    args = parser.parse_args()

    if args.path and args.force_download:
        raise SystemExit("--path and --force-download are mutually exclusive.")

    if not args.force_download:
        # Ordered by how explicit the user was about the location.
        sources: list[tuple[str, Path | None]] = [
            ("--path", args.path),
            ("recorded", recorded_path()),
            ("local cache", cached_path()),
        ]
        for source, candidate in sources:
            if candidate is None:
                continue
            ok, detail = validate_dataset_root(candidate)
            if ok:
                print(f"Using existing dataset ({source}): {candidate}", flush=True)
                print(f"  {detail}", flush=True)
                print("  No download performed; no Kaggle credentials required.",
                      flush=True)
                record(candidate, source)
                print("DOWNLOAD_COMPLETE", flush=True)
                return 0
            if source == "--path":
                # An explicit path that fails validation is a user error worth
                # stopping on, not something to silently fall through from.
                raise SystemExit(f"--path is not a usable dataset: {detail}")
            print(f"Ignoring {source} candidate {candidate}: {detail}", flush=True)

    path = download()
    ok, detail = validate_dataset_root(path)
    if not ok:
        raise SystemExit(f"Downloaded dataset failed validation: {detail}")
    print(f"Path to dataset files: {path}", flush=True)
    record(path, "kagglehub")
    print("DOWNLOAD_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
