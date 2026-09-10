"""Assemble a self-contained Hugging Face Space from this project.

A Space is its own git repository. It cannot import from the project around it,
so everything the app touches at runtime has to be copied in: the ML package,
the exported weights, the calibrated thresholds and the knowledge base.

What is deliberately *not* copied is as important as what is. The dataset, the
training code, the experiment ledger, the checkpoints of the other seventeen
models and every artefact of the research all stay behind. The Space serves one
model; shipping the rest would push a 3 GB repository to host a 16 MB network.

The result is written to a directory you then push with git. Nothing is uploaded
from here - pushing is yours to do, with your own credentials.

    python scripts/build_space.py
    python scripts/build_space.py --out ../plantguard-space
"""
from __future__ import annotations

import argparse
import json
import shutil
import stat
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import settings  # noqa: E402

README = """---
title: PlantGuard AI
emoji: 🌿
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: Plant disease detection with CBAM attention
suggested_hardware: cpu-basic
---

# PlantGuard AI

Plant disease detection from a leaf photograph, using a CNN with a Convolutional
Block Attention Module (CBAM). Covers {classes} conditions across 14 plant
species.

The point of the system is not only the classifier. It reports a **calibrated**
confidence rather than a raw softmax score, shows **where** the network looked
via Grad-CAM, and **declines to answer** when the image is unusable or unlike
anything it was trained on.

## Measured behaviour

| | |
|---|---:|
| Test accuracy ({model}) | {accuracy} |
| Macro F1 | {f1} |
| Test images | {test_n} |
| Inference (CPU) | ~32 ms |

Out-of-distribution inputs are rejected at **95%**, at a cost of **2%** of
genuine leaves being asked for a retake.

## Limitations, stated plainly

* It knows {classes} classes. A crop outside that set has no correct answer
  available; it should refuse, and usually does.
* Training images are single leaves on plain backgrounds. A measured
  domain-shift test found accuracy falls from 100% to 13% when the leaf occupies
  only a third of the frame — framing matters far more than background or
  lighting. Photograph one leaf, filling the frame.
* Educational only. Not a professional agricultural diagnosis.

Full research results — an 18-model benchmark, a five-arm attention ablation
judged against a measured noise floor, and a CBAM placement study that
replicated at full data scale — are in the
[source repository](https://github.com/AtharvDhiman/plant-disease-detection).
"""

REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.13.0
torchvision==0.28.0
fastapi>=0.115
uvicorn[standard]>=0.30
python-multipart>=0.0.9
sqlalchemy>=2.0
pydantic>=2.6
pydantic-settings>=2.2
pillow>=10.0
numpy>=1.26
scipy>=1.11
opencv-python-headless>=4.9
# Present only so the Space also starts on ZeroGPU hardware, which refuses to
# boot without a @spaces.GPU function anywhere. Unused on CPU hardware.
gradio>=5.9
spaces>=0.30
"""

GITATTRIBUTES = """*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
"""


def copy_tree(source: Path, destination: Path, *, skip: set[str]) -> int:
    """Copy a package, leaving out caches and anything named in `skip`."""
    copied = 0
    for path in source.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(source)
        if any(part in skip or part == "__pycache__" for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "deploy" / "space-build")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the output directory if it exists.")
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists():
        if not args.force and any(out.iterdir()):
            raise SystemExit(
                f"{out} already exists and is not empty. Pass --force to replace it."
            )
        # Git marks objects read-only, and Windows refuses to unlink those,
        # so a plain rmtree fails on any directory that has been git init'd.
        def _force(func, path, _exc):
            Path(path).chmod(stat.S_IWRITE)
            func(path)

        shutil.rmtree(out, onexc=_force)
    out.mkdir(parents=True)

    manifest_path = settings.exported_dir / "production.json"
    if not manifest_path.exists():
        raise SystemExit(
            "No exported model. Run: python training/export_model.py --criterion f1_macro"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint missing: {checkpoint}")

    # --- application ------------------------------------------------------
    shutil.copy2(PROJECT_ROOT / "deploy" / "space" / "app.py", out / "app.py")
    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (out / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")

    # --- the whole backend ------------------------------------------------
    # Everything, including the API routes and schemas: the Space runs the real
    # service, so the dashboard's endpoints have to be there.
    files = copy_tree(PROJECT_ROOT / "backend" / "app", out / "backend" / "app",
                      skip=set())

    # --- the built React dashboard ----------------------------------------
    # main.py mounts frontend/dist when it exists, which is what turns two
    # services into one origin. Without this the Space would serve JSON.
    dist = PROJECT_ROOT / "frontend" / "dist"
    if not (dist / "index.html").is_file():
        raise SystemExit(
            "frontend/dist is missing or empty. Build it first:\n"
            "    cd frontend && npm run build"
        )
    files += copy_tree(dist, out / "frontend" / "dist", skip=set())

    # --- weights and calibration -----------------------------------------
    weights_dir = out / "models" / "checkpoints"
    weights_dir.mkdir(parents=True)
    shutil.copy2(checkpoint, weights_dir / checkpoint.name)

    exported_dir = out / "models" / "exported"
    exported_dir.mkdir(parents=True)
    # Rewrite the checkpoint path: it is absolute and points at this machine.
    # POSIX separators, explicitly. str(Path(...)) yields backslashes on Windows,
    # which are not path separators on the Linux host that runs the Space - the
    # first deploy failed with "Checkpoint not found" for exactly this reason.
    manifest["checkpoint"] = f"models/checkpoints/{checkpoint.name}"
    (exported_dir / "production.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    metadata_dir = out / "models" / "metadata"
    metadata_dir.mkdir(parents=True)
    for name in ("ood_thresholds.json", f"ood_{manifest['model_name']}.json"):
        source = settings.metadata_dir / name
        if source.is_file():
            shutil.copy2(source, metadata_dir / name)

    # --- knowledge base ---------------------------------------------------
    data_dir = out / "data"
    data_dir.mkdir(parents=True)
    knowledge = PROJECT_ROOT / "data" / "disease_info.json"
    if not knowledge.is_file():
        raise SystemExit("data/disease_info.json missing - run scripts/build_disease_info.py")
    shutil.copy2(knowledge, data_dir / "disease_info.json")
    entries = json.loads(knowledge.read_text(encoding="utf-8"))

    # --- results the research pages read ----------------------------------
    # The benchmark, ablation and calibration pages are generated from these.
    # They are small JSON summaries, not the raw experiment outputs.
    results_out = out / "artifacts" / "results"
    results_out.mkdir(parents=True)
    for name in ("experiments.json", "summary.json", "ablation_summary.json",
                 "serving_benchmark.json", "domain_shift.json"):
        source = PROJECT_ROOT / "artifacts" / "results" / name
        if source.is_file():
            shutil.copy2(source, results_out / name)

    report = PROJECT_ROOT / "artifacts" / "dataset_report.json"
    if report.is_file():
        shutil.copy2(report, out / "artifacts" / "dataset_report.json")

    splits = PROJECT_ROOT / "data" / "splits.json"
    if splits.is_file():
        shutil.copy2(splits, data_dir / "splits.json")

    ood_plot_dir = out / "artifacts" / "plots"
    ood_plot_dir.mkdir(parents=True, exist_ok=True)

    # --- readme, with the real numbers -----------------------------------
    metrics = (manifest.get("metrics") or {})
    accuracy = metrics.get("accuracy")
    f1 = metrics.get("f1_macro")
    (out / "README.md").write_text(README.format(
        classes=entries.get("class_count", len(entries.get("diseases", {}))),
        model=manifest.get("label", manifest.get("model_name", "-")),
        accuracy=f"{accuracy * 100:.2f}%" if accuracy else "see repository",
        f1=f"{f1:.4f}" if f1 else "see repository",
        test_n=f"{metrics.get('num_samples', 0):,}" if metrics.get("num_samples") else "held-out split",
    ), encoding="utf-8")

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"Built {out}")
    print(f"  {files} package files, {total / 1048576:.1f} MB total")
    print(f"  model: {manifest.get('label')} ({checkpoint.name})")
    print("\nPush it:")
    print(f"  cd {out}")
    print("  git init && git remote add origin "
          "https://huggingface.co/spaces/AtharvDhiman/plantguard-ai")
    print("  git add -A && git commit -m 'PlantGuard AI'")
    print("  git push --force origin main")
    return 0


if __name__ == "__main__":
    sys.exit(main())
