"""Deploy the application to Google Cloud Run.

Cloud Run builds the image from `docker/cloudrun.Dockerfile` and runs one
container that serves both the API and the React dashboard. There is no local
Docker step: `gcloud run deploy --source` uploads the build context and builds
it in the cloud.

Before the first run this script re-stages two files that the image needs and
that cannot be committed as-is:

* the production checkpoint, copied to a fixed name, because the developer
  manifest holds an absolute path from whichever machine trained the model;
* a manifest pointing at that fixed in-image path.

It also checks the frontend bundle exists. Forgetting `npm run build` produces a
container that starts cleanly and serves JSON where the dashboard should be,
which is a slow thing to notice.

    python scripts/deploy_cloudrun.py --project my-gcp-project
    python scripts/deploy_cloudrun.py --project my-gcp-project --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import settings  # noqa: E402

# 1 GiB covers the measured 945 MB footprint with room for a request in flight.
# Cloud Run bills per request-second, so a larger ceiling costs nothing while
# idle and simply avoids the container being killed under load.
DEFAULT_MEMORY = "2Gi"
DEFAULT_REGION = "us-central1"


def stage_model() -> tuple[Path, dict]:
    """Copy the checkpoint to a fixed name and rewrite the manifest path."""
    manifest_path = settings.exported_dir / "production.json"
    if not manifest_path.exists():
        raise SystemExit(
            "No exported model. Run:\n"
            "    python training/export_model.py --criterion f1_macro"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint missing: {checkpoint}")

    staged = settings.exported_dir / "cloudrun-checkpoint.pt"
    shutil.copy2(checkpoint, staged)

    manifest["checkpoint"] = "models/checkpoints/model.pt"
    (settings.exported_dir / "production.cloudrun.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return staged, manifest


def check_frontend() -> None:
    index = PROJECT_ROOT / "frontend" / "dist" / "index.html"
    if not index.is_file():
        raise SystemExit(
            "frontend/dist is missing. The container would serve the API with no\n"
            "dashboard. Build it first:\n"
            "    cd frontend && npm run build"
        )


def gcloud_available() -> bool:
    try:
        subprocess.run(["gcloud", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Google Cloud project id.")
    parser.add_argument("--service", default="plantguard-ai")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--cpu", default="2")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stage everything and print the command, without deploying.")
    args = parser.parse_args()

    check_frontend()
    staged, manifest = stage_model()
    print(f"Staged {staged.name} ({staged.stat().st_size / 1048576:.1f} MB)")
    print(f"Model: {manifest.get('label')} - {manifest.get('model_name')}")

    command = [
        "gcloud", "run", "deploy", args.service,
        "--source", ".",
        "--project", args.project,
        "--region", args.region,
        "--platform", "managed",
        "--allow-unauthenticated",
        "--memory", args.memory,
        "--cpu", args.cpu,
        # The model loads at startup and takes a few hundred milliseconds; a
        # generous timeout stops a cold start being mistaken for a failure.
        "--timeout", "300",
        # Scale to zero when idle: nothing runs, nothing is billed.
        "--min-instances", "0",
        "--max-instances", "3",
        # Cloud Run injects PORT; the image reads it. Naming it here documents
        # the contract rather than relying on the default staying 8080.
        "--port", "8080",
        "--set-env-vars", "PDD_DEVICE=cpu,PDD_DATABASE_URL=sqlite:////tmp/pg.db,"
                          "PDD_UPLOAD_DIR=/tmp/pg-uploads",
    ]

    print("\nCommand:")
    print("  " + " ".join(command))

    if args.dry_run:
        print("\n(dry run - nothing deployed)")
        return 0

    if not gcloud_available():
        raise SystemExit(
            "\ngcloud is not installed. Install the Google Cloud CLI, then:\n"
            "    gcloud auth login\n"
            "    gcloud config set project <your-project-id>\n"
            "and run this script again."
        )

    print("\nDeploying - the first build takes about 10 minutes.\n")
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode == 0:
        print("\nDeployed. The service URL is printed above.")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
