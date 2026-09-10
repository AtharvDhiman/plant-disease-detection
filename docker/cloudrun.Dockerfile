# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Cloud Run image: the whole application in one container.
#
# This differs from backend.Dockerfile in three ways that matter, all forced by
# how Cloud Run works:
#
#   1. The weights are baked in. Cloud Run has no persistent volumes, so the
#      compose setup's approach of mounting models/ at run time cannot work.
#   2. The built React bundle is copied in. FastAPI mounts frontend/dist when it
#      exists, which is what lets one container serve the API and the dashboard
#      from a single origin - Cloud Run gives you exactly one port.
#   3. The server binds $PORT, which Cloud Run injects (8080 by default) and
#      does not let you choose. A hardcoded port fails the health check and the
#      revision never goes live.
#
# CPU wheels, deliberately: the model runs in 32 ms on CPU against 17.6 ms on a
# GPU, and Cloud Run's free tier is CPU-only. The CUDA wheels would add ~2.5 GB
# to an image that is otherwise about 1 GB.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PDD_DEVICE=cpu

# libgl / libglib are needed by OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch first, in its own layer: large, and it changes far less often than the
# application code, so this layer stays cached across rebuilds.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY requirements.txt ./
# torch and torchvision are already installed from the CPU index above; letting
# pip resolve them again from PyPI would pull the CUDA build over the top.
RUN grep -v -E "^(torch|torchvision)" requirements.txt > /tmp/reqs.txt \
    && pip install -r /tmp/reqs.txt

# --- application ----------------------------------------------------------
COPY backend/ ./backend/

# --- the built dashboard --------------------------------------------------
# Built on the host with `npm run build`, not here: adding Node to this image
# to rebuild something that is already compiled would roughly double its size.
COPY frontend/dist/ ./frontend/dist/

# --- model, calibration, knowledge base -----------------------------------
# The cloudrun variant of the manifest points at the fixed in-image path
# below; the developer copy holds an absolute path from the build machine.
COPY models/exported/production.cloudrun.json ./models/exported/production.json
COPY models/exported/cloudrun-checkpoint.pt ./models/checkpoints/model.pt
COPY models/metadata/ ./models/metadata/
COPY data/disease_info.json ./data/disease_info.json
COPY data/splits.json ./data/splits.json

# --- results the research pages read --------------------------------------
COPY artifacts/results/experiments.json ./artifacts/results/experiments.json
COPY artifacts/results/summary.json ./artifacts/results/summary.json
COPY artifacts/results/ablation_summary.json ./artifacts/results/ablation_summary.json
COPY artifacts/dataset_report.json ./artifacts/dataset_report.json

RUN mkdir -p /app/uploads /app/artifacts/plots \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Cloud Run injects PORT and ignores EXPOSE; this documents the default.
ENV PORT=8080
EXPOSE 8080

# Shell form so $PORT is expanded at run time rather than taken literally.
CMD exec uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8080}
