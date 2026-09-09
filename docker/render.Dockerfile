# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Render image: the whole application in one container, ONNX Runtime only.
#
# How this differs from cloudrun.Dockerfile, and why
# --------------------------------------------------
# Cloud Run is deployed with 2 GiB, so it can afford the PyTorch serving path.
# Render's free tier gives 512 MB, and the PyTorch path does not fit in it -
# measured on this project, one serving process with torch resident is ~657 MB
# and is OOM-killed before it answers a request.
#
# So this image installs no torch at all. It ships the exported ONNX graph and
# runs it through ONNX Runtime, which brings the same process to ~130 MB. The
# model is unchanged: identical predictions, verified to 2.2e-06 on probability.
#
#   requirements.txt      (torch)  ~657 MB resident, ~1.0 GB image  -> OOM at 512 MB
#   requirements-serve.txt (onnx)  ~130 MB resident, ~450 MB image  -> fits
#
# Build the model bundle on a machine that has torch, BEFORE deploying, and
# commit it - Render builds from the git repo and this image has no torch to
# read a .pt checkpoint:
#     python scripts/export_serving_onnx.py
#
# The dashboard is built here instead, in the Node stage below, because
# frontend/dist is a build artefact and does not belong in git.
# ---------------------------------------------------------------------------

# --- stage 1: the React dashboard -----------------------------------------
# Multi-stage, so Node and node_modules never reach the runtime image.
FROM node:22-slim AS frontend

WORKDIR /build
# Manifests first: this layer is cached unless the dependencies actually change.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


# --- stage 2: the serving image -------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Select the ONNX runtime path. Without this the app imports torch, which is not
# installed in this image and would fail at startup.
ENV PDD_SERVING_BACKEND=onnx \
    PDD_DEVICE=cpu

# A free instance is a fraction of a core, so extra threads buy no throughput
# and each one costs a stack plus a malloc arena. MALLOC_ARENA_MAX caps glibc,
# which otherwise creates up to 8 arenas per core and lets the resident set
# drift upward under load until the container is killed.
#
# These are separate ENV instructions rather than one continued block on
# purpose: a comment between continuation lines is legal but easy to break, and
# this image cannot be built on the development machine to catch that.
ENV PDD_ONNX_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

# opencv-python-headless still links libglib2.0; it does NOT need libgl1, which
# is why this list is shorter than the other images'.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, in their own layer, so application edits do not trigger a
# reinstall on every build.
COPY requirements-serve.txt ./
RUN pip install -r requirements-serve.txt

# --- application ----------------------------------------------------------
COPY backend/ ./backend/

# --- the built dashboard --------------------------------------------------
# From the Node stage above, so the runtime image carries the compiled bundle
# without Node, npm or node_modules.
COPY --from=frontend /build/dist/ ./frontend/dist/

# --- the model bundle -----------------------------------------------------
# Two files, and that is the whole model: the graph plus a manifest carrying
# class names, temperature and OOD thresholds. No .pt checkpoint, so nothing in
# this image needs torch to read it.
COPY models/exported/serving.onnx ./models/exported/serving.onnx
COPY models/exported/serving.json ./models/exported/serving.json
COPY models/exported/production.json ./models/exported/production.json
COPY models/metadata/ ./models/metadata/

# --- content the API serves ------------------------------------------------
COPY data/disease_info.json ./data/disease_info.json
COPY data/splits.json ./data/splits.json
COPY artifacts/dataset_report.json ./artifacts/dataset_report.json
COPY artifacts/results/ ./artifacts/results/
COPY artifacts/plots/ ./artifacts/plots/
COPY artifacts/confusion_matrices/ ./artifacts/confusion_matrices/

RUN mkdir -p /app/uploads /app/data /app/artifacts/plots /app/artifacts/logs \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Render injects PORT; the default here is only for local `docker run`.
ENV PORT=8000
EXPOSE 8000

# Shell form so $PORT expands at run time. One worker deliberately: a second
# worker is a second full copy of the model and the interpreter, which is the
# quickest way back over the 512 MB line.
CMD exec uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8000} --workers 1
