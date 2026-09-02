# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Backend image: FastAPI + PyTorch (CPU wheels).
#
# CPU wheels are the default because they are ~200 MB instead of ~2.5 GB and
# most free/small hosts have no GPU. To build a CUDA image, override the torch
# index at build time:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 ...
# and run the container with `--gpus all`.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgl / libglib are needed by OpenCV; curl is used by the container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch separately so the (large, rarely changing) layer is cached
# independently of the application requirements.
RUN pip install --index-url ${TORCH_INDEX} torch torchvision

COPY requirements.txt .
# torch/torchvision are already installed from the index above; installing them
# again from PyPI would pull a different build.
RUN grep -v -E "^(torch|torchvision)" requirements.txt > /tmp/reqs.txt \
    && pip install -r /tmp/reqs.txt

COPY backend/ ./backend/
COPY data/disease_info.json ./data/disease_info.json

# Model weights and artefacts are mounted at run time (see docker-compose.yml)
# rather than baked in: they are large and change independently of the code.
RUN mkdir -p /app/models/exported /app/models/checkpoints /app/models/metadata \
             /app/artifacts/results /app/artifacts/plots /app/artifacts/logs \
             /app/uploads /app/data

# Run as a non-root user: the process only ever needs to read the checkpoint and
# write to uploads/ and data/.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
