"""FastAPI application entry point.

The model is loaded and warmed up exactly once, in the lifespan startup hook.
Requests never load weights, and the process never trains: training lives
entirely in ``training/`` and communicates with the API only through the
exported checkpoint and the JSON artefacts.

Run:
    uvicorn app.main:app --reload --app-dir backend
"""
from __future__ import annotations

import app.core.runtime  # noqa: F401  (sets OMP env before torch)  # isort:skip

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import dashboard, predictions, research
from app.core.config import settings
from app.core.database import init_db
from app.core.logging import configure_logging, get_logger
from app.ml.registry import registry
from app.services.knowledge import knowledge_base

log = get_logger(__name__)

DESCRIPTION = """
AI-powered plant disease detection from leaf images, using a CNN with a
Convolutional Block Attention Module (CBAM).

**Pipeline for every prediction**

1. Image-quality analysis (sharpness, exposure, contrast, plant-colour coverage)
2. Deterministic preprocessing identical to validation-time preprocessing
3. Inference with the exported production model
4. Temperature-scaled confidence and HIGH/MEDIUM/LOW banding
5. Out-of-distribution assessment (max softmax probability, entropy, free energy)
6. Explainability: Grad-CAM, Grad-CAM++ and the model's own CBAM spatial attention

**Reported numbers** come from the experiment ledger in `artifacts/results/` and
were produced by the training pipeline. Nothing in this API is hard-coded.

**Disease information** is educational and general; it is not a professional
agricultural diagnosis.
"""

TAGS = [
    {"name": "Prediction", "description": "Upload an image and get a diagnosis."},
    {"name": "History", "description": "Stored predictions: list, inspect, delete."},
    {"name": "Research", "description": "Benchmark, ablation, dataset and calibration results."},
    {"name": "Dashboard", "description": "Aggregated statistics over stored predictions."},
    {"name": "System", "description": "Health and readiness probes."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start-up: create tables, load the knowledge base, warm the model."""
    configure_logging(settings.log_level, settings.logs_dir / "api.log")
    settings.ensure_dirs()
    log.info("Starting API", fields={"version": settings.api_version,
                                     "database": settings.database_url})

    init_db()
    log.info("Database ready")

    # Force the JSON load now: a malformed knowledge base should fail at start-up,
    # loudly, rather than on the first user request.
    log.info("Knowledge base ready",
             fields={"classes": knowledge_base.metadata.get("class_count", 0)})

    started = time.perf_counter()
    try:
        loaded = registry.warm_up(None if settings.serving_model == "auto" else settings.serving_model)
        app.state.model_ready = True
        app.state.model_error = None
        log.info("Model warmed up",
                 fields={"model": loaded.name, "device": str(loaded.device),
                         "warmup_ms": round((time.perf_counter() - started) * 1000, 1)})
    except Exception as exc:  # noqa: BLE001
        # The API still starts so the dashboard, docs and dataset endpoints work
        # before a model has been trained; /api/predict returns 503 until then.
        app.state.model_ready = False
        app.state.model_error = str(exc)
        log.warning("Model not available at startup", fields={"error": str(exc)})

    yield
    log.info("Shutting down API")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=DESCRIPTION,
    openapi_tags=TAGS,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    """Attach a request id, time the request and log the outcome.

    Only the method, path, status and duration are logged - never headers,
    bodies or uploaded file contents.
    """
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("Unhandled error",
                      fields={"request_id": request_id, "path": request.url.path,
                              "method": request.method})
        raise
    duration_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-ms"] = f"{duration_ms:.2f}"

    if not request.url.path.startswith(("/api/images", "/api/figures", "/docs", "/openapi")):
        log.info("request",
                 fields={"request_id": request_id, "method": request.method,
                         "path": request.url.path, "status": response.status_code,
                         "ms": round(duration_ms, 2)})
    return response


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Emit a consistent error envelope for every HTTP error."""
    detail = exc.detail
    if isinstance(detail, dict):
        payload = {"error": detail}
    else:
        payload = {"error": {"code": f"http_{exc.status_code}", "message": str(detail)}}
    payload["error"]["status"] = exc.status_code
    payload["error"]["path"] = request.url.path
    return JSONResponse(status_code=exc.status_code, content=payload,
                        headers=getattr(exc, "headers", None))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": {"code": "validation_error",
                           "message": "Request validation failed.",
                           "status": 422, "path": request.url.path,
                           "details": exc.errors()}},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Return a generic 500 without leaking internals to the client.

    The full traceback goes to the log; the response carries only the request
    path and a stable error code.
    """
    log.exception("Internal server error", fields={"path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error",
                           "message": "An internal error occurred. See the server log.",
                           "status": 500, "path": request.url.path}},
    )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/api/health", tags=["System"], summary="Liveness and readiness")
def health():
    manifest = registry.production_manifest()
    return {
        "status": "ok",
        "version": settings.api_version,
        "model_ready": getattr(app.state, "model_ready", False),
        "model_error": getattr(app.state, "model_error", None),
        "production_model": manifest.get("model_name") if manifest else None,
        "device": str(registry.resolve_device()),
        "knowledge_base_classes": knowledge_base.metadata.get("class_count", 0),
        "confidence_thresholds": {
            "high": settings.confidence_high,
            "medium": settings.confidence_medium,
        },
        "limits": {
            "max_upload_bytes": settings.max_upload_size,
            "allowed_extensions": list(settings.allowed_extensions),
        },
    }


@app.get("/", tags=["System"], include_in_schema=False)
def root():
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/api/health",
    }


app.include_router(predictions.router, prefix="/api", tags=["Prediction"])
app.include_router(research.router, prefix="/api", tags=["Research"])
app.include_router(dashboard.router, prefix="/api", tags=["Dashboard"])
