"""Shared pytest fixtures.

The tests run against a **temporary** database and uploads directory so they can
never touch the developer's real prediction history. The redirection happens
through environment variables set *before* anything from ``app`` is imported:
the settings object and the SQLAlchemy engine are both built at import time, so
patching them afterwards would be too late.

The model itself is the real exported checkpoint - these are integration tests,
not mock tests, because the thing most worth testing is that preprocessing, the
model and the response schema actually agree.
"""
from __future__ import annotations

import atexit
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

# --------------------------------------------------------------------------- #
# Environment redirection - must happen before the first `app.*` import
# --------------------------------------------------------------------------- #

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="pdd_tests_"))
(_TEST_ROOT / "uploads").mkdir(parents=True, exist_ok=True)

os.environ["PDD_DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'test.db').as_posix()}"
os.environ["PDD_UPLOAD_DIR"] = str(_TEST_ROOT / "uploads")
os.environ["PDD_LOG_LEVEL"] = "WARNING"


@atexit.register
def _cleanup_test_root() -> None:
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)


import app.core.runtime  # noqa: F401,E402  # isort:skip

from app.core.config import settings  # noqa: E402


@pytest.fixture(scope="session")
def test_root() -> Path:
    return _TEST_ROOT


@pytest.fixture(scope="session", autouse=True)
def _verify_isolation():
    """Fail loudly rather than write test data into the developer's database."""
    # The URL is built with as_posix(), so compare in posix form on Windows too.
    assert _TEST_ROOT.as_posix() in settings.database_url, (
        f"Test isolation failed: database_url is {settings.database_url!r}. "
        "conftest must be imported before any app module."
    )
    assert settings.upload_dir.resolve() == (_TEST_ROOT / "uploads").resolve()
    yield


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def model_available(client) -> bool:
    """Whether an exported model is present; model tests skip when it is not."""
    return bool(client.get("/api/health").json().get("model_ready", False))


# --------------------------------------------------------------------------- #
# Image fixtures
# --------------------------------------------------------------------------- #


def _encode(image, fmt: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture(scope="session")
def real_leaf_bytes() -> bytes:
    """A genuine leaf image from the test manifest, or a synthetic stand-in."""
    import csv

    import numpy as np
    from PIL import Image

    manifest = PROJECT_ROOT / "data" / "manifests" / "test_bench.csv"
    if manifest.exists():
        with manifest.open(encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        path = Path(row["path"])
        if path.exists():
            return path.read_bytes()

    # Fallback: a green textured square so the suite still runs without the dataset.
    rng = np.random.default_rng(0)
    base = np.zeros((256, 256, 3), dtype=np.uint8)
    base[:, :, 1] = 150
    base = np.clip(base.astype(int) + rng.integers(0, 60, base.shape), 0, 255).astype("uint8")
    return _encode(Image.fromarray(base))


@pytest.fixture
def blurred_bytes(real_leaf_bytes) -> bytes:
    from PIL import Image, ImageFilter

    image = Image.open(io.BytesIO(real_leaf_bytes)).convert("RGB")
    return _encode(image.filter(ImageFilter.GaussianBlur(9)))


@pytest.fixture
def dark_bytes(real_leaf_bytes) -> bytes:
    import numpy as np
    from PIL import Image

    image = Image.open(io.BytesIO(real_leaf_bytes)).convert("RGB")
    return _encode(Image.fromarray((np.asarray(image) * 0.08).astype("uint8")))


@pytest.fixture
def tiny_bytes(real_leaf_bytes) -> bytes:
    from PIL import Image

    image = Image.open(io.BytesIO(real_leaf_bytes)).convert("RGB")
    return _encode(image.resize((32, 32)))


@pytest.fixture
def noise_bytes() -> bytes:
    """Sharp, colourful noise - passes the quality checks but is not a leaf."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(7)
    return _encode(Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)))


@pytest.fixture
def not_an_image() -> bytes:
    return b"This is definitely not a PNG or a JPEG, it is plain text."


@pytest.fixture
def oversized_bytes() -> bytes:
    """Bytes larger than the configured upload limit."""
    return b"\xff\xd8\xff\xe0" + b"0" * (settings.max_upload_size + 2048)
