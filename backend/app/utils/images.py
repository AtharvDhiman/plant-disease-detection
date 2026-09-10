"""Upload validation and safe image handling.

Every upload is treated as hostile until proven otherwise:

* the declared content type **and** the decoded image format are both checked -
  an attacker controls the filename and the ``Content-Type`` header, but not
  what Pillow finds inside the bytes;
* the size limit is enforced while streaming, so an oversized body is rejected
  before it is fully buffered in memory;
* the stored filename is generated server-side from a UUID, never derived from
  user input, which removes path-traversal and overwrite risks entirely;
* images are re-encoded before being saved, which strips EXIF (including GPS
  coordinates) and guarantees the stored bytes are a real image rather than a
  polyglot file.
"""
from __future__ import annotations

import io
import uuid
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import settings

# Pillow format names corresponding to the configured MIME allow-list.
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}

# Refuse absurd pixel counts before decoding: a small compressed file can expand
# to gigabytes of RGB data ("decompression bomb").
MAX_PIXELS = 60_000_000


class UploadError(ValueError):
    """Raised when an upload fails validation. Carries an HTTP-ready detail."""

    def __init__(self, code: str, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def as_detail(self) -> dict:
        return {"code": self.code, "message": self.message, "hint": self.hint}


def validate_upload(filename: str | None, content_type: str | None, size: int) -> None:
    """Check the declared metadata before the bytes are decoded."""
    if size <= 0:
        raise UploadError("empty_file", "The uploaded file is empty.")
    if size > settings.max_upload_size:
        limit_mb = settings.max_upload_size / 1024**2
        raise UploadError(
            "file_too_large",
            f"File is {size / 1024**2:.1f} MB; the limit is {limit_mb:.0f} MB.",
            "Resize the photo or reduce its quality before uploading.",
        )
    if content_type and content_type.lower().split(";")[0] not in settings.allowed_content_types:
        raise UploadError(
            "unsupported_content_type",
            f"Content type {content_type!r} is not supported.",
            f"Allowed types: {', '.join(settings.allowed_content_types)}.",
        )
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix and suffix not in settings.allowed_extensions:
            raise UploadError(
                "unsupported_extension",
                f"File extension {suffix!r} is not supported.",
                f"Allowed extensions: {', '.join(settings.allowed_extensions)}.",
            )


def decode_image(data: bytes) -> Image.Image:
    """Decode bytes into a PIL image, verifying the real format."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            width, height = probe.size
    except UnidentifiedImageError as exc:
        raise UploadError(
            "not_an_image",
            "The uploaded file could not be decoded as an image.",
            "Upload a JPG, PNG or WEBP photograph.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise UploadError("corrupt_image", f"The image could not be read: {exc}") from exc

    if width <= 0 or height <= 0:
        raise UploadError(
            "corrupt_image",
            f"Image dimensions are invalid ({width}x{height}).",
            "Upload a valid JPG, PNG or WEBP photograph.",
        )
    if fmt not in ALLOWED_PIL_FORMATS:
        raise UploadError(
            "unsupported_image_format",
            f"Image format {fmt!r} is not supported.",
            "Upload a JPG, PNG or WEBP photograph.",
        )
    if width * height > MAX_PIXELS:
        raise UploadError(
            "image_too_large",
            f"Image is {width}x{height} ({width * height / 1e6:.0f} MP); "
            f"the limit is {MAX_PIXELS / 1e6:.0f} MP.",
            "Resize the photo before uploading.",
        )

    # Re-open for the actual decode; `Image.open` above only read the header.
    image = Image.open(io.BytesIO(data))
    # Honour the EXIF orientation flag, then discard EXIF entirely on save.
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def safe_filename(extension: str = ".jpg") -> str:
    """Generate a collision-free, user-input-free filename."""
    return f"{uuid.uuid4().hex}{extension}"


def store_image(image: Image.Image, directory: Path, stem: str, max_side: int = 640) -> tuple[str, int]:
    """Re-encode and save an upload; returns ``(filename, bytes_written)``.

    Downscaling to ``max_side`` keeps the uploads directory small - the model
    only ever sees 224 px - while leaving enough resolution for the user to
    review the image in the history view.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stored = image.copy()
    if max(stored.size) > max_side:
        resample = getattr(Image, "Resampling", Image).LANCZOS
        stored.thumbnail((max_side, max_side), resample)

    filename = f"{stem}.jpg"
    path = directory / filename
    # `save` on a fresh image object writes no EXIF, so GPS data in the original
    # never reaches disk.
    stored.save(path, format="JPEG", quality=88, optimize=True)
    return filename, path.stat().st_size


def resolve_upload_path(filename: str) -> Path:
    """Resolve a stored filename inside the uploads directory, rejecting traversal."""
    root = settings.upload_dir.resolve()
    candidate = (settings.upload_dir / filename).resolve()
    if not candidate.is_relative_to(root):
        raise UploadError("invalid_path", "Invalid image path.")
    if not candidate.is_file():
        raise UploadError("not_found", f"Image {filename!r} does not exist.")
    return candidate
