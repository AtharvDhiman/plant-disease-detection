"""Uploads are decoded small enough for the 512 MB host, without changing what the model sees."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

import app.core.runtime  # noqa: F401  # isort:skip
from app.utils.images import DECODE_MAX_SIDE, decode_image


def _jpeg(width: int, height: int, exif_orientation: int | None = None) -> bytes:
    rng = np.random.default_rng(0)
    pixels = (rng.random((height // 16, width // 16, 3)) * 255).astype("uint8")
    image = Image.fromarray(pixels).resize((width, height), Image.BILINEAR)
    buffer = io.BytesIO()
    if exif_orientation is None:
        image.save(buffer, format="JPEG", quality=90)
    else:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        image.save(buffer, format="JPEG", quality=90, exif=exif.tobytes())
    return buffer.getvalue()


def test_a_large_photo_is_decoded_no_bigger_than_the_cap_keeping_its_shape():
    image = decode_image(_jpeg(5312, 2988))
    assert max(image.size) == DECODE_MAX_SIDE
    assert abs(image.size[0] / image.size[1] - 5312 / 2988) < 0.01
    assert image.mode == "RGB"


def test_a_small_photo_is_left_at_its_own_size():
    assert decode_image(_jpeg(640, 480)).size == (640, 480)


def test_exif_orientation_is_still_honoured_after_reduced_decoding():
    # Orientation 6 = rotate 90 degrees clockwise: a landscape file shows as portrait.
    image = decode_image(_jpeg(4000, 3000, exif_orientation=6))
    assert image.size[1] > image.size[0]
    assert max(image.size) == DECODE_MAX_SIDE


def test_png_uploads_are_capped_too():
    buffer = io.BytesIO()
    Image.new("RGB", (3000, 2000), (60, 140, 60)).save(buffer, format="PNG")
    assert max(decode_image(buffer.getvalue()).size) == DECODE_MAX_SIDE
