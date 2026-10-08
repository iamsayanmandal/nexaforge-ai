"""Tests for image_processor — NexaForge CI/CD"""
import io
import pytest
from PIL import Image

from processors.image_processor import (
    compress_image,
    increase_image_size,
    resize_image,
    convert_image,
    rotate_image,
)


def _make_image(width=800, height=600, fmt="JPEG") -> bytes:
    """Create a test image in memory."""
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ── compress_image ────────────────────────────────────────────────────────────

def test_compress_image_under_target():
    data = _make_image(2000, 2000)
    result, fname = compress_image(data, target_kb=50)
    assert len(result) <= 50 * 1024, "Result exceeds target_kb"
    assert fname.endswith(".jpg")


def test_compress_image_small_target():
    data = _make_image()
    result, _ = compress_image(data, target_kb=5)
    # May exceed 5KB for large images, but should be very small
    assert len(result) < len(data), "Result should be smaller than original"


def test_compress_image_returns_bytes():
    data = _make_image()
    result, fname = compress_image(data, target_kb=100)
    assert isinstance(result, bytes)
    assert isinstance(fname, str)


def test_increase_image_size():
    data = _make_image(200, 200)
    orig_len = len(data)
    result, fname = increase_image_size(data, target_kb=50)
    assert len(result) >= 50 * 1024
    assert len(result) > orig_len
    # Image should still be valid and openable
    img = Image.open(io.BytesIO(result))
    assert img.size == (200, 200)


# ── resize_image ──────────────────────────────────────────────────────────────

def test_resize_exact_dimensions():
    data = _make_image(1000, 800)
    result, fname = resize_image(data, width=400, height=300)
    img = Image.open(io.BytesIO(result))
    assert img.size == (400, 300)


def test_resize_width_only_keeps_aspect():
    data = _make_image(1000, 500)  # 2:1 ratio
    result, _ = resize_image(data, width=500)
    img = Image.open(io.BytesIO(result))
    assert img.width == 500
    assert img.height == 250  # maintains 2:1 ratio


def test_resize_scale_percent():
    data = _make_image(1000, 800)
    result, _ = resize_image(data, scale_percent=50)
    img = Image.open(io.BytesIO(result))
    assert img.size == (500, 400)


def test_resize_no_params_raises():
    data = _make_image()
    with pytest.raises(ValueError):
        resize_image(data)


# ── convert_image ─────────────────────────────────────────────────────────────

def test_convert_to_png():
    data = _make_image(fmt="JPEG")
    result, fname = convert_image(data, "PNG")
    img = Image.open(io.BytesIO(result))
    assert img.format == "PNG"
    assert fname.endswith(".png")


def test_convert_to_webp():
    data = _make_image()
    result, fname = convert_image(data, "WEBP")
    img = Image.open(io.BytesIO(result))
    assert img.format == "WEBP"
    assert fname.endswith(".webp")


def test_convert_to_jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (100, 100)).save(buf, format="PNG")
    data = buf.getvalue()
    result, fname = convert_image(data, "JPEG")
    img = Image.open(io.BytesIO(result))
    assert img.format == "JPEG"


# ── rotate_image ──────────────────────────────────────────────────────────────

def test_rotate_90_swaps_dimensions():
    data = _make_image(800, 600)
    result, fname = rotate_image(data, 90)
    img = Image.open(io.BytesIO(result))
    assert img.width == 600
    assert img.height == 800
    assert "90" in fname


def test_rotate_180_keeps_dimensions():
    data = _make_image(800, 600)
    result, _ = rotate_image(data, 180)
    img = Image.open(io.BytesIO(result))
    assert img.size == (800, 600)


def test_rotate_270_swaps_dimensions():
    data = _make_image(800, 600)
    result, _ = rotate_image(data, 270)
    img = Image.open(io.BytesIO(result))
    assert img.width == 600
    assert img.height == 800
