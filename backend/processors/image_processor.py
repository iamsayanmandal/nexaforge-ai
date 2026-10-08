"""
Image Processor — NexaForge
Deterministic, in-memory image processing using Pillow.
Supports adaptive quality-preserving compression, resizing, format conversion,
rotation, and size expansion/padding.
"""
from __future__ import annotations

import io
from typing import Literal
from PIL import Image, ImageOps


# ── Helpers ──────────────────────────────────────────────────────────────────

def _open(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes))


def _to_bytes(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> bytes:
    buf = io.BytesIO()
    save_kwargs = {"format": fmt}
    if fmt in ("JPEG", "WEBP"):
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    elif fmt == "PNG":
        save_kwargs["optimize"] = True
    img.save(buf, **save_kwargs)
    return buf.getvalue()


def _ensure_rgb(img: Image.Image) -> Image.Image:
    """Convert RGBA/P/LA to RGB (required for JPEG/WEBP output)."""
    if img.mode in ("RGBA", "P", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        return background
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


# ── Feature 1: Intelligent Adaptive Compression ───────────────────────────────

def compress_image(image_bytes: bytes, target_kb: int = 100) -> tuple[bytes, str]:
    """
    Compress an image to <= target_kb while preserving high visual fidelity.
    Instead of dropping JPEG quality to severe artifacts (<60), it adaptively
    scales down image dimensions using Lanczos resampling and maintains crisp 70-85 quality.
    """
    orig_img = _open(image_bytes)
    orig_img = _ensure_rgb(orig_img)
    target_bytes = target_kb * 1024

    # Pass 1: Try binary search on quality while maintaining original dimensions
    low, high, best_bytes = 65, 95, None
    for _ in range(8):
        mid = (low + high) // 2
        buf = io.BytesIO()
        orig_img.save(buf, format="JPEG", quality=mid, optimize=True)
        if buf.tell() <= target_bytes:
            best_bytes = buf.getvalue()
            low = mid + 1
        else:
            high = mid - 1
        if low > high:
            break

    if best_bytes is not None:
        return best_bytes, f"compressed_{target_kb}kb.jpg"

    # Pass 2: If quality >= 65 is still too large, downsample dimensions adaptively
    # but keep quality crisp (75) to avoid ugly JPEG block artifacts.
    scale_factors = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]
    for scale in scale_factors:
        new_w = max(100, int(orig_img.width * scale))
        new_h = max(100, int(orig_img.height * scale))
        scaled_img = orig_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        for quality in (82, 75, 68, 60):
            buf = io.BytesIO()
            scaled_img.save(buf, format="JPEG", quality=quality, optimize=True)
            if buf.tell() <= target_bytes:
                return buf.getvalue(), f"compressed_{target_kb}kb.jpg"

    # Fallback if extremely small target requested: lowest acceptable threshold
    buf = io.BytesIO()
    min_img = orig_img.resize((max(120, int(orig_img.width * 0.15)), max(120, int(orig_img.height * 0.15))), Image.Resampling.LANCZOS)
    min_img.save(buf, format="JPEG", quality=55, optimize=True)
    return buf.getvalue(), f"compressed_{target_kb}kb.jpg"


# ── Feature 2: Increase / Pad Image File Size ─────────────────────────────────

def increase_image_size(image_bytes: bytes, target_kb: int = 200) -> tuple[bytes, str]:
    """
    Increase image file size to >= target_kb (for portal requirements).
    Uses high-quality re-encoding and benign padding blocks without degrading visuals.
    """
    target_bytes = target_kb * 1024
    if len(image_bytes) >= target_bytes:
        return image_bytes, f"expanded_{target_kb}kb.jpg"

    img = _open(image_bytes)
    img = _ensure_rgb(img)

    # 1. Try maximum quality encoding
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=98, subsampling=0)
    current_bytes = buf.getvalue()

    if len(current_bytes) >= target_bytes:
        return current_bytes, f"expanded_{target_kb}kb.jpg"

    # 2. Append standard safe non-destructive JPEG comment block padding
    needed_pad = target_bytes - len(current_bytes)
    # JPEG Comment segment marker is 0xFF 0xFE followed by 2-byte length
    # Max comment segment length is 65535 bytes
    padded_stream = io.BytesIO()
    # Write initial bytes up to before End-Of-Image (EOI 0xFFD9)
    if current_bytes.endswith(b"\xff\xd9"):
        base_body = current_bytes[:-2]
    else:
        base_body = current_bytes

    padded_stream.write(base_body)

    remaining = needed_pad
    while remaining > 0:
        chunk_size = min(remaining, 65000)
        marker = b"\xff\xfe"
        length_bytes = (chunk_size + 2).to_bytes(2, byteorder="big")
        pad_data = b"0" * chunk_size
        padded_stream.write(marker + length_bytes + pad_data)
        remaining -= chunk_size

    padded_stream.write(b"\xff\xd9")  # Re-attach EOI marker
    return padded_stream.getvalue(), f"expanded_{target_kb}kb.jpg"


# ── Feature 3: Resize ─────────────────────────────────────────────────────────

def resize_image(
    image_bytes: bytes,
    width: int | None = None,
    height: int | None = None,
    scale_percent: int | None = None,
) -> tuple[bytes, str]:
    """
    Resize image to exact dimensions OR by scale percentage.
    Maintains aspect ratio when only one dimension is given.
    """
    img = _open(image_bytes)
    orig_fmt = img.format or "JPEG"

    if scale_percent is not None:
        factor = scale_percent / 100
        new_w = max(1, int(img.width * factor))
        new_h = max(1, int(img.height * factor))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        suffix = f"resized_{scale_percent}pct"
    elif width and height:
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        suffix = f"resized_{width}x{height}"
    elif width:
        ratio = width / img.width
        img = img.resize((width, max(1, int(img.height * ratio))), Image.Resampling.LANCZOS)
        suffix = f"resized_w{width}"
    elif height:
        ratio = height / img.height
        img = img.resize((max(1, int(img.width * ratio)), height), Image.Resampling.LANCZOS)
        suffix = f"resized_h{height}"
    else:
        raise ValueError("Provide width, height, or scale_percent.")

    fmt = orig_fmt if orig_fmt in ("JPEG", "PNG", "WEBP") else "JPEG"
    img = _ensure_rgb(img) if fmt in ("JPEG", "WEBP") else img
    ext = fmt.lower().replace("jpeg", "jpg")
    return _to_bytes(img, fmt), f"{suffix}.{ext}"


# ── Feature 4: Convert Format ─────────────────────────────────────────────────

FormatType = Literal["JPEG", "PNG", "WEBP"]

def convert_image(image_bytes: bytes, target_format: FormatType = "PNG") -> tuple[bytes, str]:
    """Convert image to JPG, PNG, or WEBP."""
    fmt = target_format.upper()
    if fmt == "JPG":
        fmt = "JPEG"
    img = _open(image_bytes)
    if fmt in ("JPEG", "WEBP"):
        img = _ensure_rgb(img)
    ext = "jpg" if fmt == "JPEG" else fmt.lower()
    return _to_bytes(img, fmt), f"converted.{ext}"


# ── Feature 5: Rotate ─────────────────────────────────────────────────────────

def rotate_image(image_bytes: bytes, degrees: int = 90) -> tuple[bytes, str]:
    """Rotate image clockwise by specified degrees."""
    img = _open(image_bytes)
    orig_fmt = img.format or "JPEG"
    img = img.rotate(-degrees, expand=True)
    fmt = orig_fmt if orig_fmt in ("JPEG", "PNG", "WEBP") else "JPEG"
    if fmt in ("JPEG", "WEBP"):
        img = _ensure_rgb(img)
    ext = "jpg" if fmt == "JPEG" else fmt.lower()
    return _to_bytes(img, fmt), f"rotated_{degrees}deg.{ext}"
