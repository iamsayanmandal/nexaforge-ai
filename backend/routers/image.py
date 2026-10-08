"""Image processing router — NexaForge"""
import io
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from processors.image_processor import (
    compress_image,
    increase_image_size,
    resize_image,
    convert_image,
    rotate_image,
)

router = APIRouter()

MAX_SIZE = 10 * 1024 * 1024  # 10 MB


def _validate(file: UploadFile, data: bytes) -> None:
    if len(data) > MAX_SIZE:
        raise HTTPException(413, "File too large. Maximum size is 10 MB.")
    ct = file.content_type or ""
    if not ct.startswith("image/"):
        raise HTTPException(415, f"Expected an image file, got: {ct}")


def _stream(result_bytes: bytes, filename: str, media_type: str = "image/jpeg") -> StreamingResponse:
    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Compress ──────────────────────────────────────────────────────────────────
@router.post("/compress", summary="Compress image to target KB")
async def compress(
    file: Annotated[UploadFile, File(description="Image file to compress")],
    target_kb: Annotated[int, Form(description="Target file size in KB", ge=1, le=10000)] = 100,
):
    data = await file.read()
    _validate(file, data)
    result, fname = compress_image(data, target_kb)
    return _stream(result, fname)


# ── Increase ──────────────────────────────────────────────────────────────────
@router.post("/increase", summary="Increase image file size to minimum target KB")
async def increase(
    file: Annotated[UploadFile, File(description="Image file to expand")],
    target_kb: Annotated[int, Form(description="Minimum target size in KB", ge=1, le=50000)] = 200,
):
    data = await file.read()
    _validate(file, data)
    result, fname = increase_image_size(data, target_kb)
    return _stream(result, fname)


# ── Resize ────────────────────────────────────────────────────────────────────
@router.post("/resize", summary="Resize image dimensions")
async def resize(
    file: Annotated[UploadFile, File()],
    width: Annotated[int | None, Form(ge=1, le=10000)] = None,
    height: Annotated[int | None, Form(ge=1, le=10000)] = None,
    scale_percent: Annotated[int | None, Form(ge=1, le=500)] = None,
):
    data = await file.read()
    _validate(file, data)
    if not any([width, height, scale_percent]):
        raise HTTPException(422, "Provide at least one of: width, height, scale_percent.")
    result, fname = resize_image(data, width, height, scale_percent)
    ext = fname.rsplit(".", 1)[-1]
    mt = "image/png" if ext == "png" else "image/webp" if ext == "webp" else "image/jpeg"
    return _stream(result, fname, mt)


# ── Convert ───────────────────────────────────────────────────────────────────
@router.post("/convert", summary="Convert image format (JPG / PNG / WEBP)")
async def convert(
    file: Annotated[UploadFile, File()],
    target_format: Annotated[str, Form(description="JPEG | PNG | WEBP")] = "PNG",
):
    data = await file.read()
    _validate(file, data)
    fmt = target_format.upper().replace("JPG", "JPEG")
    if fmt not in ("JPEG", "PNG", "WEBP"):
        raise HTTPException(422, "target_format must be JPEG, PNG, or WEBP.")
    result, fname = convert_image(data, fmt)  # type: ignore[arg-type]
    mt = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}[fmt]
    return _stream(result, fname, mt)


# ── Rotate ────────────────────────────────────────────────────────────────────
@router.post("/rotate", summary="Rotate image 90 / 180 / 270 degrees")
async def rotate(
    file: Annotated[UploadFile, File()],
    degrees: Annotated[int, Form(description="90 | 180 | 270")] = 90,
):
    data = await file.read()
    _validate(file, data)
    if degrees not in (90, 180, 270):
        raise HTTPException(422, "degrees must be 90, 180, or 270.")
    result, fname = rotate_image(data, degrees)
    ext = fname.rsplit(".", 1)[-1]
    mt = "image/png" if ext == "png" else "image/webp" if ext == "webp" else "image/jpeg"
    return _stream(result, fname, mt)
