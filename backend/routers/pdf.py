"""PDF processing router — NexaForge"""
import io
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from processors.pdf_processor import (
    compress_pdf,
    increase_pdf_size,
    merge_pdfs,
    split_pdf,
    extract_pages,
    images_to_pdf,
)

router = APIRouter()

MAX_SIZE = 10 * 1024 * 1024  # 10 MB per file


def _validate_pdf(file: UploadFile, data: bytes) -> None:
    if len(data) > MAX_SIZE:
        raise HTTPException(413, "File too large. Maximum size is 10 MB per file.")
    ct = file.content_type or ""
    if "pdf" not in ct and not file.filename.endswith(".pdf"):
        raise HTTPException(415, f"Expected a PDF file, got: {ct}")


def _pdf_stream(result_bytes: bytes, filename: str) -> StreamingResponse:
    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Compress ──────────────────────────────────────────────────────────────────
@router.post("/compress", summary="Compress PDF to target KB")
async def compress(
    file: Annotated[UploadFile, File(description="PDF file to compress")],
    target_kb: Annotated[int, Form(description="Target size in KB", ge=10, le=50000)] = 300,
):
    data = await file.read()
    _validate_pdf(file, data)
    result, fname = compress_pdf(data, target_kb)
    return _pdf_stream(result, fname)


# ── Increase Size ─────────────────────────────────────────────────────────────
@router.post("/increase", summary="Increase PDF size to minimum target KB")
async def increase(
    file: Annotated[UploadFile, File(description="PDF file to expand")],
    target_kb: Annotated[int, Form(description="Minimum target size in KB", ge=10, le=50000)] = 200,
):
    data = await file.read()
    _validate_pdf(file, data)
    result, fname = increase_pdf_size(data, target_kb)
    return _pdf_stream(result, fname)


# ── Merge ─────────────────────────────────────────────────────────────────────
@router.post("/merge", summary="Merge multiple PDFs into one")
async def merge(
    files: Annotated[list[UploadFile], File(description="Two or more PDF files")],
):
    if len(files) < 2:
        raise HTTPException(422, "Upload at least 2 PDF files to merge.")
    pdf_bytes_list = []
    for f in files:
        data = await f.read()
        _validate_pdf(f, data)
        pdf_bytes_list.append(data)
    result, fname = merge_pdfs(pdf_bytes_list)
    return _pdf_stream(result, fname)


# ── Split ─────────────────────────────────────────────────────────────────────
@router.post("/split", summary="Extract a page range from PDF")
async def split(
    file: Annotated[UploadFile, File()],
    start_page: Annotated[int, Form(description="Start page (1-indexed)", ge=1)] = 1,
    end_page: Annotated[int | None, Form(description="End page (inclusive, 1-indexed)")] = None,
):
    data = await file.read()
    _validate_pdf(file, data)
    result, fname = split_pdf(data, start_page, end_page)
    return _pdf_stream(result, fname)


# ── Extract Pages ─────────────────────────────────────────────────────────────
@router.post("/extract-pages", summary="Extract specific pages from PDF")
async def extract(
    file: Annotated[UploadFile, File()],
    pages: Annotated[str, Form(description="Comma-separated 1-indexed page numbers e.g. '1,3,5'")] = "1",
):
    data = await file.read()
    _validate_pdf(file, data)
    try:
        page_list = [int(p.strip()) for p in pages.split(",") if p.strip()]
    except ValueError:
        raise HTTPException(422, "pages must be comma-separated integers e.g. '1,3,5'")
    if not page_list:
        raise HTTPException(422, "Provide at least one page number.")
    result, fname = extract_pages(data, page_list)
    return _pdf_stream(result, fname)


# ── Images → PDF ──────────────────────────────────────────────────────────────
@router.post("/images-to-pdf", summary="Combine images into a single PDF")
async def img_to_pdf(
    files: Annotated[list[UploadFile], File(description="One or more image files")],
):
    if not files:
        raise HTTPException(422, "Upload at least one image file.")
    image_bytes_list = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_SIZE:
            raise HTTPException(413, f"File '{f.filename}' is too large (max 10 MB).")
        ct = f.content_type or ""
        if not ct.startswith("image/"):
            raise HTTPException(415, f"Expected image files, got: {ct} for {f.filename}")
        image_bytes_list.append(data)
    result, fname = images_to_pdf(image_bytes_list)
    return _pdf_stream(result, fname)
