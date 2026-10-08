"""Tests for pdf_processor — NexaForge CI/CD"""
import io
import pytest
import fitz  # PyMuPDF
from PIL import Image

from processors.pdf_processor import (
    compress_pdf,
    increase_pdf_size,
    merge_pdfs,
    split_pdf,
    extract_pages,
    images_to_pdf,
)


def _make_pdf(num_pages: int = 3, with_image: bool = False) -> bytes:
    """Create a multi-page test PDF in memory."""
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1} content for NexaForge test")
        if with_image:
            # Insert a dummy image
            img = Image.new("RGB", (300, 300), color=(50, 100, 150))
            img_buf = io.BytesIO()
            img.save(img_buf, format="JPEG")
            page.insert_image(fitz.Rect(100, 100, 400, 400), stream=img_buf.getvalue())

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_image_bytes(fmt="JPEG", color=(255, 128, 0)) -> bytes:
    img = Image.new("RGB", (400, 300), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ── compress_pdf ─────────────────────────────────────────────────────────────

def test_compress_pdf_valid():
    raw_pdf = _make_pdf(num_pages=3, with_image=True)
    compressed, fname = compress_pdf(raw_pdf, target_kb=500)
    assert isinstance(compressed, bytes)
    assert len(compressed) > 0
    assert fname.endswith(".pdf")
    # Verify resulting PDF is valid
    doc = fitz.open(stream=compressed, filetype="pdf")
    assert len(doc) == 3
    doc.close()


def test_increase_pdf_size():
    raw_pdf = _make_pdf(num_pages=2)
    orig_len = len(raw_pdf)
    expanded, fname = increase_pdf_size(raw_pdf, target_kb=100)
    assert len(expanded) >= 100 * 1024
    assert len(expanded) > orig_len
    doc = fitz.open(stream=expanded, filetype="pdf")
    assert len(doc) == 2
    doc.close()


# ── merge_pdfs ───────────────────────────────────────────────────────────────

def test_merge_pdfs():
    pdf1 = _make_pdf(num_pages=2)
    pdf2 = _make_pdf(num_pages=3)
    merged, fname = merge_pdfs([pdf1, pdf2])
    assert fname == "merged.pdf"
    doc = fitz.open(stream=merged, filetype="pdf")
    assert len(doc) == 5
    doc.close()


# ── split_pdf ────────────────────────────────────────────────────────────────

def test_split_pdf_range():
    pdf = _make_pdf(num_pages=5)
    split, fname = split_pdf(pdf, start_page=2, end_page=4)
    doc = fitz.open(stream=split, filetype="pdf")
    assert len(doc) == 3  # pages 2, 3, 4
    doc.close()
    assert "split_pages_2_to_4" in fname


def test_split_pdf_to_end():
    pdf = _make_pdf(num_pages=4)
    split, _ = split_pdf(pdf, start_page=3, end_page=None)
    doc = fitz.open(stream=split, filetype="pdf")
    assert len(doc) == 2  # pages 3, 4
    doc.close()


# ── extract_pages ────────────────────────────────────────────────────────────

def test_extract_pages_specific():
    pdf = _make_pdf(num_pages=5)
    extracted, fname = extract_pages(pdf, pages=[1, 3, 5])
    doc = fitz.open(stream=extracted, filetype="pdf")
    assert len(doc) == 3
    doc.close()
    assert "pages_1_3_5" in fname


def test_extract_pages_single():
    pdf = _make_pdf(num_pages=4)
    extracted, _ = extract_pages(pdf, pages=[2])
    doc = fitz.open(stream=extracted, filetype="pdf")
    assert len(doc) == 1
    doc.close()


# ── images_to_pdf ────────────────────────────────────────────────────────────

def test_images_to_pdf():
    img1 = _make_image_bytes(fmt="JPEG", color=(200, 50, 50))
    img2 = _make_image_bytes(fmt="PNG", color=(50, 200, 50))
    pdf_bytes, fname = images_to_pdf([img1, img2])
    assert fname == "images_combined.pdf"
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    assert len(doc) == 2
    doc.close()


def test_images_to_pdf_empty_raises():
    with pytest.raises(ValueError):
        images_to_pdf([])
