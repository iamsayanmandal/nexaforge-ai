"""
PDF Processor — NexaForge
All PDF operations using PyMuPDF (fitz) + pikepdf. In-memory only.
Includes adaptive quality-preserving compression, file size expansion/padding,
merging, splitting, extracting pages, and converting images to PDF.
"""
from __future__ import annotations

import io
from PIL import Image
import fitz          # PyMuPDF
import pikepdf


# ── Feature 1: Intelligent Adaptive PDF Compression ──────────────────────────

def compress_pdf(pdf_bytes: bytes, target_kb: int = 300) -> tuple[bytes, str]:
    """
    Compress a PDF to <= target_kb while maintaining document and image clarity.
    Strategy:
      1. Lossless stream deflation, object deduplication, and garbage collection.
      2. If target is still exceeded, downsample high-resolution embedded images
         proportionally using Lanczos resampling while keeping quality at 70-80,
         avoiding severe blocky artifacts.
    """
    target_bytes = target_kb * 1024
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Step 1: Lossless deflation + garbage cleanup
    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True, deflate_images=True, deflate_fonts=True)
    if buf.tell() <= target_bytes:
        doc.close()
        return buf.getvalue(), f"compressed_{target_kb}kb.pdf"

    # Step 2: Adaptive image downsampling with high quality preservation
    # Test scales: progressively downsample dimensions, but retain good JPEG quality (75-80)
    for max_dim in (1600, 1200, 900, 700, 500):
        for page in doc:
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                try:
                    extracted = doc.extract_image(xref)
                    pil_img = Image.open(io.BytesIO(extracted["image"]))
                    w, h = pil_img.size

                    # Downsample only if larger than target max dimension
                    if w > max_dim or h > max_dim:
                        scale = min(max_dim / w, max_dim / h)
                        new_w = max(50, int(w * scale))
                        new_h = max(50, int(h * scale))
                        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                    if pil_img.mode in ("RGBA", "P"):
                        pil_img = pil_img.convert("RGB")

                    img_buf = io.BytesIO()
                    # Keep quality at 75-80 to avoid ugly block compression artifacts
                    pil_img.save(img_buf, format="JPEG", quality=75, optimize=True)
                    doc.update_stream(xref, img_buf.getvalue())
                except Exception:
                    continue

        buf = io.BytesIO()
        doc.save(buf, garbage=4, deflate=True, deflate_images=True)
        if buf.tell() <= target_bytes:
            break

    doc.close()
    return buf.getvalue(), f"compressed_{target_kb}kb.pdf"


# ── Feature 2: Increase / Pad PDF File Size ───────────────────────────────────

def increase_pdf_size(pdf_bytes: bytes, target_kb: int = 200) -> tuple[bytes, str]:
    """
    Increase PDF file size to >= target_kb to satisfy minimum upload limits
    required by portals without affecting document appearance or validity.
    """
    # Validate PDF structure first
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    doc.close()

    target_bytes = target_kb * 1024
    if len(pdf_bytes) >= target_bytes:
        return pdf_bytes, f"expanded_{target_kb}kb.pdf"

    needed_bytes = target_bytes - len(pdf_bytes)
    prefix = b"\n% --- NexaForge Safe Document Padding ---\n% "
    if needed_bytes > len(prefix):
        fill = b"0" * (needed_bytes - len(prefix))
        padded_pdf = pdf_bytes + prefix + fill
    else:
        padded_pdf = pdf_bytes + (b" " * needed_bytes)

    if len(padded_pdf) < target_bytes:
        padded_pdf += b"0" * (target_bytes - len(padded_pdf))

    return padded_pdf, f"expanded_{target_kb}kb.pdf"


# ── Feature 3: Merge PDFs ────────────────────────────────────────────────────

def merge_pdfs(pdf_bytes_list: list[bytes]) -> tuple[bytes, str]:
    """Merge multiple PDFs into a single PDF."""
    output = pikepdf.Pdf.new()
    for raw in pdf_bytes_list:
        src = pikepdf.Pdf.open(io.BytesIO(raw))
        output.pages.extend(src.pages)
    buf = io.BytesIO()
    output.save(buf)
    return buf.getvalue(), "merged.pdf"


# ── Feature 4: Split PDF ──────────────────────────────────────────────────────

def split_pdf(pdf_bytes: bytes, start_page: int = 1, end_page: int | None = None) -> tuple[bytes, str]:
    """
    Extract a page range from PDF.
    Pages are 1-indexed. end_page=None means go to last page.
    """
    src = pikepdf.Pdf.open(io.BytesIO(pdf_bytes))
    total = len(src.pages)
    s = max(1, start_page) - 1
    e = min(end_page or total, total)

    output = pikepdf.Pdf.new()
    output.pages.extend(src.pages[s:e])
    buf = io.BytesIO()
    output.save(buf)
    return buf.getvalue(), f"split_pages_{start_page}_to_{e}.pdf"


# ── Feature 5: Extract Specific Pages ────────────────────────────────────────

def extract_pages(pdf_bytes: bytes, pages: list[int]) -> tuple[bytes, str]:
    """Extract specific pages by 1-indexed page numbers."""
    src = pikepdf.Pdf.open(io.BytesIO(pdf_bytes))
    total = len(src.pages)
    output = pikepdf.Pdf.new()

    for page_num in pages:
        idx = page_num - 1
        if 0 <= idx < total:
            output.pages.append(src.pages[idx])

    buf = io.BytesIO()
    output.save(buf)
    pages_str = "_".join(str(p) for p in pages[:5])
    return buf.getvalue(), f"pages_{pages_str}.pdf"


# ── Feature 6: Images to PDF ──────────────────────────────────────────────────

def images_to_pdf(image_bytes_list: list[bytes]) -> tuple[bytes, str]:
    """Combine multiple images into a single PDF file."""
    pil_images = []
    for raw in image_bytes_list:
        img = Image.open(io.BytesIO(raw))
        if img.mode in ("RGBA", "P", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        pil_images.append(img)

    if not pil_images:
        raise ValueError("No valid images provided.")

    buf = io.BytesIO()
    pil_images[0].save(
        buf,
        format="PDF",
        save_all=True,
        append_images=pil_images[1:],
    )
    return buf.getvalue(), "images_combined.pdf"
